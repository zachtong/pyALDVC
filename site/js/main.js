// pyALDVC website: small progressive enhancements, no dependencies.
// Without this file the page still works: every tab panel shows, each clip has a <noscript> copy
// with a poster and native controls, and the zoom links open the full images. It adds the mobile
// menu, copy buttons, clip playback, tabs, an image lightbox and the active nav link. Each part
// runs on its own, so one failing does not take the others down.
(() => {
  "use strict";

  const hasIO = "IntersectionObserver" in window;

  /* ---------------------------------------------------------------- mobile menu */
  function initMenu() {
    const button = document.querySelector(".menu-button");
    const nav = document.getElementById("site-nav");
    if (!button || !nav) return;
    // While the menu is open, a scrim (html.menu-open, see style.css) dims the page; a click on it closes the menu.
    const setOpen = (open) => {
      button.setAttribute("aria-expanded", String(open));
      nav.classList.toggle("is-open", open);
      document.documentElement.classList.toggle("menu-open", open);
    };
    button.addEventListener("click", () => setOpen(button.getAttribute("aria-expanded") !== "true"));
    nav.addEventListener("click", (event) => {
      if (event.target.closest("a")) setOpen(false);
    });
    document.addEventListener("click", (event) => {
      if (nav.classList.contains("is-open") && !nav.contains(event.target) && !button.contains(event.target)) setOpen(false);
    });
    // The menu button disappears above the phone breakpoint: never leave the scrim behind.
    const wide = window.matchMedia("(min-width: 981px)");
    const onWide = () => { if (wide.matches) setOpen(false); };
    if (wide.addEventListener) wide.addEventListener("change", onWide);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && nav.classList.contains("is-open")) {
        setOpen(false);
        button.focus();
      }
    });
  }

  /* ---------------------------------------------------------------- copy buttons */
  async function writeClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return;
      } catch (err) {
        console.warn("Clipboard API refused, trying the fallback:", err);
      }
    }
    // Fallback for file://, older browsers and a refused Clipboard API.
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    if (!ok) throw new Error("copy command was refused");
  }

  function initCopy() {
    const live = document.createElement("p");
    live.className = "visually-hidden";
    live.setAttribute("aria-live", "polite");
    document.body.appendChild(live);

    document.querySelectorAll("[data-copy-target]").forEach((button) => {
      const label = button.dataset.label || button.textContent.trim();
      let timer = 0;
      button.addEventListener("click", async () => {
        const source = document.querySelector(button.dataset.copyTarget);
        if (!source) return;
        // Commands are copied without their trailing comments and alignment padding: cmd.exe (Anaconda
        // Prompt) does not treat "#" as a comment and would pass it to pip as an argument.
        const raw = source.textContent;
        const text = button.dataset.copyMode === "commands"
          ? raw.split("\n").map((line) => line.replace(/\s+#.*$/, "").trimEnd()).filter(Boolean).join("\n")
          : raw.trim();
        try {
          await writeClipboard(text);
          button.textContent = "Copied";
          button.classList.add("is-done");
          live.textContent = "Copied to the clipboard";
        } catch (err) {
          console.warn("Copy failed:", err);
          button.textContent = "Press Ctrl+C";
          live.textContent = "Copying failed. The text is selected: press Ctrl+C.";
          const range = document.createRange();
          range.selectNodeContents(source);
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
        }
        clearTimeout(timer);
        timer = setTimeout(() => {
          button.textContent = label;
          button.classList.remove("is-done");
        }, 2000);
      });
    });
  }

  /* ---------------------------------------------------------------- clips */
  // Every clip plays by itself, muted and looped, with no click (also with reduced motion: the owner's
  // choice; the clips are slow scientific animations without flashes). The markup carries autoplay, so
  // the hero plays even before this script runs. The other clips hold their file in data-src and get it
  // only when they come near the screen, so a visit does not download every clip up front. A clip plays
  // while at least a quarter of it is on screen and pauses when it leaves. A small round button pauses and
  // resumes each clip. It sits in the top-left corner of the result clips (empty there: the colour bar is on
  // the right, the axis triad at the bottom left); data-btn on the clip box moves it, for example to the
  // top-right corner of the texture-guide animations, whose notes are top left (see .vbtn in style.css).
  const BUTTON_HTML =
    '<svg class="i-pause" aria-hidden="true"><use href="#i-pause"/></svg>' +
    '<svg class="i-play" aria-hidden="true"><use href="#i-play"/></svg>';

  function attachSource(video) {
    const src = video.dataset.src;
    if (!src || video.getAttribute("src")) return;
    video.src = src;
    video.removeAttribute("data-src");
    video.preload = "auto";
    video.load(); // the autoplay attribute starts it once enough has arrived
  }

  function initClips() {
    const videos = [...document.querySelectorAll("video[data-auto]")];
    if (!videos.length) return;
    const state = new Map(); // video -> { box, button, name, visible, userPaused, blocked }

    const render = (video) => {
      const s = state.get(video);
      s.box.classList.toggle("is-paused", video.paused && (s.userPaused || s.blocked));
      s.button.setAttribute("aria-label", `${video.paused ? "Play" : "Pause"} animation: ${s.name}`);
    };

    const tryPlay = (video) => {
      const s = state.get(video);
      attachSource(video);
      const p = video.play();
      if (p && typeof p.then === "function") {
        p.then(
          () => { s.blocked = false; render(video); },
          (err) => {
            if (err && err.name === "AbortError") return; // paused again before it started
            console.warn("Clip did not start:", video.currentSrc || video.dataset.src, err && err.name);
            s.blocked = true;
            render(video);
          }
        );
      }
    };

    const sync = (video) => {
      const s = state.get(video);
      if (s.visible !== false && !s.userPaused && !document.hidden) {
        if (video.paused) tryPlay(video);
      } else if (!video.paused) {
        video.pause();
      }
    };

    videos.forEach((video) => {
      const box = video.closest("[data-clip]") || video.parentElement;
      video.muted = true; // autoplay needs muted; set the property too, not only the attribute
      const button = document.createElement("button");
      button.type = "button";
      button.className = "vbtn";
      button.innerHTML = BUTTON_HTML;
      box.appendChild(button);
      const name = video.dataset.desc || "clip";
      // visible: null until the observer has reported, so a clip the browser starts early is left alone
      state.set(video, { box, button, name, visible: hasIO ? null : true, userPaused: false, blocked: false });

      button.addEventListener("click", () => {
        const s = state.get(video);
        if (video.paused) {
          s.userPaused = false;
          tryPlay(video);
        } else {
          s.userPaused = true;
          video.pause();
        }
        render(video);
      });
      // The autoplay attribute may start a clip that has already scrolled away: stop it until it is back.
      video.addEventListener("play", () => {
        const s = state.get(video);
        if (s.visible === false || s.userPaused) video.pause();
        render(video);
      });
      video.addEventListener("pause", () => render(video));
      video.addEventListener("error", () => console.error("Clip failed to load:", video.currentSrc || video.dataset.src), true);
      render(video);
    });

    if (hasIO) {
      // Attach the file a little before the clip scrolls in, so it is ready when it arrives.
      const near = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            attachSource(entry.target);
            near.unobserve(entry.target);
          });
        },
        { rootMargin: "600px 0px" }
      );
      const seen = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            state.get(entry.target).visible = entry.isIntersecting && entry.intersectionRatio >= 0.25;
            sync(entry.target);
          });
        },
        { threshold: [0, 0.25, 0.6] }
      );
      videos.forEach((v) => {
        if (v.dataset.src) near.observe(v);
        seen.observe(v);
      });
    } else {
      videos.forEach((v) => sync(v));
    }

    document.addEventListener("visibilitychange", () => videos.forEach(sync));
  }

  /* ---------------------------------------------------------------- tabs */
  let tabsSeq = 0;
  function initTabs() {
    document.querySelectorAll("[data-tabs]").forEach((root) => {
      const panels = [...root.querySelectorAll(":scope > .tab-panel")];
      if (panels.length < 2) return;
      const id = `tabs-${++tabsSeq}`;
      const style = root.dataset.tabsStyle || "seg";
      const list = document.createElement("div");
      list.className = `tablist tablist--${style}`;
      list.setAttribute("role", "tablist");
      if (root.dataset.tabsLabel) list.setAttribute("aria-label", root.dataset.tabsLabel);

      // Images in hidden panels are lazy and would load only when shown. Fetch a panel's images
      // when the pointer or the focus reaches its tab, so switching is quick without loading
      // every panel up front.
      const warm = (panel) => panel.querySelectorAll('img[loading="lazy"]').forEach((img) => { img.loading = "eager"; });

      const tabs = panels.map((panel, i) => {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.id = `${id}-tab-${i}`;
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-controls", `${id}-panel-${i}`);
        const label = panel.dataset.label || `Part ${i + 1}`;
        if (panel.dataset.step) {
          const n = document.createElement("span");
          n.className = "step-n";
          n.setAttribute("aria-hidden", "true");
          n.textContent = panel.dataset.step;
          tab.append(n, label);
        } else {
          tab.textContent = label;
        }
        tab.addEventListener("pointerenter", () => warm(panel), { once: true });
        tab.addEventListener("focus", () => warm(panel), { once: true });
        panel.id = `${id}-panel-${i}`;
        panel.setAttribute("role", "tabpanel");
        panel.setAttribute("aria-labelledby", tab.id);
        panel.tabIndex = -1;
        list.appendChild(tab);
        return tab;
      });

      const select = (index, focus) => {
        tabs.forEach((tab, i) => {
          const on = i === index;
          tab.setAttribute("aria-selected", String(on));
          tab.tabIndex = on ? 0 : -1;
          panels[i].hidden = !on;
          // Clips in hidden panels pause; the clip observer starts the visible one.
          if (!on) panels[i].querySelectorAll("video").forEach((v) => { if (!v.paused) v.pause(); });
        });
        if (focus) tabs[index].focus();
      };

      list.addEventListener("click", (event) => {
        const tab = event.target.closest('[role="tab"]');
        if (tab) select(tabs.indexOf(tab), false);
      });
      list.addEventListener("keydown", (event) => {
        const current = tabs.indexOf(document.activeElement);
        if (current < 0) return;
        const last = tabs.length - 1;
        const next = { ArrowRight: current + 1, ArrowLeft: current - 1, Home: 0, End: last }[event.key];
        if (next === undefined) return;
        event.preventDefault();
        select(next > last ? 0 : next < 0 ? last : next, true);
      });

      root.insertBefore(list, root.firstChild);
      select(0, false);
      root.classList.add("is-ready");

      // On a phone the tab row scrolls sideways; it fades out at each end where more tabs wait
      // (.tablist.more-left / .more-right in style.css), so the row shows that it scrolls.
      const edges = () => {
        const room = list.scrollWidth - list.clientWidth;
        list.classList.toggle("more-left", room > 2 && list.scrollLeft > 2);
        list.classList.toggle("more-right", room > 2 && list.scrollLeft < room - 2);
      };
      list.addEventListener("scroll", edges, { passive: true });
      window.addEventListener("resize", edges);
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(edges, () => {});
      edges();
    });
  }

  /* ---------------------------------------------------------------- lightbox */
  function initLightbox() {
    const dialog = document.querySelector("dialog.lightbox");
    if (!dialog || typeof dialog.showModal !== "function") return; // links then open the image itself
    const caption = dialog.querySelector(".lightbox-caption");
    const close = dialog.querySelector(".lightbox-close");
    const narrow = window.matchMedia("(max-width: 720px)");
    const SCROLL_WIDTH = 960; // px: on a phone, the figures' axis text is about 13 px at this width
    const CAPTION_ROOM = 96; // px of dialog height kept for the caption on a large screen

    document.addEventListener("click", (event) => {
      const link = event.target.closest("a.zoom");
      if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
      event.preventDefault();
      const thumb = link.querySelector("img");
      const figure = link.closest("figure");
      const text = figure?.querySelector("figcaption")?.textContent.replace(/\s+/g, " ").trim() || "";
      // A fresh <img> per opening: the page never holds an image without a source.
      const img = document.createElement("img");
      img.decoding = "async";
      img.addEventListener("error", () => console.error("Image failed to load:", link.getAttribute("href")));
      // Never enlarge an image beyond its pixels. On a large screen it is shrunk to fit the window; on a
      // phone the figures are far wider than the screen, so they show at up to SCROLL_WIDTH px and the
      // dialog scrolls (the viewer can also pinch-zoom).
      const isPaper = !!link.closest(".plate--paper");
      const pad = isPaper ? 24 : 0; // the paper mat (.lightbox.is-paper img)
      img.addEventListener("load", () => {
        const natural = img.naturalWidth;
        const naturalH = img.naturalHeight;
        if (!natural || !naturalH) return;
        let width;
        if (narrow.matches) {
          width = Math.min(natural, SCROLL_WIDTH);
        } else {
          const fitW = window.innerWidth * 0.96;
          const fitH = (window.innerHeight * 0.94 - CAPTION_ROOM) * (natural / naturalH);
          width = Math.min(natural, fitW - pad, Math.max(fitH, 320));
        }
        img.style.width = `${Math.floor(width + pad)}px`;
        dialog.classList.toggle("is-scroll", width + pad > window.innerWidth * 0.96);
      });
      dialog.classList.remove("is-scroll");
      img.src = link.getAttribute("href");
      img.alt = thumb ? thumb.alt : "";
      dialog.querySelector("img")?.remove();
      dialog.insertBefore(img, caption);
      caption.textContent = text;
      dialog.classList.toggle("is-paper", isPaper);
      dialog.showModal();
      close.focus();
    });
    close.addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener("close", () => { dialog.querySelector("img")?.remove(); });
  }

  /* ---------------------------------------------------------------- active nav link */
  function initScrollSpy() {
    if (!hasIO) return;
    const links = new Map();
    document.querySelectorAll('.site-nav a[href^="#"]').forEach((a) => {
      const section = document.querySelector(a.getAttribute("href"));
      if (section) links.set(section, a);
    });
    const hero = document.getElementById("top");
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          links.forEach((a) => a.removeAttribute("aria-current"));
          const link = links.get(entry.target); // the hero has no link: nothing is marked
          if (link) link.setAttribute("aria-current", "location");
        });
      },
      { rootMargin: "-40% 0px -55% 0px" }
    );
    links.forEach((_, section) => io.observe(section));
    if (hero) io.observe(hero);
  }

  function init() {
    // Tabs first: they hide panels, so the clip observer never starts a clip in a hidden panel.
    [initMenu, initCopy, initTabs, initClips, initLightbox, initScrollSpy].forEach((step) => {
      try {
        step();
      } catch (err) {
        console.error(`pyALDVC site: ${step.name} failed`, err);
      }
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
