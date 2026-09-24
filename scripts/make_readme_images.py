#!/usr/bin/env python
"""README images drawn with the website's own style sheet, so the README looks like the site.

Renders small HTML pages that load ``site/css/style.css`` (and the site's web fonts) in a headless
Chrome or Edge and saves them as PNG at two device pixels per CSS pixel::

    assets/readme/button-website.png   the call to action: "Visit the website"
    assets/readme/button-start.png     "Get started"
    assets/readme/button-download.png  "Windows download"
    assets/readme/button-guide.png     "User guide"
    assets/readme/stats.png            the website's key-numbers strip, copied from site/index.html

The README shows each image at half its pixel width. Buttons have a transparent background and a
filled face, so they read on GitHub's light and dark themes alike. The numbers strip is taken from
the page itself, so run this script again after ``scripts/sync_site_numbers.py`` changes a number::

    python scripts/make_readme_images.py [--browser PATH]
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUT = ROOT / "assets" / "readme"
SCALE = 2  # device pixels per CSS pixel; the README draws every image at half its width
FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=Newsreader:ital,opsz,wght@0,28,400;1,60,400&display=swap"
)
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
)
STATS_WIDTH = 860  # CSS px: the site's two-column layout of the numbers; the README draws it 800 px wide
ARROW = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"'
    ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>'
)

# name, label, primary (filled accent) or secondary (raised face with a border), trailing arrow
BUTTONS = (
    ("button-website", "Visit the website", True, True),
    ("button-start", "Get started", False, False),
    ("button-download", "Windows download", False, False),
    ("button-guide", "User guide", False, False),
)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<link rel="stylesheet" href="{fonts}">
<link rel="stylesheet" href="{css}">
<style>
  html, body {{ background: transparent !important; margin: 0; padding: 0; }}
  body {{ padding: 8px; }}
  .readme-btn {{ font-size: {font}px; padding: {pad_y}px {pad_x}px; border-radius: 9px; gap: 10px; }}
  .readme-btn.btn-ghost {{ background: var(--panel); border-color: var(--border-2); color: var(--text); }}
  .readme-card {{ width: {width}px; background: var(--bg); border: 1px solid var(--border); border-radius: 14px;
                  overflow: hidden; }}
  .readme-card .stats {{ border: 0; }}
  .readme-card .wrap {{ padding-inline: 28px; }}
</style></head>
<body>{body}</body></html>
"""


def find_browser(given: str | None) -> str:
    for cand in (given,) if given else BROWSERS:
        path = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if path:
            return path
    sys.exit("no Chrome or Edge found; pass --browser PATH")


def shoot(browser: str, body: str, out: Path, width: int, height: int, **style) -> Path:
    """Render ``body`` on a transparent page and save it cropped to what was drawn."""
    page = PAGE.format(
        fonts=html.escape(FONTS),
        css=(SITE / "css" / "style.css").as_uri(),
        body=body,
        font=style.get("font", 19),
        pad_y=style.get("pad_y", 16),
        pad_x=style.get("pad_x", 28),
        width=style.get("card_width", STATS_WIDTH),
    )
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "page.html"
        src.write_text(page, encoding="utf-8")
        raw = Path(tmp) / "shot.png"
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={Path(tmp) / 'profile'}",
            f"--force-device-scale-factor={SCALE}",
            "--default-background-color=00000000",
            "--virtual-time-budget=8000",  # time for the web fonts
            f"--window-size={width},{height}",
            f"--screenshot={raw}",
            src.as_uri(),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if not raw.is_file():
            sys.exit(f"browser made no screenshot for {out.name}:\n{run.stderr[-2000:]}")
        with Image.open(raw) as im:
            rgba = im.convert("RGBA")
    alpha = np.asarray(rgba)[..., 3]
    rows, cols = np.nonzero(alpha > 0)
    if rows.size == 0:
        sys.exit(f"{out.name}: nothing was drawn")
    box = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    if box[2] >= rgba.width - 1 or box[3] >= rgba.height - 1:
        sys.exit(f"{out.name}: the drawing reaches the window edge; enlarge the window")
    cropped = rgba.crop(box)
    if cropped.width % 2:  # an even width, so the README's half-width is a whole pixel
        pad = Image.new("RGBA", (cropped.width + 1, cropped.height), (0, 0, 0, 0))
        pad.paste(cropped, (0, 0))
        cropped = pad
    cropped.save(out, optimize=True)
    return out


def stats_markup() -> str:
    """The key-numbers section of the website, verbatim."""
    page = (SITE / "index.html").read_text(encoding="utf-8")
    found = re.search(r'<section class="stats".*?</section>', page, re.S)
    if not found:
        sys.exit("site/index.html has no stats section")
    return found.group(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--browser", help="path to Chrome or Edge (found automatically when left out)")
    args = ap.parse_args()
    browser = find_browser(args.browser)
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for name, label, primary, arrow in BUTTONS:
        kind = "btn-primary" if primary else "btn-ghost"
        body = f'<a class="btn {kind} readme-btn">{html.escape(label)}{ARROW if arrow else ""}</a>'
        style = {"font": 21, "pad_y": 18, "pad_x": 34} if primary else {"font": 17, "pad_y": 15, "pad_x": 24}
        made.append(shoot(browser, body, OUT / f"{name}.png", 520, 140, **style))
    card = f'<div class="readme-card">{stats_markup()}</div>'
    made.append(shoot(browser, card, OUT / "stats.png", STATS_WIDTH + 40, 700, card_width=STATS_WIDTH))
    for p in made:
        with Image.open(p) as im:
            print(f"{p.relative_to(ROOT)}  {im.width} x {im.height} px  {p.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
