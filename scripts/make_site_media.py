#!/usr/bin/env python
"""Media for the project website (site/) and the README (assets/readme/).

Sources
    The owner's local materials folder (``--materials``, default ``assets/pyALDVC_flyer_materials``; it is
    gitignored and never committed): screen recordings of pyALDVC results (foam compression, hydrogel
    indentation, synthetic rigid-body rotation, laser-induced cavitation), a slice screenshot with the
    node grid, and three figures adapted from the DVC Challenge 2.0 manuscript.
    Tracked files: ``assets/videos/`` (the hydrogel and rotation recordings, used when the materials
    folder lacks them), ``assets/banner.png`` and ``assets/icon/pyALDVC-master.png``.

Writes
    site/media/   <clip>.mp4 (H.264, 15 fps, faststart) and <clip>.jpg posters; hydrogel-orbit.webm (VP9)
    site/img/     three paper figures (.webp), foam-mesh-grid.webp, favicon.png (32 px),
                  apple-touch-icon.png (180 px), logo.png (96 px, header and footer),
                  og.png (1200 x 630 social preview), thumb-*.webp (case index thumbnails, cropped
                  from the posters in site/media, so the site-video group must have run once)
    assets/readme/ hero and gallery GIFs, banner.png (<= 400 kB)

The recordings are cropped to the 3-D viewport, so no application chrome, screen-recorder label or
tooltip reaches the outputs; composites put the colour bar next to the object. Crops are in source
pixels and were measured on every frame of the recordings. Every output is checked against a size
budget (README images must stay under the 5 MB image-proxy cap) and a table of outputs is printed.

Requires imageio-ffmpeg (ffmpeg 7.1 with libx264 and libvpx-vp9), Pillow and NumPy.

    python scripts/make_site_media.py                 # everything
    python scripts/make_site_media.py --only readme   # one group: site-video, site-img or readme
    python scripts/make_site_media.py --sheets out/   # also first/middle/last-frame sheets for review
    python scripts/make_site_media.py --only readme --clip hydrogel-orbit   # one clip's outputs only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
TRACKED_VIDEOS = ASSETS / "videos"

FPS = 15  # the recordings are variable frame rate (~14 fps); every clip is resampled to this
MB = 1e6
SITE_MP4_BUDGET = 3.2 * MB
README_HERO_BUDGET = 3.0 * MB
README_TILE_BUDGET = 1.2 * MB
README_TOTAL_BUDGET = 7.0 * MB
README_FILE_CAP = 5.0 * MB  # GitHub's and PyPI's image proxy (camo) refuses larger files
BANNER_BUDGET = 400e3

BG_DARKEST = (11, 15, 26)  # theme.py BG_DARKEST, #0b0f1a
FOAM_VIEW_BG = "0x1a1c23"  # background of the foam recordings' 3-D view

ORBIT_FRAMES = 179  # one full camera turn of the hydrogel orbit recording, at FPS
HG = "Jin Yang hydorgel indentation"  # folder names as the owner supplied them
FOAM = "Alex_foam_compression"


# --------------------------------------------------------------------------- clip definitions
@dataclass(frozen=True)
class Clip:
    """A web video: a filter graph from the source to the finished frames at ``FPS``."""

    name: str
    source: str  # path relative to the materials folder
    graph: str  # from [0:v]; no output label
    fallback: str | None = None  # path relative to assets/videos when the materials copy is absent
    pingpong: bool = False  # play forward then backward, so a partial orbit loops without a jump
    poster_at: float = 0.0  # poster frame as a fraction of the clip (0 = first frame)
    webm: bool = False
    crf: int = 23


def _composite(obj: str, bar: str, tail: str = "", head: str = "") -> str:
    """Object crop and colour-bar crop of one recording side by side (drops the empty gap between)."""
    return f"[0:v]setsar=1,fps={FPS}{head},split[s1][s2];[s1]crop={obj}[o];[s2]crop={bar}[c];[o][c]hstack=inputs=2{tail}"


CLIPS = (
    # Deformed lattice with arrows, camera orbit. ORBIT_FRAMES frames at 15 fps are one turn, so the
    # loop is seamless (of all late frames, frame 179 matches frame 0 best). Browsers with VP9 load the
    # WebM (about 1.3 MB); CRF 28 keeps the MP4 for the others near 1.5 MB (CRF 23 gave 2.7 MB).
    Clip(
        "hydrogel-orbit",
        f"{HG}/indentation_deformed_lattice_orbit_with_arrow.mp4",
        f"[0:v]setsar=1,fps={FPS},crop=1272:786:668:538,trim=end_frame={ORBIT_FRAMES},setpts=PTS-STARTPTS",
        fallback="indentation_deformed_lattice_orbit_with_arrow.mp4",
        webm=True,
        crf=28,
    ),
    # Displacement magnitude over orthogonal CT slices while the XY slice sweeps along z.
    Clip("foam-slice-sweep", f"{FOAM}/slice_sweep.gif", _composite("440:670:690:222", "100:670:1420:222"), poster_at=0.5, crf=22),
    # Deformed lattice (drawn with the deformation exaggerated 2x in the recording). The recording
    # holds 2.5 passes of the reference -> deformed animation with a jump back between passes; frames
    # 30-71 at 15 fps are one whole pass, played forward and back.
    Clip(
        "foam-lattice",
        f"{FOAM}/deformed_lattice_under_compression.gif",
        _composite("430:640:695:252", "100:640:1420:252", head=",trim=start_frame=30:end_frame=72,setpts=PTS-STARTPTS"),
        pingpong=True,
        poster_at=0.5,
        crf=22,
    ),
    # w over an XY slice swept through the gel.
    Clip(
        "hydrogel-sweep-z",
        f"{HG}/indentation_sweep_z.mp4",
        f"[0:v]setsar=1,fps={FPS},crop=1236:656:714:530",
        fallback="indentation_sweep_z.mp4",
        poster_at=0.67,
    ),
    # u over the rotating cube, incremental tracking, ping-pong. Same frame size as lic-strain.
    Clip(
        "rigid-rotation",
        "rigid_rotation_frame_animation.mp4",
        f"[0:v]setsar=1,fps={FPS},crop=1280:736:690:550",
        fallback="rotation_frame_animation.mp4",
        pingpong=True,
        poster_at=0.5,
    ),
    # Residual von Mises strain, partial orbit, ping-pong. The crop drops the 1-px light top row. A
    # one-sentence teaser, so CRF 29 keeps it near 1.3 MB (CRF 23 gave 2.7 MB).
    Clip(
        "lic-strain",
        "LIC_residual_VM_strain.mp4",
        _composite("1700:1066:150:74", "152:1066:2180:74", ",scale=1280:-2:flags=lanczos"),
        pingpong=True,
        crf=29,
    ),
)


@dataclass(frozen=True)
class Gif:
    """A README GIF made from a clip's frames."""

    name: str
    clip: str
    width: int
    speed: float  # playback speed-up relative to the recording
    fps: int
    colors: int
    budget: float
    blur: float = 0.0  # a light blur keeps dithering noise, and so the file size, down
    bayer: int = 5
    pad: str = ""  # pad filter applied before scaling, to give paired tiles the same aspect ratio
    # "full": one palette for the whole GIF; "single": a palette per frame, which keeps smooth colour
    # ramps (the hydrogel dimple) from breaking into bands; every frame is then written whole
    stats: str = "full"


GIFS = (
    # Hero: one full turn in exactly 4 s (40 frames, so the wrap step equals the others). 800 px keeps
    # it under 3 MB; a slower turn does not fit the budget. A palette per frame: one shared palette
    # broke the colour ramp of the dimple into visible rings (and the per-frame file is even smaller).
    Gif(
        "hydrogel-orbit",
        "hydrogel-orbit",
        800,
        ORBIT_FRAMES / (FPS * 4.0),
        10,
        96,
        README_HERO_BUDGET,
        blur=0.5,
        stats="single",
    ),
    # Gallery, two rows of equal aspect: the foam pair (portrait) and rotation + LIC (landscape).
    Gif("foam-slice-sweep", "foam-slice-sweep", 480, 2.5, 8, 80, README_TILE_BUDGET, blur=0.6),
    Gif(
        "foam-lattice",
        "foam-lattice",
        480,
        1.0,
        8,
        96,
        README_TILE_BUDGET,
        blur=0.5,
        pad=f"pad=540:670:5:15:color={FOAM_VIEW_BG}",
    ),
    Gif("rigid-rotation", "rigid-rotation", 480, 1.0, 12, 256, README_TILE_BUDGET, bayer=3),
    Gif("lic-strain", "lic-strain", 480, 3.0, 8, 96, README_TILE_BUDGET, blur=0.5),
)

PAPER_FIGURES = (  # (source, output stem); all three adapted from the DVC Challenge 2.0 manuscript
    (f"{FOAM}/image (1).png", "foam-xct-setup"),
    (f"{FOAM}/image (2).png", "foam-compression-steps"),
    (f"{HG}/image (3).png", "hydrogel-indentation-setup"),
)
THUMBS = (  # (poster in site/media, output stem, 3:2 crop box l, t, r, b in poster pixels, around the specimen)
    ("foam-slice-sweep", "thumb-foam", (20, 230, 380, 470)),
    ("hydrogel-orbit", "thumb-hydrogel", (150, 140, 930, 660)),
    ("rigid-rotation", "thumb-rotation", (226, 87, 946, 567)),
    ("lic-strain", "thumb-cavitation", (210, 86, 1050, 646)),
)
THUMB_SIZE = (240, 160)  # shown at 120 x 80 CSS pixels
LOGO_PX = 96  # the page shows the logo at 28-30 CSS pixels; 96 px stays sharp at 3x
MESH_SCREENSHOT = f"{FOAM}/GUI_showing_mesh_grid.png"
MESH_PANELS = ((885, 286, 1485, 906), (1620, 286, 2045, 906))  # XY and YZ slice panels (l, t, r, b)
MESH_GAP = 24


# --------------------------------------------------------------------------- ffmpeg helpers
def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit("imageio-ffmpeg is required: pip install imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


FF = ffmpeg_exe()


def run_ffmpeg(args: list[str]) -> str:
    """Run ffmpeg; return its log (stderr). Raise with the log tail on failure."""
    proc = subprocess.run(
        [FF, "-hide_banner", "-nostdin", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(args[:6])} ...):\n{proc.stderr[-2000:]}")
    return proc.stderr


def count_frames(src: Path, graph: str) -> int:
    log = run_ffmpeg(["-i", str(src), "-filter_complex", graph + "[v]", "-map", "[v]", "-f", "null", "-"])
    found = re.findall(r"frame=\s*(\d+)", log)
    if not found:
        raise RuntimeError(f"could not count the frames of {src.name}")
    return int(found[-1])


def pingpong(n_frames: int) -> str:
    """Forward, then backward without repeating either end frame, so the loop has no hold."""
    if n_frames < 3:
        raise ValueError("a ping-pong loop needs at least 3 frames")
    return (
        f",split[pf][pr];[pr]reverse,trim=start_frame=1:end_frame={n_frames - 1},setpts=PTS-STARTPTS[prv];"
        f"[pf][prv]concat=n=2:v=1:a=0,setpts=N/({FPS}*TB)"
    )  # even timestamps: no frame lost at the join


def probe(path: Path) -> dict:
    """Frame size, frame count and duration of a video or GIF, read back from the written file."""
    if path.suffix == ".gif":
        with Image.open(path) as im:
            durations = []
            for i in range(im.n_frames):
                im.seek(i)
                durations.append(im.info.get("duration", 0))
            return {"size": im.size, "frames": im.n_frames, "seconds": sum(durations) / 1000}
    log = run_ffmpeg(["-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "null", "-"])
    w, h = map(int, re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", log).groups())
    frames = int(re.findall(r"frame=\s*(\d+)", log)[-1])
    hh, mm, ss = re.search(r"Duration: (\d+):(\d+):([\d.]+)", log).groups()
    return {"size": (w, h), "frames": frames, "seconds": int(hh) * 3600 + int(mm) * 60 + float(ss)}


def moov_before_mdat(path: Path) -> bool:
    """True when the MP4 index comes first (faststart), so playback starts before the download ends."""
    with path.open("rb") as fh:
        while header := fh.read(8):
            size, kind = int.from_bytes(header[:4], "big"), header[4:8]
            if kind in (b"moov", b"mdat"):
                return kind == b"moov"
            if size == 1:
                size = int.from_bytes(fh.read(8), "big") - 8
            fh.seek(size - 8, 1)
    return False


# --------------------------------------------------------------------------- videos and GIFs
def resolve_source(clip: Clip, materials: Path) -> Path:
    path = materials / clip.source
    if path.is_file():
        return path
    if clip.fallback and (TRACKED_VIDEOS / clip.fallback).is_file():
        return TRACKED_VIDEOS / clip.fallback
    raise FileNotFoundError(f"{clip.name}: source not found ({path})")


def clip_graph(clip: Clip, src: Path, cache: dict[str, str]) -> str:
    """The clip's graph with the ping-pong applied (needs the frame count, so it is cached)."""
    if clip.name not in cache:
        graph = clip.graph
        if clip.pingpong:
            graph += pingpong(count_frames(src, clip.graph))
        cache[clip.name] = graph
    return cache[clip.name]


def make_site_video(clip: Clip, src: Path, graph: str, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    mp4, jpg = out / f"{clip.name}.mp4", out / f"{clip.name}.jpg"
    even = ",crop=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[v]"
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            graph + even,
            "-map",
            "[v]",
            "-an",
            "-map_metadata",
            "-1",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(clip.crf),
            "-profile:v",
            "high",
            "-level:v",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4),
        ]
    )
    written = [mp4]
    if clip.webm:
        webm = out / f"{clip.name}.webm"
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(src),
                "-filter_complex",
                graph + even,
                "-map",
                "[v]",
                "-an",
                "-map_metadata",
                "-1",
                "-r",
                str(FPS),
                "-c:v",
                "libvpx-vp9",
                "-b:v",
                "0",
                "-crf",
                "38",
                "-row-mt",
                "1",
                "-deadline",
                "good",
                "-cpu-used",
                "2",
                "-pix_fmt",
                "yuv420p",
                str(webm),
            ]
        )
        written.append(webm)
    n = probe(mp4)["frames"]
    k = min(n - 1, round(clip.poster_at * (n - 1)))
    run_ffmpeg(["-y", "-i", str(mp4), "-vf", f"select=eq(n\\,{k})", "-frames:v", "1", "-q:v", "3", str(jpg)])
    written.append(jpg)
    return written


def make_gif(gif: Gif, src: Path, graph: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{gif.name}.gif"
    pre = f",{gif.pad}" if gif.pad else ""
    blur = f",gblur=sigma={gif.blur}" if gif.blur > 0 else ""
    if gif.stats not in ("full", "single"):
        raise ValueError(f"{gif.name}: stats must be 'full' or 'single', not {gif.stats!r}")
    if gif.stats == "single":
        # Every frame is written whole with its own palette. The encoder's inter-frame shortcuts (only the
        # changed rectangle, unchanged pixels transparent) compare palette indices, which mean different
        # colours from one frame's palette to the next: with them on, the colour bar and the page corners
        # came out garbled. Whole frames are no larger here, since an orbit changes nearly every pixel.
        palette_use, encoder = "diff_mode=none:new=1", ["-gifflags", "-offsetting-transdiff"]
    else:
        palette_use, encoder = "diff_mode=rectangle", []
    tail = (
        f"{pre},setpts=PTS/{gif.speed},fps={gif.fps},scale={gif.width}:-2:flags=lanczos{blur},split[ga][gb];"
        f"[ga]palettegen=max_colors={gif.colors}:stats_mode={gif.stats}[gp];"
        f"[gb][gp]paletteuse=dither=bayer:bayer_scale={gif.bayer}:{palette_use}[v]"
    )
    run_ffmpeg(["-y", "-i", str(src), "-filter_complex", graph + tail, "-map", "[v]", *encoder, "-loop", "0", str(dst)])
    return dst


# --------------------------------------------------------------------------- still images
def _trim(im: Image.Image, background: tuple[int, int, int], pad: int, tol: int = 10) -> Image.Image:
    """Crop to the content box (pixels differing from ``background``), then add ``pad`` px of it back."""
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    mask = np.abs(a - np.array(background, np.int16)).max(axis=-1) > tol
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("image has no content")
    box = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    canvas = Image.new("RGB", (box.width + 2 * pad, box.height + 2 * pad), background)
    canvas.paste(box, (pad, pad))
    return canvas


def _fit_width(im: Image.Image, width: int) -> Image.Image:
    """Downscale to ``width`` if wider; never upscale."""
    if im.width <= width:
        return im
    return im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)


def make_paper_figures(materials: Path, out: Path) -> list[Path]:
    written = []
    for rel, stem in PAPER_FIGURES:
        with Image.open(materials / rel) as src:
            rgba = src.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        fig = _fit_width(_trim(flat, (255, 255, 255), pad=16), 1200)
        dst = out / f"{stem}.webp"
        fig.save(dst, "WEBP", quality=90, method=6)
        written.append(dst)
    return written


def make_mesh_grid(materials: Path, out: Path) -> Path:
    """The XY and YZ slice panels with the node grid, side by side, without the application around them."""
    with Image.open(materials / MESH_SCREENSHOT) as src:
        shot = src.convert("RGB")
    panels = [shot.crop(box) for box in MESH_PANELS]
    background = shot.getpixel((MESH_PANELS[0][0] + 5, MESH_PANELS[0][1] + 5))
    width = sum(p.width for p in panels) + MESH_GAP
    canvas = Image.new("RGB", (width, max(p.height for p in panels)), background)
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width + MESH_GAP, 0))
    dst = out / "foam-mesh-grid.webp"
    _trim(canvas, background, pad=20, tol=20).save(dst, "WEBP", quality=90, method=6)
    return dst


def _resize_rgba(im: Image.Image, size: int) -> Image.Image:
    """Resize with premultiplied alpha, so the transparent corners do not bleed dark fringes."""
    return im.convert("RGBa").resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")


def make_icons(out: Path) -> list[Path]:
    with Image.open(ASSETS / "icon" / "pyALDVC-master.png") as src:
        master = src.convert("RGBA")
    fav = out / "favicon.png"
    _resize_rgba(master, 32).save(fav, optimize=True)
    touch = Image.new("RGBA", (180, 180), BG_DARKEST + (255,))  # iOS fills transparency with black
    touch.alpha_composite(_resize_rgba(master, 180))
    apple = out / "apple-touch-icon.png"
    touch.convert("RGB").save(apple, optimize=True)
    logo = out / "logo.png"
    _resize_rgba(master, LOGO_PX).save(logo, optimize=True)
    return [fav, apple, logo]


def make_thumbs(posters: Path, out: Path) -> list[Path]:
    """Small 3:2 crops of the video posters, for the case index at the top of the Cases section."""
    written = []
    for poster, stem, box in THUMBS:
        src = posters / f"{poster}.jpg"
        if not src.is_file():
            raise FileNotFoundError(f"{src} is missing: build the site-video group first")
        left, top, right, bottom = box
        if abs((right - left) * THUMB_SIZE[1] - (bottom - top) * THUMB_SIZE[0]) > 1:
            raise ValueError(f"{stem}: crop box {box} is not {THUMB_SIZE[0]}:{THUMB_SIZE[1]}")
        with Image.open(src) as im:
            if right > im.width or bottom > im.height:
                raise ValueError(f"{stem}: crop box {box} exceeds the poster ({im.width} x {im.height})")
            thumb = im.convert("RGB").crop(box).resize(THUMB_SIZE, Image.Resampling.LANCZOS)
        dst = out / f"{stem}.webp"
        thumb.save(dst, "WEBP", quality=82, method=6)
        written.append(dst)
    return written


def make_og_image(out: Path) -> Path:
    """1200 x 630 social preview: the banner on its own background, extended above and below."""
    with Image.open(ASSETS / "banner.png") as src:
        banner = src.convert("RGB")
    W, H = 1200, 630
    b = np.asarray(banner.resize((W, round(banner.height * W / banner.width)), Image.Resampling.LANCZOS), np.float32)
    top = (H - b.shape[0]) // 2
    bottom = H - b.shape[0] - top

    def edge(rows: np.ndarray) -> np.ndarray:
        """Row colour smoothed along x, so the extension carries the banner's vignette but no detail."""
        row = rows.mean(axis=0)
        k = np.exp(-0.5 * (np.arange(-60, 61) / 20.0) ** 2)
        k /= k.sum()
        padded = np.pad(row, ((60, 60), (0, 0)), mode="edge")
        return np.stack([np.convolve(padded[:, c], k, mode="valid") for c in range(3)], axis=-1)

    canvas = np.empty((H, W, 3), np.float32)
    canvas[:top] = edge(b[:6])
    canvas[top + b.shape[0] :] = edge(b[-6:])
    canvas[top : top + b.shape[0]] = b
    # Feather the seams: blend the first and last 24 banner rows with the extension colour.
    ramp = np.linspace(0.0, 1.0, 24, dtype=np.float32)[:, None, None]
    canvas[top : top + 24] = ramp * b[:24] + (1 - ramp) * edge(b[:6])
    canvas[top + b.shape[0] - 24 : top + b.shape[0]] = ramp[::-1] * b[-24:] + (1 - ramp[::-1]) * edge(b[-6:])
    # Fade the extensions a little towards the outer edges, like the banner's own vignette.
    fade = np.ones(H, np.float32)
    fade[:top] = np.linspace(0.8, 1.0, top)
    fade[H - bottom :] = np.linspace(1.0, 0.8, bottom)
    canvas *= fade[:, None, None]
    dst = out / "og.png"
    Image.fromarray(np.clip(canvas + 0.5, 0, 255).astype(np.uint8)).save(dst, optimize=True)
    return dst


def make_readme_banner(out: Path) -> Path:
    """assets/banner.png at the widest size whose full-colour PNG fits the budget.

    A 256-colour palette would fit at 1600 px, but it bands the glow of the displacement panel, so the
    banner stays full colour and gives up pixels instead (the README shows it about 850 px wide).
    """
    with Image.open(ASSETS / "banner.png") as src:
        icc = src.info.get("icc_profile")
        banner = src.convert("RGB")
    dst = out / "banner.png"
    for width in (1600, 1440, 1360, 1280, 1200, 1120, 1040, 960):
        small = banner.resize((width, round(banner.height * width / banner.width)), Image.Resampling.LANCZOS)
        small.save(dst, optimize=True, icc_profile=icc)
        if dst.stat().st_size <= BANNER_BUDGET:
            break
    return dst


# --------------------------------------------------------------------------- review and report
def frame_sheet(path: Path, dst: Path) -> None:
    """First, middle and last frame of a video or GIF side by side, for a visual check."""
    n = probe(path)["frames"]
    picks = sorted({0, n // 2, n - 1})
    select = "+".join(f"eq(n\\,{k})" for k in picks)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(path),
            "-vf",
            f"select={select},scale=640:-2,tile={len(picks)}x1",
            "-frames:v",
            "1",
            "-fps_mode",
            "passthrough",
            str(dst),
        ]
    )


def describe(path: Path) -> tuple[str, str, str]:
    if path.suffix in (".mp4", ".webm", ".gif"):
        p = probe(path)
        w, h = p["size"]
        return f"{w}x{h}", f"{p['frames']} fr / {p['seconds']:.1f} s", f"{path.stat().st_size / MB:.2f} MB"
    with Image.open(path) as im:
        w, h = im.size
    return f"{w}x{h}", "-", f"{path.stat().st_size / 1e3:.0f} kB"


def check_budgets(site_videos: list[Path], readme: list[Path]) -> list[str]:
    problems = []
    for p in site_videos:
        if p.suffix == ".mp4":
            if p.stat().st_size > SITE_MP4_BUDGET:
                problems.append(f"{p.name}: {p.stat().st_size / MB:.2f} MB > {SITE_MP4_BUDGET / MB:.1f} MB")
            if not moov_before_mdat(p):
                problems.append(f"{p.name}: moov atom is not before mdat (no faststart)")
    budgets = {f"{g.name}.gif": g.budget for g in GIFS} | {"banner.png": BANNER_BUDGET}
    for p in readme:
        size = p.stat().st_size
        if size > budgets.get(p.name, README_FILE_CAP) or size >= README_FILE_CAP:
            problems.append(f"{p.name}: {size / MB:.2f} MB over its budget")
    total = sum(p.stat().st_size for p in readme)
    if total > README_TOTAL_BUDGET:
        problems.append(f"README media total {total / MB:.2f} MB > {README_TOTAL_BUDGET / MB:.1f} MB")
    return problems


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--materials",
        type=Path,
        default=Path("assets/pyALDVC_flyer_materials"),
        help="owner's materials folder (relative paths are taken from the repository root)",
    )
    ap.add_argument("--out-site", type=Path, default=Path("site/media"), help="website videos and posters")
    ap.add_argument("--out-img", type=Path, default=Path("site/img"), help="website still images")
    ap.add_argument("--out-readme", type=Path, default=Path("assets/readme"), help="README GIFs and banner")
    ap.add_argument(
        "--only",
        choices=("site-video", "site-img", "readme"),
        action="append",
        help="build only these groups (repeatable); default all",
    )
    ap.add_argument("--sheets", type=Path, help="also write first/middle/last-frame sheets here")
    ap.add_argument(
        "--clip",
        action="append",
        choices=[c.name for c in CLIPS],
        help="rebuild only these clips' site videos and README GIFs (repeatable); the README banner is then left as it is",
    )
    args = ap.parse_args(argv)

    materials = _resolve(args.materials)
    out_site, out_img, out_readme = (_resolve(p) for p in (args.out_site, args.out_img, args.out_readme))
    groups = set(args.only or ("site-video", "site-img", "readme"))
    if not materials.is_dir():
        print(f"materials folder not found: {materials}", file=sys.stderr)
        return 2

    clips = {c.name: c for c in CLIPS}
    picked = set(args.clip or clips)
    sources = {c.name: resolve_source(c, materials) for c in CLIPS if c.name in picked}
    graphs: dict[str, str] = {}
    site_videos, site_imgs, readme = [], [], []

    if "site-video" in groups:
        for clip in CLIPS:
            if clip.name not in picked:
                continue
            print(f"video  {clip.name}  <- {sources[clip.name].name}", flush=True)
            site_videos += make_site_video(clip, sources[clip.name], clip_graph(clip, sources[clip.name], graphs), out_site)
    if "site-img" in groups:
        out_img.mkdir(parents=True, exist_ok=True)
        print("images paper figures, mesh grid, icons, og.png, case thumbnails", flush=True)
        site_imgs += make_paper_figures(materials, out_img)
        site_imgs.append(make_mesh_grid(materials, out_img))
        site_imgs += make_icons(out_img)
        site_imgs.append(make_og_image(out_img))
        site_imgs += make_thumbs(out_site, out_img)
    if "readme" in groups:
        for gif in GIFS:
            if gif.clip not in picked:
                continue
            clip = clips[gif.clip]
            print(f"gif    {gif.name}", flush=True)
            readme.append(make_gif(gif, sources[clip.name], clip_graph(clip, sources[clip.name], graphs), out_readme))
        if args.clip is None:
            readme.append(make_readme_banner(out_readme))

    outputs = site_videos + site_imgs + readme
    if args.sheets:
        sheets = _resolve(args.sheets)
        sheets.mkdir(parents=True, exist_ok=True)
        for p in outputs:
            if p.suffix in (".mp4", ".webm", ".gif"):
                frame_sheet(p, sheets / f"{p.parent.name}_{p.stem}{p.suffix.replace('.', '_')}.png")

    print(f"\n{'file':44s} {'pixels':>10s} {'frames / duration':>18s} {'size':>9s}")
    for p in outputs:
        dims, timing, size = describe(p)
        name = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
        print(f"{name:44s} {dims:>10s} {timing:>18s} {size:>9s}")
    # the README budget covers every GIF and the banner, also those this run left as they were
    readme_all = sorted({*readme, *(out_readme / f"{g.name}.gif" for g in GIFS), out_readme / "banner.png"})
    readme_all = [p for p in readme_all if p.is_file()] if readme else []
    if readme_all:
        print(f"README media total: {sum(p.stat().st_size for p in readme_all) / MB:.2f} MB")
    if site_videos or site_imgs:
        print(f"site media total (this run): {sum(p.stat().st_size for p in site_videos + site_imgs) / MB:.2f} MB")
    problems = check_budgets(site_videos, readme_all)
    for line in problems:
        print("BUDGET:", line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
