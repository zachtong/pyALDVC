#!/usr/bin/env python
"""Branding assets: application icon, README banner, screenshots and a workflow GIF.

Writes
    assets/icon/pyALDVC.svg, pyALDVC.ico, png/pyALDVC-<size>.png   (also copied into the package)
        from assets/icon/pyALDVC-master.svg or pyALDVC-master.png when present (the hand-made icon),
        otherwise from the built-in SVG below
    assets/banner.png                                              (1280 x 400, README hero)
    assets/screenshot_main.png, screenshot_strain.png, screenshot_3d.png
    assets/pyALDVC_demo.gif                                        (workflow in seven frames)

Everything is rendered offscreen from synthetic data (an open-cell foam with a localised vortex
under compression), so the assets are reproducible.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ASSETS = ROOT / "assets"
PKG_ASSETS = ROOT / "src" / "al_dvc" / "gui" / "assets"

BG_DARKEST = "#0b0f1a"
BG_PANEL = "#141929"
BG_INPUT = "#1a1f33"
ACCENT = "#6366f1"
ACCENT_LIGHT = "#818cf8"
TEAL = "#2dd4bf"
TEXT_PRIMARY = "#e2e8f0"
TEXT_SECONDARY = "#94a3b8"

ICON_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#1e2440"/>
      <stop offset="1" stop-color="#0f1322"/>
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="240" height="240" rx="52" fill="url(#bg)" stroke="{ACCENT}" stroke-width="6"/>
  <g fill="none" stroke="{ACCENT_LIGHT}" stroke-width="5" stroke-linejoin="round" opacity="0.9">
    <path d="M166 40 L214 62 L214 112 L166 134 L118 112 L118 62 Z"/>
    <path d="M118 62 L166 84 L214 62 M166 84 L166 134"/>
  </g>
  <text x="34" y="132" font-family="Segoe UI, Helvetica Neue, Arial, sans-serif" font-weight="800" font-size="92"
        fill="{TEXT_PRIMARY}" letter-spacing="-2">AL</text>
  <text x="34" y="216" font-family="Segoe UI, Helvetica Neue, Arial, sans-serif" font-weight="800" font-style="italic"
        font-size="84" fill="{TEAL}" letter-spacing="-1">DVC</text>
</svg>
"""


ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
MASTER_CANDIDATES = ("pyALDVC-master.svg", "pyALDVC-master.png")


def _icon_master(icon_dir: Path) -> Path | None:
    """The hand-made icon master, if the designer dropped one in ``assets/icon``."""
    for name in MASTER_CANDIDATES:
        if (icon_dir / name).is_file():
            return icon_dir / name
    return None


def _render_master(master: Path | None, size: int):
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    if master is not None and master.suffix.lower() == ".png":
        src = QImage(str(master))
        if src.isNull():
            raise ValueError(f"cannot read icon master {master}")
        return src.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    svg = master.read_bytes() if master is not None else ICON_SVG.encode("utf-8")
    renderer = QSvgRenderer(QByteArray(svg))
    if not renderer.isValid():
        raise ValueError(f"invalid SVG icon master {master}")
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    renderer.render(painter)
    painter.end()
    return img


def make_icon() -> Path:
    """PNG sizes, the Windows ICO and the package copy, from the master or the built-in SVG."""
    from PIL import Image

    from al_dvc.gui.app import create_application

    create_application([])  # the themed application: the screenshots must show the dark theme
    icon_dir = ASSETS / "icon"
    (icon_dir / "png").mkdir(parents=True, exist_ok=True)
    master = _icon_master(icon_dir)
    print("icon master:", master if master else "built-in SVG")
    svg_out = icon_dir / "pyALDVC.svg"
    if master is None:
        svg_out.write_text(ICON_SVG, encoding="utf-8", newline="\n")
    elif master.suffix.lower() == ".svg":
        shutil.copy(master, svg_out)
    elif svg_out.is_file():
        svg_out.unlink()  # a PNG master has no vector version
    pngs = []
    for size in ICON_SIZES:
        path = icon_dir / "png" / f"pyALDVC-{size}.png"
        _render_master(master, size).save(str(path))
        pngs.append(path)
    frames = [Image.open(p).convert("RGBA") for p in pngs if int(p.stem.split("-")[1]) <= 256]
    frames[-1].save(
        icon_dir / "pyALDVC.ico", format="ICO", sizes=[(f.width, f.height) for f in frames], append_images=frames[:-1]
    )
    PKG_ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy(icon_dir / "png" / "pyALDVC-256.png", PKG_ASSETS / "pyALDVC.png")
    pkg_svg = PKG_ASSETS / "pyALDVC.svg"
    if svg_out.is_file():
        shutil.copy(svg_out, pkg_svg)
    elif pkg_svg.is_file():
        pkg_svg.unlink()
    return icon_dir / "pyALDVC.ico"


# --------------------------------------------------------------------------- synthetic result
DEMO_SHAPE = (200, 224, 256)  # (nz, ny, nx)
DEMO_PARAMS = dict(winsize=24, winstepsize=8, search_radius=10, admm_max_iter=3, verbose=False)
DEMO_ARROWS = dict(stride=2, scale=2.0)  # arrow subsampling and length for the 3-D looks
DEMO_WARP_SCALE = 6.0  # exaggeration of the deformed lattice in the gif finale
LATTICE_RATE = (2, 2, 2)  # every second node in the banner lattice panel
DEMO_MARGIN = (12, 16, 16)  # voxels outside the region of interest per side (z, y, x)


def foam_volume(shape=DEMO_SHAPE, seed=7, strut_sigma=4.0, fill=0.55) -> np.ndarray:
    """A micro-CT-like open-cell foam: thresholded Gaussian random field with textured struts."""
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal(shape), sigma=strut_sigma, mode="nearest")
    struts = (field > np.quantile(field, 1 - fill)).astype(np.float32)
    struts = gaussian_filter(struts, sigma=1.2, mode="nearest")
    texture = gaussian_filter(rng.standard_normal(shape), sigma=1.6, mode="nearest")
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    vol = 0.15 + 0.85 * struts * (0.7 + 0.3 * texture) + 0.08 * texture * (1 - struts)
    vol += 0.01 * rng.standard_normal(shape)
    return np.ascontiguousarray(np.clip(vol, 0, 1), dtype=np.float32)


def vortex_displacement(shape=DEMO_SHAPE, angle_deg=18.0, radius=45.0, height=45.0, compress=0.012, poisson=0.35):
    """A localised twist about the z axis that fades away from the centre, on top of uniaxial compression.

    The displacement magnitude is a torus around the centre: rings on the xy planes, two lobes on
    the xz and yz planes and a doughnut as an iso-surface.
    """
    nz, ny, nx = shape
    cx, cy, cz = (nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2
    theta0 = np.deg2rad(angle_deg)

    def disp(x, y, z):
        xr, yr, zr = x - cx, y - cy, z - cz
        theta = theta0 * np.exp(-(xr**2 + yr**2) / (2 * radius**2)) * np.exp(-(zr**2) / (2 * height**2))
        c, s = np.cos(theta), np.sin(theta)
        u = xr * c - yr * s - xr + poisson * compress * xr
        v = xr * s + yr * c - yr + poisson * compress * yr
        w = -compress * zr
        return u, v, w

    return disp


def _synthetic_result():
    from al_dvc.core.config import dvcpara_default
    from al_dvc.core.pipeline import run_aldvc
    from al_dvc.synthetic import warp_volume_lagrangian

    ref = foam_volume()
    dfm = warp_volume_lagrangian(ref, vortex_displacement())
    result = run_aldvc(dvcpara_default(**DEMO_PARAMS), [ref, dfm])
    return ref, dfm, result


def _render_3d(kind: str, ref, result, size=(520, 520)) -> np.ndarray:
    """Offscreen pyvista renders for the banner panels: volume slices, node lattice, displacement."""
    import pyvista as pv

    from al_dvc.gui.view3d_scene import SceneOptions, build_scene, node_grid, volume_slice_planes

    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background(BG_PANEL)
    if kind == "volume":
        planes = volume_slice_planes(ref, {"z": ref.shape[0] // 2, "y": ref.shape[1] // 2, "x": ref.shape[2] // 2}, (1, 1, 1))
        for plane in planes.values():
            pl.add_mesh(plane, scalars="intensity", cmap="gray", show_scalar_bar=False)
        nz, ny, nx = ref.shape
        box = pv.ImageData(dimensions=(2, 2, 2), spacing=(nx - 1, ny - 1, nz - 1)).outline()
        pl.add_mesh(box, color=ACCENT_LIGHT, line_width=2)
    elif kind == "lattice":
        grid = node_grid(result, 0, ("disp_magnitude",))
        pl.add_mesh(grid.outline(), color=ACCENT_LIGHT, line_width=2)
        nx, ny, nz = grid.dimensions
        grid = grid.extract_subset((0, nx - 1, 0, ny - 1, 0, nz - 1), LATTICE_RATE)  # readable at the demo step
        pl.add_mesh(
            pv.PolyData(np.asarray(grid.points)), style="points", point_size=7.0, render_points_as_spheres=True, color=TEAL
        )
        edges = grid.extract_all_edges()
        pl.add_mesh(edges, color=ACCENT, line_width=1, opacity=0.35)
    else:
        opts = SceneOptions(
            field="disp_magnitude",
            mode="slices",
            show_arrows=True,
            arrow_stride=DEMO_ARROWS["stride"],
            arrow_scale=DEMO_ARROWS["scale"],
            colormap="turbo",
            background=BG_PANEL,
            show_outline=True,
            title="Displacement",
        )
        build_scene(pl, result, opts, None)
        pl.remove_scalar_bar()
        pl.hide_axes()
    pl.view_isometric()
    pl.camera.zoom(1.15 if kind == "lattice" else 1.25)
    return np.asarray(pl.screenshot(return_img=True))


def make_banner(ref, result) -> Path:
    fig = plt.figure(figsize=(16, 5), dpi=80, facecolor=BG_DARKEST)
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    ax_bg.axis("off")
    Y, X = np.mgrid[0:200, 0:400] / 200.0
    R = np.sqrt((X - 0.35) ** 2 + (Y - 0.5) ** 2)
    gradient = np.zeros((*R.shape, 4))
    base = np.array([0.043, 0.059, 0.102])
    glow = np.array([0.388, 0.400, 0.945])
    for i in range(3):
        gradient[:, :, i] = base[i] + (glow[i] - base[i]) * np.exp(-(R**2) / 0.15) * 0.12
    gradient[:, :, 3] = 1.0
    ax_bg.imshow(gradient, extent=[0, 1, 0, 1], aspect="auto", interpolation="bicubic")
    rng = np.random.default_rng(42)
    speckle = gaussian_filter(rng.standard_normal((100, 200)), sigma=2.0)
    speckle = (speckle - speckle.min()) / (speckle.max() - speckle.min())
    ax_bg.imshow(speckle, extent=[0, 1, 0, 1], aspect="auto", cmap="gray", alpha=0.03, interpolation="bicubic")
    panels = [("volume", "Volume"), ("lattice", "Node lattice"), ("displacement", "Displacement")]
    x, w, gap, y, h = 0.04, 0.165, 0.03, 0.12, 0.66
    for i, (kind, title) in enumerate(panels):
        img = _render_3d(kind, ref, result)
        ax = fig.add_axes([x + i * (w + gap), y, w, h])
        ax.imshow(img)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(ACCENT)
        fig.text(
            x + i * (w + gap) + w / 2, y + h + 0.05, title, ha="center", color=TEXT_SECONDARY, fontsize=13, family="monospace"
        )
        if i < 2:
            fig.text(
                x + (i + 1) * (w + gap) - gap / 2, y + h / 2, "\u2192", ha="center", va="center", color=ACCENT_LIGHT, fontsize=22
            )
    # logo + title on the right
    from matplotlib.image import imread

    logo = imread(str(ASSETS / "icon" / "png" / "pyALDVC-256.png"))
    ax_logo = fig.add_axes([0.605, 0.30, 0.10, 0.32])
    ax_logo.imshow(logo)
    ax_logo.axis("off")
    fig.text(0.72, 0.60, "pyALDVC", color=TEXT_PRIMARY, fontsize=40, weight="bold", family="monospace", va="center")
    fig.text(
        0.72,
        0.42,
        "Augmented Lagrangian\nDigital Volume Correlation\nin Python",
        color=TEXT_SECONDARY,
        fontsize=14,
        va="center",
        linespacing=1.4,
    )
    ax_bg.plot([0.04, 0.96], [0.06, 0.06], color=ACCENT, lw=1.2, alpha=0.7)
    out = ASSETS / "banner.png"
    fig.savefig(out, dpi=80, facecolor=BG_DARKEST)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- screenshots + gif
def make_screens(ref, dfm) -> list[Path]:
    from PIL import Image
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow, create_application
    from al_dvc.gui.names import select_key

    app = QApplication.instance() or create_application([])
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    frames: list[Path] = []
    tmp = Path(tempfile.mkdtemp(prefix="pyaldvc_brand_"))

    def grab(name: str) -> Path:
        app.processEvents()
        path = tmp / f"{name}.png"
        window.grab().save(str(path))
        frames.append(path)
        return path

    grab("01_empty")
    window.state.set_volume_arrays([ref, dfm], ["scan_000.tif", "scan_001.tif"])
    app.processEvents()
    grab("02_loaded")
    nz, ny, nx = ref.shape
    mz, my, mx = DEMO_MARGIN
    mask = np.zeros(ref.shape, dtype=bool)
    mask[mz : nz - mz, my : ny - my, mx : nx - mx] = True
    window.state.set_mask(0, mask=mask)
    window.viewer.mask_tools.set_tool("rectangle")
    app.processEvents()
    grab("03_roi")
    window.state.set_params(**DEMO_PARAMS)
    window.run_panel.start()
    window.run_panel.wait(600_000)
    app.processEvents()
    main_shot = grab("04_result")
    shutil.copy(main_shot, ASSETS / "screenshot_main.png")
    sw = window.open_strain_window()
    sw.resize(1300, 800)
    sw.compute()
    sw.wait(600_000)
    app.processEvents()
    select_key(sw.field, "von_mises")
    app.processEvents()
    strain_shot = tmp / "05_strain.png"
    sw.grab().save(str(strain_shot))
    frames.append(strain_shot)
    shutil.copy(strain_shot, ASSETS / "screenshot_strain.png")
    sw.close()
    window.center_tabs.setCurrentIndex(1)
    select_key(window.view3d.mode, "slices")
    window.view3d.arrows.setChecked(True)
    window.view3d.stride.setValue(DEMO_ARROWS["stride"])
    window.view3d.arrow_scale.setValue(DEMO_ARROWS["scale"])
    app.processEvents()
    window.view3d.refresh()
    shot3d = grab("06_view3d_slices")
    shutil.copy(shot3d, ASSETS / "screenshot_3d.png")
    select_key(window.view3d.mode, "warped")
    window.view3d.arrows.setChecked(False)
    window.view3d.warp_scale.setValue(DEMO_WARP_SCALE)
    app.processEvents()
    window.view3d.refresh()
    grab("07_view3d_warped")
    window.close()
    images = [Image.open(p).convert("RGB").resize((960, 600), Image.Resampling.LANCZOS) for p in frames]
    images[0].save(
        ASSETS / "pyALDVC_demo.gif",
        save_all=True,
        append_images=images[1:],
        duration=[1200, 1500, 1500, 2500, 2500, 2200, 2800],
        loop=0,
        optimize=True,
    )
    return frames


def main(argv=None) -> int:
    ASSETS.mkdir(exist_ok=True)
    ico = make_icon()
    print("icon:", ico)
    ref, dfm, result = _synthetic_result()
    print("banner:", make_banner(ref, result))
    frames = make_screens(ref, dfm)
    print("screenshots:", len(frames), "gif:", ASSETS / "pyALDVC_demo.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
