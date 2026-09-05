#!/usr/bin/env python
"""Branding assets: application icon, README banner, screenshots and a workflow GIF.

Writes
    assets/icon/pyALDVC.svg, pyALDVC.ico, png/pyALDVC-<size>.png   (also copied into the package)
    assets/banner.png                                              (1280 x 400, README hero)
    assets/screenshot_main.png, screenshot_strain.png, screenshot_3d.png
    assets/pyALDVC_demo.gif                                        (workflow in six frames)

Everything is rendered offscreen from synthetic data, so the assets are reproducible.
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


def make_icon() -> Path:
    from PIL import Image
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    from al_dvc.gui.app import create_application

    create_application([])  # the themed application: the screenshots must show the dark theme
    icon_dir = ASSETS / "icon"
    (icon_dir / "png").mkdir(parents=True, exist_ok=True)
    (icon_dir / "pyALDVC.svg").write_text(ICON_SVG, encoding="utf-8", newline="\n")
    renderer = QSvgRenderer(QByteArray(ICON_SVG.encode("utf-8")))
    pngs = []
    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        img = QImage(size, size, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        path = icon_dir / "png" / f"pyALDVC-{size}.png"
        img.save(str(path))
        pngs.append(path)
    frames = [Image.open(p).convert("RGBA") for p in pngs if int(p.stem.split("-")[1]) <= 256]
    frames[-1].save(
        icon_dir / "pyALDVC.ico", format="ICO", sizes=[(f.width, f.height) for f in frames], append_images=frames[:-1]
    )
    PKG_ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy(icon_dir / "png" / "pyALDVC-256.png", PKG_ASSETS / "pyALDVC.png")
    shutil.copy(icon_dir / "pyALDVC.svg", PKG_ASSETS / "pyALDVC.svg")
    return icon_dir / "pyALDVC.ico"


# --------------------------------------------------------------------------- synthetic result
def _synthetic_result():
    from al_dvc.core.config import dvcpara_default
    from al_dvc.core.pipeline import run_aldvc
    from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

    shape = (72, 80, 88)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=21)
    affine = affine_displacement(np.diag([0.02, -0.008, 0.01]), (0.6, -0.4, 0.3), centre)

    def disp(x, y, z):  # a smooth non-uniform field: stretch + a bulge, so the displacement picture has structure
        u, v, w = affine(x, y, z)
        bulge = 1.5 * np.exp(-(((x - 44) / 18) ** 2 + ((y - 40) / 16) ** 2 + ((z - 36) / 14) ** 2))
        return u + bulge, v + 0.5 * bulge, w - 0.4 * bulge

    dfm = warp_volume_lagrangian(ref, disp)
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, admm_max_iter=2, verbose=False)
    result = run_aldvc(para, [ref, dfm])
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
        pl.add_mesh(
            pv.PolyData(np.asarray(grid.points)), style="points", point_size=7.0, render_points_as_spheres=True, color=TEAL
        )
        edges = grid.extract_all_edges()
        pl.add_mesh(edges, color=ACCENT, line_width=1, opacity=0.35)
    else:
        opts = SceneOptions(field="disp_magnitude", mode="slices", colormap="turbo", background=BG_PANEL, show_outline=True)
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
    mask = np.zeros(ref.shape, dtype=bool)
    mask[6 : nz - 6, 8 : ny - 8, 8 : nx - 8] = True
    window.state.set_mask(0, mask=mask)
    window.viewer.mask_tools.set_tool("rectangle")
    app.processEvents()
    grab("03_roi")
    window.state.set_params(winsize=16, winstepsize=8, search_radius=5, admm_max_iter=2, verbose=False)
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
    from al_dvc.gui.names import select_key

    select_key(sw.field, "von_mises")
    app.processEvents()
    strain_shot = tmp / "05_strain.png"
    sw.grab().save(str(strain_shot))
    frames.append(strain_shot)
    shutil.copy(strain_shot, ASSETS / "screenshot_strain.png")
    sw.close()
    window.center_tabs.setCurrentIndex(1)
    window.view3d.mode.setCurrentIndex(0)
    app.processEvents()
    window.view3d.refresh()
    shot3d = grab("06_view3d")
    shutil.copy(shot3d, ASSETS / "screenshot_3d.png")
    window.close()
    images = [Image.open(p).convert("RGB").resize((960, 600), Image.Resampling.LANCZOS) for p in frames]
    images[0].save(
        ASSETS / "pyALDVC_demo.gif",
        save_all=True,
        append_images=images[1:],
        duration=[1200, 1500, 1500, 2500, 2500, 2500],
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
