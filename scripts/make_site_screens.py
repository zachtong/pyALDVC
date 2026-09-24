#!/usr/bin/env python
"""Screenshots of the application for the website, from a real analysis of real data.

Drives the real application offscreen: loads a reference and a deformed volume, draws a box-shaped
region of interest with the Rectangle tool, runs the analysis from the Run button's code path (backend
``auto``: the GPU when there is one), then captures

    app-slices.jpg      main window, Slices tab: displacement w over the reference volume
    app-3d.jpg          main window, 3-D view tab: the deformed node lattice with displacement arrows
    app-strain.jpg      post-processing window, Strain tab, after computing the strain
    app-statistics.jpg  post-processing window, Statistics tab: two drawn regions, the Regions page
    app-texture.jpg     texture analysis window on the reference volume (RVE sweep, autocorrelation)
    app-setup.jpg       main window before the run: region of interest and node-grid preview on the slices;
                        written to --raw (not to the site) with app-setup.json, the panels' rectangles in
                        device pixels of the capture, for annotated figures

The website's captures use the hydrogel indentation pair of the DVC Challenge 2.0 dataset (confocal,
1024 x 1024 x 306 voxels). The region of interest stops at z = 215: above it, next to the contact
with the sphere, the two frames no longer correlate (the top node layers of a full-depth run stall)::

    python scripts/make_site_screens.py --ref <Box>/Indentation/20190504_cut_01.tiff ^
        --def <Box>/Indentation/20190504_cut_02.tiff --roi-z 0:215 --out site/img

Windows are rendered at ``--scale`` device pixels per logical pixel and saved as JPG ``--width``
pixels wide, at the best quality that keeps each file under ``--max-kb``. The main window is
``--size`` logical pixels: the Slices tab needs about 1260 px for its control row and the left
column about 490 px for the tracking-mode text, so below about 2100 px some labels are cut.
``--result`` keeps the displacement result in a pickle and reuses it on the next call (to iterate
on the look without running the analysis again); ``--view3d`` and ``--suffix`` compare 3-D looks.

Every capture is checked for text that does not fit its widget; the script prints what it finds.
A few widths and one rendering setting the application gets wrong at any window size are worked
around here and reported (``work_around_layout_bugs``, ``widen_*_sidebar``, ``opaque_3d_fields``);
remove each once the application is fixed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAIN_SIZE = (2120, 1325)  # logical px (16:10): the Slices tab's control row needs ~1260 px between the columns
POST_SIZE = (1680, 1050)  # the post-processing and texture windows
LEFT_EXTRA = 50  # px beyond the left column's minimum: room for the longest combo text (tracking mode)
RIGHT_WIDTH = 360  # px of the right column (run, results, console)
SHOTS = ("setup", "slices", "3d", "strain", "statistics", "texture")
# the 3-D view: the deformed lattice, opaque (a translucent lattice blends the colours of its inside into its
# faces), with arrows every third node
VIEW3D = {
    "mode": "warped",
    "field": "disp_w",
    "opacity": 1.0,
    "arrows": True,
    "edges": False,
    "stride": 3,
    "arrow_scale": 4.0,
    "warp": 1.0,
    "azimuth": 25,
    "elevation": -5,
    "zoom": 1.15,
}
OVERLAY_ALPHA = 0.65  # field over the grey volume: the beads stay visible through it
REGION_RADIUS = 150.0  # voxels: the disc under the sphere; the far-field square has the same width
STATS_SLICE_BELOW = 2  # node layers below the slices of the main window: the Statistics tab's XY slice
JPG_QUALITIES = (92, 90, 88, 86, 84, 82, 80, 77, 74, 70)
SPIN_ARROWS, SPIN_PADDING = 18, 16  # px of a spin box that are not text (up/down buttons; frame and padding)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref", required=True, help="reference volume (any format al_dvc reads)")
    p.add_argument("--def", dest="deformed", required=True, help="deformed volume")
    p.add_argument("--out", default="site/img", help="output folder, relative to the repository (default site/img)")
    p.add_argument("--subset", type=int, default=43, help="subset size in voxels, odd (default 43)")
    p.add_argument("--step", type=int, default=12, help="node spacing in voxels (default 12)")
    p.add_argument("--search", type=int, default=14, help="search radius in voxels (default 14)")
    p.add_argument("--margin", type=int, default=8, help="voxels between the region of interest and the x, y edges")
    p.add_argument("--roi-z", default=None, help="first:last slice of the region of interest (default: all)")
    p.add_argument("--size", default="x".join(map(str, MAIN_SIZE)), help="main window size in logical px, WxH")
    p.add_argument("--scale", type=float, default=2.0, help="device pixels per logical pixel (default 2)")
    p.add_argument("--width", type=int, default=1600, help="width of the saved JPGs (default 1600)")
    p.add_argument("--max-kb", type=int, default=390, help="largest JPG size in KB (default 390)")
    p.add_argument("--result", default=None, help="pickle of the displacement result: written after a run, reused if present")
    p.add_argument("--raw", default=None, help="also keep the full-resolution PNG captures in this folder")
    p.add_argument("--only", default=",".join(SHOTS), help=f"comma-separated subset of {','.join(SHOTS)}")
    p.add_argument("--view3d", default="{}", help="JSON overrides of the 3-D view look (mode, stride, arrow_scale, ...)")
    p.add_argument("--suffix", default="", help="appended to every file name (to compare variants)")
    return p.parse_args(argv)


def configure_qt(scale: float) -> None:
    """Offscreen platform, the Windows fonts and the device pixel ratio; before Qt is imported."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if os.name == "nt":
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    os.environ["QT_SCALE_FACTOR"] = f"{scale:g}"


# --------------------------------------------------------------------------- capture
def pump(n: int = 30) -> None:
    from PySide6.QtWidgets import QApplication

    for _ in range(n):
        QApplication.processEvents()


def pixmap_to_pil(pixmap):
    from PIL import Image
    from PySide6.QtCore import QBuffer, QIODevice

    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGB")


def save_jpg(img, path: Path, width: int, max_kb: int) -> tuple[int, int]:
    """Resize to ``width`` and save at the best quality under ``max_kb``; returns (quality, bytes)."""
    from PIL import Image

    if img.width != width:
        img = img.resize((width, round(img.height * width / img.width)), Image.Resampling.LANCZOS)
    data = b""
    for quality in JPG_QUALITIES:
        for subsampling in (0, 2):  # 4:4:4 keeps coloured text sharp; 4:2:0 only when it does not fit
            out = io.BytesIO()
            img.save(out, "JPEG", quality=quality, subsampling=subsampling, optimize=True, progressive=True)
            data = out.getvalue()
            if len(data) <= max_kb * 1024:
                path.write_bytes(data)
                return quality, len(data)
    path.write_bytes(data)
    return JPG_QUALITIES[-1], len(data)


def truncated_texts(top) -> list[str]:
    """Labels, buttons, check boxes, combo and spin boxes of ``top`` whose text is wider than their room."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QAbstractButton,
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QLabel,
        QToolButton,
        QWidget,
    )

    found = []
    for w in top.findChildren(QWidget):
        if not w.isVisibleTo(top) or w.width() <= 1:
            continue
        fm = w.fontMetrics()
        if isinstance(w, QLabel):
            text = w.text()
            if not text or w.wordWrap() or "<" in text:  # wrapped or rich text is laid out by Qt
                continue
            need, room = max(fm.horizontalAdvance(line) for line in text.split("\n")), w.contentsRect().width()
        elif isinstance(w, QAbstractButton):
            text = w.text().replace("&", "")
            if not text or (isinstance(w, QToolButton) and w.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly):
                continue
            extra = 22 if isinstance(w, QCheckBox) else 12
            extra += w.iconSize().width() + 4 if not w.icon().isNull() else 0
            need, room = fm.horizontalAdvance(text) + extra, w.width()
        elif isinstance(w, QComboBox):
            text = w.currentText()
            if not text:
                continue
            need, room = fm.horizontalAdvance(text) + 28, w.width()
        elif isinstance(w, QAbstractSpinBox):
            text = w.text()
            arrows = 0 if w.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons else SPIN_ARROWS
            need, room = fm.horizontalAdvance(text), w.width() - arrows - SPIN_PADDING
        else:
            continue
        if text and need > room + 1:
            found.append(f"{type(w).__name__} {text[:60]!r}: needs {need} px, has {room}")
    return found


class Shooter:
    """Grabs a window, reports cut texts, saves the JPG (and optionally the full-resolution PNG)."""

    def __init__(self, out: Path, width: int, max_kb: int, raw: Path | None, suffix: str = "") -> None:
        self.out, self.width, self.max_kb, self.raw, self.suffix = out, width, max_kb, raw, suffix
        self.written: list[dict] = []

    def __call__(self, widget, name: str) -> Path:
        pump(40)
        cut = truncated_texts(widget)
        for line in cut:
            print(f"  [{name}] text cut: {line}")
        img = pixmap_to_pil(widget.grab())
        if self.raw is not None:
            img.save(self.raw / f"{name}.png")
        path = self.out / f"{name}.jpg"
        quality, size = save_jpg(img, path, self.width, self.max_kb)
        self.written.append({"file": path.name, "bytes": size, "quality": quality, "captured": img.size, "cut": len(cut)})
        print(
            f"  {path.name}: captured {img.size[0]} x {img.size[1]}, saved {self.width} px wide, q{quality}, {size / 1024:.0f} KB"
        )
        return path


# --------------------------------------------------------------------------- work-arounds
def work_around_layout_bugs(window) -> None:
    """Widths the application gets wrong at any window size (reported; remove each once fixed in the GUI).

    * the viewer's Background combo sizes itself on first show, before the frames exist, so
      "Reference (frame 0)" is cut;
    * the elapsed-time label keeps the last in-run estimate "(~N s left)" after the run finished.
    """
    from PySide6.QtWidgets import QComboBox

    combo = window.viewer.background_frame
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    combo.setMinimumWidth(max(combo.minimumWidth(), combo.sizeHint().width()))
    rp = window.run_panel
    text = rp._elapsed.text()
    if "(" in text:
        rp._elapsed.setText(text.split("(")[0].strip())


def opaque_3d_fields() -> None:
    """The 3-D view draws every field with ``nan_opacity=0``, which puts the mesh in VTK's translucent pass even at
    opacity 1: the far faces of the deformed lattice blend through the near ones (the dimple's centre, w = -11,
    renders dark red instead of dark purple). A lattice has no NaN left to hide (its NaN cells are cut away before
    it is drawn), so an opaque field is drawn with ``nan_opacity=1``."""
    import pyvista as pv

    original = pv.Plotter.add_mesh
    if getattr(original, "_site_screens_patch", False):
        return

    def add_mesh(self, mesh, *args, **kwargs):
        if kwargs.get("nan_opacity") == 0.0 and float(kwargs.get("opacity", 1.0)) >= 1.0:
            kwargs["nan_opacity"] = 1.0
        return original(self, mesh, *args, **kwargs)

    add_mesh._site_screens_patch = True
    pv.Plotter.add_mesh = add_mesh


def widen_sidebar(window, anchor, extra: int) -> None:
    """Widen the fixed-width sidebar (a scroll area holding a fixed-width widget) that contains ``anchor``."""
    from PySide6.QtWidgets import QScrollArea

    for area in window.findChildren(QScrollArea):
        side = area.widget()
        if side is not None and side.isAncestorOf(anchor):
            side.setFixedWidth(side.width() + extra)
            area.setFixedWidth(area.width() + extra)
            pump()
            return
    raise RuntimeError(f"no sidebar holds {anchor!r}")


def widen_strain_sidebar(sw) -> None:
    """The strain window's sidebar is fixed at 330 px, but its fit-window row (three spin boxes and 'Cube') needs
    248 px beside a 150 px label: the spin boxes overlap and 'Cube' is cut. Widen the sidebar by the difference."""
    row = sw.fit_window_lock.parentWidget()
    widen_sidebar(sw, row, max(0, row.minimumSizeHint().width() - row.width()) + 8)


def widen_statistics_sidebar(post) -> None:
    """The Statistics tab's sidebar is fixed at 330 px: the region buttons ('+ Sphere', '+ Cylinder', 'Rectangle')
    are cut in its four-column grid. Widen it until the widest button fits."""
    from PySide6.QtWidgets import QPushButton

    panel = post.analysis.regions_panel
    buttons = [b for b in panel.findChildren(QPushButton) if b.text() and b.isVisibleTo(post)]
    short = max((b.sizeHint().width() - b.width() for b in buttons), default=0)
    if short > 0:
        widen_sidebar(post, panel, 4 * short + 8)


# --------------------------------------------------------------------------- analysis
def load_pair(ref: str, deformed: str):
    from al_dvc.io.volume_io import load_volume

    vols = []
    for path in (ref, deformed):
        t0 = time.perf_counter()
        vol = load_volume(path)
        print(f"read {Path(path).name}: {vol.shape} {vol.dtype} in {time.perf_counter() - t0:.1f} s")
        vols.append(vol)
    if vols[0].shape != vols[1].shape:
        raise SystemExit(f"the volumes differ in size: {vols[0].shape} and {vols[1].shape}")
    return vols


def draw_roi(window, shape, margin: int, depth: tuple[int, int] | None) -> None:
    """A rectangle on the XY slice, through ``depth`` (first, last slice) or the whole volume, as the
    Rectangle tool with its Depth control draws it."""
    from al_dvc.gui.mask_editor import MaskOp

    nz, ny, nx = shape
    tools = window.viewer.mask_tools
    tools.set_tool("rectangle")
    if depth is not None:
        tools.set_depth("range", depth[0], depth[1])
    else:
        tools.set_depth("all")
    op = MaskOp(
        "rectangle",
        plane="xy",
        points=((margin, margin), (nx - 1 - margin, ny - 1 - margin)),
        depth=depth,
        mode="replace",
    )
    window.state.apply_mask_op(op)
    tools.set_tool("none")
    pump()


def set_parameters(window, args) -> None:
    half = (args.subset - 1) // 2 * 2  # the panel shows the odd span 2h + 1; the parameter is the even 2h
    window.state.set_params(winsize=(half,) * 3, winstepsize=(args.step,) * 3, search_radius=(args.search,) * 3, backend="auto")
    pump()


def run_analysis(window) -> dict:
    """The Run button's code path; waits for the worker and returns what was measured."""
    from al_dvc.gui.app_state import RunState

    t0 = time.perf_counter()
    window.run_panel.start()
    if window.state.run_state != RunState.RUNNING:
        raise SystemExit("the run did not start (see the log above)")
    while not window.run_panel.wait(250):
        pump(5)
    pump(60)
    elapsed = time.perf_counter() - t0
    res = window.state.results
    if res is None or not res.result_disp:
        raise SystemExit(f"the run failed: {window.run_panel._message.text()}")
    return {"elapsed_s": round(elapsed, 1), "nodes": int(res.dvc_mesh.n_nodes), "timings": dict(res.timings)}


def describe(res) -> dict:
    """The numbers the Results panel reports, and the w range."""
    import numpy as np

    from al_dvc.export.export_utils import converged_fraction

    fr = res.result_disp[0]
    w = fr.U[:, 2]
    return {
        "grid": tuple(int(v) for v in res.dvc_mesh.grid_shape[::-1]),
        "converged": round(float(converged_fraction(res, fr)), 3),
        "median_zncc": round(float(np.nanmedian(fr.zncc)), 3),
        "w_p0.5": round(float(np.nanpercentile(w, 0.5)), 2),
        "w_p99.5": round(float(np.nanpercentile(w, 99.5)), 2),
        "w_min": round(float(np.nanmin(w)), 2),
    }


def dimple(res) -> tuple[float, float, int]:
    """(x, y) of the indentation centre -- the centroid of the most negative w on the second-highest node layer
    inside the region of interest -- and the z of that layer (the slices go through it)."""
    import numpy as np

    mesh = res.dvc_mesh
    valid = mesh.to_grid(np.asarray(mesh.node_valid, dtype=float)) > 0.5
    w = np.where(valid, mesh.to_grid(res.result_disp[0].U[:, 2]), np.nan)
    layers = np.flatnonzero(valid.any(axis=(1, 2)))
    k = int(layers[-2] if layers.size > 1 else layers[-1])
    layer = w[k]
    cut = np.nanpercentile(layer, 2)
    iy, ix = np.nonzero(layer <= cut)
    return float(np.mean(mesh.x0[ix])), float(np.mean(mesh.y0[iy])), int(mesh.z0[k])


# --------------------------------------------------------------------------- scenes
SETUP_PANELS = {  # attribute paths on the main window whose rectangles app-setup.json records
    "volumes": "volume_panel",
    "parameters": "param_panel",
    "left_column": "_left_column",
    "viewer": "center_tabs",
    "right_column": "_right_column",
    "run": "run_panel",
    "results": "results_panel",
    "console": "console",
    "roi_tools": "viewer.mask_tools",
    "grid_toggle": "viewer.show_mesh",
    "post_processing": "results_panel._analysis_group",
}


def panel_rects(window, scale: float) -> dict:
    """Rectangles [x, y, w, h] of SETUP_PANELS in device pixels of a grab of ``window``."""
    from PySide6.QtCore import QPoint

    rects = {}
    for name, path in SETUP_PANELS.items():
        widget = window
        for attr in path.split("."):
            widget = getattr(widget, attr, None)
            if widget is None:
                break
        if widget is None or not widget.isVisible():
            continue
        top_left = widget.mapTo(window, QPoint(0, 0))
        rects[name] = [round(v * scale) for v in (top_left.x(), top_left.y(), widget.width(), widget.height())]
    return rects


def shot_setup(window, shoot, scale: float) -> None:
    """Before the run: the reference volume on the Slices tab, the region of interest and the node grid the
    parameters would place (the preview), as a user sees them just before pressing Run."""
    window.viewer.show_mesh.setChecked(True)
    window.param_panel.sections["units"].set_expanded(False)  # voxel units: nothing to read there
    window.center_tabs.setCurrentIndex(0)
    arrange(window)
    work_around_layout_bugs(window)
    tidy_console(window)
    path = shoot(window, "app-setup" + shoot.suffix)
    rects = panel_rects(window, scale)
    target = path.with_suffix(".json")
    target.write_text(json.dumps({"scale": scale, "panels": rects}, indent=1), encoding="utf-8")
    print(f"  {target.name}: {len(rects)} panel rectangles")


def arrange(window) -> None:
    """Column widths: the left one a little wider than its minimum (the longest combo text fits), the right one
    fixed, the view in the middle takes the rest (the splitter keeps the view at its own minimum)."""
    pump()
    total = sum(window._splitter.sizes())
    left = window._left_column.minimumWidth() + LEFT_EXTRA
    window._splitter.setSizes([left, total - left - RIGHT_WIDTH, RIGHT_WIDTH])
    pump()


def tidy_console(window) -> None:
    """The console from its left edge, scrolled to the latest line."""
    pump()
    view = window.console._view
    view.horizontalScrollBar().setValue(0)
    view.verticalScrollBar().setValue(view.verticalScrollBar().maximum())


def shot_slices(window, where, clim, shoot) -> None:
    """Slices tab: w over the reference volume, slices through the indentation, no node grid."""
    st = window.state
    x, y, z = where
    st.slice_index.update({"z": int(z), "y": int(round(y)), "x": int(round(x))})
    window.viewer.show_mesh.setChecked(False)
    window.results_panel.select_field("disp_w")
    # the reference under the field: the pairing in which both are drawn where the material was
    st.set_display(
        background_frame=0, color_auto=False, color_min=clim[0], color_max=clim[1], overlay_alpha=OVERLAY_ALPHA, colormap="turbo"
    )
    window.viewer._fill_background_choices()
    window.viewer._load_background(reset_sliders=True)
    window.results_panel.refresh()
    window.param_panel.sections["units"].set_expanded(False)  # voxel units: nothing to read there
    window.center_tabs.setCurrentIndex(0)
    arrange(window)
    work_around_layout_bugs(window)
    tidy_console(window)
    shoot(window, "app-slices" + shoot.suffix)


def shot_3d(window, shoot, opts: dict) -> None:
    """3-D view tab: the deformed node lattice coloured by the field, with displacement arrows."""
    import al_dvc.gui.panels.view3d as view3d_module
    from al_dvc.gui.names import select_key

    v = window.view3d
    if opts.get("opaque_fix", True):
        opaque_3d_fields()
    window.results_panel.select_field(opts.get("field", "disp_w"))
    window.state.set_display(overlay_alpha=float(opts.get("opacity", 1.0)))
    window.results_panel.refresh()
    window.center_tabs.setCurrentIndex(1)
    arrange(window)
    pump()
    # the offscreen (static) backend renders a fixed 900 x 640 image and scales it into the view; render it at
    # the view's own size instead, as the desktop's interactive view does
    view3d_module.STATIC_SIZE = (v._image.width(), v._image.height())
    v._updating = True  # set every control, then draw once
    try:
        select_key(v.mode, opts.get("mode", "warped"))
        v.arrows.setChecked(opts.get("arrows", True))
        v.edges.setChecked(opts.get("edges", False))
        v.stride.setValue(opts.get("stride", VIEW3D["stride"]))
        v.arrow_scale.setValue(opts.get("arrow_scale", VIEW3D["arrow_scale"]))
        v.warp_scale.setValue(opts.get("warp", VIEW3D["warp"]))
        v.iso.setValue(opts.get("iso", 0.5))
        v.volume_slices.setChecked(opts.get("volume_slices", False))
        v.outline.setChecked(True)
        for axis, show in opts.get("planes", {}).items():  # slices mode: which of the XY (z), XZ (y), YZ (x) planes
            v.slice_visible[axis].setChecked(bool(show))
        if "slice_z" in opts:
            window.state.slice_index["z"] = int(opts["slice_z"])
            v._sync_slice_spins()
        select_key(v.camera, "iso")
        v.azimuth.setValue(opts.get("azimuth", 0))
        v.elevation.setValue(opts.get("elevation", 0))
        v.zoom.setValue(opts.get("zoom", 1.0))
    finally:
        v._updating = False
    v._camera = "iso"
    v._camera_reset_pending = True
    v._update_enabled()
    v.refresh()
    tidy_console(window)
    shoot(window, "app-3d" + shoot.suffix)


def shot_strain(window, shoot, field: str, where) -> None:
    """Post-processing window, Strain tab, after Compute strain with the default settings."""
    from al_dvc.gui.names import select_key

    sw = window.open_strain_window()
    sw.resize(*POST_SIZE)
    sw.show_tab("strain")
    pump()
    widen_strain_sidebar(sw)
    t0 = time.perf_counter()
    sw.compute()
    while not sw.wait(250):
        pump(5)
    pump(60)
    print(f"  strain computed in {time.perf_counter() - t0:.1f} s")
    select_key(sw.field, field)
    x, y, z = where
    sw.canvas.set_slices(int(z), int(round(y)), int(round(x)))
    pump()
    shoot(sw, "app-strain" + shoot.suffix)


def shot_statistics(window, shoot, regions: list[dict], field: str, where) -> None:
    """Post-processing window, Statistics tab: two regions drawn on the XY slice, the Regions page."""
    from PySide6.QtWidgets import QScrollArea

    from al_dvc.gui.names import select_key

    post = window.open_statistics()
    post.resize(*POST_SIZE)
    tab = post.analysis
    pump()
    panel = tab.regions_panel
    widen_statistics_sidebar(post)
    for spec in regions:
        region = panel.add_drawn(spec["plane"], spec["outline"], spec["points"])  # what a drawn outline adds
        if region is None:
            raise SystemExit(f"region {spec['name']!r} was refused: {panel.hint.text()}")
        pump()
        panel._edits["name"].setText(spec["name"])  # renamed in the region editor, as a user would
        if not panel.apply_editor():
            raise SystemExit(f"region {spec['name']!r}: {panel.hint.text()}")
        pump()
    select_key(tab.shown, field)
    x, y, z = where
    tab.canvas.set_slices(int(z), int(round(y)), int(round(x)))
    tab.results_tabs.setCurrentIndex(3)  # Regions
    tab.sections["motion"].set_expanded(False)  # nothing removed here; the regions list moves into view
    tab.wait(120_000)
    pump(60)
    for area in post.findChildren(QScrollArea):  # the sidebar: scrolled so the regions list is in view
        if area.widget() is not None and area.widget().isAncestorOf(panel):
            area.ensureWidgetVisible(panel, 0, 8)
    pump()
    shoot(post, "app-statistics" + shoot.suffix)


def shot_texture(window, shoot, cube: int) -> None:
    """Texture analysis window on the reference volume: RVE sweep, then the autocorrelation of the cube the sweep
    settled on (or of a ``cube`` of that edge, when not 0)."""
    from PySide6.QtWidgets import QScrollArea

    tw = window.open_texture_window()
    tw.resize(*POST_SIZE)
    pump()
    tw.use_dvc_roi()
    tw.go_to_step(tw.TAB_SWEEP)
    tw.centre_on_region()
    pump()
    t0 = time.perf_counter()
    tw.run_sweep_analysis()
    while not tw.wait(250):
        pump(5)
    pump(30)
    if tw.sweep_size() is not None:
        tw.use_sweep_size()
    tw.go_to_step(tw.TAB_ACF)
    if cube:
        tw.cube_size.setValue(cube)
    tw.analyse()
    while not tw.wait(250):
        pump(5)
    pump(60)
    res = tw.result
    if res is None:
        raise SystemExit("texture analysis: no result")
    print(
        f"  texture analysis in {time.perf_counter() - t0:.1f} s: {res.status}, "
        f"1/e length x {res.length('x'):.1f}, y {res.length('y'):.1f}, z {res.length('z'):.1f} voxel"
    )
    for area in tw.findChildren(QScrollArea):  # the sidebar: scrolled so the subset suggestion is in view
        if area.widget() is not None and area.widget().isAncestorOf(tw._suggestion_box):
            area.ensureWidgetVisible(tw._suggestion_box, 0, 8)
    pump()
    shoot(tw, "app-texture" + shoot.suffix)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    args = parse_args(argv)
    configure_qt(args.scale)
    import numpy as np

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw) if args.raw else None
    if raw is not None:
        raw.mkdir(parents=True, exist_ok=True)
    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = set(wanted) - set(SHOTS)
    if unknown:
        raise SystemExit(f"unknown shots {sorted(unknown)}; use {SHOTS}")
    depth = tuple(int(v) for v in args.roi_z.split(":")) if args.roi_z else None

    ref, dfm = load_pair(args.ref, args.deformed)  # before any window exists: nothing waits on the read

    from al_dvc.gui.app import MainWindow, create_application

    app = create_application([sys.argv[0]])
    print(f"Qt platform {app.platformName()}, device pixel ratio {app.devicePixelRatio():g}")
    window = MainWindow()
    window.resize(*(int(v) for v in args.size.lower().split("x")))
    window.show()
    pump()
    window.state.set_volume_arrays([ref, dfm], [Path(args.ref).name, Path(args.deformed).name])
    pump()
    draw_roi(window, ref.shape, args.margin, depth)
    set_parameters(window, args)

    shoot = Shooter(out, args.width, args.max_kb, raw, args.suffix)
    if "setup" in wanted:  # before any result exists; a figure source, so it goes to --raw, not the site
        folder = raw if raw is not None else out
        shot_setup(window, Shooter(folder, args.width, args.max_kb, raw, args.suffix), args.scale)

    cache = Path(args.result) if args.result else None
    if cache is not None and cache.is_file():
        with cache.open("rb") as fh:
            result, info = pickle.load(fh)
        window.state.set_results(result, uids=[v.uid for v in window.state.volumes])
        print(f"reused the result in {cache}")
    else:
        info = run_analysis(window)
        if cache is not None:
            with cache.open("wb") as fh:
                pickle.dump((window.state.results, info), fh)
    pump(60)
    res = window.state.results
    info = {**info, **describe(res)}
    print(
        f"analysis: {info['nodes']:,} nodes {info['grid']} in {info['elapsed_s']} s; converged {info['converged']:.1%}, "
        f"median ZNCC {info['median_zncc']}; w from {info['w_min']} (0.5 % {info['w_p0.5']}) to {info['w_p99.5']} voxel"
    )
    x, y, z = dimple(res)
    print(f"indentation centre x {x:.0f}, y {y:.0f}; slices at z {z}")
    clim = (float(np.floor(info["w_p0.5"])), float(np.ceil(max(info["w_p99.5"], 0.5))))

    if "slices" in wanted:
        shot_slices(window, (x, y, z), clim, shoot)
    if "3d" in wanted:
        shot_3d(window, shoot, {**VIEW3D, **json.loads(args.view3d)})
    if "strain" in wanted:
        shot_strain(window, shoot, "ezz", (x, y, z))
    if "statistics" in wanted:
        # two regions drawn on the XY slice and extruded through the node grid (what drawing one does): a disc
        # under the sphere and a square in a corner, far from it
        mesh = res.dvc_mesh
        r = REGION_RADIUS
        corner = (float(mesh.x0[1]), float(mesh.y0[1]))
        regions = [
            {"name": "under the sphere", "plane": "xy", "outline": "ellipse", "points": [[x - r, y - r], [x + r, y + r]]},
            {
                "name": "far field",
                "plane": "xy",
                "outline": "rect",
                "points": [list(corner), [corner[0] + 2 * r, corner[1] + 2 * r]],
            },
        ]
        shot_statistics(window, shoot, regions, "disp_w", (x, y, z - STATS_SLICE_BELOW * float(mesh.spacing[2])))
    if "texture" in wanted:
        shot_texture(window, shoot, 0)  # 0: the cube the RVE analysis settled on
    window.settle_workers(30_000)
    window.state.dirty = False
    for w in (getattr(window, "strain_window", None), getattr(window, "texture_window", None)):
        if w is not None:
            w.close()
    window.close()
    total = sum(item["bytes"] for item in shoot.written)
    files = ", ".join(f"{i['file']} ({i['bytes'] / 1024:.0f} KB)" for i in shoot.written)
    print(f"written: {files}; total {total / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
