#!/usr/bin/env python
"""Report on the light theme and the Settings menu: every window of the application in the dark and the light theme.

Builds the application offscreen with private settings, draws a region of interest on a synthetic
speckle pair, runs a small analysis through the GUI worker, opens the post-processing, texture and
guide windows, captures everything in the dark theme, switches live to the light theme (as
*Settings > Theme > Light* does) and captures it again, then switches back and checks that the dark
stylesheet is the one it started from. Writes ``reports/theme.pdf``: a text page (where the setting
lives, what follows the switch, timings, checks, limitations), the two palettes with their contrast,
and one page per window with the dark capture above the light one. ``--shots DIR`` keeps the PNG
captures. Reports are generated, never hand-edited.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":  # the offscreen platform has no font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
os.environ["PYALDVC_LANGUAGE"] = "en"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402

REPORT = ROOT / "reports" / "theme.pdf"
SHAPE = (56, 64, 72)  # (nz, ny, nx)
QUICK_SHAPE = (40, 44, 48)
# the dark stylesheet of release 1.0.1, arrow folder replaced by "<arrows>": the dark theme must not change
# (the same value as tests/test_theme.py DARK_101_SHA256; change both together, and only on purpose)
DARK_101_SHA256 = "0eeab19fa14d850782cff78c637b598cd6460d8384de4957c3321405d37a0efc"
PALETTE_ROWS = (
    "BG_DARKEST",
    "BG_SIDE_COLUMN",
    "BG_PANEL",
    "BG_INPUT",
    "BG_HOVER",
    "BG_CANVAS",
    "ACCENT",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "TEXT_MUTED",
    "CANVAS_TEXT",
    "BORDER",
    "DANGER",
    "WARNING",
    "SUCCESS",
)


def _pump(n: int = 30) -> None:
    from PySide6.QtWidgets import QApplication

    for _ in range(n):
        QApplication.processEvents()


def _private_settings(folder: Path) -> None:
    """The report never reads or writes the user's settings (every store goes to a temporary folder)."""
    from al_dvc.gui.settings_store import use_private_settings

    use_private_settings(folder)


def _luminance(colour: str) -> float:
    rgb = [int(colour.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _dark_hash(css: str) -> str:
    from al_dvc.gui.theme import ARROWS_DIR

    return hashlib.sha256(css.replace(ARROWS_DIR.as_posix(), "<arrows>").encode("utf-8")).hexdigest()


def _synthetic_pair(shape):
    from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=11)
    F = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])
    return ref, warp_volume_lagrangian(ref, affine_displacement(F, (1.3, -0.7, 0.4), centre))


class Shots:
    """Captures per window and theme, saved as PNG in ``folder``."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.paths: dict[tuple[str, str], Path] = {}

    def grab(self, widget, key: str, theme: str) -> None:
        _pump()
        path = self.folder / f"theme_{key}_{theme}.png"
        widget.grab().save(str(path))
        self.paths[(key, theme)] = path


def _capture_menus(window, shots: Shots, theme: str) -> None:
    from PySide6.QtCore import QPoint

    for key in ("settings", "theme"):
        menu = window._menus[key]
        menu.popup(QPoint(40, 40))
        shots.grab(menu, f"menu_{key}", theme)
        menu.hide()


def _capture_windows(window, post, texture, guide, shots: Shots, theme: str) -> None:
    """Every window of the analysed session in the theme in use."""
    window.center_tabs.setCurrentWidget(window.viewer)
    shots.grab(window, "main", theme)
    window.center_tabs.setCurrentWidget(window.view3d)
    _pump()
    window.view3d.refresh()
    shots.grab(window, "main_3d", theme)
    window.center_tabs.setCurrentWidget(window.viewer)
    post.show_tab("strain")
    shots.grab(post, "post_strain", theme)
    post.show_tab("analysis")
    post.analysis.results_tabs.setCurrentIndex(1)  # the histograms: a chart in the theme's colours
    shots.grab(post, "post_analysis", theme)
    texture.go_to_step(texture.TAB_ACF)
    shots.grab(texture, "texture", theme)
    for movie in guide.movies:  # the demonstrations play: show their first frame in both themes
        movie.setPaused(True)
        movie.jumpToFrame(0)
    shots.grab(guide, "guide", theme)
    _capture_menus(window, shots, theme)


def _timed_switch(manager, name: str) -> tuple[float, float]:
    """(seconds in set_theme, seconds until the queued repaints and redraws are done)."""
    t0 = time.perf_counter()
    manager.set_theme(name, persist=False)
    t1 = time.perf_counter()
    _pump(60)
    return t1 - t0, time.perf_counter() - t0


def run(shape, shots: Shots) -> dict:
    from al_dvc.gui.app import MainWindow, create_application
    from al_dvc.gui.theme import DARK, LIGHT, build_stylesheet, current_theme
    from al_dvc.gui.theme_manager import theme_manager

    app = create_application([sys.argv[0]])
    manager = theme_manager()
    manager.set_theme("dark", persist=False)
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    ref, dfm = _synthetic_pair(shape)
    window.state.set_volume_arrays([ref, dfm], ["reference", "deformed"])
    nz, ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    mask[nz // 6 : 5 * nz // 6, ny // 5 : 4 * ny // 5, nx // 6 : 5 * nx // 6] = True  # the drawn region of interest
    window.state.set_mask(0, mask=mask)
    window.state.set_params(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    window.viewer.show_subset.setChecked(True)
    for name in ("dark", "light"):  # before the run: the node-grid preview and the region tint on the scan
        manager.set_theme(name, persist=False)
        shots.grab(window, "setup", name)
    manager.set_theme("dark", persist=False)
    window.viewer.show_subset.setChecked(False)
    dark_css = app.styleSheet()
    with tempfile.TemporaryDirectory(prefix="pyaldvc_theme_report_", ignore_cleanup_errors=True) as tmp:
        window.state.set_output_dir(tmp)
        window.state.write_checkpoints = False
        t0 = time.perf_counter()
        window.run_panel.start()
        window.run_panel.wait(600_000)
        run_seconds = time.perf_counter() - t0
        _pump()
        res = window.state.results
        post = window.open_strain_window()
        post.resize(1300, 850)
        post.compute()
        post.wait(600_000)
        _pump()
        post.field.setCurrentIndex(max(0, post.field.findData("exx")))
        post.show_tab("analysis")
        post.analysis.wait(120_000)
        texture = window.open_texture_window()
        texture.resize(1360, 880)
        texture.go_to_step(texture.TAB_ACF)
        texture.analyse()
        texture.wait(300_000)
        guide = window.open_guide_window()
        _pump()
        _capture_windows(window, post, texture, guide, shots, "dark")
        to_light = _timed_switch(manager, "light")
        light_ok = current_theme() == "light" and app.styleSheet() == build_stylesheet(LIGHT)
        _capture_windows(window, post, texture, guide, shots, "light")
        to_dark = _timed_switch(manager, "dark")
        facts = {
            "nodes": res.dvc_mesh.n_nodes,
            "run_seconds": run_seconds,
            "to_light": to_light,
            "to_dark": to_dark,
            "light_ok": light_ok,
            "round_trip": app.styleSheet() == dark_css,
            "dark_is_101": _dark_hash(build_stylesheet(DARK)) == DARK_101_SHA256,
            "dark_chars": len(build_stylesheet(DARK)),
            "light_chars": len(build_stylesheet(LIGHT)),
            "texture_background": texture.plot_background.currentData(),
            "view3d_background": window.view3d.background_key(),
        }
        window.state.dirty = False
        window.close()
    return facts


# ---------------------------------------------------------------------- pages
def _text_page(pdf, facts: dict, shape, shots: Shots) -> None:
    ms = 1000.0
    lines = [
        f"pyALDVC {__version__} -- light theme and the Settings menu",
        "",
        "Settings, the menu right after View: Theme > Dark | Light (one checked), Language > seven languages,",
        "Notify when a task finishes. Language and notifications moved there from View; View keeps the columns.",
        "The theme is saved in the application settings (QSettings, key ui/theme, 'dark' or 'light', default dark)",
        "and applied by create_application before the first window is built; PYALDVC_THEME overrides it (scripts).",
        "",
        "A switch is live, no restart: the application stylesheet is rebuilt from the palette, widgets with a",
        "style of their own (step strip, headings, notices, guide text, sticky titles) and icons are drawn again,",
        "the console is rewritten in the new colours, every matplotlib canvas and chart redraws (slices, strain,",
        "statistics, texture plots and region slices), the matplotlib toolbars pick their icons again, the",
        "Windows title bars follow, and the 3-D view and the texture plots switch their background unless the user",
        "picked one. Open or hidden secondary windows (post-processing, texture, guide, batch, export) follow too.",
        "",
        "Light palette: window #ffffff, side columns #f6f7f9, group boxes #eef0f4, inputs #ffffff, borders #d1d5db,",
        "text #111827 / #4b5563 / #6b7280, accent #4f46e5 (hover #6366f1, pressed #4338ca), selection #e0e7ff,",
        "canvases white with #111827 ticks, labels, spines and titles; spin and combo arrows in dark variants.",
        "",
        f"Synthetic pair {shape[::-1]} (x, y, z), subset 17, step 8: {facts['nodes']} nodes, run {facts['run_seconds']:.1f} s.",
        "Switch with the main, post-processing, texture and guide windows open (offscreen, this machine):",
        *(
            f"  {label}  {facts[key][0] * ms:6.0f} ms in set_theme, {facts[key][1] * ms:6.0f} ms with the repaints"
            for label, key in (("dark -> light", "to_light"), ("light -> dark", "to_dark"))
        ),
        "Checks:",
        f"  light stylesheet applied after the switch ............ {facts['light_ok']}",
        f"  dark stylesheet after dark -> light -> dark identical  {facts['round_trip']}",
        f"  dark stylesheet identical to release 1.0.1 ........... {facts['dark_is_101']} ({facts['dark_chars']} characters;"
        f" light {facts['light_chars']})",
        f"  texture plots / 3-D view background after the switch back: {facts['texture_background']} / "
        f"{facts['view3d_background']}",
        "",
        "Boundary conditions: an unknown saved value starts dark; a background the user picked in the 3-D view or",
        "the texture plots stays in either theme; the switch does not touch a running analysis.",
        "Limitations: the guide's demonstrations are pre-rendered on the dark canvas and keep their dark frame;",
        "the title-bar colour is Windows only (DWM); native file dialogs follow the operating system; matplotlib",
        "toolbar icons are chosen through matplotlib's private toolbar API (without it they keep their colours).",
        "One visible change in the dark theme: the texture plots' toolbar icons are light now (they were black",
        "on the dark background, chosen before the stylesheet reached the toolbar).",
    ]
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.965, "Light theme and Settings menu", fontsize=16, weight="bold", va="top")
    fig.text(0.06, 0.93, "\n".join(lines), fontsize=7.6, va="top", family="monospace", linespacing=1.45)
    _menus(fig, shots)
    pdf.savefig(fig)
    plt.close(fig)


def _menus(fig, shots: Shots, dpi: float = 110.0) -> None:
    """The Settings menu and its Theme submenu in both themes, at a fixed scale, under the text of the first page."""
    fig.text(0.06, 0.345, "Settings menu and its Theme submenu", fontsize=10, weight="bold", va="bottom")
    width, height = fig.get_size_inches()
    for col, theme in enumerate(("dark", "light")):
        x = 0.06 + col * 0.46
        for key in ("menu_settings", "menu_theme"):
            img = mpimg.imread(str(shots.paths[(key, theme)]))
            h, w = img.shape[:2]
            fw, fh = w / dpi / width, h / dpi / height
            ax = fig.add_axes([x, 0.33 - fh, fw, fh])
            ax.imshow(img)
            ax.set_axis_off()
            x += fw + 0.02
    fig.text(
        0.06,
        0.16,
        "Left: dark theme, right: light theme. The checked theme follows a switch made anywhere; Language and\n"
        "'Notify when a task finishes' keep their behaviour and their saved values.",
        fontsize=8,
        va="top",
    )


def _palette_page(pdf) -> None:
    from al_dvc.gui.theme import DARK, LIGHT

    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.965, "The two palettes", fontsize=16, weight="bold", va="top")
    fig.text(
        0.06,
        0.94,
        "Contrast (WCAG ratio) of the primary / secondary / muted text on each background; 4.5 is the usual floor for text.",
        fontsize=8,
        va="top",
    )
    for col, (name, pal) in enumerate((("Dark", DARK), ("Light", LIGHT))):
        x0 = 0.06 + col * 0.47
        fig.text(x0, 0.895, name, fontsize=12, weight="bold", va="bottom")
        for row, field in enumerate(PALETTE_ROWS):
            colour = getattr(pal, field)
            y = 0.84 - row * 0.052
            fig.add_artist(Rectangle((x0, y), 0.07, 0.038, transform=fig.transFigure, fc=colour, ec="#888888", lw=0.6))
            text = f"{field}  {colour}"
            if field.startswith("BG_"):
                ratios = [_contrast(getattr(pal, t), colour) for t in ("TEXT_PRIMARY", "TEXT_SECONDARY", "TEXT_MUTED")]
                text += "   " + " / ".join(f"{r:.1f}" for r in ratios)
            elif field == "ACCENT":
                text += f"   white text {_contrast('#ffffff', colour):.1f}"
            fig.text(x0 + 0.08, y + 0.012, text, fontsize=7.2, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _pair_page(pdf, title: str, note: str, dark: Path, light: Path) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=13, weight="bold")
    for k, (label, path) in enumerate((("Dark", dark), ("Light", light))):
        img = mpimg.imread(str(path))
        top = 0.945 - k * 0.455
        ax = fig.add_axes([0.03, top - 0.43, 0.94, 0.42])
        ax.imshow(img)
        ax.set_axis_off()
        fig.text(0.03, top + 0.004, label, fontsize=9, weight="bold", va="bottom")
    fig.text(0.03, 0.045, "\n".join(textwrap.wrap(note, 120)), fontsize=8, va="top")
    pdf.savefig(fig)
    plt.close(fig)


NOTES = {
    "setup": (
        "Main window before the run: region of interest, node grid, subsets",
        "The node grid (light blue), the subset outlines (yellow), the region tint and its red edge sit on the grey scan "
        "and read the same in both themes. Light: white window, grey side columns, black ticks and labels.",
    ),
    "main": (
        "Main window with the result (Slices tab)",
        "The field overlay and its colour bar; the canvas turns white with black ticks, labels, spines and titles; the "
        "icons of the region tools turn dark; the console is rewritten in the theme's colours.",
    ),
    "main_3d": (
        "Main window, 3-D view tab",
        "The 3-D background follows the theme (dark or white) until the user picks one; the outline, the colour-bar text "
        "and the axes follow the background.",
    ),
    "post_strain": (
        "Post-processing window, Strain tab",
        "The private field canvas redraws in the new colours; the side column is grey in the light theme.",
    ),
    "post_analysis": (
        "Post-processing window, Analysis tab (histograms)",
        "Statistics charts, tables (tinted selection in the light theme) and the slice canvas follow the switch.",
    ),
    "texture": (
        "Texture analysis window, autocorrelation step",
        "The plots switch to their white background (unless one was picked); the step strip, headings, notice and "
        "toolbar icons follow the theme.",
    ),
    "guide": (
        "Texture analysis guide",
        "Text, formulas and rules follow the theme; the demonstrations are pre-rendered on the dark canvas and keep it.",
    ),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPORT))
    ap.add_argument("--shots", default="", help="folder that keeps the PNG captures (default: a temporary one)")
    ap.add_argument("--quick", action="store_true", help=f"a smaller volume {QUICK_SHAPE} (z, y, x)")
    args = ap.parse_args(argv)
    shape = QUICK_SHAPE if args.quick else SHAPE
    with tempfile.TemporaryDirectory(prefix="pyaldvc_theme_settings_", ignore_cleanup_errors=True) as settings_dir:
        _private_settings(Path(settings_dir))
        shots_dir = Path(args.shots) if args.shots else Path(tempfile.mkdtemp(prefix="pyaldvc_theme_shots_"))
        shots = Shots(shots_dir)
        facts = run(shape, shots)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        _text_page(pdf, facts, shape, shots)
        _palette_page(pdf)
        for key, (title, note) in NOTES.items():
            _pair_page(pdf, title, note, shots.paths[(key, "dark")], shots.paths[(key, "light")])
    print(f"report: {out}")
    print(f"captures: {shots.folder}")
    for key in ("to_light", "to_dark"):
        print(f"{key}: {facts[key][0] * 1000:.0f} ms in set_theme, {facts[key][1] * 1000:.0f} ms with repaints")
    print({k: v for k, v in facts.items() if k not in ("to_light", "to_dark")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
