#!/usr/bin/env python
"""Report on the graphical application: offscreen screenshots of its states and timings.

Builds the main window offscreen, loads a synthetic pair, runs the pipeline
through the GUI worker, exports, and captures the window at each stage.
Writes ``reports/gui.pdf``. Reports are generated, never hand-edited.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":  # the offscreen platform has no font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402

SHAPE = (96, 104, 112)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "gui.pdf"))
    ap.add_argument("--language", default="en")
    args = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow, create_application
    from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

    app = create_application([sys.argv[0]])
    app._pyaldvc_lang_mgr.load(args.language)
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    shots: list[tuple[str, np.ndarray]] = []

    def grab(title: str) -> None:
        for _ in range(10):
            QApplication.processEvents()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            name = fh.name
        window.grab().save(name)
        shots.append((title, mpimg.imread(name)))
        os.unlink(name)

    grab("1. Empty window")
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=11)
    F = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])
    dfm = warp_volume_lagrangian(ref, affine_displacement(F, (1.3, -0.7, 0.4), centre))
    window.state.set_volume_arrays([ref, dfm], ["reference", "deformed"])
    window.state.set_params(winsize=24, winstepsize=12, search_radius=6, verbose=False)
    grab("2. Volumes loaded, parameters set")
    with tempfile.TemporaryDirectory(prefix="pyaldvc_gui_report_") as tmp:
        window.state.set_output_dir(tmp)
        t0 = time.perf_counter()
        window.run_panel.start()
        window.run_panel.wait(600_000)
        run_time = time.perf_counter() - t0
        for _ in range(30):
            QApplication.processEvents()
        window.state.set_display(display_field="disp_magnitude")
        grab("3. Result: displacement magnitude overlay")
        window.state.set_display(display_field="exx", colormap="coolwarm")
        grab("4. Result: exx overlay")
        window.results_panel.export("report")
        grab("5. After exporting the PDF report")
        window.center_tabs.setCurrentWidget(window.view3d)
        window.view3d.arrows.setChecked(True)
        grab("6. 3-D view tab (off-screen fallback renderer)")
        window.center_tabs.setCurrentWidget(window.viewer)
    res = window.state.results
    window.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        lines = [
            f"pyALDVC {__version__} -- graphical application (al-dvc-gui)",
            "",
            "Standalone PySide6 window laid out like pyALDIC: volumes and folding parameter sections",
            "(fixed-width inputs, wheel only when focused) on the left, the slice viewer in the middle, run",
            "controls, results, exports and the console on the right. The analysed box follows the region",
            "of interest drawn on the slices; results stay in memory and each export asks where to write.",
            "Three-plane slice viewer with result overlays,",
            "result summary and exports (npz, mat, csv, vti, PDF). Sessions (.aldvc) store volumes,",
            "masks, parameters, export folder and display state. Languages: English, 简体中文.",
            "",
            f"Screenshots below were captured offscreen (QT_QPA_PLATFORM=offscreen), language '{args.language}'.",
            f"Synthetic pair {SHAPE[::-1]} (x,y,z), subset 24, step 12: {res.dvc_mesh.n_nodes} nodes,",
            f"run through the GUI worker in {run_time:.1f} s (JIT already warm).",
            "",
            "Limitations: the slice viewer shows node-grid fields as blocks on the mid-planes; the 3-D tab",
            "renders off-screen (static) when Qt has no OpenGL context, as in these screenshots;",
            "large volumes are loaded fully into memory when added; exports run on the UI thread.",
        ]
        y = 0.95
        fig.text(0.05, y, "Graphical application", fontsize=15, weight="bold", va="top")
        y -= 0.04
        for line in lines:
            fig.text(0.05, y, line, fontsize=8.5, family="monospace", va="top")
            y -= 0.017
        pdf.savefig(fig)
        plt.close(fig)
        for title, img in shots:
            fig = plt.figure(figsize=(11, 7))
            ax = fig.add_axes([0.01, 0.01, 0.98, 0.93])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(title, fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
