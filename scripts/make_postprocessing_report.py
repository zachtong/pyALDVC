#!/usr/bin/env python
"""Report on the post-processing windows: strain window and export dialog (offscreen screenshots, timings).

Writes ``reports/postprocessing.pdf``: a text page (what the windows do, strain timings per method,
export timings per format), a screenshot of the strain window for two fields, a screenshot of the
export dialog and one exported slice image.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
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
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

REPORT = ROOT / "reports" / "postprocessing.pdf"
SHAPE = (72, 80, 88)


def _grab(widget, path: Path) -> Path:
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    widget.grab().save(str(path))
    return path


def _text_page(pdf, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, title, fontsize=16, weight="bold", va="top")
    fig.text(0.05, 0.9, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _image_page(pdf, title: str, path: Path) -> None:
    img = mpimg.imread(str(path))
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.5, 0.95, title, ha="center", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    from al_dvc.gui.app import MainWindow, create_application

    app = create_application([])
    tmp = Path(tempfile.mkdtemp(prefix="pyaldvc_post_"))
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=11)
    frames = [ref]
    for k in (1, 2):
        disp = affine_displacement(np.diag([0.01 * k, -0.005 * k, 0.005 * k]), (0.4 * k, -0.3 * k, 0.2 * k), centre)
        frames.append(warp_volume_lagrangian(ref, disp))
    window.state.set_volume_arrays(frames, ["reference", "deformed 1", "deformed 2"])
    window.state.set_params(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    t0 = time.perf_counter()
    window.run_panel.start()
    window.run_panel.wait(600_000)
    app.processEvents()
    t_run = time.perf_counter() - t0
    res = window.state.results
    lines = [
        f"pyALDVC {__version__} -- post-processing windows (strain window, export dialog)",
        "",
        "Strain is not part of the run: the strain window (Analysis > Strain post-processing, Ctrl+T)",
        "computes it on demand from the displacement results with its own parameters and writes",
        "it back so the main viewer and the exports see it. The export dialog (Ctrl+E) gathers the",
        "destination, formats, fields and frames and writes everything on a worker thread.",
        "",
        f"Synthetic sequence {SHAPE} (z, y, x), 3 frames, subset 17 / step 8: {res.dvc_mesh.n_nodes} nodes,",
        f"run (displacement only) {t_run:.1f} s.",
        "",
        "Strain timings (all frames):",
    ]
    sw = window.open_strain_window()
    sw.resize(1300, 800)
    app.processEvents()
    shots = []
    for method in ("plane_fit", "fd", "fem", "direct"):
        sw.method.setCurrentText(method)
        t0 = time.perf_counter()
        sw.compute()
        sw.wait(600_000)
        app.processEvents()
        dt = time.perf_counter() - t0
        sr = window.state.results.result_strain[0]
        exx = sr.field("exx")
        med = float(np.nanmedian(exx)) if np.isfinite(exx).any() else float("nan")
        n_valid = int(np.sum(sr.strain_valid))
        lines.append(f"  {method:10s} {dt:6.2f} s   median exx frame 1 = {med:+.4f} (truth +0.0100), valid {n_valid}")
    sw.method.setCurrentText("plane_fit")
    sw.compute()
    sw.wait(600_000)
    app.processEvents()
    for field in ("exx", "von_mises"):
        i = sw.field.findData(field)
        sw.field.setCurrentIndex(i)
        sw.frame_slider.setValue(1)
        app.processEvents()
        shots.append((f"Strain window: {field}, frame 2", _grab(sw, tmp / f"strain_{field}.png")))
    # export dialog
    dialog = window.open_export_dialog()
    dialog.folder.setText(str(tmp / "export"))
    for key in ("mat", "csv", "vtk", "report", "images"):
        dialog.checks[key].setChecked(True)
    dialog._check_fields(lambda n: n in ("disp_u", "disp_magnitude", "exx"))
    app.processEvents()
    shots.append(("Export dialog", _grab(dialog, tmp / "export_dialog.png")))
    t0 = time.perf_counter()
    dialog.start()
    dialog.wait(600_000)
    app.processEvents()
    t_export = time.perf_counter() - t0
    written = sorted(p.relative_to(tmp / "export").as_posix() for p in (tmp / "export").rglob("*") if p.is_file())
    lines += ["", f"Export of every format ({len(written)} files) took {t_export:.1f} s:"] + [f"  {p}" for p in written[:24]]
    if len(written) > 24:
        lines.append(f"  ... {len(written) - 24} more")
    lines += [
        "",
        "Limitations: the strain window recomputes every frame on each Compute (no per-frame cache);",
        "the export dialog writes on one worker thread (formats one after another); images use the",
        "slice positions of the main viewer.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(REPORT) as pdf:
        _text_page(pdf, "Post-processing: strain window and export dialog", lines)
        for title, path in shots:
            _image_page(pdf, title, path)
        images = sorted((tmp / "export" / "images").glob("*.png"))
        if images:
            _image_page(pdf, f"Exported slice image: {images[0].name}", images[0])
    dialog.close()
    sw.close()
    window.close()
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
