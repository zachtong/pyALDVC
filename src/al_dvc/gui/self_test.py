"""Self-test of an installation: Qt, kernels, a small run through the GUI worker, exports.

``al-dvc-gui --self-test`` (or Help > Run self-test) runs every check offscreen
and writes a short report. Exit code 0 means all checks passed.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable


class CheckFailed(Exception):
    """A self-test check failed."""


def check_qt() -> str:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":  # the offscreen platform has no font database of its own
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    from PySide6 import QtCore

    from .app import create_application

    app = create_application([sys.argv[0]])
    from .theme import build_stylesheet

    if len(build_stylesheet()) < 1000:
        raise CheckFailed("stylesheet is empty")
    return f"PySide6 {QtCore.__version__}, Qt {QtCore.qVersion()}, style ok ({type(app).__name__})"


def check_numba() -> str:
    from al_dvc._numba_compat import HAS_NUMBA, JIT_CACHE, get_num_threads

    if not HAS_NUMBA:
        raise CheckFailed("numba is not installed")
    import numba

    return f"numba {numba.__version__}, {get_num_threads()} threads, cache {'on' if JIT_CACHE else 'off'}"


def check_languages() -> str:
    from .i18n import SUPPORTED_LANGUAGES, load_table

    counts = {code: len(load_table(code)) for code in SUPPORTED_LANGUAGES}
    for code, n in counts.items():
        if code != "en" and n == 0:
            raise CheckFailed(f"translation table for {code} is empty")
    return ", ".join(f"{code}: {n} strings" for code, n in counts.items())


def check_mini_run() -> str:
    """A small synthetic pair through the GUI worker, then a report export."""
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

    from .app import MainWindow, create_application

    create_application([sys.argv[0]])
    shape = (48, 52, 56)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=3)
    dfm = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, -0.005, 0.005]), (0.4, -0.3, 0.2), centre))
    window = MainWindow()
    window.state.set_volume_arrays([ref, dfm], ["reference", "deformed"])
    window.state.set_params(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    with tempfile.TemporaryDirectory(prefix="pyaldvc_selftest_") as tmp:
        window.state.set_output_dir(tmp)
        window.state.write_checkpoints = False
        t0 = time.perf_counter()
        window.run_panel.start()
        if not window.run_panel.wait(300_000):
            raise CheckFailed("the pipeline worker did not finish within 5 minutes")
        for _ in range(20):
            QApplication.processEvents()
        result = window.state.results
        if result is None:
            raise CheckFailed("no result came back from the worker")
        fr = result.result_disp[0]
        conv = float(np.mean(fr.status == 0))
        if conv < 0.9:
            raise CheckFailed(f"only {100 * conv:.0f}% of the nodes converged")
        path = window.results_panel.export("report")
        if path is None or not Path(path).is_file():
            raise CheckFailed("report export failed")
        window.viewer.redraw()
        elapsed = time.perf_counter() - t0
    window.close()
    return f"{result.dvc_mesh.n_nodes} nodes, {100 * conv:.0f}% converged, report written, {elapsed:.1f} s"


def check_view3d() -> str:
    """Off-screen pyvista render (optional dependency: reported, never a failure, when missing)."""
    from .view3d_scene import import_error

    reason = import_error()
    if reason is not None:
        return f"pyvista not available ({reason}); optional: pip install al-dvc[gui3d]"
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkVersion  # the 'vtk' facade package is not bundled

    pl = pv.Plotter(off_screen=True, window_size=(96, 72))
    pl.add_mesh(pv.Cube(), color="orange")
    img = pl.screenshot(None, return_img=True)
    pl.close()
    if img is None or img.std() < 1.0:
        raise CheckFailed("off-screen render produced a blank image")
    return f"pyvista {pv.__version__}, VTK {vtkVersion.GetVTKVersion()}, off-screen render ok"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("Qt and theme", check_qt),
    ("Numba kernels", check_numba),
    ("Translations", check_languages),
    ("3-D view (pyvista)", check_view3d),
    ("Mini run through the GUI worker", check_mini_run),
]


def run_self_test(report_path: str | Path | None = None) -> int:
    lines = [
        f"pyALDVC self-test  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python {sys.version.split()[0]} on {platform.platform()}",
        "",
    ]
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            lines.append(f"[ok]   {name}: {detail}")
        except Exception as exc:  # every failure is reported, none aborts the others
            failures += 1
            lines.append(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    lines.append("")
    lines.append("all checks passed" if failures == 0 else f"{failures} check(s) failed")
    text = "\n".join(lines)
    print(text)
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(text + "\n", encoding="utf-8")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(run_self_test(Path("pyaldvc_self_test.txt")))
