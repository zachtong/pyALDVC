"""Visual report of batch runs: several sessions through BatchRunner, outcomes, timings, the dialog.

    python scripts/make_batch_report.py [--out reports/batch.pdf] [--quick]

Three sessions with different subset sizes (plus one deliberately broken
session) are written to a temporary folder, run through ``BatchRunner`` --
the same code behind ``File > Batch run...`` and ``al-dvc batch`` -- and
summarised: per-session status, node count, convergence and time, followed
by a screenshot of the dialog after the run (captured offscreen).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from al_dvc import __version__  # noqa: E402
from al_dvc.gui.app_state import AppState  # noqa: E402
from al_dvc.gui.batch import BatchRunner  # noqa: E402
from al_dvc.gui.mask_editor import MaskOp  # noqa: E402
from al_dvc.gui.session import save_session  # noqa: E402
from al_dvc.io.volume_io import save_volume  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE_FULL = (96, 104, 112)
SHAPE_QUICK = (48, 52, 56)
F = np.diag([0.02, -0.01, 0.01])
T = (0.8, -0.5, 0.3)


def _write_sessions(root: Path, shape, quick: bool) -> list[Path]:
    nz, ny, nx = shape
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=41)
    fn = affine_displacement(F, T, centre)
    frames = [ref, warp_volume_lagrangian(ref, fn)]
    paths = [root / f"frame{i}.npy" for i in range(len(frames))]
    for p, v in zip(paths, frames):
        save_volume(p, v)
    sizes = (12, 16, 20) if quick else (16, 24, 32)
    sessions = []
    for ws in sizes:
        st = AppState()
        st.add_volume_paths([str(p) for p in paths])
        st.set_params(winsize=ws, winstepsize=ws // 2, search_radius=6, verbose=False)
        st.set_output_dir(root / f"out_ws{ws}")
        if ws == sizes[-1]:  # the largest subset also carries a drawn mask
            st.set_mask_display(target="all")
            st.apply_mask_op(MaskOp("ellipse", "xy", ((nx * 0.05, ny * 0.05), (nx * 0.95, ny * 0.95))))
        sessions.append(save_session(st, root / f"subset_{ws}.aldvc"))
    broken = AppState()
    broken.add_volume_paths([str(paths[0]), str(root / "missing.npy")])
    broken.set_output_dir(root / "out_broken")
    sessions.append(save_session(broken, root / "broken_missing_volume.aldvc"))
    return sessions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "batch.pdf"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    shape = SHAPE_QUICK if args.quick else SHAPE_FULL

    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow, create_application

    _app = create_application([sys.argv[0]])  # keeps Qt alive for the dialog
    with tempfile.TemporaryDirectory(prefix="pyaldvc_batch_report_") as tmp:
        root = Path(tmp)
        sessions = _write_sessions(root, shape, args.quick)
        # 1. headless runner (what al-dvc batch does)
        t0 = time.perf_counter()
        jobs = BatchRunner(sessions, exports=("npz", "summary"), checkpoints=False).run()
        total = time.perf_counter() - t0
        table = BatchRunner.summary_table(jobs)
        # 2. the dialog, offscreen
        window = MainWindow()
        window.resize(1440, 900)
        window.show()
        dialog = window.open_batch_dialog()
        dialog.resize(1000, 640)
        dialog.add_sessions([str(s) for s in sessions])
        dialog.exports["report"].setChecked(False)
        dialog.checkpoints.setChecked(False)
        dialog.start()
        dialog.wait(1_800_000)
        for _ in range(30):
            QApplication.processEvents()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            shot_name = fh.name
        dialog.grab().save(shot_name)
        shot = mpimg.imread(shot_name)
        os.unlink(shot_name)
        dialog_jobs = list(dialog.jobs)
        window.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        lines = (
            [
                f"pyALDVC {__version__} -- batch runs (al_dvc.gui.batch, dialogs/batch_dialog.py, al-dvc batch)",
                "",
                "A batch is a list of .aldvc session files. Each is loaded, its volumes and masks",
                "(files, then drawn operations) are read, the pipeline runs with the session's own",
                "parameters and the chosen exports go to the session's output folder. Jobs are",
                "independent: a failure is recorded and the next session starts; Stop ends the",
                "running session early (pipeline stop flag) and skips the rest.",
                "",
                f"Synthetic pair {shape[::-1]} (x,y,z); sessions with subsets {'12/16/20' if args.quick else '16/24/32'}",
                "(the largest also carries a drawn elliptical mask) and one session whose second",
                f"volume does not exist. Headless runner: {total:.1f} s in total.",
                "",
            ]
            + table.splitlines()
            + [
                "",
                "Dialog (screenshot below): the same queue through the worker thread; statuses:",
                "  " + ", ".join(f"{j.session.name}: {j.status}" for j in dialog_jobs),
                "",
                "Limitations: sessions run one after another (no parallel jobs: each run already uses",
                "every core); results are written to disk, not loaded into the window (open a finished",
                "session with 'Open in window' and load its exports); relative paths in a session",
                "resolve against the session file, so moving a session without its volumes breaks it.",
            ]
        )
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, "Batch runs", fontsize=15, weight="bold", va="top")
        fig.text(0.05, 0.90, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        ok = [j for j in jobs if j.status == "done"]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
        names = [j.session.stem.replace("subset_", "ws ") for j in ok]
        axes[0].bar(names, [j.elapsed for j in ok], color="#4c72b0")
        axes[0].set_ylabel("time (s)")
        axes[0].set_title("run time per session", fontsize=10)
        axes[1].bar(names, [100 * j.converged for j in ok], color="#55a868")
        axes[1].set_ylabel("converged nodes (%)")
        axes[1].set_ylim(0, 105)
        axes[1].set_title("convergence per session", fontsize=10)
        for ax, vals in zip(axes, ([j.n_nodes for j in ok], [j.n_nodes for j in ok])):
            for k, n in enumerate(vals):
                ax.text(k, ax.get_ylim()[1] * 0.02, f"{n} nodes", ha="center", fontsize=7, color="white")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.imshow(shot)
        ax.axis("off")
        ax.set_title("Batch dialog after the run (offscreen capture)", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
