#!/usr/bin/env python
"""Report on per-frame checkpoints (``run_aldvc(..., checkpoint_dir=...)``).

A synthetic 5-frame sequence is solved once straight through, once
interrupted after two frames and resumed, and once with everything already
checkpointed. The report shows the wall times of the three runs, the size of
the checkpoint files and that the resumed results are bit-identical. Writes
``reports/checkpoint.pdf``. Reports are generated, never hand-edited.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402
from al_dvc.core.checkpoint import Checkpoint  # noqa: E402
from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE = (96, 104, 112)
N_FRAMES = 5
STOP_AFTER = 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "checkpoint.pdf"))
    args = ap.parse_args(argv)
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=11)
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    frames = [ref]
    for k in range(1, N_FRAMES):
        F = np.array([[0.006 * k, 0.001 * k, 0.0], [0.0, -0.004 * k, 0.001 * k], [0.001 * k, 0.0, 0.003 * k]])
        frames.append(warp_volume_lagrangian(ref, affine_displacement(F, (0.5 * k, -0.3 * k, 0.4 * k), centre)))
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False, admm_max_iter=2)
    ck = Path(tempfile.mkdtemp(prefix="aldvc_ck_"))
    try:
        t0 = time.perf_counter()
        straight = run_aldvc(para, frames)
        t_straight = time.perf_counter() - t0
        polls = {"n": 0}

        def stop_fn():
            polls["n"] += 1
            return polls["n"] > 2 * STOP_AFTER + 1

        t0 = time.perf_counter()
        partial = run_aldvc(para, frames, stop_fn=stop_fn, checkpoint_dir=ck)
        t_partial = time.perf_counter() - t0
        done = Checkpoint(ck).completed_frames()
        t0 = time.perf_counter()
        resumed = run_aldvc(para, frames, checkpoint_dir=ck)
        t_resumed = time.perf_counter() - t0
        t0 = time.perf_counter()
        run_aldvc(para, frames, checkpoint_dir=ck)  # every frame comes from the checkpoint
        t_reloaded = time.perf_counter() - t0
        sizes = {p.name: p.stat().st_size / 1e6 for p in sorted(ck.glob("frame_*.npz"))}
        max_diff = max(float(np.nanmax(np.abs(a.U_accum - b.U_accum))) for a, b in zip(resumed.result_disp, straight.result_disp))
        max_diff_strain = max(
            float(np.nanmax(np.abs(a.exx - b.exx))) for a, b in zip(resumed.result_strain, straight.result_strain)
        )
    finally:
        shutil.rmtree(ck, ignore_errors=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        lines = (
            [
                f"pyALDVC {__version__} -- per-frame checkpoints",
                "",
                f"Synthetic sequence: {N_FRAMES} frames of {SHAPE[::-1]} (x,y,z), accumulative mode, subset 16, step 8,",
                f"{straight.dvc_mesh.n_nodes} nodes, ADMM 2 steps, strain computed.",
                "",
                f"{'run':<44} {'frames solved':>13} {'wall time [s]':>13}",
                f"{'straight through':<44} {N_FRAMES - 1:>13} {t_straight:13.1f}",
                f"{'interrupted after frame %d (checkpointing)' % STOP_AFTER:<44} {partial.n_frames:>13} {t_partial:13.1f}",
                f"{'resumed from the checkpoint':<44} {N_FRAMES - 1 - len(done):>13} {t_resumed:13.1f}",
                f"{'all frames already checkpointed':<44} {0:>13} {t_reloaded:13.1f}",
                "",
                f"max |U_accum(resumed) - U_accum(straight)| = {max_diff:.2e} voxel, "
                f"max |exx difference| = {max_diff_strain:.2e}",
                "",
                "Checkpoint files (compressed npz, one per frame pair):",
            ]
            + [f"  {name:<20} {size:6.2f} MB" for name, size in sizes.items()]
            + [
                "",
                "Contents per frame: U, F, U_local, F_local, U0, ZNCC, status, U_std, ADMM diagnostics (beta sweep,",
                "residuals, per-pass iteration counts) and the node validity of the reference mesh; meta.json holds the",
                "parameters, volume shape, frame schedule and node grid. A directory written by a different run is",
                "rejected (CheckpointMismatch) unless resume=False. Cumulative displacements and strains are recomputed",
                "from the loaded frames, so the stored files never go stale.",
                "",
                "Limitation: the check compares parameters, shape, schedule and grid, not the voxel data; pointing a",
                "checkpoint directory at a different scan with identical settings is the user's responsibility.",
            ]
        )
        y = 0.95
        fig.text(0.05, y, "Per-frame checkpoints", fontsize=15, weight="bold", va="top")
        y -= 0.04
        for line in lines:
            fig.text(0.05, y, line, fontsize=8.5, family="monospace", va="top")
            y -= 0.017
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        labels = ["straight", f"interrupted after {STOP_AFTER}", "resumed", "all checkpointed"]
        ax.bar(labels, [t_straight, t_partial, t_resumed, t_reloaded], color="#4c72b0")
        ax.set_ylabel("wall time [s]")
        ax.set_title("Checkpoint timing on the synthetic sequence")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"report: {out}")
    print(
        f"straight {t_straight:.1f} s, partial {t_partial:.1f} s, resumed {t_resumed:.1f} s, "
        f"reloaded {t_reloaded:.1f} s, max diff {max_diff:.1e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
