#!/usr/bin/env python
"""Report on the large-volume mode (``gradient_mode="on_the_fly"``).

Measures wall time and peak resident memory of the two gradient modes on
synthetic volumes and tabulates the memory model for scan sizes and RAM
budgets. Writes ``reports/large_volume.pdf``. Reports are generated, never
hand-edited.

Usage::

    python scripts/make_large_volume_report.py [--quick]
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402
from al_dvc.io.volume_ops import memory_model  # noqa: E402

CHILD = r"""
import json, sys, time
import numpy as np, psutil
sys.path.insert(0, %r)
from al_dvc.core.config import dvcpara_default
from al_dvc.core.pipeline import run_aldvc
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian
n, mode, winsize = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
shape = (n, n, n)
f = generate_speckle_volume(shape, sigma=2.0, seed=11)
c = tuple((s - 1) / 2 for s in shape[::-1])
disp = affine_displacement(np.array([[0.02, 0.003, 0.0], [0.0, -0.01, 0.002], [0.001, 0.0, 0.01]]), (0.7, -0.4, 0.3), c)
g = warp_volume_lagrangian(f, disp)
para = dvcpara_default(winsize=winsize, winstepsize=winsize // 2, search_radius=6, verbose=False, gradient_mode=mode)
run_aldvc(para, [f, g], compute_strain=False)  # warm-up (JIT)
import gc, threading
gc.collect()
proc = psutil.Process()
rss0 = proc.memory_info().rss
peak_box = [rss0]
box = {}
def work():
    box["r"] = run_aldvc(para, [f, g], compute_strain=False)
th = threading.Thread(target=work)
t0 = time.perf_counter()
th.start()
while th.is_alive():  # sample the resident size while the pipeline runs
    peak_box[0] = max(peak_box[0], proc.memory_info().rss)
    time.sleep(0.02)
th.join()
dt = time.perf_counter() - t0
r = box["r"]
peak = peak_box[0] - rss0  # resident memory the run added on top of the loaded volumes
print(json.dumps({"n": n, "mode": mode, "time": dt, "rss0": rss0, "peak": peak, "nodes": r.dvc_mesh.n_nodes,
                  "local": r.timings.get("local_icgn", 0.0), "subpb1": r.timings.get("subpb1", 0.0),
                  "pre": r.timings.get("reference_precompute", 0.0)}))
"""


def measure(n: int, mode: str, winsize: int) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", CHILD % str(ROOT / "src"), str(n), mode, str(winsize)],
        capture_output=True,
        text=True,
        check=True,
    )
    return __import__("json").loads(out.stdout.strip().splitlines()[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports" / "large_volume.pdf"))
    args = ap.parse_args(argv)
    sizes = [192] if args.quick else [192, 256, 320]
    winsize = 32
    rows = []
    for n in sizes:
        for mode in ("stored", "on_the_fly"):
            gc.collect()
            rows.append(measure(n, mode, winsize))
            r = rows[-1]
            print(f"{n}^3 {mode:<11} {r['time']:6.1f} s  local {r['local']:5.1f} s  run RSS {r['peak'] / 1e9:5.2f} GB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        lines = [f"pyALDVC {__version__} -- large-volume mode (gradient_mode='on_the_fly')", ""]
        lines.append("The three reference-gradient volumes (12 bytes/voxel) are dropped; the kernels evaluate the")
        lines.append("7-point stencil on the reference at the subset voxels instead (identical solution, see tests).")
        lines.append(f"Synthetic speckle, affine 2 %, subset {winsize}, step {winsize // 2}, AL-DVC with 2-4 ADMM steps,")
        lines.append("each measurement in a fresh process after a JIT warm-up run; run RSS = resident memory added while")
        lines.append("the pipeline runs (input volumes already loaded), sampled every 20 ms.")
        lines.append("")
        lines.append(
            f"{'volume':>8} {'mode':<11} {'nodes':>6} {'total [s]':>9} {'precompute':>10} "
            f"{'local':>7} {'3-DOF':>7} {'run RSS [GB]':>13}"
        )
        for r in rows:
            lines.append(
                f"{str(r['n']) + '^3':>8} {r['mode']:<11} {r['nodes']:>6} {r['time']:9.1f} {r['pre']:10.1f} {r['local']:7.1f} "
                f"{r['subpb1']:7.1f} {r['peak'] / 1e9:13.2f}"
            )
        lines.append("")
        lines.append("Memory model (resident bytes per voxel for a frame pair: normalised f and g, mask, gradients):")
        lines.append(f"{'scan':>14} {'stored [GB]':>12} {'on_the_fly [GB]':>16}   largest cubic scan per RAM budget:")
        budgets = (16, 32, 64, 128)
        for n in (512, 768, 1024, 1536, 2048):
            ms = memory_model((n, n, n), "stored")["total_gb"]
            mf = memory_model((n, n, n), "on_the_fly")["total_gb"]
            lines.append(f"{str(n) + '^3':>14} {ms:12.1f} {mf:16.1f}")
        lines.append("")
        lines.append(
            f"{'RAM [GB]':>10} {'stored':>10} {'on_the_fly':>12}   (edge length of the largest cubic scan, 70 % of RAM usable)"
        )
        for b in budgets:
            ns = int((0.7 * b * 1e9 / memory_model((1, 1, 1), "stored")["bytes_per_voxel"]) ** (1 / 3))
            nf = int((0.7 * b * 1e9 / memory_model((1, 1, 1), "on_the_fly")["bytes_per_voxel"]) ** (1 / 3))
            lines.append(f"{b:>10} {ns:>10} {nf:>12}")
        lines.append("")
        lines.append("Not counted: the raw input volumes while they are loaded (use the streaming provider), the")
        lines.append("B-spline coefficient array (+4 bytes/voxel with interp_method='bspline'), the masked copy of the")
        lines.append("deformed volume (+4 bytes/voxel with a deformed-frame mask) and the per-node arrays (~1.3 KB/node).")
        fig = plt.figure(figsize=(8.5, 11))
        y = 0.95
        fig.text(0.05, y, "Large-volume mode", fontsize=15, weight="bold", va="top")
        y -= 0.04
        for line in lines:
            fig.text(0.05, y, line, fontsize=8.2, family="monospace", va="top")
            y -= 0.0165
        pdf.savefig(fig)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for mode, color in (("stored", "#4c72b0"), ("on_the_fly", "#dd8452")):
            sel = [r for r in rows if r["mode"] == mode]
            axes[0].plot([r["n"] for r in sel], [r["time"] for r in sel], "o-", color=color, label=mode)
            axes[1].plot([r["n"] for r in sel], [r["peak"] / 1e9 for r in sel], "o-", color=color, label=mode)
        axes[0].set_xlabel("volume edge [voxel]")
        axes[0].set_ylabel("wall time [s]")
        axes[0].legend()
        axes[1].set_xlabel("volume edge [voxel]")
        axes[1].set_ylabel("resident memory added by the run [GB]")
        axes[1].legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
