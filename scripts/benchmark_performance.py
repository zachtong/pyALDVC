"""Performance benchmark: throughput vs volume size / subset / node count.

Writes ``reports/performance_<timestamp>.csv`` and ``reports/performance.pdf``.

Run:  python scripts/benchmark_performance.py [--quick]
"""

from __future__ import annotations

import argparse
import csv
import logging
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from al_dvc import __version__, dvcpara_default, run_aldvc
from al_dvc._numba_compat import get_num_threads
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

logging.basicConfig(level=logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]


def bench(shape, winsize, step, admm=True):
    vol = generate_speckle_volume(shape, sigma=2.0, seed=3)
    c = tuple((s - 1) / 2 for s in shape[::-1])
    F = np.array([[0.01, 0.005, 0.0], [0.0, -0.01, 0.0], [0.0, 0.0, 0.005]])
    g = warp_volume_lagrangian(vol, affine_displacement(F, (2.3, -1.4, 0.8), c), order=3)
    para = dvcpara_default(winsize=winsize, winstepsize=step, search_radius=6, verbose=False, use_global_step=admm)
    t0 = time.perf_counter()
    res = run_aldvc(para, [vol, g], compute_strain=True)
    total = time.perf_counter() - t0
    n = res.dvc_mesh.n_nodes
    fr = res.result_disp[0]
    n_local_iter = float(np.mean(fr.admm.local_info[0].n_iter)) if fr.admm else float(np.mean(res.result_disp[0].status == 0))
    return {
        "shape": "x".join(map(str, shape)), "voxels": int(np.prod(shape)), "winsize": winsize, "step": step,
        "nodes": n, "admm": admm, "total_s": total,
        "init_s": res.timings.get("init_guess", 0.0), "precompute_s": res.timings.get("reference_precompute", 0.0),
        "local_s": res.timings.get("local_icgn", 0.0), "subpb1_s": res.timings.get("subpb1", 0.0),
        "subpb2_s": res.timings.get("subpb2", 0.0), "strain_s": res.timings.get("strain", 0.0),
        "nodes_per_s_local": n / max(res.timings.get("local_icgn", 1e-9), 1e-9),
        "mean_iter_local": n_local_iter, "admm_steps": fr.admm.n_steps if fr.admm else 0,
    }


def main(quick: bool) -> None:
    # JIT warm-up
    bench((48, 48, 48), 16, 16, admm=True)
    cases = [
        ((128, 128, 128), 16, 8, True),
        ((128, 128, 128), 32, 16, True),
        ((192, 192, 192), 32, 16, True),
        ((256, 256, 256), 32, 16, True),
        ((256, 256, 256), 32, 8, True),
        ((256, 256, 256), 48, 16, True),
        ((256, 256, 256), 32, 16, False),
    ]
    if not quick:
        cases += [((384, 384, 384), 32, 16, True), ((384, 384, 384), 48, 24, True)]
    rows = []
    for shape, ws, st, admm in cases:
        r = bench(shape, ws, st, admm)
        rows.append(r)
        print(f"{r['shape']:>13} ws={ws:<3} step={st:<3} admm={admm!s:<5} nodes={r['nodes']:>6} total={r['total_s']:7.1f}s "
              f"init={r['init_s']:6.1f} pre={r['precompute_s']:6.1f} local={r['local_s']:6.1f} sub1={r['subpb1_s']:6.1f} "
              f"sub2={r['subpb2_s']:5.2f} strain={r['strain_s']:5.2f} | local {r['nodes_per_s_local']:8.0f} nodes/s")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"performance_{stamp}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pdf_path = out_dir / "performance.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.95, f"pyALDVC {__version__} performance ({platform.node()}, {get_num_threads()} threads)",
                 fontsize=13, weight="bold", va="top")
        fig.text(0.04, 0.925, f"{platform.processor()}", fontsize=8, va="top", color="gray")
        y = 0.9
        hdr = f"{'shape':>13} {'ws':>3} {'step':>4} {'admm':>5} {'nodes':>6} {'total':>7} {'init':>6} {'pre':>6} {'local':>6} {'sub1':>6} {'sub2':>5} {'strain':>6} {'local nodes/s':>13}"
        fig.text(0.04, y, hdr, fontsize=8, family="monospace", va="top"); y -= 0.025
        for r in rows:
            fig.text(0.04, y, f"{r['shape']:>13} {r['winsize']:>3} {r['step']:>4} {str(r['admm']):>5} {r['nodes']:>6} {r['total_s']:7.1f} "
                     f"{r['init_s']:6.1f} {r['precompute_s']:6.1f} {r['local_s']:6.1f} {r['subpb1_s']:6.1f} {r['subpb2_s']:5.2f} "
                     f"{r['strain_s']:6.2f} {r['nodes_per_s_local']:13.0f}", fontsize=8, family="monospace", va="top")
            y -= 0.025
        fig.text(0.04, y - 0.02, "Times in seconds (JIT already warm). 'local' = 12-DOF IC-GN over all nodes; 'sub1' = all 3-DOF ADMM passes.",
                 fontsize=8, va="top")
        pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        sub = [r for r in rows if r["admm"]]
        axes[0].bar([f"{r['shape']}\nws{r['winsize']} s{r['step']}" for r in sub],
                    [r["local_s"] for r in sub], label="local IC-GN")
        axes[0].bar([f"{r['shape']}\nws{r['winsize']} s{r['step']}" for r in sub],
                    [r["subpb1_s"] for r in sub], bottom=[r["local_s"] for r in sub], label="ADMM 3-DOF passes")
        axes[0].bar([f"{r['shape']}\nws{r['winsize']} s{r['step']}" for r in sub],
                    [r["init_s"] + r["precompute_s"] for r in sub],
                    bottom=[r["local_s"] + r["subpb1_s"] for r in sub], label="init guess + precompute")
        axes[0].tick_params(axis="x", labelsize=6, rotation=45); axes[0].set_ylabel("seconds"); axes[0].legend(fontsize=8)
        axes[0].set_title("Stage breakdown")
        vox = [r["nodes"] * (r["winsize"] + 1) ** 3 for r in sub]
        axes[1].loglog(vox, [r["local_s"] for r in sub], "o", label="local 12-DOF pass")
        axes[1].set_xlabel("nodes x subset voxels"); axes[1].set_ylabel("seconds"); axes[1].grid(alpha=0.3, which="both")
        axes[1].legend(); axes[1].set_title("Local pass scales with total subset voxels")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {csv_path} and {pdf_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
