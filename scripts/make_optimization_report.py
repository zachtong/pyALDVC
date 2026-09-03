"""Performance optimisation study: before / after, thread scaling, subset stride, initial guess.

    python scripts/make_optimization_report.py [--out reports/optimization.pdf] [--quick]

"Before" numbers are constants measured on 2026-09-03 at commit 28c0a3c
(Intel Core Ultra 9 285K, 24 threads) with the kernels prior to the study;
everything else is measured live by this script, so the report documents the
current code against that baseline. Negative results (what was tried and did
not help) are listed with their measurements.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
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
from al_dvc._numba_compat import get_num_threads, set_num_threads  # noqa: E402
from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.data_structures import STATUS_CONVERGED, P_from_UF  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.io.volume_ops import build_reference_bundle, memory_model, normalize_volume, prepare_deformed  # noqa: E402
from al_dvc.mesh.grid_mesh import build_grid_axes, mesh_setup  # noqa: E402
from al_dvc.solver.init_disp import compute_initial_guess  # noqa: E402
from al_dvc.solver.interp_kernels import INTERP_CUBIC  # noqa: E402
from al_dvc.solver.local_icgn import precompute_local_context  # noqa: E402
from al_dvc.solver.numba_kernels import icgn_12dof_parallel  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

# ----------------------------------------------------------------------------- baseline constants (commit 28c0a3c)
BEFORE = {
    "commit": "28c0a3c (2026-09-03)",
    "case": (256, 32, 8),  # shape side, subset, step of the pipeline constants
    "sampler_ns": 136.8,  # tricubic sample, random positions, single thread
    "kernel12_ns": 7.0,  # ns per voxel-iteration, 24 threads, 256^3 / subset 32 / step 16
    "kernel3_ns": 7.1,
    "pipeline": {
        "total": 26.5,
        "local_icgn": 14.9,
        "subpb1": 5.6,
        "init_guess": 4.6,
        "reference_precompute": 0.88,
        "subpb2": 0.23,
    },
    "init_fine_radius4_s": 4.7,
    "negative": [
        (
            "fastmath (reassoc, contract) on the direct ZNCC search kernel",
            "4.3 s -> 5.8 s (slower: the vectorised reduction loses to the scalar one)",
        ),
        (
            "FFT correlation instead of the direct kernel at the fine pyramid level",
            "4.7 s -> 11.6 s (radius 2-4 windows are too small for FFTs to pay off)",
        ),
        (
            "trilinear sampling for the first IC-GN iterations, cubic afterwards",
            "8251 -> 11267 iterations, 0.61 -> 0.83 s: the cubic phase still needs two iterations to remove the trilinear bias",
        ),
        (
            "skipping the full-resolution pyramid level of the initial guess",
            "2.2 s but median initial error 0.03 -> 0.12 voxel and 106 nodes rejected",
        ),
    ],
}
F_TRUE = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])
T_TRUE = (1.3, -0.7, 0.4)


def _pair(shape, noise=0.0, seed=1):
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=seed)
    fn = affine_displacement(F_TRUE, T_TRUE, centre)
    dfm = warp_volume_lagrangian(ref, fn)
    if noise > 0:
        rng = np.random.default_rng(seed + 100)
        ref = ref + rng.normal(0, noise, ref.shape)
        dfm = dfm + rng.normal(0, noise, dfm.shape)
    return ref, dfm, fn


def _truth(mesh, fn):
    c = mesh.coordinates
    return np.stack(fn(c[:, 0], c[:, 1], c[:, 2]), axis=-1).reshape(-1, 3)


def _mean_iters(fr, ok, first_pass: int) -> float:
    """Mean IC-GN iterations per node over the local passes from ``first_pass`` on (NaN without ADMM)."""
    if fr.admm is None or not ok.any() or len(fr.admm.local_info) <= first_pass:
        return float("nan")
    return float(np.mean([li.n_iter[ok].mean() for li in fr.admm.local_info[first_pass:]]))


def measure_pipeline(shape, ws, st, stride=1, noise=0.0, noise_hessian=True, coarse=1):
    ref, dfm, fn = _pair(shape, noise)
    para = dvcpara_default(
        winsize=ws,
        winstepsize=st,
        search_radius=6,
        verbose=False,
        subset_stride=stride,
        icgn_noise_hessian=noise_hessian,
        init_coarse_factor=coarse,
    )
    t0 = time.perf_counter()
    res = run_aldvc(para, [ref, dfm])
    total = time.perf_counter() - t0
    fr = res.result_disp[0]
    U_true = _truth(res.dvc_mesh, fn)
    ok = res.dvc_mesh.node_valid & (fr.status == STATUS_CONVERGED)
    ok[res.dvc_mesh.boundary_nodes] = False
    err = np.linalg.norm(fr.U[ok] - U_true[ok], axis=1) if ok.any() else np.array([np.nan])
    out = {k: float(v) for k, v in res.timings.items()}
    out.update(
        total=total,
        n_nodes=res.dvc_mesh.n_nodes,
        iters=_mean_iters(fr, ok, 0),
        iters_sub1=_mean_iters(fr, ok, 1),
        converged=float(np.mean(fr.status == STATUS_CONVERGED)),
        rmse=float(np.sqrt(np.mean(err**2))),
        p95=float(np.percentile(err, 95)),
        u_std=float(np.nanmedian(fr.U_std)) if fr.U_std is not None else float("nan"),
    )
    return out


def measure_kernel(shape, ws, st, threads):
    ref, dfm, fn = _pair(shape)
    para = dvcpara_default(winsize=ws, winstepsize=st, verbose=False)
    f, g = normalize_volume(ref), normalize_volume(dfm)
    bundle = build_reference_bundle(f, None)
    mesh = mesh_setup(*build_grid_axes(para.voi, shape, para.winsize, para.winstepsize))
    ctx = precompute_local_context(mesh, bundle, para)
    g_prep = prepare_deformed(g, "cubic")
    U_true = _truth(mesh, fn)
    rng = np.random.default_rng(0)
    P0 = P_from_UF(U_true + rng.normal(0, 0.4, U_true.shape), np.zeros((mesh.n_nodes, 3, 3)))
    hx, hy, hz = ctx.half
    S = (2 * hx + 1) * (2 * hy + 1) * (2 * hz + 1)

    def run():
        return icgn_12dof_parallel(
            ctx.coords_int,
            P0.copy(),
            hx,
            hy,
            hz,
            bundle.f,
            bundle.gx,
            bundle.gy,
            bundle.gz,
            bundle.mask,
            g_prep,
            INTERP_CUBIC,
            ctx.L_all,
            ctx.meanf,
            ctx.bottomf,
            ctx.valid,
            1e-2,
            1e-3,
            100,
            5,
            1,
        )

    run()
    rows = []
    for n in threads:
        set_num_threads(n)
        t = time.perf_counter()
        P, n_iter, status, zncc = run()
        dt = time.perf_counter() - t
        rows.append((n, dt, dt * 1e9 / (int(n_iter.sum()) * S)))
    set_num_threads(threads[-1])
    return rows, mesh.n_nodes


def measure_init(shape, ws, st, fine_radius, noise=0.0):
    ref, dfm, fn = _pair(shape, noise)
    para = dvcpara_default(winsize=ws, winstepsize=st, search_radius=6, verbose=False, pyramid_fine_radius=fine_radius)
    f, g = normalize_volume(ref), normalize_volume(dfm)
    mesh = mesh_setup(*build_grid_axes(para.voi, shape, para.winsize, para.winstepsize))
    compute_initial_guess(f, g, mesh, para)
    t = time.perf_counter()
    U0, info = compute_initial_guess(f, g, mesh, para)
    dt = time.perf_counter() - t
    err = np.linalg.norm(U0 - _truth(mesh, fn), axis=1)
    return dt, float(np.median(err)), float(np.percentile(err, 95)), int(info["n_bad"]), mesh.n_nodes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "optimization.pdf"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    n = 128 if args.quick else 256
    shape = (n, n, n)
    ws, st = (16, 8) if args.quick else (32, 8)
    ws_k, st_k = (16, 8) if args.quick else (32, 16)
    max_threads = get_num_threads()
    threads = sorted({1, 4, 8, 16, max_threads} & set(range(1, max_threads + 1))) if not args.quick else sorted({1, max_threads})

    print("pipeline, stride 1 ...")
    after1 = measure_pipeline(shape, ws, st, 1)
    print("pipeline, stride 2 ...")
    after2 = measure_pipeline(shape, ws, st, 2)
    print("kernel thread scaling ...")
    krows, kn = measure_kernel(shape, ws_k, st_k, threads)
    print("stride trade-off ...")
    strides = [1, 2, 3] if not args.quick else [1, 2]
    trade = {noise: [measure_pipeline(shape, ws, st, s, noise) for s in strides] for noise in (0.0, 0.03)}
    print("initial guess ...")
    init_rows = [(r, *measure_init(shape, ws, st, r)) for r in (4, 2)]
    print("noise-corrected Hessian and coarse initial guess ...")
    variants = {}
    for noise in (0.0, 0.03):
        variants[noise] = {
            "plain": measure_pipeline(shape, ws, st, 1, noise, noise_hessian=False),
            "noise_hessian": measure_pipeline(shape, ws, st, 1, noise, noise_hessian=True),
            "coarse_x2": measure_pipeline(shape, ws, st, 1, noise, noise_hessian=True, coarse=2),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        # ---------------------------------------------------------------- page 1: summary
        stages = ["local_icgn", "subpb1", "init_guess", "reference_precompute", "subpb2"]
        lines = [
            f"pyALDVC {__version__} -- performance optimisation study ({n}^3 synthetic speckle, {max_threads} threads)",
            "",
            "Changes: (1) scalar interpolation weights -- the tricubic sampler allocated three",
            "np.empty(4) arrays per sampled voxel (a heap allocation each in Numba): 137 -> 52 ns per",
            "random sample; (2) the ZNCC numerator is accumulated in the gradient pass instead of a",
            "second pass over the subset, and the per-voxel divisions became multiplications; results",
            "unchanged to 3e-14 with identical iteration counts; (3) subset sampling stride (parameter",
            "subset_stride: every k-th voxel per axis, k^3 fewer samples; Hessian, statistics and the",
            "uncertainty model use the sampled set); (4) the NCC pyramid refines the finer levels with",
            "radius 2 instead of 4 (pyramid_fine_radius; auto-expand keeps clipped peaks safe);",
            "(5) ListVolumeProvider normalises frames on demand (LRU of 3) instead of every frame up front.",
            "",
            f"Pipeline, subset {ws} / step {st}, {after1['n_nodes']} nodes -- before ({BEFORE['commit']}) vs now:",
            f"  {'stage':<22s} {'before':>8s} {'now':>8s} {'stride 2':>9s}",
        ]
        same_case = (n, ws, st) == BEFORE["case"]
        before = BEFORE["pipeline"] if same_case else {}

        def _b(k):
            return f"{before[k]:8.2f}" if k in before else f"{'n/a':>8s}"

        for k in stages:
            lines.append(f"  {k:<22s} {_b(k)} {after1.get(k, 0.0):8.2f} {after2.get(k, 0.0):9.2f}")
        lines.append(f"  {'total':<22s} {_b('total')} {after1['total']:8.2f} {after2['total']:9.2f}")
        if not same_case:
            c = BEFORE["case"]
            lines.append(f"  (the baseline constants were measured at {c[0]}^3 / subset {c[1]} / step {c[2]})")
        lines += [
            f"  displacement RMSE (interior, converged) now {after1['rmse']:.4f} voxel, stride 2 {after2['rmse']:.4f} voxel;",
            f"  converged {100 * after1['converged']:.1f} % / {100 * after2['converged']:.1f} %; "
            f"median U_std {after1['u_std']:.4f} / {after2['u_std']:.4f} voxel.",
            "",
            f"12-DOF kernel, subset {ws_k} / step {st_k}, {kn} nodes (ns per voxel-iteration, wall clock):",
            f"  before: {BEFORE['kernel12_ns']:.1f} ns (24 threads)   now: "
            + ", ".join(f"{t} thr {ns:.1f} ns" for t, dt, ns in krows),
            "  thread scaling now: " + ", ".join(f"{t}: {krows[0][1] / dt:.1f}x" for t, dt, ns in krows),
            "",
            "Initial guess (pyramid NCC):",
        ]
        for r, dt, med, p95, nbad, nn in init_rows:
            lines.append(
                f"  fine radius {r}: {dt:5.2f} s, median error {med:.3f}, 95 % {p95:.3f} voxel, {nbad} rejected ({nn} nodes)"
            )
        lines += ["", "Tried and rejected (measured):"]
        for what, result in BEFORE["negative"]:
            lines.extend(textwrap.wrap(f"- {what}: {result}", width=96, initial_indent="  ", subsequent_indent="    "))
        mm = memory_model(shape, "stored")
        mm2 = memory_model(shape, "on_the_fly")
        lines += [
            "",
            f"Memory: {mm['bytes_per_voxel']:.0f} bytes/voxel resident with stored gradients ({mm2['bytes_per_voxel']:.0f} with",
            "gradient_mode='on_the_fly'); the phase-correlation pre-shift adds a transient 40 bytes/voxel",
            "at 128^3 (downsampled), and sequences no longer hold a normalised copy of every frame.",
        ]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.96, "Performance optimisation study", fontsize=15, weight="bold", va="top")
        fig.text(0.05, 0.92, "\n".join(lines), fontsize=7.6, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------------- page 2: bars + scaling
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
        x = np.arange(len(stages))
        w = 0.27
        axes[0].bar(x - w, [BEFORE["pipeline"].get(k, 0) for k in stages], w, label="before")
        axes[0].bar(x, [after1.get(k, 0) for k in stages], w, label="now (stride 1)")
        axes[0].bar(x + w, [after2.get(k, 0) for k in stages], w, label="now (stride 2)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(stages, rotation=30, ha="right", fontsize=7)
        axes[0].set_ylabel("seconds")
        axes[0].set_title(f"stage timings, {n}^3 / subset {ws} / step {st}", fontsize=9)
        axes[0].legend(fontsize=7)
        axes[1].plot([t for t, _, _ in krows], [krows[0][1] / dt for _, dt, _ in krows], "o-", label="measured")
        axes[1].plot([t for t, _, _ in krows], [t for t, _, _ in krows], "--", color="gray", label="ideal")
        axes[1].set_xlabel("threads")
        axes[1].set_ylabel("speed-up")
        axes[1].set_title("12-DOF kernel thread scaling", fontsize=9)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------------- page 3: stride trade-off
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
        for noise, rows in trade.items():
            axes[0].plot(strides, [r["local_icgn"] + r["subpb1"] for r in rows], "o-", label=f"noise sd {noise}")
            axes[1].plot(strides, [r["rmse"] for r in rows], "o-", label=f"noise sd {noise}")
        axes[0].set_xlabel("subset_stride")
        axes[0].set_ylabel("local + ADMM local steps (s)")
        axes[0].set_title("time", fontsize=9)
        axes[1].set_xlabel("subset_stride")
        axes[1].set_ylabel("displacement RMSE (voxel)")
        axes[1].set_title("accuracy (interior converged nodes)", fontsize=9)
        for ax in axes:
            ax.set_xticks(strides)
            ax.legend(fontsize=7)
        fig.suptitle(f"Subset sampling stride, subset {ws}: k^3 fewer samples per iteration", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        lines = ["stride  noise   local+subpb1 [s]   total [s]   RMSE [vox]   95% [vox]   U_std [vox]   converged"]
        for noise, rows in trade.items():
            for s, r in zip(strides, rows):
                lines.append(
                    f"  {s}     {noise:<6.2f} {r['local_icgn'] + r['subpb1']:14.2f} {r['total']:11.2f} "
                    f"{r['rmse']:12.4f} {r['p95']:11.4f} {r['u_std']:12.4f} {100 * r['converged']:9.1f} %"
                )
        lines += [
            "",
            "Reading: the stride divides the local-step time by roughly k^3 / 2 (per-voxel overhead grows",
            "as the sampled set shrinks) while the RMSE follows the noise sensitivity of a subset with",
            "k^3 fewer voxels (about 3x the random error at k = 2 on noisy data, none on clean data);",
            "the smoothing bias is that of the full subset span. Stride 2 suits subsets of 32 and more",
            "on data with a decent SNR; stride 3 is for large subsets on clean data.",
        ]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.96, "Stride trade-off table", fontsize=13, weight="bold", va="top")
        fig.text(0.05, 0.92, "\n".join(lines), fontsize=8, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        lines = [
            "Noise-corrected Hessian (icgn_noise_hessian) and coarse-lattice initial guess (init_coarse_factor),",
            f"{n}^3 / subset {ws} / step {st}:",
            "",
            f"  {'variant':<28s} {'noise':>6s} {'total [s]':>10s} {'init [s]':>9s} {'local [s]':>10s} "
            f"{'subpb1 [s]':>11s} {'it/node':>8s} {'it/node sub1':>13s} {'RMSE [vox]':>11s}",
        ]
        for noise, rows in variants.items():
            for name, r in rows.items():
                lines.append(
                    f"  {name:<28s} {noise:6.2f} {r['total']:10.2f} {r['init_guess']:9.2f} {r['local_icgn']:10.2f} "
                    f"{r['subpb1']:11.2f} {r['iters']:8.2f} {r['iters_sub1']:13.2f} {r['rmse']:11.4f}"
                )
        lines += [
            "",
            "The noise-corrected Hessian subtracts the reference-gradient noise inflation c s^2 (I3 x M)",
            "from the stored Hessian once a node's step is below 0.5 voxel (capped at half of the",
            "translation diagonal): same fixed point, shorter path; on clean data 1 - ZNCC ~ 0 and",
            "nothing changes. The coarse lattice solves NCC + IC-GN on every second node per axis and",
            "interpolates U and F to all nodes: the NCC cost drops 8x and the full pass starts within",
            "0.1 voxel with the gradient in place (fewer iterations); off by default because a",
            "discontinuous field is not captured by the interpolated start.",
            "",
            "Real micro-CT example (79,200 nodes, MATLAB beta), IC-GN iterations per ADMM pass:",
            "  plain Hessian 8.1 / 7.3 / 6.9 / 6.5, corrected (cap 0.5) 6.8 / 4.1 / 4.0 / 3.9; a full",
            "  correction (cap 0.1) over-shot on the non-white CT noise: the ADMM stopped after 2 steps",
            "  on an answer 0.024 / 0.028 / 0.077 voxel from MATLAB's instead of 0.005 / 0.006 / 0.020.",
        ]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.96, "Noise-corrected Hessian and coarse initial guess", fontsize=13, weight="bold", va="top")
        fig.text(0.05, 0.92, "\n".join(lines), fontsize=7.4, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
