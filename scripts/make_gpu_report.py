"""Visual report of the CUDA backend: parity with the CPU kernels and speed-ups.

    python scripts/make_gpu_report.py [--out reports/gpu.pdf] [--quick]

Runs the 12-DOF, 3-DOF and Hessian-precompute kernels on the CPU and on the
GPU for several cases (subset sizes, noise, stride, interpolation, masks,
gradient modes) and compares statuses, iteration counts and displacements;
then times the whole pipeline with ``backend="numba"`` and ``"auto"``.
Without a usable CUDA device the report documents that fact and exits.
"""

from __future__ import annotations

import argparse
import sys
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
from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.data_structures import STATUS_CONVERGED, P_from_UF  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.io.volume_ops import build_reference_bundle, normalize_volume, prepare_deformed  # noqa: E402
from al_dvc.mesh.grid_mesh import build_grid_axes, mesh_setup  # noqa: E402
from al_dvc.solver import cuda_kernels as ck  # noqa: E402
from al_dvc.solver import numba_kernels as nk  # noqa: E402
from al_dvc.solver.interp_kernels import INTERP_MODE_BY_NAME  # noqa: E402
from al_dvc.solver.local_icgn import precompute_local_context  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

F_TRUE = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])
T_TRUE = (1.3, -0.7, 0.4)


def _case(n, ws, st, noise=0.0, interp="cubic", stride=1, gradient_mode="stored", masks=False):
    shape = (n, n, n)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=1)
    fn = affine_displacement(F_TRUE, T_TRUE, centre)
    dfm = warp_volume_lagrangian(ref, fn)
    if noise > 0:
        rng = np.random.default_rng(3)
        ref = ref + rng.normal(0, noise, ref.shape)
        dfm = dfm + rng.normal(0, noise, dfm.shape)
    para = dvcpara_default(
        winsize=ws,
        winstepsize=st,
        interp_method=interp,
        verbose=False,
        subset_stride=stride,
        gradient_mode=gradient_mode,
        backend="numba",
    )
    f, g = normalize_volume(ref), normalize_volume(dfm)
    rmask = dmask = None
    if masks:
        rmask = np.ones(shape, dtype=bool)
        rmask[:, :, : n // 4] = False
        dmask = np.ones(shape, dtype=bool)
        dmask[: n // 5] = False
    bundle = build_reference_bundle(f, rmask, gradient_mode)
    mesh = mesh_setup(*build_grid_axes(para.voi, shape, para.winsize, para.winstepsize))
    ctx = precompute_local_context(mesh, bundle, para)
    c = mesh.coordinates
    U_true = np.stack(fn(c[:, 0], c[:, 1], c[:, 2]), axis=-1).reshape(-1, 3)
    return dict(
        ref=ref, dfm=dfm, bundle=bundle, g=prepare_deformed(g, interp, dmask), ctx=ctx, para=para, U_true=U_true, mesh=mesh
    )


def _timeit(fn, repeat=2):
    fn()
    best = np.inf
    out = None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def compare_kernels(case):
    b, ctx, para = case["bundle"], case["ctx"], case["para"]
    pattern, gain = ctx.noise_args(para)
    mode = INTERP_MODE_BY_NAME[para.interp_method]
    hx, hy, hz = ctx.half
    P0 = P_from_UF(case["U_true"] + np.random.default_rng(0).normal(0, 0.4, case["U_true"].shape), np.zeros((ctx.n_nodes, 3, 3)))
    a12 = (
        ctx.coords_int,
        P0,
        hx,
        hy,
        hz,
        b.f,
        b.gx,
        b.gy,
        b.gz,
        b.mask,
        case["g"],
        mode,
        ctx.L_all,
        ctx.meanf,
        ctx.bottomf,
        ctx.valid,
        1e-2,
        1e-3,
        100,
        5,
        ctx.stride,
        ctx.H_all,
        pattern,
        gain,
        True,
    )
    tc, cpu = _timeit(lambda: nk.icgn_12dof_parallel(*a12))
    tg, gpu = _timeit(lambda: ck.icgn_12dof_cuda(*a12))
    ok = (cpu[2] == STATUS_CONVERGED) & (gpu[2] == STATUS_CONVERGED)
    row12 = dict(
        cpu_s=tc,
        gpu_s=tg,
        same_status=float(np.mean(cpu[2] == gpu[2])),
        same_iter=float(np.mean(cpu[1] == gpu[1])),
        du_med=float(np.median(np.abs(gpu[0][ok, 9:] - cpu[0][ok, 9:]))),
        du_max=float(np.max(np.abs(gpu[0][ok, 9:] - cpu[0][ok, 9:]))),
        err_cpu=float(np.median(np.linalg.norm(cpu[0][ok, 9:] - case["U_true"][ok], axis=1))),
        err_gpu=float(np.median(np.linalg.norm(gpu[0][ok, 9:] - case["U_true"][ok], axis=1))),
        conv=float(ok.mean()),
    )
    N = ctx.n_nodes
    F12 = cpu[0][:, :9].reshape(N, 3, 3)
    a3 = (
        ctx.coords_int,
        cpu[0][:, 9:] + 0.2,
        F12,
        np.zeros((N, 3)),
        hx,
        hy,
        hz,
        b.f,
        b.gx,
        b.gy,
        b.gz,
        b.mask,
        case["g"],
        mode,
        ctx.H_all,
        ctx.meanf,
        ctx.bottomf,
        ctx.valid,
        para.mu,
        1e-2,
        1e-3,
        100,
        5,
        ctx.stride,
        float(pattern[9, 9]),
        gain,
        True,
    )
    tc3, cpu3 = _timeit(lambda: nk.icgn_3dof_parallel(*a3))
    tg3, gpu3 = _timeit(lambda: ck.icgn_3dof_cuda(*a3))
    ok3 = (cpu3[2] == STATUS_CONVERGED) & (gpu3[2] == STATUS_CONVERGED)
    row3 = dict(
        cpu_s=tc3,
        gpu_s=tg3,
        same_status=float(np.mean(cpu3[2] == gpu3[2])),
        same_iter=float(np.mean(cpu3[1] == gpu3[1])),
        du_max=float(np.max(np.abs(gpu3[0][ok3] - cpu3[0][ok3]))),
    )
    ap = (ctx.coords_int, hx, hy, hz, b.f, b.gx, b.gy, b.gz, b.mask, 0.5, 1e12, ctx.stride)
    tcp, cpp = _timeit(lambda: nk.precompute_nodes(*ap))
    tgp, gpp = _timeit(lambda: ck.precompute_nodes_cuda(*ap))
    rowp = dict(
        cpu_s=tcp,
        gpu_s=tgp,
        same_valid=float(np.mean(cpp[5] == gpp[5])),
        dh_rel=float(np.max(np.abs(gpp[0][cpp[5]] - cpp[0][cpp[5]])) / max(np.abs(cpp[0][cpp[5]]).max(), 1e-30)),
    )
    return row12, row3, rowp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "gpu.pdf"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not ck.cuda_available():
        with PdfPages(str(out)) as pdf:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.05, 0.95, "CUDA backend", fontsize=15, weight="bold", va="top")
            fig.text(
                0.05,
                0.9,
                f"pyALDVC {__version__}: no usable CUDA device on this machine ({ck.unavailable_reason()}).",
                fontsize=10,
                va="top",
            )
            pdf.savefig(fig)
        print(f"wrote {out} (no CUDA)")
        return 0
    n = 128 if args.quick else 256
    cases = [
        ("subset 32 / step 8", dict(n=n, ws=32, st=8)),
        ("subset 32 / step 8, noise sd 0.03", dict(n=n, ws=32, st=8, noise=0.03)),
        ("subset 32 / step 8, stride 2", dict(n=n, ws=32, st=8, stride=2)),
        ("subset 16 / step 8", dict(n=n, ws=16, st=8)),
        ("subset 48 / step 16", dict(n=n, ws=48, st=16)),
        ("subset 32 / step 16, B-spline", dict(n=n, ws=32, st=16, interp="bspline")),
        ("subset 32 / step 16, on-the-fly gradients", dict(n=n, ws=32, st=16, gradient_mode="on_the_fly")),
        ("subset 32 / step 16, reference + deformed masks", dict(n=n, ws=32, st=16, masks=True)),
    ]
    rows = []
    for label, kw in cases:
        print(label, "...")
        case = _case(**kw)
        r12, r3, rp = compare_kernels(case)
        rows.append((label, case["ctx"].n_nodes, r12, r3, rp))
    print("pipeline ...")
    shape = (n, n, n)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=1)
    dfm = warp_volume_lagrangian(ref, affine_displacement(F_TRUE, T_TRUE, centre))
    pipe = {}
    for backend in ("numba", "auto"):
        para = dvcpara_default(winsize=32, winstepsize=8, search_radius=6, verbose=False, backend=backend)
        run_aldvc(
            dvcpara_default(winsize=16, winstepsize=8, search_radius=4, verbose=False, backend=backend, admm_max_iter=2),
            [ref[:48, :48, :48], dfm[:48, :48, :48]],
        )
        t = time.perf_counter()
        res = run_aldvc(para, [ref, dfm])
        pipe[backend] = (time.perf_counter() - t, {k: float(v) for k, v in res.timings.items()}, res)
    fc, fg = pipe["numba"][2].result_disp[0], pipe["auto"][2].result_disp[0]
    ok = (fc.status == STATUS_CONVERGED) & (fg.status == STATUS_CONVERGED)
    pipe_du = float(np.max(np.abs(fg.U[ok] - fc.U[ok])))

    with PdfPages(str(out)) as pdf:
        lines = [
            f"pyALDVC {__version__} -- CUDA backend on {ck.device_name()} ({n}^3 synthetic speckle)",
            "",
            "One thread block per node: threads stride over the subset voxels, sampled values go to a",
            "per-block scratch row, statistics and gradient components are block-reduced in shared",
            "memory, thread 0 runs the control (12x12 Cholesky in float64, warp composition, stopping,",
            "stall, noise correction, look-ahead). Sampling and accumulation are float32. Same masks,",
            "NaN handling, stride, interpolation modes and gradient modes as the CPU kernels.",
            "",
            f"  {'case':<46s} {'nodes':>6s} {'12-DOF CPU':>10s} {'GPU':>7s} {'x':>5s} "
            f"{'3-DOF CPU':>9s} {'GPU':>7s} {'x':>5s} {'precomp CPU':>11s} {'GPU':>7s}",
        ]
        for label, nn, r12, r3, rp in rows:
            lines.append(
                f"  {label:<46s} {nn:6d} {r12['cpu_s']:10.3f} {r12['gpu_s']:7.3f} {r12['cpu_s'] / r12['gpu_s']:5.1f} "
                f"{r3['cpu_s']:9.3f} {r3['gpu_s']:7.3f} {r3['cpu_s'] / r3['gpu_s']:5.1f} {rp['cpu_s']:11.3f} {rp['gpu_s']:7.3f}"
            )
        lines += [
            "",
            "Parity (converged nodes): same status / same iteration count / |dU| median, max [voxel] / median error CPU vs GPU:",
        ]
        for label, nn, r12, r3, rp in rows:
            lines.append(
                f"  {label:<46s} 12-DOF {100 * r12['same_status']:5.1f} % / {100 * r12['same_iter']:5.1f} % / "
                f"{r12['du_med']:.1e}, {r12['du_max']:.1e} / {r12['err_cpu']:.4f} vs {r12['err_gpu']:.4f};  "
                f"3-DOF {100 * r3['same_status']:5.1f} % / {100 * r3['same_iter']:5.1f} % / max {r3['du_max']:.1e};  "
                f"H rel {rp['dh_rel']:.1e}"
            )
        tn, timings_n, _ = pipe["numba"]
        tg, timings_g, _ = pipe["auto"]
        lines += [
            "",
            f"Whole pipeline, subset 32 / step 8 ({fc.U.shape[0]} nodes): CPU {tn:.2f} s, GPU {tg:.2f} s ({tn / tg:.1f}x); "
            f"max |dU| between the two {pipe_du:.1e} voxel.",
            "  stage        CPU [s]   GPU [s]",
        ]
        for k in ("init_guess", "reference_precompute", "local_icgn", "subpb1", "subpb2", "strain"):
            lines.append(f"  {k:<20s} {timings_n.get(k, 0):7.2f} {timings_g.get(k, 0):9.2f}")
        lines += [
            "",
            "On the GPU the remaining time is the initial guess (the pyramid bookkeeping, phase",
            "correlation, block down-sampling and median cleaning stay on the CPU; only the direct",
            "ZNCC search runs on the GPU), the reference upload inside the precompute, and the global",
            "step. Installations without numba-cuda or without an NVIDIA GPU run the CPU kernels;",
            "backend='auto' decides once per process.",
        ]
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.03, 0.95, "CUDA backend", fontsize=15, weight="bold", va="top")
        fig.text(0.03, 0.9, "\n".join(lines), fontsize=6.6, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        labels = [r[0] for r in rows]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r[2]["cpu_s"] / r[2]["gpu_s"] for r in rows], 0.4, label="12-DOF")
        ax.bar(x + 0.2, [r[3]["cpu_s"] / r[3]["gpu_s"] for r in rows], 0.4, label="3-DOF")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("GPU speed-up over the 24-thread CPU kernel")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
