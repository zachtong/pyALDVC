#!/usr/bin/env python
"""Tutorial: a complete pyALDVC run on a volume pair, step by step.

The same steps as ``examples/tutorial_real_data.ipynb``. Point ``--reference``
and ``--deformed`` at your own volumes (TIFF stack, slice folder, MATLAB
``.mat``, ``.npy``); without them a synthetic speckle pair with a known
affine deformation is generated so the script runs anywhere::

    python examples/scripting/tutorial_real_data.py                     # synthetic demo
    python examples/scripting/tutorial_real_data.py --reference ref.tif --deformed def.tif --voxel-size 5 --units um

Outputs go to ``--output`` (default ``tutorial_output/``): results.npz, VTK
files for ParaView, a PDF report and the checkpoint directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Volumes
# ---------------------------------------------------------------------------


def load_or_synthesise(args):
    from al_dvc import load_volume
    from al_dvc.io.volume_io import volume_info

    if args.reference and args.deformed:
        ref = load_volume(args.reference)
        dfm = load_volume(args.deformed)
        print("reference:", volume_info(args.reference))
        truth = None
    else:
        # A synthetic pair: speckle volume, 2 % affine deformation applied as an exact Lagrangian warp.
        from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian

        shape = (64, 72, 80) if args.quick else (96, 104, 112)  # (nz, ny, nx)
        centre = tuple((s - 1) / 2 for s in shape[::-1])  # (x, y, z)
        F = np.array([[0.02, 0.004, 0.0], [0.003, -0.01, 0.002], [0.0, -0.002, 0.01]])
        truth = affine_displacement(F, (1.3, -0.7, 0.4), centre)
        ref = generate_speckle_volume(shape, sigma=2.0, seed=11)
        dfm = warp_volume_lagrangian(ref, truth)
        print(f"synthetic pair {shape} (nz, ny, nx), affine deformation with 2 % strain")
    print("reference volume:", ref.shape, ref.dtype, "deformed volume:", dfm.shape, dfm.dtype)
    return ref, dfm, truth


# ---------------------------------------------------------------------------
# 2. Parameters
# ---------------------------------------------------------------------------


def make_parameters(args, shape):
    from al_dvc import dvcpara_default
    from al_dvc.io.volume_ops import memory_model

    para = dvcpara_default(
        winsize=args.winsize,  # subset size in voxels (scalar or (x, y, z))
        winstepsize=args.step,  # node spacing
        voxel_size=(args.voxel_size,) * 3,  # physical scale of exported displacements and strains
        units=args.units,
        search_radius=8,  # NCC search half-width at the coarsest pyramid level; grows automatically
        interp_method="cubic",  # tricubic (MATLAB ba_interp3); "bspline" is more accurate, "linear" fastest
        use_global_step=True,  # AL-DVC: local subsets + global compatibility through ADMM
        strain_method="plane_fit",
        verbose=not args.quick,
    )
    mem = memory_model(shape, para.gradient_mode, para.interp_method)
    print(f"resident memory for this pair: {mem['bytes_per_voxel']:.0f} bytes/voxel, {mem['total_gb']:.2f} GB")
    return para


# ---------------------------------------------------------------------------
# 3. Run (with a checkpoint directory so an interrupted run resumes)
# ---------------------------------------------------------------------------


def run(para, ref, dfm, out_dir):
    from al_dvc import run_aldvc

    def progress(fraction, message):
        print(f"  [{100 * fraction:5.1f}%] {message}")

    result = run_aldvc(para, [ref, dfm], progress_fn=progress, checkpoint_dir=out_dir / "checkpoints")
    return result


# ---------------------------------------------------------------------------
# 4. Inspect the result
# ---------------------------------------------------------------------------


def inspect(result, truth):
    from al_dvc.core.data_structures import STATUS_NAMES
    from al_dvc.synthetic import evaluate_at_nodes

    mesh = result.dvc_mesh
    fr = result.result_disp[0]
    print(f"nodes: {mesh.n_nodes} on a {mesh.grid_shape} (z, y, x) grid, spacing {mesh.spacing}")
    codes, counts = np.unique(fr.status, return_counts=True)
    print("node status:", {STATUS_NAMES.get(int(c), int(c)): int(n) for c, n in zip(codes, counts)})
    print(f"median ZNCC {np.nanmedian(fr.zncc):.3f}")
    if fr.admm is not None:
        print(
            f"ADMM: beta {fr.admm.beta:.3g}, {fr.admm.n_steps} steps, displacement updates {np.round(fr.admm.update_global, 4)}"
        )
    std = fr.U_std[np.all(np.isfinite(fr.U_std), axis=1)]
    print(f"predicted displacement uncertainty (median, voxel): {np.median(std, axis=0).round(4)}")
    if truth is not None:
        U_gt = evaluate_at_nodes(truth, mesh.coordinates)
        ok = mesh.node_valid & np.all(np.isfinite(fr.U), axis=1)
        rmse = np.sqrt(np.mean((fr.U - U_gt)[ok] ** 2, axis=0))
        print(f"RMSE against the known deformation (voxel): {rmse.round(4)}")
    strain = result.result_strain[0]
    exx = strain.field("exx")
    print(f"exx: mean {np.nanmean(exx):.4f}, std {np.nanstd(exx):.4f}")
    print("timings [s]:", {k: round(v, 2) for k, v in result.timings.items() if not k.startswith("frame_")})


# ---------------------------------------------------------------------------
# 5. Export
# ---------------------------------------------------------------------------


def export(result, out_dir):
    from al_dvc.export import export_npz, export_report, export_vtk

    export_npz(result, out_dir / "results.npz")
    export_vtk(result, out_dir / "vtk", fields=["disp_magnitude", "disp_std", "exx", "eyy", "ezz", "von_mises"])
    export_report(result, out_dir / "report.pdf")
    print("written:", out_dir / "results.npz", out_dir / "vtk" / "aldvc.pvd", out_dir / "report.pdf")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference")
    ap.add_argument("--deformed")
    ap.add_argument("--output", default="tutorial_output")
    ap.add_argument("--winsize", type=int, default=24)
    ap.add_argument("--step", type=int, default=12)
    ap.add_argument("--voxel-size", type=float, default=1.0)
    ap.add_argument("--units", default="voxel")
    ap.add_argument("--quick", action="store_true", help="smaller synthetic volume, quieter")
    args = ap.parse_args(argv)
    if args.quick:
        args.winsize, args.step = 16, 8
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref, dfm, truth = load_or_synthesise(args)
    para = make_parameters(args, ref.shape)
    result = run(para, ref, dfm, out_dir)
    inspect(result, truth)
    export(result, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
