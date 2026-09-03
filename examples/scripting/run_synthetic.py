"""End-to-end example on synthetic data with ground truth.

python examples/scripting/run_synthetic.py [output_dir]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from al_dvc import dvcpara_default, run_aldvc, warmup
from al_dvc.export import export_csv, export_npz, export_report, export_run_summary, export_vtk
from al_dvc.synthetic import (
    affine_displacement,
    evaluate_at_nodes,
    generate_speckle_volume,
    gradient_at_nodes,
    warp_volume_lagrangian,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")


def main(out_dir: Path) -> None:
    warmup()  # compile the Numba kernels once (cached afterwards)

    # --- synthetic pair: 2 % stretch in x with Poisson contraction, plus a translation
    shape = (96, 104, 112)  # (nz, ny, nx)
    ref = generate_speckle_volume(shape, sigma=2.0, seed=0)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    F_true = np.diag([0.02, -0.006, -0.006])
    disp = affine_displacement(F_true, (1.5, -0.7, 2.2), centre)
    dfm = warp_volume_lagrangian(ref, disp)

    para = dvcpara_default(
        winsize=24,
        winstepsize=8,
        search_radius=6,
        voxel_size=(2.5, 2.5, 2.5),
        units="um",
    )
    t0 = time.perf_counter()
    result = run_aldvc(para, [ref, dfm])
    print(f"solved {result.dvc_mesh.n_nodes} nodes in {time.perf_counter() - t0:.1f}s")

    # --- compare with ground truth (voxels)
    mesh = result.dvc_mesh
    fr = result.result_disp[0]
    U_gt = evaluate_at_nodes(disp, mesh.coordinates)
    F_gt = gradient_at_nodes(disp, mesh.coordinates)
    interior = ~np.isin(np.arange(mesh.n_nodes), mesh.boundary_nodes)
    rmse_u = np.sqrt(np.mean((fr.U - U_gt)[interior] ** 2, axis=0))
    rmse_F = np.sqrt(np.mean((fr.F - F_gt)[interior] ** 2))
    rmse_F_local = np.sqrt(np.mean((fr.F_local - F_gt)[interior] ** 2))
    print(f"displacement RMSE (u, v, w) = {rmse_u} voxel")
    print(f"gradient RMSE: AL-DVC {rmse_F:.2e}   local IC-GN {rmse_F_local:.2e}")
    print(f"ADMM: beta={fr.admm.beta:.3e}, steps={fr.admm.n_steps}, median ZNCC={np.nanmedian(fr.zncc):.4f}")
    sr = result.result_strain[0]
    print(f"mean exx (valid nodes) = {np.nanmean(sr.exx[sr.strain_valid]):.5f}  (truth {F_true[0, 0]})")

    # --- exports
    out_dir.mkdir(parents=True, exist_ok=True)
    export_npz(result, out_dir / "synthetic.npz")
    export_vtk(result, out_dir / "vtk", "synthetic")
    export_csv(result, out_dir / "csv", "synthetic")
    export_run_summary(result, out_dir / "synthetic_summary.json")
    export_report(result, out_dir / "synthetic_report.pdf", gt={"U": [U_gt], "F": [F_gt]})
    print(f"exports written to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("aldvc_synthetic_out"))
