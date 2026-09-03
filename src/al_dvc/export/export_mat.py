"""Export to MATLAB ``.mat`` in a layout close to the MATLAB ALDVC results.

Both the Python layout (``U`` (N,3), ``F`` (N,3,3) row-major) and the MATLAB
ALDVC interleaved layout (``U_interleaved`` (3N,1), ``F_interleaved`` (9N,1)
column-major ``[F11,F21,F31,F12,...]``) are written. Node/element indices
are converted to 1-based.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from ..core.config import para_to_dict
from ..core.data_structures import F_to_matlab_order, PipelineResult
from .export_utils import ensure_dir


def export_mat(result: PipelineResult, path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".mat":
        p = p.with_suffix(".mat")
    ensure_dir(p.parent)
    mesh = result.dvc_mesh
    para = {k: (v if v is not None else "") for k, v in para_to_dict(result.dvc_para).items()}
    data: dict = {
        "coordinatesFEM": mesh.coordinates,
        "elementsFEM": mesh.elements + 1,
        "gridShapeZYX": np.asarray(mesh.grid_shape),
        "x0": mesh.x0,
        "y0": mesh.y0,
        "z0": mesh.z0,
        "winsize": np.asarray(result.dvc_para.winsize),
        "winstepsize": np.asarray(result.dvc_para.winstepsize),
        "voxelSize": np.asarray(result.dvc_para.voxel_size),
        "refIndices": np.asarray(result.frame_schedule.ref_indices) + 1,
        "DVCpara": para,
    }
    n = len(result.result_disp)
    U = np.empty((n,), dtype=object)
    U_acc = np.empty((n,), dtype=object)
    F = np.empty((n,), dtype=object)
    U_int = np.empty((n,), dtype=object)
    F_int = np.empty((n,), dtype=object)
    zncc = np.empty((n,), dtype=object)
    for k, fr in enumerate(result.result_disp):
        U[k] = fr.U
        U_acc[k] = fr.U_accum if fr.U_accum is not None else fr.U
        F[k] = fr.F
        U_int[k] = fr.U.reshape(-1, 1)
        F_int[k] = F_to_matlab_order(fr.F).reshape(-1, 1)
        zncc[k] = fr.zncc if fr.zncc is not None else np.array([])
    data.update(
        {
            "ResultDisp": U,
            "ResultDispAccum": U_acc,
            "ResultDefGrad": F,
            "ResultDisp_interleaved": U_int,
            "ResultDefGrad_interleaved": F_int,
            "ResultZNCC": zncc,
        }
    )
    if result.result_strain:
        m = len(result.result_strain)
        strain = np.empty((m,), dtype=object)
        for k, sr in enumerate(result.result_strain):
            strain[k] = {
                "exx": sr.exx,
                "eyy": sr.eyy,
                "ezz": sr.ezz,
                "exy": sr.exy,
                "exz": sr.exz,
                "eyz": sr.eyz,
                "principal": sr.principal,
                "vonMises": sr.von_mises,
                "maxShear": sr.max_shear,
                "volumetric": sr.volumetric,
                "detF": sr.det_F,
                "rotationDeg": sr.rotation_deg,
                "valid": sr.strain_valid.astype(np.uint8),
                "F": sr.F,
                "dispPhysical": np.column_stack([sr.disp_u, sr.disp_v, sr.disp_w]),
                "strainType": sr.strain_type,
                "method": sr.method,
            }
        data["ResultStrain"] = strain
    savemat(str(p), data, do_compression=True, long_field_names=True)
    return p
