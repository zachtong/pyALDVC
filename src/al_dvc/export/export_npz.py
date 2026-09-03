"""Export to a NumPy ``.npz`` archive (self-describing, lossless)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.config import para_to_dict
from ..core.data_structures import PipelineResult
from .export_utils import ensure_dir


def export_npz(result: PipelineResult, path: str | Path) -> Path:
    """Write every frame's displacement/gradient/strain plus the mesh and parameters.

    Layout: ``coordinates`` (N,3), ``grid_shape`` (3,), ``x0/y0/z0``, and per
    frame ``U_k`` (N,3 voxels), ``U_accum_k``, ``F_k`` (N,3,3), ``zncc_k``,
    ``status_k`` and strain fields ``exx_k`` ... in physical units. The
    parameter set is stored as a JSON string under ``para_json``.
    """
    p = Path(path)
    if p.suffix.lower() != ".npz":
        p = p.with_suffix(".npz")
    ensure_dir(p.parent)
    mesh = result.dvc_mesh
    arrays: dict[str, np.ndarray] = {
        "coordinates": mesh.coordinates,
        "grid_shape": np.asarray(mesh.grid_shape, dtype=np.int64),
        "x0": mesh.x0,
        "y0": mesh.y0,
        "z0": mesh.z0,
        "spacing": np.asarray(mesh.spacing, dtype=np.float64),
        "node_valid": mesh.node_valid,
        "elements": mesh.elements,
        "voxel_size": np.asarray(result.dvc_para.voxel_size, dtype=np.float64),
        "ref_indices": np.asarray(result.frame_schedule.ref_indices, dtype=np.int64),
        "volume_shape": np.asarray(result.volume_shape, dtype=np.int64),
        "para_json": np.array(json.dumps(para_to_dict(result.dvc_para))),
    }
    for k, fr in enumerate(result.result_disp):
        tag = f"_{k + 1}"
        arrays["U" + tag] = fr.U
        arrays["F" + tag] = fr.F
        if fr.U_accum is not None:
            arrays["U_accum" + tag] = fr.U_accum
        if fr.zncc is not None:
            arrays["zncc" + tag] = fr.zncc
        if fr.status is not None:
            arrays["status" + tag] = fr.status
        if fr.U_local is not None:
            arrays["U_local" + tag] = fr.U_local
    for k, sr in enumerate(result.result_strain):
        tag = f"_{k + 1}"
        for name in ("exx", "eyy", "ezz", "exy", "exz", "eyz", "von_mises", "max_shear", "volumetric", "det_F", "rotation_deg"):
            arrays[name + tag] = getattr(sr, name)
        arrays["principal" + tag] = sr.principal
        arrays["strain_valid" + tag] = sr.strain_valid
        arrays["disp_phys" + tag] = np.column_stack([sr.disp_u, sr.disp_v, sr.disp_w])
        arrays["F_phys" + tag] = sr.F
    np.savez_compressed(str(p), **arrays)
    return p


def load_npz_result(path: str | Path) -> dict:
    """Load an exported archive into a plain dict (arrays + ``para``)."""
    with np.load(str(path), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["para"] = json.loads(str(d.pop("para_json")))
    return d
