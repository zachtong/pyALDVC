"""Export to a NumPy ``.npz`` archive (self-describing, lossless)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.config import para_from_dict, para_to_dict
from ..core.data_structures import DVCMesh, FrameResult, FrameSchedule, PipelineResult, StrainResult
from .export_utils import ensure_dir

_STRAIN_ARRAYS = ("exx", "eyy", "ezz", "exy", "exz", "eyz", "von_mises", "max_shear", "volumetric", "det_F", "rotation_deg")


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
        "boundary_nodes": np.asarray(mesh.boundary_nodes, dtype=np.int64),
        "edge_ok": np.asarray(mesh.edge_ok, dtype=bool),
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
        if fr.split_fraction is not None:
            arrays["split_fraction" + tag] = fr.split_fraction
        if fr.status is not None:
            arrays["status" + tag] = fr.status
        if fr.U_local is not None:
            arrays["U_local" + tag] = fr.U_local
        if fr.U_std is not None:
            arrays["U_std" + tag] = fr.U_std
        if fr.outlier is not None:
            arrays["outlier" + tag] = np.asarray(fr.outlier, dtype=bool)
    for k, sr in enumerate(result.result_strain):
        tag = f"_{k + 1}"
        for name in _STRAIN_ARRAYS:
            arrays[name + tag] = getattr(sr, name)
        arrays["principal" + tag] = sr.principal
        arrays["strain_valid" + tag] = sr.strain_valid
        arrays["disp_phys" + tag] = np.column_stack([sr.disp_u, sr.disp_v, sr.disp_w])
        arrays["F_phys" + tag] = sr.F
    np.savez_compressed(str(p), **arrays)
    return p


def result_from_npz(path: str | Path) -> PipelineResult:
    """Rebuild the :class:`PipelineResult` an archive was written from, so scripts and ``al-dvc stats`` can
    analyse an exported run. Timings and ADMM diagnostics are not archived and come back empty."""
    with np.load(str(path), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    para = para_from_dict(json.loads(str(d["para_json"])))
    empty_edges = np.empty((0, 3), dtype=bool)
    mesh = DVCMesh(
        coordinates=d["coordinates"],
        elements=d["elements"],
        grid_shape=tuple(int(v) for v in d["grid_shape"]),
        x0=d["x0"],
        y0=d["y0"],
        z0=d["z0"],
        spacing=tuple(float(v) for v in d["spacing"]),
        node_valid=np.asarray(d["node_valid"], dtype=bool),
        boundary_nodes=np.asarray(d.get("boundary_nodes", np.empty(0, dtype=np.int64)), dtype=np.int64),
        edge_ok=np.asarray(d.get("edge_ok", empty_edges), dtype=bool),
    )
    ref_indices = tuple(int(v) for v in d["ref_indices"])
    frames = []
    k = 1
    while f"U_{k}" in d:
        tag = f"_{k}"
        frames.append(
            FrameResult(
                U=d["U" + tag],
                F=d["F" + tag],
                ref_frame=ref_indices[k - 1] if k - 1 < len(ref_indices) else 0,
                U_accum=d.get("U_accum" + tag),
                U_local=d.get("U_local" + tag),
                zncc=d.get("zncc" + tag),
                U_std=d.get("U_std" + tag),
                status=d.get("status" + tag),
                split_fraction=d.get("split_fraction" + tag),
                outlier=None if d.get("outlier" + tag) is None else np.asarray(d["outlier" + tag], dtype=bool),
            )
        )
        k += 1
    strains = []
    k = 1
    while f"exx_{k}" in d:
        tag = f"_{k}"
        disp = d["disp_phys" + tag]
        strains.append(
            StrainResult(
                disp_u=disp[:, 0],
                disp_v=disp[:, 1],
                disp_w=disp[:, 2],
                F=d["F_phys" + tag],
                principal=d["principal" + tag],
                strain_valid=np.asarray(d["strain_valid" + tag], dtype=bool),
                strain_type=para.strain_type,
                method=para.strain_method,
                **{name: d[name + tag] for name in _STRAIN_ARRAYS},
            )
        )
        k += 1
    return PipelineResult(
        dvc_para=para,
        dvc_mesh=mesh,
        result_disp=frames,
        result_strain=strains,
        frame_schedule=FrameSchedule(ref_indices=ref_indices),
        volume_shape=tuple(int(v) for v in d["volume_shape"]),
    )


def load_npz_result(path: str | Path) -> dict:
    """Load an exported archive into a plain dict (arrays + ``para``)."""
    with np.load(str(path), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["para"] = json.loads(str(d.pop("para_json")))
    return d
