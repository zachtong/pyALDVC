"""Shared helpers for exporters."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import FrameResult, PipelineResult, StrainResult

DISP_FIELDS = ("disp_u", "disp_v", "disp_w", "disp_magnitude")
STD_FIELDS = ("disp_std_u", "disp_std_v", "disp_std_w", "disp_std")
STRAIN_FIELDS = (
    "exx",
    "eyy",
    "ezz",
    "exy",
    "exz",
    "eyz",
    "e1",
    "e2",
    "e3",
    "max_shear",
    "von_mises",
    "volumetric",
    "det_F",
    "rotation_deg",
)
ALL_FIELDS = DISP_FIELDS + STD_FIELDS + STRAIN_FIELDS


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def frame_tag(i: int, n: int) -> str:
    width = max(2, len(str(n)))
    return f"frame_{i + 1:0{width}d}"


def displacement_physical(fr: FrameResult, voxel_size) -> NDArray[np.float64]:
    """Cumulative displacement in physical units ``(N, 3)``."""
    U = fr.U_accum if fr.U_accum is not None else fr.U
    return np.asarray(U, dtype=np.float64) * np.asarray(voxel_size, dtype=np.float64)[None, :]


def field_array(result: PipelineResult, frame: int, name: str, trimmed: bool = True) -> NDArray[np.float64]:
    """Per-node array of a named field for one frame (displacement or strain)."""
    fr = result.result_disp[frame]
    if name in DISP_FIELDS:
        U = displacement_physical(fr, result.dvc_para.voxel_size)
        if name == "disp_u":
            return U[:, 0]
        if name == "disp_v":
            return U[:, 1]
        if name == "disp_w":
            return U[:, 2]
        return np.linalg.norm(U, axis=1)
    if name in STD_FIELDS:
        if fr.U_std is None:
            raise ValueError("displacement uncertainty is not available for this result (U_std is None)")
        S = np.asarray(fr.U_std, dtype=np.float64) * np.asarray(result.dvc_para.voxel_size, dtype=np.float64)[None, :]
        comp = {"disp_std_u": 0, "disp_std_v": 1, "disp_std_w": 2}.get(name)
        return S[:, comp] if comp is not None else np.linalg.norm(S, axis=1)
    if frame >= len(result.result_strain):
        raise ValueError(f"strain field '{name}' requested but strain was not computed")
    sr: StrainResult = result.result_strain[frame]
    return sr.field(name, trimmed=trimmed)


def result_summary(result: PipelineResult) -> dict:
    """Small JSON-friendly summary for reports."""
    mesh = result.dvc_mesh
    out = {
        "n_frames": result.n_frames,
        "n_nodes": mesh.n_nodes,
        "grid_shape_zyx": list(mesh.grid_shape),
        "spacing_xyz": list(mesh.spacing),
        "volume_shape_zyx": list(result.volume_shape),
        "timings": {k: float(v) for k, v in result.timings.items()},
        "stopped_early": result.stopped_early,
        "frames": [],
    }
    for i, fr in enumerate(result.result_disp):
        entry = {"index": i + 1, "ref_frame": fr.ref_frame}
        if fr.zncc is not None:
            z = np.asarray(fr.zncc)
            entry["median_zncc"] = float(np.nanmedian(z)) if np.isfinite(z).any() else None
        if fr.status is not None:
            entry["frac_converged"] = float(np.mean(np.asarray(fr.status) == 0))
        if fr.admm is not None:
            entry["beta"] = fr.admm.beta
            entry["admm_steps"] = fr.admm.n_steps
            entry["update_global"] = list(map(float, fr.admm.update_global))
        out["frames"].append(entry)
    return out
