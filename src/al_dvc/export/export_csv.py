"""Per-node CSV export (one file per frame)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..core.data_structures import PipelineResult
from .export_utils import ALL_FIELDS, ensure_dir, field_array, frame_tag


def export_csv(
    result: PipelineResult,
    out_dir: str | Path,
    basename: str = "aldvc",
    fields: list[str] | None = None,
    frames: list[int] | None = None,
    trimmed: bool = True,
) -> list[Path]:
    """Write ``<basename>_<frame>.csv`` with x,y,z (voxels) and requested fields."""
    out = ensure_dir(out_dir)
    n = len(result.result_disp)
    if fields is None:
        fields = ["disp_u", "disp_v", "disp_w"]
        if result.result_strain:
            fields += ["exx", "eyy", "ezz", "exy", "exz", "eyz", "von_mises"]
    unknown = [f for f in fields if f not in ALL_FIELDS]
    if unknown:
        raise ValueError(f"Unknown field(s) {unknown}; valid: {ALL_FIELDS}")
    coords = result.dvc_mesh.coordinates
    paths = []
    for k in (frames if frames is not None else range(n)):
        cols = [field_array(result, k, f, trimmed=trimmed) for f in fields]
        p = out / f"{basename}_{frame_tag(k, n)}.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["x", "y", "z", "valid"] + fields)
            valid = result.dvc_mesh.node_valid
            for i in range(coords.shape[0]):
                w.writerow([coords[i, 0], coords[i, 1], coords[i, 2], int(valid[i]) if valid.size else 1]
                           + [("" if not np.isfinite(c[i]) else repr(float(c[i]))) for c in cols])
        paths.append(p)
    return paths
