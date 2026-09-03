"""VTK ImageData (``.vti``) export for ParaView / VisIt / pyvista.

The node grid is a regular image grid, so every field is written as point
data of a ``vtkImageData`` with ``Origin = (x0[0], y0[0], z0[0]) * voxel_size``
and ``Spacing = spacing * voxel_size``. Binary base64 payloads keep the
files compact; no external dependency is needed.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import numpy as np

from ..core.data_structures import PipelineResult
from .export_utils import ALL_FIELDS, ensure_dir, field_array, frame_tag


def _encode(arr: np.ndarray) -> str:
    raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    header = struct.pack("<I", len(raw))
    return base64.b64encode(header + raw).decode("ascii")


def write_vti(path: Path, grid_shape: tuple[int, int, int], origin, spacing, point_data: dict[str, np.ndarray]) -> Path:
    """Write a ``.vti`` file. ``point_data`` values are ``(N,)`` or ``(N, 3)``
    node arrays ordered ``n = iz*ny*nx + iy*nx + ix`` (VTK's own x-fastest order)."""
    nz, ny, nx = grid_shape
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt32">',
        f'  <ImageData WholeExtent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}" '
        f'Origin="{origin[0]} {origin[1]} {origin[2]}" Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">',
        f'    <Piece Extent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}">',
        "      <PointData>",
    ]
    for name, arr in point_data.items():
        arr = np.asarray(arr, dtype=np.float32)
        ncomp = 1 if arr.ndim == 1 else arr.shape[1]
        arr = np.where(np.isfinite(arr), arr, np.nan)
        lines.append(f'        <DataArray type="Float32" Name="{name}" NumberOfComponents="{ncomp}" format="binary">')
        lines.append("          " + _encode(arr))
        lines.append("        </DataArray>")
    lines += ["      </PointData>", "    </Piece>", "  </ImageData>", "</VTKFile>", ""]
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def export_vtk(
    result: PipelineResult,
    out_dir: str | Path,
    basename: str = "aldvc",
    fields: list[str] | None = None,
    frames: list[int] | None = None,
    trimmed: bool = True,
) -> list[Path]:
    """Write one ``.vti`` per frame (plus a ``.pvd`` time-series index)."""
    out = ensure_dir(out_dir)
    mesh = result.dvc_mesh
    vs = np.asarray(result.dvc_para.voxel_size, dtype=np.float64)
    origin = (mesh.x0[0] * vs[0], mesh.y0[0] * vs[1], mesh.z0[0] * vs[2])
    spacing = (mesh.spacing[0] * vs[0], mesh.spacing[1] * vs[1], mesh.spacing[2] * vs[2])
    n = len(result.result_disp)
    if fields is None:
        fields = ["disp_magnitude"]
        if result.result_strain:
            fields += ["exx", "eyy", "ezz", "exy", "exz", "eyz", "von_mises", "volumetric"]
    unknown = [f for f in fields if f not in ALL_FIELDS]
    if unknown:
        raise ValueError(f"Unknown field(s) {unknown}; valid: {ALL_FIELDS}")
    paths = []
    for k in (frames if frames is not None else range(n)):
        fr = result.result_disp[k]
        U = (fr.U_accum if fr.U_accum is not None else fr.U) * vs[None, :]
        data: dict[str, np.ndarray] = {"displacement": U, "node_valid": mesh.node_valid.astype(np.float32)}
        if fr.zncc is not None:
            data["zncc"] = fr.zncc
        for f in fields:
            data[f] = field_array(result, k, f, trimmed=trimmed)
        p = out / f"{basename}_{frame_tag(k, n)}.vti"
        write_vti(p, mesh.grid_shape, origin, spacing, data)
        paths.append(p)
    pvd = out / f"{basename}.pvd"
    entries = "\n".join(
        f'    <DataSet timestep="{i + 1}" group="" part="0" file="{p.name}"/>' for i, p in enumerate(paths)
    )
    pvd.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        f"  <Collection>\n{entries}\n  </Collection>\n</VTKFile>\n",
        encoding="ascii",
    )
    return paths
