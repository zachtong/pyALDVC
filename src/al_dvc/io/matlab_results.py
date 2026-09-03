"""Read result files written by the MATLAB ALDVC code (``results_ws*_st*.mat``).

The MATLAB code stores, per frame pair, the interleaved displacement vector
``U`` (3N,) = ``[u1, v1, w1, u2, v2, w2, ...]``, the interleaved gradient
vector ``F`` (9N,) with per-node order
``[F11, F21, F31, F12, F22, F32, F13, F23, F33]`` (``Fij = du_i/dx_j``) and
1-based node coordinates ``coordinatesFEM`` (N, 3) = ``[x, y, z]``.
This module converts them to the pyALDVC layout: 0-based coordinates,
``U`` (N, 3) and ``F`` (N, 3, 3), so that MATLAB results can be compared
with, or plotted next to, pyALDVC results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MatlabFrame",
    "MatlabResults",
    "interleaved_to_U",
    "interleaved_to_F",
    "load_matlab_results",
    "match_nodes",
]


@dataclass
class MatlabFrame:
    """One frame pair of a MATLAB ALDVC result (pyALDVC layout)."""

    U: NDArray[np.float64]
    U_local: NDArray[np.float64] | None = None
    U0: NDArray[np.float64] | None = None
    F: NDArray[np.float64] | None = None
    F_local: NDArray[np.float64] | None = None
    beta: float | None = None
    mu: float | None = None
    conv_iter: NDArray[np.float64] | None = None
    """IC-GN iterations per node and ADMM step, ``(N, n_steps)``."""


@dataclass
class MatlabResults:
    """Contents of a MATLAB ``results_ws*_st*.mat`` file in pyALDVC layout."""

    path: Path
    file_names: list[str]
    para: dict[str, Any]
    coordinates: NDArray[np.float64]
    """(N, 3) node coordinates ``[x, y, z]``, 0-based voxel units."""
    elements: NDArray[np.int64]
    """(E, 8) hex8 connectivity, 0-based."""
    winsize: tuple[int, int, int]
    winstepsize: tuple[int, int, int]
    grid_range: dict[str, tuple[int, int]] = field(default_factory=dict)
    """MATLAB ``gridRange`` converted to 0-based inclusive ``(lo, hi)`` per axis."""
    frames: list[MatlabFrame] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return int(self.coordinates.shape[0])


def interleaved_to_U(vec: NDArray) -> NDArray[np.float64]:
    """MATLAB ``(3N,)`` ``[u1, v1, w1, ...]`` -> ``(N, 3)``."""
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size % 3:
        raise ValueError(f"interleaved displacement length {v.size} is not a multiple of 3")
    return v.reshape(-1, 3)


def interleaved_to_F(vec: NDArray) -> NDArray[np.float64]:
    """MATLAB ``(9N,)`` column-major per-node gradient -> ``(N, 3, 3)`` with ``F[n, i, j] = du_i/dx_j``.

    Inverse of :func:`al_dvc.core.data_structures.F_to_matlab_order`.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size % 9:
        raise ValueError(f"interleaved gradient length {v.size} is not a multiple of 9")
    return np.ascontiguousarray(np.transpose(v.reshape(-1, 3, 3), (0, 2, 1)))


def _to_python(obj: Any) -> Any:
    """Recursively convert ``scipy.io.loadmat`` objects (mat_struct, arrays) to plain Python."""
    if hasattr(obj, "_fieldnames"):
        return {k: _to_python(getattr(obj, k)) for k in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_to_python(o) for o in obj.ravel()]
        if obj.size == 1:
            return obj.reshape(-1)[0].item()
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _cell_items(obj: Any) -> list[Any]:
    """Items of a (squeezed) MATLAB cell array; a lone struct becomes a one-item list."""
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [o for o in obj.ravel()]
    return [obj]


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size != 3:
        raise ValueError(f"MATLAB DVCpara.{name} must have 1 or 3 entries, got {arr.size}")
    return (int(arr[0]), int(arr[1]), int(arr[2]))


def _get(struct: Any, name: str, default: Any = None) -> Any:
    return getattr(struct, name) if hasattr(struct, name) else default


def load_matlab_results(path: str | Path) -> MatlabResults:
    """Load a MATLAB ALDVC result file into pyALDVC layout (0-based, ``(N, 3)`` / ``(N, 3, 3)``).

    Raises ``FileNotFoundError`` / ``KeyError`` with the missing variable name
    when the file is not an ALDVC result file.
    """
    from scipy.io import loadmat

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"MATLAB result file not found: {p}")
    m = loadmat(str(p), squeeze_me=True, struct_as_record=False)
    for key in ("DVCpara", "DVCmesh", "ResultDisp"):
        if key not in m:
            raise KeyError(f"{p}: variable '{key}' not found; not an ALDVC result file")

    para_struct = m["DVCpara"]
    para = _to_python(para_struct)
    winsize = _triple(_get(para_struct, "winsize", 32), "winsize")
    winstepsize = _triple(_get(para_struct, "winstepsize", 16), "winstepsize")

    grid_range: dict[str, tuple[int, int]] = {}
    gr = _get(para_struct, "gridRange")
    if gr is not None:
        for axis in "xyz":
            rng = _get(gr, f"grid{axis}Range")
            if rng is not None:
                arr = np.asarray(rng, dtype=np.float64).reshape(-1)
                grid_range[axis] = (int(arr[0]) - 1, int(arr[-1]) - 1)

    mesh = m["DVCmesh"]
    coords = np.asarray(mesh.coordinatesFEM, dtype=np.float64).reshape(-1, 3) - 1.0
    elements = np.asarray(_get(mesh, "elementsFEM", np.zeros((0, 8))), dtype=np.int64).reshape(-1, 8) - 1

    disp_items = _cell_items(m["ResultDisp"])
    grad_items = _cell_items(m["ResultDefGrad"]) if "ResultDefGrad" in m else []
    mubeta_items = _cell_items(m["ResultMuBeta"]) if "ResultMuBeta" in m else []
    conv_items = _cell_items(m["ResultConvItPerEle"]) if "ResultConvItPerEle" in m else []

    frames: list[MatlabFrame] = []
    n_nodes = coords.shape[0]
    for k, d in enumerate(disp_items):
        U = interleaved_to_U(d.U)
        if U.shape[0] != n_nodes:
            raise ValueError(f"{p}: frame {k + 1} has {U.shape[0]} nodes, mesh has {n_nodes}")
        fr = MatlabFrame(U=U)
        if hasattr(d, "U_local_ICGN"):
            fr.U_local = interleaved_to_U(d.U_local_ICGN)
        if hasattr(d, "U0_crosscorr"):
            fr.U0 = interleaved_to_U(d.U0_crosscorr)
        if k < len(grad_items):
            gk = grad_items[k]
            if hasattr(gk, "F"):
                fr.F = interleaved_to_F(gk.F)
            if hasattr(gk, "F_local_ICGN"):
                fr.F_local = interleaved_to_F(gk.F_local_ICGN)
        if k < len(mubeta_items):
            mb = mubeta_items[k]
            fr.beta = float(_get(mb, "ALVarBeta", np.nan))
            fr.mu = float(_get(mb, "ALVarMu", np.nan))
        if k < len(conv_items):
            ci = np.asarray(_get(conv_items[k], "ConvItPerEle", np.zeros((n_nodes, 0))), dtype=np.float64)
            fr.conv_iter = ci.reshape(n_nodes, -1)
        frames.append(fr)

    names = m.get("fileNameAll", [])
    file_names = [str(s) for s in np.atleast_1d(names)] if not isinstance(names, str) else [names]

    return MatlabResults(
        path=p,
        file_names=file_names,
        para=para,
        coordinates=coords,
        elements=elements,
        winsize=winsize,
        winstepsize=winstepsize,
        grid_range=grid_range,
        frames=frames,
    )


def match_nodes(coords_a: NDArray, coords_b: NDArray, decimals: int = 3) -> NDArray[np.int64]:
    """For every node of ``coords_a`` return its row in ``coords_b`` (or -1).

    Coordinates are matched after rounding to ``decimals`` decimals.
    """
    a = np.round(np.asarray(coords_a, dtype=np.float64).reshape(-1, 3), decimals)
    b = np.round(np.asarray(coords_b, dtype=np.float64).reshape(-1, 3), decimals)
    lookup = {tuple(row): i for i, row in enumerate(b.tolist())}
    out = np.fromiter((lookup.get(tuple(row), -1) for row in a.tolist()), dtype=np.int64, count=a.shape[0])
    return out
