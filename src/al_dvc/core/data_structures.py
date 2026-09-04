"""Core data structures for the AL-DVC pipeline.

All containers are dataclasses; parameter containers are frozen.

Layout contracts (see docs/design.md, section 4):
    - Volumes are ``(nz, ny, nx)`` arrays, ``vol[z, y, x]``.
    - Node coordinates are ``(N, 3)`` with columns ``[x, y, z]`` (voxels).
    - Grid fields are ``(nz, ny, nx)``; node ``n = iz*ny*nx + iy*nx + ix``.
    - Displacement ``U`` is ``(N, 3)`` = ``[u, v, w]``.
    - Displacement gradient ``F`` is ``(N, 3, 3)``, ``F[n, i, j] = du_i/dx_j``.
    - All indices are 0-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Node status codes (shared by every solver stage)
# ---------------------------------------------------------------------------

STATUS_CONVERGED = 0
STATUS_MAX_ITER = 1
STATUS_OUT_OF_BOUNDS = 2
STATUS_INVALID_SUBSET = 3
STATUS_SINGULAR = 4
STATUS_NAN = 5
STATUS_SKIPPED = 6  # node not solved (masked out before the solver ran)
STATUS_STALLED = 7  # no objective improvement for icgn_patience iterations (hopeless node)

STATUS_NAMES: dict[int, str] = {
    STATUS_CONVERGED: "converged",
    STATUS_MAX_ITER: "max_iter",
    STATUS_OUT_OF_BOUNDS: "out_of_bounds",
    STATUS_INVALID_SUBSET: "invalid_subset",
    STATUS_SINGULAR: "singular",
    STATUS_NAN: "nan",
    STATUS_SKIPPED: "skipped",
    STATUS_STALLED: "stalled",
}


# ---------------------------------------------------------------------------
# Volume of interest
# ---------------------------------------------------------------------------


VOI_EXTRA_MARGIN = 6  # voxels beyond the subset half-width and search range around a mask's bounding box


def voi_from_mask(mask, winsize, search_radius, shape=None) -> "VOIRange | None":
    """Bounding box of the valid voxels of ``mask`` grown by the subset half-width, the search range and
    :data:`VOI_EXTRA_MARGIN`, clamped to the volume; ``None`` when the mask is empty or the box is the whole volume.

    This is how the GUI turns the region of interest drawn on the slices into the analysed box: subsets near
    the region's edge keep their full support and the search window, and everything else is cropped away
    (memory and time scale with the box, not with the scan).
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"mask must be 3-D (got shape {m.shape})")
    shape = tuple(int(s) for s in (shape if shape is not None else m.shape))
    if not m.any():
        return None
    zs = np.flatnonzero(m.any(axis=(1, 2)))
    ys = np.flatnonzero(m.any(axis=(0, 2)))
    xs = np.flatnonzero(m.any(axis=(0, 1)))
    ws = np.broadcast_to(np.asarray(winsize, dtype=np.int64), (3,))
    sr = np.broadcast_to(np.asarray(search_radius, dtype=np.int64), (3,))
    margins = [int(ws[i] // 2 + sr[i] + VOI_EXTRA_MARGIN) for i in range(3)]  # (x, y, z)
    nz, ny, nx = shape
    box = VOIRange(
        x=(max(0, int(xs[0]) - margins[0]), min(nx - 1, int(xs[-1]) + margins[0])),
        y=(max(0, int(ys[0]) - margins[1]), min(ny - 1, int(ys[-1]) + margins[1])),
        z=(max(0, int(zs[0]) - margins[2]), min(nz - 1, int(zs[-1]) + margins[2])),
    )
    if box.x == (0, nx - 1) and box.y == (0, ny - 1) and box.z == (0, nz - 1):
        return None
    return box


@dataclass(frozen=True)
class VOIRange:
    """Volume of interest as inclusive voxel index ranges.

    Attributes:
        x: ``(xmin, xmax)`` inclusive, along the last array axis.
        y: ``(ymin, ymax)`` inclusive, along the middle axis.
        z: ``(zmin, zmax)`` inclusive, along the first axis.

    A range of ``(0, -1)`` means "whole extent" and is resolved against the
    volume shape by :func:`clamp`.
    """

    x: tuple[int, int] = (0, -1)
    y: tuple[int, int] = (0, -1)
    z: tuple[int, int] = (0, -1)

    @property
    def is_whole(self) -> bool:
        """True for the default ``(0, -1)`` ranges, i.e. no cropping requested."""
        return self.x == (0, -1) and self.y == (0, -1) and self.z == (0, -1)

    def clamp(self, shape: tuple[int, int, int]) -> "VOIRange":
        """Resolve sentinels and clamp every range to ``shape``."""
        nz, ny, nx = shape

        def _one(rng: tuple[int, int], n: int) -> tuple[int, int]:
            lo, hi = int(rng[0]), int(rng[1])
            if hi < 0:
                hi = n - 1
            lo = max(0, min(lo, n - 1))
            hi = max(lo, min(hi, n - 1))
            return lo, hi

        return VOIRange(x=_one(self.x, nx), y=_one(self.y, ny), z=_one(self.z, nz))

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        """``(z, y, x)`` slices for indexing a volume (ranges must be clamped)."""
        return (
            slice(self.z[0], self.z[1] + 1),
            slice(self.y[0], self.y[1] + 1),
            slice(self.x[0], self.x[1] + 1),
        )

    @property
    def extent(self) -> tuple[int, int, int]:
        """``(nz, ny, nx)`` size of the VOI (ranges must be clamped)."""
        return (
            self.z[1] - self.z[0] + 1,
            self.y[1] - self.y[0] + 1,
            self.x[1] - self.x[0] + 1,
        )


# ---------------------------------------------------------------------------
# Frame provider protocol
# ---------------------------------------------------------------------------


class VolumeProvider(Protocol):
    """Per-frame normalised-volume access used by :func:`run_aldvc`.

    Frames may be materialised eagerly (:class:`al_dvc.io.volume_ops.ListVolumeProvider`)
    or streamed from disk (:class:`al_dvc.io.volume_io.FileVolumeProvider`).
    """

    def __len__(self) -> int: ...

    @property
    def shape(self) -> tuple[int, int, int]: ...

    @property
    def clamped_voi(self) -> VOIRange: ...

    def get_normalized(self, idx: int) -> NDArray[np.float32]: ...

    def get_mask(self, idx: int) -> NDArray[np.bool_] | None: ...


# ---------------------------------------------------------------------------
# Frame schedule (ported from pyALDIC)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSchedule:
    """Reference-frame pairing for multi-frame tracking.

    ``ref_indices[i]`` is the 0-based reference frame for deformed frame
    ``i + 1``. The DAG constraint ``0 <= ref_indices[i] <= i`` guarantees no
    frame references a future frame.
    """

    ref_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for i, ref in enumerate(self.ref_indices):
            if not isinstance(ref, (int, np.integer)):
                raise TypeError(f"ref_indices[{i}] must be int (got {type(ref).__name__})")
            if ref < 0:
                raise ValueError(f"ref_indices[{i}]={ref} is negative")
            if ref > i:
                raise ValueError(f"ref_indices[{i}]={ref} references a future frame (must be <= {i})")

    @classmethod
    def from_mode(cls, mode: str, n_frames: int) -> "FrameSchedule":
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2 (got {n_frames})")
        n_pairs = n_frames - 1
        if mode == "accumulative":
            return cls(ref_indices=tuple(0 for _ in range(n_pairs)))
        if mode == "incremental":
            return cls(ref_indices=tuple(range(n_pairs)))
        raise ValueError(f"Unknown reference mode '{mode}'. Use 'accumulative' or 'incremental'.")

    @classmethod
    def from_every_n(cls, n: int, n_frames: int) -> "FrameSchedule":
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n})")
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2 (got {n_frames})")
        return cls(ref_indices=tuple((d - 1) // n * n for d in range(1, n_frames)))

    def parent(self, frame: int) -> int:
        if frame < 1 or frame > len(self.ref_indices):
            raise IndexError(f"frame={frame} out of range [1, {len(self.ref_indices)}]")
        return self.ref_indices[frame - 1]

    def path_to_root(self, frame: int) -> list[int]:
        path = [frame]
        current = frame
        while current > 0:
            current = self.parent(current)
            path.append(current)
        return path

    @property
    def ref_frame_set(self) -> set[int]:
        return set(self.ref_indices) | {0}

    def __len__(self) -> int:
        return len(self.ref_indices)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


@dataclass
class DVCMesh:
    """Uniform hexahedral (hex8) mesh on a regular node grid.

    Attributes:
        coordinates: ``(N, 3)`` node coordinates ``[x, y, z]`` in voxels.
        elements: ``(E, 8)`` hex8 connectivity (0-based). Rows of ``-1`` are
            dropped elements (kept so element indices stay stable).
        grid_shape: ``(nz, ny, nx)`` number of nodes along each axis.
        x0, y0, z0: 1-D node coordinate axes.
        spacing: ``(hx, hy, hz)`` node spacing in voxels.
        node_valid: ``(N,)`` bool, False for nodes outside the mask or with
            insufficient subset coverage.
        boundary_nodes: indices of nodes on the outer surface of the node grid
            or adjacent to invalid nodes (used to exclude edge effects from
            beta tuning and diagnostics).
    """

    coordinates: NDArray[np.float64]
    elements: NDArray[np.int64]
    grid_shape: tuple[int, int, int]
    x0: NDArray[np.float64]
    y0: NDArray[np.float64]
    z0: NDArray[np.float64]
    spacing: tuple[float, float, float]
    node_valid: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, dtype=bool))
    boundary_nodes: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def n_nodes(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def n_elements(self) -> int:
        return int(np.sum(self.elements[:, 0] >= 0)) if self.elements.size else 0

    def to_grid(self, values: NDArray) -> NDArray:
        """Reshape a per-node array ``(N, ...)`` to ``(nz, ny, nx, ...)``."""
        values = np.asarray(values)
        return values.reshape(self.grid_shape + values.shape[1:])

    def node_index(self, ix: int, iy: int, iz: int) -> int:
        nz, ny, nx = self.grid_shape
        return iz * ny * nx + iy * nx + ix


# ---------------------------------------------------------------------------
# Reference-frame bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceBundle:
    """Everything the local solver needs about a reference frame.

    Attributes:
        f: normalised reference volume ``(nz, ny, nx)`` float32.
        gx, gy, gz: gradients of ``f`` (same shape/dtype).
        mask: ``(nz, ny, nx)`` uint8, 1 = valid.
    """

    f: NDArray[np.float32]
    gx: NDArray[np.float32]
    gy: NDArray[np.float32]
    gz: NDArray[np.float32]
    mask: NDArray[np.uint8]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.f.shape)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalSolveInfo:
    """Diagnostics from one local IC-GN pass over all nodes."""

    n_iter: NDArray[np.int32]
    status: NDArray[np.int8]
    zncc: NDArray[np.float64]
    solve_time: float
    n_bad: int

    def summary(self) -> dict[str, float | int]:
        n = int(self.status.size)
        conv = int(np.sum(self.status == STATUS_CONVERGED))
        return {
            "n_nodes": n,
            "n_converged": conv,
            "frac_converged": conv / n if n else 0.0,
            "mean_iter": float(np.mean(self.n_iter[self.status == STATUS_CONVERGED])) if conv else float("nan"),
            "median_zncc": float(np.nanmedian(self.zncc)) if n else float("nan"),
            "n_bad": int(self.n_bad),
            "solve_time_s": float(self.solve_time),
        }


@dataclass(frozen=True)
class ADMMInfo:
    """Per-frame ADMM diagnostics."""

    beta: float
    mu: float
    n_steps: int
    update_global: tuple[float, ...]
    update_local: tuple[float, ...]
    primal_residual_u: tuple[float, ...]
    primal_residual_f: tuple[float, ...]
    local_info: tuple[LocalSolveInfo, ...]
    subpb2_time: float
    # L-curve sweep used to pick beta (keys: betas, err1, err2, score, k_best);
    # None when beta was given explicitly or reused from an earlier frame.
    beta_sweep: dict | None = None


@dataclass(frozen=True)
class FrameResult:
    """Displacement result for one frame pair.

    Attributes:
        U: ``(N, 3)`` displacement of the pair (reference -> deformed), voxels.
        F: ``(N, 3, 3)`` displacement gradient of the pair.
        U_accum: ``(N, 3)`` cumulative displacement from frame 0 (filled by
            the pipeline after all pairs are solved).
        U_local: ``(N, 3)`` local (subset-only) IC-GN result before ADMM.
        F_local: ``(N, 3, 3)`` local IC-GN gradient.
        U0: ``(N, 3)`` initial guess used.
        U_std: ``(N, 3)`` standard deviation of u, v, w (voxels) from the
            IC-GN normal equations at converged nodes, NaN elsewhere.
        zncc: ``(N,)`` final zero-normalised cross-correlation per node.
        status: ``(N,)`` node status codes of the final local pass.
        ref_frame: index of the reference frame of this pair.
        admm: ADMM diagnostics (None when the global step was disabled).
    """

    U: NDArray[np.float64]
    F: NDArray[np.float64]
    ref_frame: int = 0
    U_accum: NDArray[np.float64] | None = None
    U_local: NDArray[np.float64] | None = None
    F_local: NDArray[np.float64] | None = None
    U0: NDArray[np.float64] | None = None
    zncc: NDArray[np.float64] | None = None
    U_std: NDArray[np.float64] | None = None
    status: NDArray[np.int8] | None = None
    admm: ADMMInfo | None = None


@dataclass(frozen=True)
class StrainResult:
    """Strain of one frame (all arrays ``(N,)`` unless stated).

    Components follow ``e_ij`` with ``i, j in {x, y, z}``; displacements and
    gradients are in physical units when ``voxel_size != 1``.
    """

    disp_u: NDArray[np.float64]
    disp_v: NDArray[np.float64]
    disp_w: NDArray[np.float64]
    F: NDArray[np.float64]  # (N, 3, 3) displacement gradient (physical units)
    exx: NDArray[np.float64]
    eyy: NDArray[np.float64]
    ezz: NDArray[np.float64]
    exy: NDArray[np.float64]
    exz: NDArray[np.float64]
    eyz: NDArray[np.float64]
    principal: NDArray[np.float64]  # (N, 3) sorted descending e1 >= e2 >= e3
    max_shear: NDArray[np.float64]
    von_mises: NDArray[np.float64]
    volumetric: NDArray[np.float64]
    det_F: NDArray[np.float64]
    rotation_deg: NDArray[np.float64]  # rotation angle from polar decomposition
    strain_valid: NDArray[np.bool_]
    strain_type: str
    method: str

    _FIELD_ALIASES = {
        "strain_exx": "exx",
        "strain_eyy": "eyy",
        "strain_ezz": "ezz",
        "strain_exy": "exy",
        "strain_exz": "exz",
        "strain_eyz": "eyz",
        "strain_von_mises": "von_mises",
        "strain_max_shear": "max_shear",
        "strain_volumetric": "volumetric",
        "strain_rotation": "rotation_deg",
        "disp_x": "disp_u",
        "disp_y": "disp_v",
        "disp_z": "disp_w",
    }

    def field(self, name: str, trimmed: bool = True) -> NDArray[np.float64]:
        """Return a named field; ``trimmed`` NaNs out low-confidence nodes."""
        key = self._FIELD_ALIASES.get(name, name)
        if key == "disp_magnitude":
            vals = np.sqrt(self.disp_u**2 + self.disp_v**2 + self.disp_w**2)
        elif key in ("e1", "e2", "e3"):
            vals = self.principal[:, int(key[1]) - 1]
        else:
            vals = getattr(self, key)
        vals = np.asarray(vals, dtype=np.float64).copy()
        if trimmed and self.strain_valid is not None and vals.shape[0] == self.strain_valid.shape[0]:
            if key not in ("disp_u", "disp_v", "disp_w", "disp_magnitude"):
                vals[~self.strain_valid] = np.nan
        return vals


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated output of :func:`run_aldvc`."""

    dvc_para: "object"  # DVCPara (avoid import cycle)
    dvc_mesh: DVCMesh
    result_disp: list[FrameResult]
    result_strain: list[StrainResult]
    frame_schedule: FrameSchedule
    volume_shape: tuple[int, int, int]
    timings: dict[str, float] = field(default_factory=dict)
    stopped_early: bool = False
    stopped_at_frame: int | None = None
    stop_reason: str = ""

    @property
    def n_frames(self) -> int:
        return len(self.result_disp)


# ---------------------------------------------------------------------------
# Small helpers for the layout contracts
# ---------------------------------------------------------------------------


def P_from_UF(U: NDArray[np.float64], F: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pack ``(N, 3)`` U and ``(N, 3, 3)`` F into ``(N, 12)`` warp parameters."""
    n = U.shape[0]
    P = np.empty((n, 12), dtype=np.float64)
    P[:, 0:9] = F.reshape(n, 9)
    P[:, 9:12] = U
    return P


def UF_from_P(P: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unpack ``(N, 12)`` warp parameters into ``(N, 3)`` U and ``(N, 3, 3)`` F."""
    n = P.shape[0]
    return P[:, 9:12].copy(), P[:, 0:9].reshape(n, 3, 3).copy()


def F_to_matlab_order(F: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(N, 3, 3)`` row-major F -> ``(9 N,)`` MATLAB column-major interleave.

    MATLAB ALDVC stores per node ``[F11, F21, F31, F12, F22, F32, F13, F23, F33]``
    where ``Fij = du_i/dx_j``.
    """
    return np.transpose(F, (0, 2, 1)).reshape(-1)
