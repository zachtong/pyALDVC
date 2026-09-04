"""Input validation at the pipeline boundary."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.config import DVCPara
from ..core.data_structures import VOIRange


def validate_volume_list(volumes: list[NDArray]) -> tuple[int, int, int]:
    """Check a list of raw volumes and return the common ``(nz, ny, nx)`` shape."""
    if not isinstance(volumes, (list, tuple)) or len(volumes) < 2:
        raise ValueError("At least 2 volumes (reference + deformed) are required.")
    shape = None
    for i, v in enumerate(volumes):
        arr = np.asarray(v)
        if arr.ndim != 3:
            raise ValueError(f"volume {i} must be 3-D (got shape {arr.shape}).")
        if arr.size == 0:
            raise ValueError(f"volume {i} is empty.")
        if shape is None:
            shape = tuple(int(s) for s in arr.shape)
        elif tuple(arr.shape) != shape:
            raise ValueError(f"volume {i} has shape {arr.shape}, expected {shape}.")
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ValueError(f"volume {i} contains NaN or Inf.")
    return shape  # type: ignore[return-value]


def validate_para_against_volume(para: DVCPara, shape: tuple[int, int, int]) -> None:
    """Raise when subset/step cannot fit inside the (clamped) VOI of ``shape`` (``voi=None``: the whole volume)."""
    voi = (para.voi if para.voi is not None else VOIRange()).clamp(shape)
    ext = voi.extent  # (nz, ny, nx)
    for name, w, e in zip("xyz", para.winsize, ext[::-1]):
        if w + 2 * 5 + 1 > e:
            raise ValueError(
                f"winsize[{name}]={w} is too large for the VOI extent {e} along {name} "
                f"(need at least {w + 11} voxels including margins)."
            )
    if min(shape) < 16:
        raise ValueError(f"Volumes must be at least 16 voxels along every axis (got {shape}).")


def validate_mask(mask: NDArray | None, shape: tuple[int, int, int], idx: int) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.shape != tuple(shape):
        raise ValueError(f"mask {idx} has shape {m.shape}, expected {tuple(shape)}.")
    m = m.astype(bool)
    if not m.any():
        raise ValueError(f"mask {idx} has no valid voxels.")
    return m
