"""Volume normalisation, gradients and eager frame providers.

Port of MATLAB ``funNormalizeImg3.m`` and ``funImgGradient3.m`` (``stencil7``).

Conventions:
    - Volumes are ``(nz, ny, nx)`` arrays; gradient ``gx`` is along the last
      axis (x), ``gy`` along the middle axis, ``gz`` along the first axis.
    - Normalised volumes are stored as float32 (memory), all downstream
      arithmetic is float64.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import correlate1d, gaussian_filter, spline_filter

from ..core.data_structures import ReferenceBundle, VOIRange

# 7-point central finite-difference kernel (MATLAB funImgGradient3 'stencil7')
STENCIL7 = np.array([-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60], dtype=np.float64)
GRADIENT_BORDER = 3  # voxels on each side where the stencil is unreliable


def as_float32_volume(vol: NDArray) -> NDArray[np.float32]:
    """Cast any 3-D array to a C-contiguous float32 volume (no rescaling)."""
    arr = np.asarray(vol)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {arr.shape}.")
    return np.ascontiguousarray(arr, dtype=np.float32)


def compute_clamped_voi(shape: tuple[int, int, int], voi: VOIRange) -> VOIRange:
    return voi.clamp(shape)


def normalize_volume(vol: NDArray, voi: VOIRange | None = None) -> NDArray[np.float32]:
    """``(vol - mean_voi) / std_voi`` computed over the (clamped) VOI.

    Statistics are accumulated in float64; the result is float32.
    """
    arr = np.asarray(vol)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {arr.shape}.")
    if voi is None:
        patch = arr
    else:
        patch = arr[voi.clamp(arr.shape).slices]
    mean = float(np.mean(patch, dtype=np.float64))
    std = float(np.std(patch, dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Volume contains NaN/Inf inside the VOI; cannot normalise.")
    if std < 1e-12:
        std = 1.0
    out = np.empty(arr.shape, dtype=np.float32)
    np.subtract(arr, mean, out=out, dtype=np.float32, casting="unsafe")
    out /= np.float32(std)
    return out


def compute_gradients(
    f: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """7-point central-difference gradients of ``f`` along x, y, z.

    The 3-voxel border, where the stencil would need out-of-volume samples,
    is set to zero (MATLAB crops it; we keep full-size arrays so ``gx[z,y,x]``
    is the gradient at voxel ``(z, y, x)`` with no offset arithmetic).
    """
    f64 = np.asarray(f, dtype=np.float32)
    gx = correlate1d(f64, STENCIL7, axis=2, mode="nearest", output=np.float32)
    gy = correlate1d(f64, STENCIL7, axis=1, mode="nearest", output=np.float32)
    gz = correlate1d(f64, STENCIL7, axis=0, mode="nearest", output=np.float32)
    b = GRADIENT_BORDER
    for g in (gx, gy, gz):
        g[:b, :, :] = 0.0
        g[-b:, :, :] = 0.0
        g[:, :b, :] = 0.0
        g[:, -b:, :] = 0.0
        g[:, :, :b] = 0.0
        g[:, :, -b:] = 0.0
    return gx, gy, gz


def presmooth_volume(vol: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    """Gaussian pre-smoothing (``sigma`` in voxels); identity when ``sigma <= 0``.

    Reduces the noise amplification of the 7-point gradient stencil on
    low-SNR volumes at the cost of some spatial resolution. Applied to both
    the reference and the deformed volume so the correlation stays unbiased.
    """
    if sigma is None or sigma <= 0:
        return np.ascontiguousarray(vol, dtype=np.float32)
    return np.ascontiguousarray(gaussian_filter(np.asarray(vol, dtype=np.float32), sigma=float(sigma), mode="nearest",
                                                output=np.float32), dtype=np.float32)


def prefilter_bspline(vol: NDArray[np.float32]) -> NDArray[np.float32]:
    """Cubic B-spline coefficients of ``vol`` (for ``interp_method='bspline'``).

    Sampling these coefficients with the cubic B-spline basis reproduces
    ``scipy.ndimage.map_coordinates(vol, ..., order=3, mode='mirror')``.
    """
    coeffs = spline_filter(np.asarray(vol, dtype=np.float32), order=3, output=np.float32, mode="mirror")
    return np.ascontiguousarray(coeffs, dtype=np.float32)


def prepare_deformed(vol: NDArray[np.float32], interp_method: str) -> NDArray[np.float32]:
    """Return the array the interpolation kernel samples for ``interp_method``."""
    if interp_method == "bspline":
        return prefilter_bspline(vol)
    return np.ascontiguousarray(vol, dtype=np.float32)


def build_reference_bundle(
    f: NDArray[np.float32],
    mask: NDArray[np.bool_] | None,
) -> ReferenceBundle:
    """Gradients + mask for a normalised reference volume."""
    f = np.ascontiguousarray(f, dtype=np.float32)
    gx, gy, gz = compute_gradients(f)
    if mask is None:
        m = np.ones(f.shape, dtype=np.uint8)
    else:
        if mask.shape != f.shape:
            raise ValueError(f"mask shape {mask.shape} != volume shape {f.shape}")
        m = np.ascontiguousarray(mask, dtype=np.uint8)
    return ReferenceBundle(f=f, gx=gx, gy=gy, gz=gz, mask=m)


class ListVolumeProvider:
    """Eager provider over in-memory volumes (normalised once, up front)."""

    def __init__(
        self,
        volumes: list[NDArray],
        voi: VOIRange | None = None,
        masks: list[NDArray[np.bool_] | None] | None = None,
    ) -> None:
        if len(volumes) == 0:
            raise ValueError("volumes list is empty")
        shape = tuple(int(s) for s in np.asarray(volumes[0]).shape)
        if len(shape) != 3:
            raise ValueError(f"volumes must be 3-D, got shape {shape}")
        for i, v in enumerate(volumes):
            if tuple(np.asarray(v).shape) != shape:
                raise ValueError(f"volume {i} has shape {np.asarray(v).shape}, expected {shape}")
        self._shape: tuple[int, int, int] = shape  # type: ignore[assignment]
        self._voi = (voi or VOIRange()).clamp(self._shape)
        self._normalized = [normalize_volume(v, self._voi) for v in volumes]
        if masks is None:
            self._masks: list[NDArray[np.bool_] | None] = [None] * len(volumes)
        else:
            if len(masks) != len(volumes):
                raise ValueError("masks must have the same length as volumes")
            self._masks = []
            for i, m in enumerate(masks):
                if m is None:
                    self._masks.append(None)
                    continue
                m = np.asarray(m)
                if m.shape != self._shape:
                    raise ValueError(f"mask {i} has shape {m.shape}, expected {self._shape}")
                self._masks.append(m.astype(bool))

    def __len__(self) -> int:
        return len(self._normalized)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def clamped_voi(self) -> VOIRange:
        return self._voi

    def get_normalized(self, idx: int) -> NDArray[np.float32]:
        return self._normalized[idx]

    def get_mask(self, idx: int) -> NDArray[np.bool_] | None:
        return self._masks[idx]
