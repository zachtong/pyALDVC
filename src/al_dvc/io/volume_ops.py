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

from .._numba_compat import HAS_NUMBA, JIT_CACHE, njit, prange
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


def _voi_bounds(shape: tuple[int, int, int], voi: VOIRange | None) -> tuple[int, int, int, int, int, int]:
    """``(z0, z1, y0, y1, x0, x1)`` half-open index bounds of the clamped VOI."""
    if voi is None:
        return 0, shape[0], 0, shape[1], 0, shape[2]
    sz, sy, sx = voi.clamp(shape).slices
    z0, z1, _ = sz.indices(shape[0])
    y0, y1, _ = sy.indices(shape[1])
    x0, x1, _ = sx.indices(shape[2])
    return z0, z1, y0, y1, x0, x1


@njit(parallel=True, cache=JIT_CACHE)
def _voi_moments(arr, z0, z1, y0, y1, x0, x1, pilot):
    """Sum and sum of squares of ``arr - pilot`` over the VOI (float64, per-slice partial sums)."""
    nz = z1 - z0
    s = np.zeros(nz, dtype=np.float64)
    ss = np.zeros(nz, dtype=np.float64)
    for iz in prange(nz):
        acc = 0.0
        acc2 = 0.0
        for y in range(y0, y1):
            for x in range(x0, x1):
                v = float(arr[z0 + iz, y, x]) - pilot
                acc += v
                acc2 += v * v
        s[iz] = acc
        ss[iz] = acc2
    return s.sum(), ss.sum()


@njit(parallel=True, cache=JIT_CACHE)
def _normalize_into(arr, mean, inv_std, out):
    nz, ny, nx = arr.shape
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                out[z, y, x] = np.float32((float(arr[z, y, x]) - mean) * inv_std)


def voi_mean_std(vol: NDArray, voi: VOIRange | None = None) -> tuple[float, float]:
    """Mean and (population) standard deviation of ``vol`` inside the clamped VOI, in float64."""
    arr = np.asarray(vol)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {arr.shape}.")
    z0, z1, y0, y1, x0, x1 = _voi_bounds(arr.shape, voi)
    n = (z1 - z0) * (y1 - y0) * (x1 - x0)
    if n <= 0:
        raise ValueError("VOI is empty after clamping to the volume.")
    if HAS_NUMBA and arr.dtype.kind in "iuf" and arr.dtype.itemsize >= 2:
        arr_c = np.ascontiguousarray(arr)
        pilot = float(arr_c[(z0 + z1) // 2, (y0 + y1) // 2, (x0 + x1) // 2])
        s, ss = _voi_moments(arr_c, z0, z1, y0, y1, x0, x1, pilot)
        mean_shift = s / n
        var = max(ss / n - mean_shift * mean_shift, 0.0)
        mean, std = pilot + mean_shift, float(np.sqrt(var))
    else:
        patch = arr[z0:z1, y0:y1, x0:x1]
        mean = float(np.mean(patch, dtype=np.float64))
        std = float(np.std(patch, dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Volume contains NaN/Inf inside the VOI; cannot normalise.")
    return mean, std


def normalize_volume(vol: NDArray, voi: VOIRange | None = None) -> NDArray[np.float32]:
    """``(vol - mean_voi) / std_voi`` computed over the (clamped) VOI.

    Statistics are accumulated in float64; the result is float32. Uses the
    Numba kernels (parallel, single pass) when available.
    """
    arr = np.asarray(vol)
    mean, std = voi_mean_std(arr, voi)
    if std < 1e-12:
        std = 1.0
    out = np.empty(arr.shape, dtype=np.float32)
    if HAS_NUMBA and arr.dtype.kind in "iuf" and arr.dtype.itemsize >= 2:
        _normalize_into(np.ascontiguousarray(arr), float(mean), 1.0 / float(std), out)
    else:
        np.subtract(arr, mean, out=out, dtype=np.float32, casting="unsafe")
        out /= np.float32(std)
    return out


@njit(parallel=True, cache=JIT_CACHE)
def _gradient_stencil7(f, gx, gy, gz, border):
    """7-point central differences into ``gx, gy, gz``; zero within ``border`` of any face."""
    nz, ny, nx = f.shape
    c1 = 3.0 / 4.0
    c2 = -3.0 / 20.0
    c3 = 1.0 / 60.0
    for z in prange(nz):
        z_in = z >= border and z < nz - border
        for y in range(ny):
            y_in = y >= border and y < ny - border
            for x in range(nx):
                if z_in and y_in and x >= border and x < nx - border:
                    gx[z, y, x] = np.float32(
                        c1 * (float(f[z, y, x + 1]) - float(f[z, y, x - 1]))
                        + c2 * (float(f[z, y, x + 2]) - float(f[z, y, x - 2]))
                        + c3 * (float(f[z, y, x + 3]) - float(f[z, y, x - 3]))
                    )
                    gy[z, y, x] = np.float32(
                        c1 * (float(f[z, y + 1, x]) - float(f[z, y - 1, x]))
                        + c2 * (float(f[z, y + 2, x]) - float(f[z, y - 2, x]))
                        + c3 * (float(f[z, y + 3, x]) - float(f[z, y - 3, x]))
                    )
                    gz[z, y, x] = np.float32(
                        c1 * (float(f[z + 1, y, x]) - float(f[z - 1, y, x]))
                        + c2 * (float(f[z + 2, y, x]) - float(f[z - 2, y, x]))
                        + c3 * (float(f[z + 3, y, x]) - float(f[z - 3, y, x]))
                    )
                else:
                    gx[z, y, x] = 0.0
                    gy[z, y, x] = 0.0
                    gz[z, y, x] = 0.0


def compute_gradients_np(
    f: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """NumPy/SciPy reference implementation of :func:`compute_gradients`."""
    f32 = np.asarray(f, dtype=np.float32)
    gx = correlate1d(f32, STENCIL7, axis=2, mode="nearest", output=np.float32)
    gy = correlate1d(f32, STENCIL7, axis=1, mode="nearest", output=np.float32)
    gz = correlate1d(f32, STENCIL7, axis=0, mode="nearest", output=np.float32)
    b = GRADIENT_BORDER
    for g in (gx, gy, gz):
        g[:b, :, :] = 0.0
        g[-b:, :, :] = 0.0
        g[:, :b, :] = 0.0
        g[:, -b:, :] = 0.0
        g[:, :, :b] = 0.0
        g[:, :, -b:] = 0.0
    return gx, gy, gz


def compute_gradients(
    f: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """7-point central-difference gradients of ``f`` along x, y, z.

    The 3-voxel border, where the stencil would need out-of-volume samples,
    is set to zero (MATLAB crops it; we keep full-size arrays so ``gx[z,y,x]``
    is the gradient at voxel ``(z, y, x)`` with no offset arithmetic).
    Arithmetic is float64, storage float32; parallel Numba kernel when
    available, :func:`compute_gradients_np` otherwise.
    """
    f32 = np.ascontiguousarray(f, dtype=np.float32)
    if f32.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {f32.shape}.")
    if not HAS_NUMBA or min(f32.shape) <= 2 * GRADIENT_BORDER:
        return compute_gradients_np(f32)
    gx = np.empty(f32.shape, dtype=np.float32)
    gy = np.empty(f32.shape, dtype=np.float32)
    gz = np.empty(f32.shape, dtype=np.float32)
    _gradient_stencil7(f32, gx, gy, gz, GRADIENT_BORDER)
    return gx, gy, gz


def presmooth_volume(vol: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    """Gaussian pre-smoothing (``sigma`` in voxels); identity when ``sigma <= 0``.

    Reduces the noise amplification of the 7-point gradient stencil on
    low-SNR volumes at the cost of some spatial resolution. Applied to both
    the reference and the deformed volume so the correlation stays unbiased.
    """
    if sigma is None or sigma <= 0:
        return np.ascontiguousarray(vol, dtype=np.float32)
    return np.ascontiguousarray(
        gaussian_filter(np.asarray(vol, dtype=np.float32), sigma=float(sigma), mode="nearest", output=np.float32),
        dtype=np.float32,
    )


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
