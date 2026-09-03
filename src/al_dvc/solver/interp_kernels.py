"""Numba sub-voxel interpolation kernels for ``(nz, ny, nx)`` volumes.

Three schemes, selected by an integer ``mode`` so the IC-GN kernels can
branch without function-pointer arguments:

    INTERP_LINEAR  = 0  trilinear, 8 taps
    INTERP_CUBIC   = 1  Keys cubic convolution (a = -0.5, Catmull-Rom), 64 taps
                        -- what MATLAB ``ba_interp3(..., 'cubic')`` computes
    INTERP_BSPLINE = 2  cubic B-spline basis on *pre-filtered* coefficients,
                        64 taps -- equals ``scipy.ndimage.map_coordinates(order=3)``

Valid sampling domain for every mode: ``1 <= x <= n - 2`` on each axis
(callers check ``interp_margin_ok`` before sampling). Out-of-domain queries
are never clamped silently: the IC-GN kernels treat them as a failure.
"""

from __future__ import annotations

import numpy as np

from .._numba_compat import JIT_CACHE, njit

INTERP_LINEAR = 0
INTERP_CUBIC = 1
INTERP_BSPLINE = 2

INTERP_MODE_BY_NAME = {"linear": INTERP_LINEAR, "cubic": INTERP_CUBIC, "bspline": INTERP_BSPLINE}

# Samples must satisfy lo <= coord <= n - 1 - hi_margin on each axis.
SAMPLE_LO = 1.0
SAMPLE_HI_MARGIN = 2.0


@njit(cache=JIT_CACHE, inline="always")
def _keys_weights(t):
    """Catmull-Rom (Keys, a=-0.5) weights for taps -1, 0, 1, 2 as four scalars."""
    w0 = ((-0.5 * t + 1.0) * t - 0.5) * t
    w1 = (1.5 * t - 2.5) * t * t + 1.0
    w2 = ((-1.5 * t + 2.0) * t + 0.5) * t
    w3 = (0.5 * t - 0.5) * t * t
    return w0, w1, w2, w3


@njit(cache=JIT_CACHE, inline="always")
def _bspline_weights(t):
    """Cubic B-spline basis weights for taps -1, 0, 1, 2 as four scalars."""
    t2 = t * t
    t3 = t2 * t
    w0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0
    w1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
    w2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0
    w3 = t3 / 6.0
    return w0, w1, w2, w3


@njit(cache=JIT_CACHE, inline="always")
def _row4(vol, zz, yy, xb, wx0, wx1, wx2, wx3):
    return vol[zz, yy, xb] * wx0 + vol[zz, yy, xb + 1] * wx1 + vol[zz, yy, xb + 2] * wx2 + vol[zz, yy, xb + 3] * wx3


@njit(cache=JIT_CACHE, inline="always")
def _plane4(vol, zz, yb, xb, wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3):
    return (
        _row4(vol, zz, yb, xb, wx0, wx1, wx2, wx3) * wy0
        + _row4(vol, zz, yb + 1, xb, wx0, wx1, wx2, wx3) * wy1
        + _row4(vol, zz, yb + 2, xb, wx0, wx1, wx2, wx3) * wy2
        + _row4(vol, zz, yb + 3, xb, wx0, wx1, wx2, wx3) * wy3
    )


@njit(cache=JIT_CACHE, inline="always")
def sample_volume(vol, z, y, x, mode):
    """Interpolate ``vol`` at real coordinate ``(z, y, x)``.

    The caller guarantees ``1 <= c <= n-2`` on every axis (see
    :func:`interp_margin_ok`); this function does not re-check. The 4-tap
    weights are scalars on purpose: a ``np.empty(4)`` per axis and sample
    costs a heap allocation each, which made the sampler 2.6x slower.
    """
    nz = vol.shape[0]
    ny = vol.shape[1]
    nx = vol.shape[2]

    ix = int(np.floor(x))
    iy = int(np.floor(y))
    iz = int(np.floor(z))
    # keep the 4-tap stencil inside the volume when the coordinate is exactly
    # on the last admissible integer
    if ix > nx - 3:
        ix = nx - 3
    if iy > ny - 3:
        iy = ny - 3
    if iz > nz - 3:
        iz = nz - 3
    fx = x - ix
    fy = y - iy
    fz = z - iz

    if mode == INTERP_LINEAR:
        c00 = vol[iz, iy, ix] * (1.0 - fx) + vol[iz, iy, ix + 1] * fx
        c10 = vol[iz, iy + 1, ix] * (1.0 - fx) + vol[iz, iy + 1, ix + 1] * fx
        c01 = vol[iz + 1, iy, ix] * (1.0 - fx) + vol[iz + 1, iy, ix + 1] * fx
        c11 = vol[iz + 1, iy + 1, ix] * (1.0 - fx) + vol[iz + 1, iy + 1, ix + 1] * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        return c0 * (1.0 - fz) + c1 * fz

    if mode == INTERP_CUBIC:
        wx0, wx1, wx2, wx3 = _keys_weights(fx)
        wy0, wy1, wy2, wy3 = _keys_weights(fy)
        wz0, wz1, wz2, wz3 = _keys_weights(fz)
    else:
        wx0, wx1, wx2, wx3 = _bspline_weights(fx)
        wy0, wy1, wy2, wy3 = _bspline_weights(fy)
        wz0, wz1, wz2, wz3 = _bspline_weights(fz)
    xb = ix - 1
    yb = iy - 1
    zb = iz - 1
    return (
        _plane4(vol, zb, yb, xb, wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3) * wz0
        + _plane4(vol, zb + 1, yb, xb, wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3) * wz1
        + _plane4(vol, zb + 2, yb, xb, wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3) * wz2
        + _plane4(vol, zb + 3, yb, xb, wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3) * wz3
    )


@njit(cache=JIT_CACHE, inline="always")
def interp_margin_ok(z, y, x, nz, ny, nx):
    """True when ``(z, y, x)`` lies inside the admissible sampling domain."""
    if x < SAMPLE_LO or x > nx - 1 - SAMPLE_HI_MARGIN:
        return False
    if y < SAMPLE_LO or y > ny - 1 - SAMPLE_HI_MARGIN:
        return False
    if z < SAMPLE_LO or z > nz - 1 - SAMPLE_HI_MARGIN:
        return False
    return True


@njit(cache=JIT_CACHE)
def sample_points(vol, zs, ys, xs, mode):
    """Vectorised sampler for tests/utilities: returns NaN outside the domain."""
    n = zs.shape[0]
    out = np.empty(n)
    nz = vol.shape[0]
    ny = vol.shape[1]
    nx = vol.shape[2]
    for i in range(n):
        if interp_margin_ok(zs[i], ys[i], xs[i], nz, ny, nx):
            out[i] = sample_volume(vol, zs[i], ys[i], xs[i], mode)
        else:
            out[i] = np.nan
    return out
