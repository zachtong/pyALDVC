"""Initial displacement guess by normalised cross-correlation (Section 3).

Replaces MATLAB ``funIntegerSearch3.m`` / ``funIntegerSearch3Multigrid.m``:

* :func:`phase_correlation_shift` -- global rigid pre-shift on a downsampled
  VOI (the "level 1" of the MATLAB multigrid search).
* :func:`ncc_search` -- per-node FFT normalised cross-correlation of the
  reference subset against a search window of the deformed volume,
  27-point quadratic sub-voxel refinement, PCE quality factor and a
  "peak clipped at the search boundary" flag.
* :func:`pyramid_search` -- coarse-to-fine wrapper (block-mean pyramid)
  that handles displacements far larger than the search radius.

Everything is vectorised over chunks of nodes with ``scipy.fft``.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

from .._numba_compat import HAS_NUMBA, JIT_CACHE, njit, prange
from ..core.data_structures import VOIRange
from ..utils.inpaint import fill_nan_grid
from ..utils.outlier_detection import universal_median_test

logger = logging.getLogger(__name__)

PYRAMID_FINE_RADIUS = 4
MIN_PYRAMID_SUBSET = 8
DEFAULT_INIT_SUBSET = 16  # NCC template size used when init_subset is None (capped at winsize)
CLIPPED_EXPAND_FRACTION = 0.05  # expand the search radius when more than this fraction of peaks is clipped


DIRECT_NCC_MAX_OPS = 3.0e7  # offsets x template voxels above which the FFT path is cheaper


@njit(parallel=True, cache=JIT_CACHE)
def _zncc_direct(f, g, coords, hx, hy, hz, rx, ry, rz, wx0, wy0, wz0, t_ok, out):
    """Spatial-domain ZNCC maps ``out[n, dz, dy, dx]`` for a batch of nodes.

    Window sums come from a per-node summed-area table, so each offset costs
    one multiply-add per template voxel (the cross term only). Nodes with
    ``t_ok == False`` or a flat template get a map of ``-2``.
    """
    N = coords.shape[0]
    sx = 2 * hx + 1
    sy = 2 * hy + 1
    sz = 2 * hz + 1
    vx = 2 * rx + 1
    vy = 2 * ry + 1
    vz = 2 * rz + 1
    wx = sx + 2 * rx
    wy = sy + 2 * ry
    wz = sz + 2 * rz
    n_tpl = float(sx * sy * sz)
    for n in prange(N):
        if not t_ok[n]:
            for a in range(vz):
                for b in range(vy):
                    for c in range(vx):
                        out[n, a, b, c] = -2.0
            continue
        x0 = coords[n, 0]
        y0 = coords[n, 1]
        z0 = coords[n, 2]
        # centred template
        T = np.empty((sz, sy, sx))
        sf = 0.0
        for k in range(sz):
            for j in range(sy):
                for i in range(sx):
                    v = float(f[z0 - hz + k, y0 - hy + j, x0 - hx + i])
                    T[k, j, i] = v
                    sf += v
        mf = sf / n_tpl
        ssf = 0.0
        for k in range(sz):
            for j in range(sy):
                for i in range(sx):
                    T[k, j, i] -= mf
                    ssf += T[k, j, i] * T[k, j, i]
        if ssf < 1e-20:
            for a in range(vz):
                for b in range(vy):
                    for c in range(vx):
                        out[n, a, b, c] = -2.0
            continue
        # summed-area tables of the window and its square
        S1 = np.zeros((wz + 1, wy + 1, wx + 1))
        S2 = np.zeros((wz + 1, wy + 1, wx + 1))
        zb = wz0[n]
        yb = wy0[n]
        xb = wx0[n]
        for z in range(wz):
            for y in range(wy):
                r1 = 0.0
                r2 = 0.0
                for x in range(wx):
                    v = float(g[zb + z, yb + y, xb + x])
                    r1 += v
                    r2 += v * v
                    S1[z + 1, y + 1, x + 1] = S1[z, y + 1, x + 1] + S1[z + 1, y, x + 1] - S1[z, y, x + 1] + r1
                    S2[z + 1, y + 1, x + 1] = S2[z, y + 1, x + 1] + S2[z + 1, y, x + 1] - S2[z, y, x + 1] + r2
        for dz in range(vz):
            for dy in range(vy):
                for dx in range(vx):
                    z1 = dz + sz
                    y1 = dy + sy
                    x1 = dx + sx
                    sg = (
                        S1[z1, y1, x1]
                        - S1[dz, y1, x1]
                        - S1[z1, dy, x1]
                        - S1[z1, y1, dx]
                        + S1[dz, dy, x1]
                        + S1[dz, y1, dx]
                        + S1[z1, dy, dx]
                        - S1[dz, dy, dx]
                    )
                    sgg = (
                        S2[z1, y1, x1]
                        - S2[dz, y1, x1]
                        - S2[z1, dy, x1]
                        - S2[z1, y1, dx]
                        + S2[dz, dy, x1]
                        + S2[dz, y1, dx]
                        + S2[z1, dy, dx]
                        - S2[dz, dy, dx]
                    )
                    var = sgg - sg * sg / n_tpl
                    if var < 1e-20:
                        out[n, dz, dy, dx] = 0.0
                        continue
                    sfg = 0.0
                    for k in range(sz):
                        for j in range(sy):
                            for i in range(sx):
                                sfg += T[k, j, i] * float(g[zb + dz + k, yb + dy + j, xb + dx + i])
                    val = sfg / np.sqrt(var * ssf)
                    if val > 1.0:
                        val = 1.0
                    elif val < -1.0:
                        val = -1.0
                    out[n, dz, dy, dx] = val


@njit(parallel=True, cache=JIT_CACHE)
def _window_box_stats(W, sz, sy, sx, out_sum, out_sq):
    """Per-node sums of ``W`` and ``W^2`` over every template-sized box.

    ``W`` is ``(n, wz, wy, wx)``; outputs are ``(n, wz-sz+1, wy-sy+1, wx-sx+1)``.
    Uses a per-node summed-area table (float64) so only the valid positions
    are touched by the caller.
    """
    n = W.shape[0]
    wz = W.shape[1]
    wy = W.shape[2]
    wx = W.shape[3]
    vz = wz - sz + 1
    vy = wy - sy + 1
    vx = wx - sx + 1
    for k in prange(n):
        S1 = np.zeros((wz + 1, wy + 1, wx + 1))
        S2 = np.zeros((wz + 1, wy + 1, wx + 1))
        for z in range(wz):
            for y in range(wy):
                r1 = 0.0
                r2 = 0.0
                for x in range(wx):
                    v = float(W[k, z, y, x])
                    r1 += v
                    r2 += v * v
                    S1[z + 1, y + 1, x + 1] = S1[z, y + 1, x + 1] + S1[z + 1, y, x + 1] - S1[z, y, x + 1] + r1
                    S2[z + 1, y + 1, x + 1] = S2[z, y + 1, x + 1] + S2[z + 1, y, x + 1] - S2[z, y, x + 1] + r2
        for z in range(vz):
            for y in range(vy):
                for x in range(vx):
                    z1 = z + sz
                    y1 = y + sy
                    x1 = x + sx
                    out_sum[k, z, y, x] = (
                        S1[z1, y1, x1]
                        - S1[z, y1, x1]
                        - S1[z1, y, x1]
                        - S1[z1, y1, x]
                        + S1[z, y, x1]
                        + S1[z, y1, x]
                        + S1[z1, y, x]
                        - S1[z, y, x]
                    )
                    out_sq[k, z, y, x] = (
                        S2[z1, y1, x1]
                        - S2[z, y1, x1]
                        - S2[z1, y, x1]
                        - S2[z1, y1, x]
                        + S2[z, y, x1]
                        + S2[z, y1, x]
                        + S2[z1, y, x]
                        - S2[z, y, x]
                    )


# 27-point quadratic design matrix and its pseudo-inverse (for sub-voxel peaks)
_dz, _dy, _dx = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
_dx, _dy, _dz = _dx.ravel().astype(float), _dy.ravel().astype(float), _dz.ravel().astype(float)
_DESIGN = np.column_stack(
    [
        np.ones(27),
        _dx,
        _dy,
        _dz,
        _dx * _dy,
        _dx * _dz,
        _dy * _dz,
        _dx**2,
        _dy**2,
        _dz**2,
    ]
)
_PINV = np.linalg.pinv(_DESIGN)  # (10, 27)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def block_downsample(vol: NDArray, factor: int) -> NDArray[np.float32]:
    """Block-mean downsampling by an integer ``factor`` (trailing voxels dropped)."""
    if factor <= 1:
        return np.ascontiguousarray(vol, dtype=np.float32)
    v = np.asarray(vol, dtype=np.float32)
    nz, ny, nx = (s // factor * factor for s in v.shape)
    v = v[:nz, :ny, :nx]
    out = v.reshape(nz // factor, factor, ny // factor, factor, nx // factor, factor).mean(axis=(1, 3, 5), dtype=np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def phase_correlation_shift(
    f: NDArray,
    g: NDArray,
    voi: VOIRange | None = None,
    max_dim: int = 128,
) -> NDArray[np.float64]:
    """Global integer shift ``(dx, dy, dz)`` of ``g`` relative to ``f`` (voxels).

    The VOI is block-downsampled so that no axis exceeds ``max_dim``; the
    shift is recovered from the phase-correlation peak and scaled back.
    """
    f = np.asarray(f, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    if voi is not None:
        sl = voi.clamp(f.shape).slices
        f = f[sl]
        g = g[sl]
    factor = max(1, int(math.ceil(max(f.shape) / max_dim)))
    fs = block_downsample(f, factor)
    gs = block_downsample(g, factor)
    fs = fs - fs.mean()
    gs = gs - gs.mean()
    # Hann window: the two volumes share identical non-periodic borders, which
    # otherwise produce a spurious zero-shift peak after spectral whitening.
    win = np.ones(fs.shape, dtype=np.float32)
    for ax, n in enumerate(fs.shape):
        w1 = np.hanning(n).astype(np.float32)
        shape = [1, 1, 1]
        shape[ax] = n
        win = win * w1.reshape(shape)
    fs = fs * win
    gs = gs * win
    Ff = sfft.fftn(fs, workers=-1)
    Fg = sfft.fftn(gs, workers=-1)
    R = Fg * np.conj(Ff)
    mag = np.abs(R)
    R = R / np.where(mag > 1e-12, mag, 1.0)
    r = np.real(sfft.ifftn(R, workers=-1))
    peak = np.unravel_index(int(np.argmax(r)), r.shape)
    shift = np.array(peak, dtype=np.float64)
    for ax, n in enumerate(r.shape):
        if shift[ax] > n // 2:
            shift[ax] -= n
    # r[k] peaks at the shift that maps f onto g: g(x) = f(x - shift)
    dz, dy, dx = shift * factor
    return np.array([dx, dy, dz], dtype=np.float64)


def _subvoxel_peak(neigh: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quadratic sub-voxel refinement for ``(n, 27)`` peak neighbourhoods."""
    n = neigh.shape[0]
    coef = neigh @ _PINV.T  # (n, 10)
    a = coef[:, 1:4]
    H = np.empty((n, 3, 3))
    H[:, 0, 0] = 2 * coef[:, 7]
    H[:, 1, 1] = 2 * coef[:, 8]
    H[:, 2, 2] = 2 * coef[:, 9]
    H[:, 0, 1] = H[:, 1, 0] = coef[:, 4]
    H[:, 0, 2] = H[:, 2, 0] = coef[:, 5]
    H[:, 1, 2] = H[:, 2, 1] = coef[:, 6]
    out = np.zeros((n, 3))
    det = np.linalg.det(H)
    ok = np.abs(det) > 1e-12
    if ok.any():
        try:
            sol = np.linalg.solve(H[ok], -a[ok][:, :, None])[:, :, 0]
            good = np.all(np.abs(sol) <= 1.0, axis=1) & np.all(np.isfinite(sol), axis=1)
            tmp = np.zeros((int(ok.sum()), 3))
            tmp[good] = sol[good]
            out[ok] = tmp
        except np.linalg.LinAlgError:
            pass
    return out


# ---------------------------------------------------------------------------
# Single-level NCC search
# ---------------------------------------------------------------------------


def ncc_search(
    f: NDArray[np.float32],
    g: NDArray[np.float32],
    coords: NDArray[np.int64],
    subset: tuple[int, int, int],
    radius: tuple[int, int, int],
    shift0: NDArray[np.int64] | None = None,
    chunk: int = 64,
) -> dict:
    """Per-node normalised cross-correlation search.

    Args:
        f, g: reference / deformed volumes ``(nz, ny, nx)``.
        coords: ``(N, 3)`` int node centres ``[x, y, z]`` in ``f``.
        subset: template size ``(wx, wy, wz)`` (even; template spans ``w+1``).
        radius: search half-width ``(rx, ry, rz)``.
        shift0: ``(N, 3)`` int prior displacement; the search window is
            centred at ``coords + shift0``.
        chunk: nodes per FFT batch.

    Returns:
        dict with ``disp`` (N,3) float (NaN where not solvable), ``cc`` (N,)
        peak NCC, ``pce`` (N,), ``clipped`` (N,) bool, ``ok`` (N,) bool.
    """
    f = np.asarray(f, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    nz, ny, nx = f.shape
    N = coords.shape[0]
    hx, hy, hz = (int(w) // 2 for w in subset)
    rx, ry, rz = (int(r) for r in radius)
    sx, sy, sz = 2 * hx + 1, 2 * hy + 1, 2 * hz + 1
    wx, wy, wz = sx + 2 * rx, sy + 2 * ry, sz + 2 * rz
    if shift0 is None:
        shift0 = np.zeros((N, 3), dtype=np.int64)
    shift0 = np.asarray(np.round(shift0), dtype=np.int64)

    disp = np.full((N, 3), np.nan)
    cc = np.full(N, np.nan)
    pce = np.full(N, np.nan)
    clipped = np.zeros(N, dtype=bool)
    ok = np.zeros(N, dtype=bool)
    if wx > nx or wy > ny or wz > nz:
        return {"disp": disp, "cc": cc, "pce": pce, "clipped": clipped, "ok": ok}

    cx, cy, cz = coords[:, 0], coords[:, 1], coords[:, 2]
    # template bounds inside f
    t_ok = (cx - hx >= 0) & (cx + hx < nx) & (cy - hy >= 0) & (cy + hy < ny) & (cz - hz >= 0) & (cz + hz < nz)
    # window start (clamped into g); a peak on a clamped side sits on the volume
    # boundary, not on the search boundary, so it must not trigger an expansion
    dx0 = cx + shift0[:, 0] - hx - rx
    dy0 = cy + shift0[:, 1] - hy - ry
    dz0 = cz + shift0[:, 2] - hz - rz
    wx0 = np.clip(dx0, 0, nx - wx)
    wy0 = np.clip(dy0, 0, ny - wy)
    wz0 = np.clip(dz0, 0, nz - wz)
    free_lo = np.column_stack([dx0 >= 0, dy0 >= 0, dz0 >= 0])
    free_hi = np.column_stack([dx0 <= nx - wx, dy0 <= ny - wy, dz0 <= nz - wz])
    idx_all = np.flatnonzero(t_ok)
    n_tpl = sx * sy * sz
    vz, vy, vx = 2 * rz + 1, 2 * ry + 1, 2 * rx + 1
    # zero-padded FFT size with fast (2,3,5-smooth) lengths: prime window sizes
    # would otherwise fall back to slow Bluestein transforms
    fft_shape = tuple(sfft.next_fast_len(int(w), real=True) for w in (wz, wy, wx))
    # spatial-domain kernel (one MAC per template voxel and offset, parallel over
    # nodes) beats the FFT path unless offsets x template is very large
    use_direct = HAS_NUMBA and (vz * vy * vx) * n_tpl <= DIRECT_NCC_MAX_OPS
    if use_direct:
        chunk = max(chunk, 4096)

    for start in range(0, idx_all.size, chunk):
        ids = idx_all[start : start + chunk]
        n = ids.size
        if use_direct:
            ncc = np.empty((n, vz, vy, vx))
            _zncc_direct(f, g, coords[ids], hx, hy, hz, rx, ry, rz, wx0[ids], wy0[ids], wz0[ids], np.ones(n, dtype=np.bool_), ncc)
            t_norm = np.where(ncc[:, 0, 0, 0] > -1.5, 1.0, 0.0)  # flat templates were marked -2
            ncc[ncc < -1.0] = 0.0
        else:
            T = np.empty((n, sz, sy, sx), dtype=np.float32)
            W = np.empty((n, wz, wy, wx), dtype=np.float32)
            for k, i in enumerate(ids):
                T[k] = f[cz[i] - hz : cz[i] + hz + 1, cy[i] - hy : cy[i] + hy + 1, cx[i] - hx : cx[i] + hx + 1]
                W[k] = g[wz0[i] : wz0[i] + wz, wy0[i] : wy0[i] + wy, wx0[i] : wx0[i] + wx]
            T -= T.mean(axis=(1, 2, 3), keepdims=True, dtype=np.float64).astype(np.float32)
            t_norm = np.sqrt(np.sum(T.astype(np.float64) ** 2, axis=(1, 2, 3)))
            Ff = sfft.rfftn(W, s=fft_shape, axes=(1, 2, 3), workers=-1)
            Ft = sfft.rfftn(T, s=fft_shape, axes=(1, 2, 3), workers=-1)
            Ff *= np.conj(Ft)
            C = sfft.irfftn(Ff, s=fft_shape, axes=(1, 2, 3), workers=-1)[:, :vz, :vy, :vx].astype(np.float64)
            s1 = np.empty((n, vz, vy, vx))
            s2 = np.empty((n, vz, vy, vx))
            _window_box_stats(W, sz, sy, sx, s1, s2)
            var = s2 - s1 * s1 / n_tpl
            np.maximum(var, 0.0, out=var)
            denom = np.sqrt(var) * t_norm[:, None, None, None]
            good = denom > 1e-10
            ncc = np.zeros_like(C)
            np.divide(C, denom, out=ncc, where=good)
            np.clip(ncc, -1.0, 1.0, out=ncc)
        flat = ncc.reshape(n, -1)
        pk = np.argmax(flat, axis=1)
        pz, py, px = np.unravel_index(pk, ncc.shape[1:])
        peak_val = flat[np.arange(n), pk]
        # quality: peak-to-correlation-energy on the min-shifted map
        cmin = flat - flat.min(axis=1, keepdims=True)
        energy = np.mean(cmin * cmin, axis=1)
        pk_min = cmin[np.arange(n), pk]
        pce_k = np.divide(pk_min * pk_min, energy, out=np.zeros(n), where=energy > 1e-30)
        lo = free_lo[ids]
        hi = free_hi[ids]
        on_edge = (
            (
                ((px == 0) & lo[:, 0])
                | ((px == 2 * rx) & hi[:, 0])
                | ((py == 0) & lo[:, 1])
                | ((py == 2 * ry) & hi[:, 1])
                | ((pz == 0) & lo[:, 2])
                | ((pz == 2 * rz) & hi[:, 2])
            )
            if (rx > 0 or ry > 0 or rz > 0)
            else np.zeros(n, dtype=bool)
        )
        sub = np.zeros((n, 3))
        at_border = (px == 0) | (px == 2 * rx) | (py == 0) | (py == 2 * ry) | (pz == 0) | (pz == 2 * rz)
        interior = ~at_border & (rx > 0) & (ry > 0) & (rz > 0)
        if interior.any():
            ii = np.flatnonzero(interior)
            neigh = np.empty((ii.size, 27))
            for kk, k in enumerate(ii):
                neigh[kk] = ncc[k, pz[k] - 1 : pz[k] + 2, py[k] - 1 : py[k] + 2, px[k] - 1 : px[k] + 2].ravel()
            sub[ii] = _subvoxel_peak(neigh)
        # displacement = (window start + peak offset + half) - node centre
        disp[ids, 0] = wx0[ids] + px + hx - cx[ids] + sub[:, 0]
        disp[ids, 1] = wy0[ids] + py + hy - cy[ids] + sub[:, 1]
        disp[ids, 2] = wz0[ids] + pz + hz - cz[ids] + sub[:, 2]
        cc[ids] = peak_val
        pce[ids] = pce_k
        clipped[ids] = on_edge
        ok[ids] = np.isfinite(peak_val) & (t_norm > 1e-10)
    disp[~ok] = np.nan
    return {"disp": disp, "cc": cc, "pce": pce, "clipped": clipped, "ok": ok}


def ncc_search_expanding(
    f: NDArray[np.float32],
    g: NDArray[np.float32],
    coords: NDArray[np.int64],
    subset: tuple[int, int, int],
    radius: tuple[int, int, int],
    shift0: NDArray[np.int64] | None = None,
    auto_expand: bool = True,
    max_expand: int = 3,
    chunk: int = 64,
) -> dict:
    """:func:`ncc_search` plus automatic radius doubling for clipped peaks.

    Only the nodes whose peak sits on the search boundary are re-searched
    with a doubled radius (up to ``max_expand`` times), so the cost of an
    expansion is proportional to the number of affected nodes.
    """
    res = ncc_search(f, g, coords, subset, radius, shift0, chunk=chunk)
    res["expansions"] = 0
    res["radius"] = tuple(int(r) for r in radius)
    rad = res["radius"]
    for _ in range(max_expand if auto_expand else 0):
        sel = res["clipped"] & res["ok"]
        n_ok = int(res["ok"].sum())
        if n_ok == 0 or sel.sum() <= CLIPPED_EXPAND_FRACTION * n_ok:
            break
        rad = tuple(2 * r for r in rad)
        logger.info("NCC: %d/%d peaks clipped; re-searching them with radius %s", int(sel.sum()), n_ok, rad)
        sub_shift = None if shift0 is None else shift0[sel]
        sub = ncc_search(f, g, coords[sel], subset, rad, sub_shift, chunk=chunk)
        if not sub["ok"].any():
            break  # window no longer fits in the volume; keep the previous peaks
        idx = np.flatnonzero(sel)[sub["ok"]]
        for key in ("disp", "cc", "pce", "clipped", "ok"):
            res[key][idx] = sub[key][sub["ok"]]
        res["expansions"] += 1
        res["radius"] = rad
    return res


def _clean_field(disp: NDArray[np.float64], grid_shape: tuple[int, int, int], threshold: float) -> NDArray[np.float64]:
    """Median-test outlier removal + spring inpainting on the node grid."""
    d = disp.reshape(grid_shape + (3,)).copy()
    valid = np.all(np.isfinite(d), axis=-1)
    if threshold > 0 and valid.sum() > 27:
        flag = universal_median_test(d, valid, threshold)
        d[flag] = np.nan
    out = np.empty_like(d)
    for c in range(3):
        out[..., c] = fill_nan_grid(d[..., c])
    return out.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Coarse-to-fine wrapper
# ---------------------------------------------------------------------------


MIN_COARSE_STD_RATIO = 0.35  # a coarse level must retain this fraction of the texture contrast


def auto_pyramid_levels(
    shape: tuple[int, int, int],
    subset: tuple[int, int, int],
    radius: tuple[int, int, int],
    max_levels: int = 3,
    f: NDArray | None = None,
    min_std_ratio: float = MIN_COARSE_STD_RATIO,
) -> int:
    """Number of coarse levels that (a) fit the search window and (b) keep texture.

    Block averaging by ``2**level`` washes out a speckle pattern whose
    correlation length is smaller than the block; correlating such a level
    only produces random peaks. When ``f`` is given, a level is accepted only
    if the downsampled volume keeps at least ``min_std_ratio`` of the
    full-resolution standard deviation.
    """
    levels = 0
    std0 = float(np.std(f)) if f is not None else None
    for lv in range(1, max_levels + 1):
        fac = 2**lv
        fits = True
        for n, w, r in zip(shape[::-1], subset, radius):
            w_l = max(MIN_PYRAMID_SUBSET, (w // fac) // 2 * 2)
            need = w_l + 1 + 2 * max(r, 1) + 4  # search window + margin must fit
            if n // fac < need:
                fits = False
        if fits and std0 is not None and std0 > 0:
            if float(np.std(block_downsample(f, fac))) < min_std_ratio * std0:
                fits = False
        if fits:
            levels = lv
        else:
            break
    return levels


def pyramid_search(
    f: NDArray[np.float32],
    g: NDArray[np.float32],
    coords: NDArray[np.int64],
    grid_shape: tuple[int, int, int],
    subset: tuple[int, int, int],
    radius: tuple[int, int, int],
    levels: int = 0,
    global_shift: NDArray[np.float64] | None = None,
    outlier_threshold: float = 2.0,
    auto_expand: bool = True,
    max_expand: int = 3,
) -> dict:
    """Coarse-to-fine NCC search. Returns ``disp`` (N, 3) in full-resolution voxels + info.

    ``levels == 0`` selects the number of levels automatically. The coarsest
    level uses the full ``radius``; finer levels refine with a small radius
    around the up-scaled estimate. The estimate is median-cleaned and
    inpainted between levels so every node has a usable prior.
    """
    f = np.asarray(f, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    if levels <= 0:
        levels = auto_pyramid_levels(f.shape, subset, radius, f=f)
        logger.info("Pyramid: %d coarse level(s) selected automatically", levels)
    N = coords.shape[0]
    disp_full = np.zeros((N, 3), dtype=np.float64)
    if global_shift is not None:
        disp_full += np.asarray(global_shift, dtype=np.float64)[None, :]
    info: dict = {"levels": levels, "per_level": [], "expansions": 0}

    for lv in range(levels, -1, -1):
        fac = 2**lv
        fl = block_downsample(f, fac) if fac > 1 else f
        gl = block_downsample(g, fac) if fac > 1 else g
        subset_l = tuple(max(MIN_PYRAMID_SUBSET, (w // fac) // 2 * 2) for w in subset)
        rad_l = tuple(int(r) for r in radius) if lv == levels else (PYRAMID_FINE_RADIUS,) * 3
        coords_l = np.round(coords / fac).astype(np.int64)
        shift_l = np.round(disp_full / fac).astype(np.int64)
        res = ncc_search_expanding(fl, gl, coords_l, subset_l, rad_l, shift_l, auto_expand, max_expand)
        rad_l = res["radius"]
        info["expansions"] += res["expansions"]
        d = res["disp"] * fac
        d[~res["ok"]] = np.nan
        n_ok = int(res["ok"].sum())
        if n_ok == 0:
            logger.warning("Pyramid level %d produced no valid NCC peaks; keeping previous estimate.", lv)
            info["per_level"].append({"level": lv, "n_ok": 0, "radius": rad_l, "subset": subset_l})
            continue
        disp_full = _clean_field(d, grid_shape, outlier_threshold)
        info["per_level"].append(
            {
                "level": lv,
                "n_ok": n_ok,
                "radius": rad_l,
                "subset": subset_l,
                "median_cc": float(np.nanmedian(res["cc"])),
                "clipped_frac": float(np.mean(res["clipped"][res["ok"]])),
            }
        )
        info["cc"] = res["cc"]
        info["pce"] = res["pce"]
        info["ok"] = res["ok"]
        info["clipped"] = res["clipped"]
    info["disp"] = disp_full
    return info
