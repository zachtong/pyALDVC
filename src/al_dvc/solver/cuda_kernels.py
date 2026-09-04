"""CUDA (numba-cuda) versions of the local IC-GN kernels: one thread block per node.

Optional backend: ``pip install al-dvc[gpu]`` installs ``numba-cuda`` with the
CUDA 12 wheels; without it, or without an NVIDIA GPU, :func:`cuda_available`
is False and the CPU kernels run. Everything here is imported lazily so a
CPU-only installation never touches CUDA.

Design (mirrors ``numba_kernels.py`` exactly, including the dynamic masked
voxel set, the sampling stride, the noise-corrected Hessian, the stall rule
and the look-ahead stop):

* a block of ``BLOCK`` threads solves one node; its threads stride over the
  subset voxels, the sampled values go to a per-block scratch row, and the
  ZNSSD statistics / the 12 gradient components are block-reduced in shared
  memory;
* thread 0 runs the per-iteration control (12x12 Cholesky solve in float64,
  warp composition, stopping rules) and broadcasts the decision;
* sampling and accumulation are float32 (consumer GPUs run float64 at 1/64
  rate); the per-thread partial sums cover ~100 voxels and the tree
  reduction keeps the relative error near 1e-6, i.e. ~1e-5 voxel on the
  displacement, well below the 1e-3 voxel increment tolerance;
* launches are chunked (``CHUNK_NODES`` nodes per launch) so a display GPU
  never runs one kernel for more than a fraction of a second (Windows TDR)
  and the host can report progress.

Reference volumes are uploaded once per array object and cached
(:class:`DeviceCache`), so the ADMM passes of a frame reuse them.
"""

from __future__ import annotations

import logging
import math
import threading
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np

from ..core.data_structures import (
    STATUS_CONVERGED,
    STATUS_INVALID_SUBSET,
    STATUS_MAX_ITER,
    STATUS_NAN,
    STATUS_OUT_OF_BOUNDS,
    STATUS_SINGULAR,
    STATUS_SKIPPED,
    STATUS_STALLED,
)
from .interp_kernels import INTERP_CUBIC, INTERP_LINEAR, SAMPLE_HI_MARGIN, SAMPLE_LO
from .numba_kernels import (
    ABS_TOL,
    LM_DAMPING_3DOF,
    MIN_CORRECTED_FRACTION,
    MIN_SUBSET_VOXELS,
    MIN_VALID_FRACTION,
    NOISE_CORR_STEP,
    PREDICT_CONTRACTION,
    STALL_STEP_DECAY,
    STALL_ZNCC_EPS,
)

logger = logging.getLogger(__name__)

BLOCK = 256  # threads per node
CHUNK_NODES = 2048  # nodes per launch (bounded kernel time, progress granularity)
RED_Q = 16  # block-reduced quantities per pass (<= 16)
TPL_MAX = 6859  # largest NCC template held in shared memory: 19^3 voxels
DEVICE_CACHE_SIZE = 8
_STENCIL = (-1.0 / 60, 3.0 / 20, -3.0 / 4, 0.0, 3.0 / 4, -3.0 / 20, 1.0 / 60)

_available: bool | None = None
_unavailable_reason = ""
_probe_lock = threading.Lock()
# numba's driver logs every cuMemFree at INFO; that is noise in an application log
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)


def _quiet_performance_warnings() -> None:
    """Small last chunks and the probe launch few blocks; numba-cuda warns about occupancy on every such launch."""
    try:
        from numba.core.errors import NumbaPerformanceWarning

        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    except Exception:  # numba without CUDA support: nothing to silence
        pass


_quiet_performance_warnings()


# --------------------------------------------------------------------------- availability
def cuda_available() -> bool:
    """True when numba-cuda is installed, a CUDA device is present and a kernel compiles for it."""
    global _available, _unavailable_reason
    if _available is not None:
        return _available
    with _probe_lock:  # the GUI warm-up thread, the status label and the worker may all ask at once
        if _available is not None:
            return _available
        return _probe_once()


def _probe_once() -> bool:
    global _available, _unavailable_reason
    try:
        from numba import cuda

        if not cuda.is_available():
            raise RuntimeError("numba.cuda reports no usable CUDA device")
        dev = cuda.get_current_device()
        _probe_kernel()
        name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
        logger.info("CUDA backend: %s (compute capability %s)", name, ".".join(str(c) for c in dev.compute_capability))
        _available = True
    except Exception as exc:  # missing package, no driver, unsupported device, compile failure
        _unavailable_reason = f"{type(exc).__name__}: {exc}"
        logger.warning("CUDA backend unavailable, using the CPU kernels (%s)", _unavailable_reason)
        _available = False
    return _available


def unavailable_reason() -> str:
    cuda_available()
    return _unavailable_reason


def _probe_kernel() -> None:
    from numba import cuda

    @cuda.jit
    def _probe(out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = i * 2.0

    out = cuda.device_array(64, np.float32)
    _probe[1, 64](out)
    cuda.synchronize()
    if float(out.copy_to_host()[3]) != 6.0:
        raise RuntimeError("probe kernel returned wrong values")


def device_name() -> str:
    if not cuda_available():
        return ""
    from numba import cuda

    dev = cuda.get_current_device()
    return dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)


# --------------------------------------------------------------------------- device array cache
class DeviceCache:
    """Host array -> device array, keyed by the host object; bounded LRU."""

    def __init__(self, size: int = DEVICE_CACHE_SIZE) -> None:
        self._items: OrderedDict[tuple, tuple[Any, Any]] = OrderedDict()
        self._size = size

    def get(self, arr: np.ndarray):
        from numba import cuda

        arr = np.ascontiguousarray(arr)
        key = (id(arr), arr.shape, str(arr.dtype))
        hit = self._items.get(key)
        if hit is not None and hit[0] is arr:
            self._items.move_to_end(key)
            return hit[1]
        dev = cuda.to_device(arr)
        self._items[key] = (arr, dev)  # keep the host array alive so its id stays unique
        while len(self._items) > self._size:
            self._items.popitem(last=False)
        return dev

    def clear(self) -> None:
        self._items.clear()


_cache = DeviceCache()


def clear_device_cache() -> None:
    _cache.clear()


# --------------------------------------------------------------------------- kernels (compiled lazily)
_kernels: dict[str, Any] = {}


def _build_kernels() -> dict[str, Any]:
    if _kernels:
        return _kernels
    from numba import cuda, float32, float64, int32

    F32 = float32
    LO = F32(SAMPLE_LO)
    HI = F32(SAMPLE_HI_MARGIN)
    W0, W1, W2, W3, W4, W5, W6 = (F32(w) for w in _STENCIL)

    @cuda.jit(device=True, inline=True)
    def keys4(t):
        w0 = ((F32(-0.5) * t + F32(1.0)) * t - F32(0.5)) * t
        w1 = (F32(1.5) * t - F32(2.5)) * t * t + F32(1.0)
        w2 = ((F32(-1.5) * t + F32(2.0)) * t + F32(0.5)) * t
        w3 = (F32(0.5) * t - F32(0.5)) * t * t
        return w0, w1, w2, w3

    @cuda.jit(device=True, inline=True)
    def bspline4(t):
        t2 = t * t
        t3 = t2 * t
        w0 = (F32(1.0) - F32(3.0) * t + F32(3.0) * t2 - t3) * F32(1.0 / 6.0)
        w1 = (F32(4.0) - F32(6.0) * t2 + F32(3.0) * t3) * F32(1.0 / 6.0)
        w2 = (F32(1.0) + F32(3.0) * t + F32(3.0) * t2 - F32(3.0) * t3) * F32(1.0 / 6.0)
        w3 = t3 * F32(1.0 / 6.0)
        return w0, w1, w2, w3

    @cuda.jit(device=True, inline=True)
    def row4(vol, zz, yy, xb, wx0, wx1, wx2, wx3):
        return vol[zz, yy, xb] * wx0 + vol[zz, yy, xb + 1] * wx1 + vol[zz, yy, xb + 2] * wx2 + vol[zz, yy, xb + 3] * wx3

    @cuda.jit(device=True, inline=True)
    def sample(vol, z, y, x, mode):
        nz = vol.shape[0]
        ny = vol.shape[1]
        nx = vol.shape[2]
        ix = int32(math.floor(x))
        iy = int32(math.floor(y))
        iz = int32(math.floor(z))
        if ix > nx - 3:
            ix = nx - 3
        if iy > ny - 3:
            iy = ny - 3
        if iz > nz - 3:
            iz = nz - 3
        fx = x - F32(ix)
        fy = y - F32(iy)
        fz = z - F32(iz)
        if mode == INTERP_LINEAR:
            c00 = vol[iz, iy, ix] * (F32(1.0) - fx) + vol[iz, iy, ix + 1] * fx
            c10 = vol[iz, iy + 1, ix] * (F32(1.0) - fx) + vol[iz, iy + 1, ix + 1] * fx
            c01 = vol[iz + 1, iy, ix] * (F32(1.0) - fx) + vol[iz + 1, iy, ix + 1] * fx
            c11 = vol[iz + 1, iy + 1, ix] * (F32(1.0) - fx) + vol[iz + 1, iy + 1, ix + 1] * fx
            c0 = c00 * (F32(1.0) - fy) + c10 * fy
            c1 = c01 * (F32(1.0) - fy) + c11 * fy
            return c0 * (F32(1.0) - fz) + c1 * fz
        if mode == INTERP_CUBIC:
            wx0, wx1, wx2, wx3 = keys4(fx)
            wy0, wy1, wy2, wy3 = keys4(fy)
            wz0, wz1, wz2, wz3 = keys4(fz)
        else:
            wx0, wx1, wx2, wx3 = bspline4(fx)
            wy0, wy1, wy2, wy3 = bspline4(fy)
            wz0, wz1, wz2, wz3 = bspline4(fz)
        xb = ix - 1
        yb = iy - 1
        zb = iz - 1
        val = F32(0.0)
        for k in range(4):
            zz = zb + k
            wzk = wz0 if k == 0 else (wz1 if k == 1 else (wz2 if k == 2 else wz3))
            plane = (
                row4(vol, zz, yb, xb, wx0, wx1, wx2, wx3) * wy0
                + row4(vol, zz, yb + 1, xb, wx0, wx1, wx2, wx3) * wy1
                + row4(vol, zz, yb + 2, xb, wx0, wx1, wx2, wx3) * wy2
                + row4(vol, zz, yb + 3, xb, wx0, wx1, wx2, wx3) * wy3
            )
            val += wzk * plane
        return val

    @cuda.jit(device=True, inline=True)
    def inside(z, y, x, nz, ny, nx):
        if x < LO or x > F32(nx - 1) - HI:
            return False
        if y < LO or y > F32(ny - 1) - HI:
            return False
        if z < LO or z > F32(nz - 1) - HI:
            return False
        return True

    @cuda.jit(device=True, inline=True)
    def grad_at(f, gx, gy, gz, stored, zz, yy, xx):
        if stored:
            return gx[zz, yy, xx], gy[zz, yy, xx], gz[zz, yy, xx]
        nz = f.shape[0]
        ny = f.shape[1]
        nx = f.shape[2]
        dfx = F32(0.0)
        dfy = F32(0.0)
        dfz = F32(0.0)
        if xx >= 3 and xx <= nx - 4:
            dfx = (
                W0 * f[zz, yy, xx - 3]
                + W1 * f[zz, yy, xx - 2]
                + W2 * f[zz, yy, xx - 1]
                + W4 * f[zz, yy, xx + 1]
                + W5 * f[zz, yy, xx + 2]
                + W6 * f[zz, yy, xx + 3]
            )
        if yy >= 3 and yy <= ny - 4:
            dfy = (
                W0 * f[zz, yy - 3, xx]
                + W1 * f[zz, yy - 2, xx]
                + W2 * f[zz, yy - 1, xx]
                + W4 * f[zz, yy + 1, xx]
                + W5 * f[zz, yy + 2, xx]
                + W6 * f[zz, yy + 3, xx]
            )
        if zz >= 3 and zz <= nz - 4:
            dfz = (
                W0 * f[zz - 3, yy, xx]
                + W1 * f[zz - 2, yy, xx]
                + W2 * f[zz - 1, yy, xx]
                + W4 * f[zz + 1, yy, xx]
                + W5 * f[zz + 2, yy, xx]
                + W6 * f[zz + 3, yy, xx]
            )
        return dfx, dfy, dfz

    # ---- float64 12x12 linear algebra on shared arrays (thread 0 only) ----
    @cuda.jit(device=True)
    def cholesky12(H, L):
        for i in range(12):
            for j in range(12):
                L[i, j] = 0.0
        for j in range(12):
            s = H[j, j]
            for k in range(j):
                s -= L[j, k] * L[j, k]
            if not (s > 0.0) or math.isnan(s) or math.isinf(s):
                return False
            ljj = math.sqrt(s)
            L[j, j] = ljj
            for i in range(j + 1, 12):
                t = H[i, j]
                for k in range(j):
                    t -= L[i, k] * L[j, k]
                L[i, j] = t / ljj
        return True

    @cuda.jit(device=True)
    def chol_solve12(L, b, x):
        for i in range(12):
            s = b[i]
            for k in range(i):
                s -= L[i, k] * x[k]
            x[i] = s / L[i, i]
        for i in range(11, -1, -1):
            s = x[i]
            for k in range(i + 1, 12):
                s -= L[k, i] * x[k]
            x[i] = s / L[i, i]

    @cuda.jit(device=True)
    def corrected_cholesky12(H, pattern, corr, H0, L0):
        limit = 1.0
        for i in range(9, 12):
            c = corr * pattern[i, i]
            if c > 0.0:
                lim = (1.0 - MIN_CORRECTED_FRACTION) * H[i, i] / c
                if lim < limit:
                    limit = lim
        if limit <= 0.0:
            return False
        for i in range(12):
            for j in range(12):
                H0[i, j] = H[i, j] - corr * limit * pattern[i, j]
        return cholesky12(H0, L0)

    @cuda.jit(device=True)
    def compose_warp(P, dP, M):
        """``P <- W(P) W(dP)^-1``; ``M`` is a 3x9 float64 scratch (A, dA^-1, A_new). False if singular."""
        # A = I + F,  dA = I + dF
        for i in range(3):
            for j in range(3):
                M[0, 3 * i + j] = P[3 * i + j] + (1.0 if i == j else 0.0)
                M[1, 3 * i + j] = dP[3 * i + j] + (1.0 if i == j else 0.0)
        a00 = M[1, 0]
        a01 = M[1, 1]
        a02 = M[1, 2]
        a10 = M[1, 3]
        a11 = M[1, 4]
        a12 = M[1, 5]
        a20 = M[1, 6]
        a21 = M[1, 7]
        a22 = M[1, 8]
        det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
        if det == 0.0 or math.isnan(det):
            return False
        inv = 1.0 / det
        M[1, 0] = (a11 * a22 - a12 * a21) * inv
        M[1, 1] = (a02 * a21 - a01 * a22) * inv
        M[1, 2] = (a01 * a12 - a02 * a11) * inv
        M[1, 3] = (a12 * a20 - a10 * a22) * inv
        M[1, 4] = (a00 * a22 - a02 * a20) * inv
        M[1, 5] = (a02 * a10 - a00 * a12) * inv
        M[1, 6] = (a10 * a21 - a11 * a20) * inv
        M[1, 7] = (a01 * a20 - a00 * a21) * inv
        M[1, 8] = (a00 * a11 - a01 * a10) * inv
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += M[0, 3 * i + k] * M[1, 3 * k + j]
                M[2, 3 * i + j] = s
        for i in range(3):
            s = 0.0
            for k in range(3):
                s += M[2, 3 * i + k] * dP[9 + k]
            P[9 + i] = P[9 + i] - s
        for i in range(3):
            for j in range(3):
                P[3 * i + j] = M[2, 3 * i + j]
            P[3 * i + i] -= 1.0
        return True

    @cuda.jit(device=True, inline=True)
    def block_reduce(red, nq):
        """Tree-sum ``red[q, :]`` for ``q < nq`` into ``red[q, 0]`` (all threads must call)."""
        cuda.syncthreads()
        s = BLOCK // 2
        tid = cuda.threadIdx.x
        while s > 0:
            if tid < s:
                for q in range(nq):
                    red[q, tid] += red[q, tid + s]
            cuda.syncthreads()
            s //= 2

    @cuda.jit
    def icgn12_kernel(
        idx,
        coords,
        P_io,
        hx,
        hy,
        hz,
        stride,
        f,
        gx,
        gy,
        gz,
        stored,
        mask,
        g,
        mode,
        L_all,
        H_all,
        meanf_all,
        bottomf_all,
        pattern,
        noise_gain,
        tol,
        dp_tol,
        max_iter,
        patience,
        predictive,
        gbuf,
        n_iter_out,
        status_out,
        zncc_out,
    ):
        blk = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        n = idx[blk]
        red = cuda.shared.array((RED_Q, BLOCK), float32)
        P = cuda.shared.array(12, float64)
        dP = cuda.shared.array(12, float64)
        bvec = cuda.shared.array(12, float64)
        negb = cuda.shared.array(12, float64)
        P_best = cuda.shared.array(12, float64)
        H0 = cuda.shared.array((12, 12), float64)
        L0 = cuda.shared.array((12, 12), float64)
        M = cuda.shared.array((3, 9), float64)
        st = cuda.shared.array(8, float64)  # meanf_d, bottomf_d, meang, bottomg, inv_bf, inv_bg, n_valid, spare
        ctrl = cuda.shared.array(4, int32)  # [0] -1 continue / >=0 stop with status, [1] warp coefficients ready
        warp = cuda.shared.array(12, float32)  # a00..a22, tx, ty, tz in float32 for the voxel loops
        if tid == 0:
            finite = True
            for k in range(12):
                P[k] = P_io[n, k]
                if math.isnan(P[k]) or math.isinf(P[k]):
                    finite = False
            ctrl[0] = -1 if finite else STATUS_NAN
        cuda.syncthreads()
        x0 = coords[n, 0]
        y0 = coords[n, 1]
        z0 = coords[n, 2]
        nz = g.shape[0]
        ny = g.shape[1]
        nx = g.shape[2]
        sx = (2 * hx) // stride + 1
        sy = (2 * hy) // stride + 1
        sz = (2 * hz) // stride + 1
        S = sx * sy * sz
        half_scale = float64(max(hx, max(hy, hz)))
        n_full = pattern[9, 9]
        norm_init = -1.0
        best_zncc = -2.0
        best_dp = 1e300
        last_dp = 1e300
        dp_prev = -1.0
        stall = 0
        zncc = math.nan
        it_done = 0
        status = STATUS_MAX_ITER
        if ctrl[0] >= 0:
            status = ctrl[0]
        for it in range(1, max_iter + 1):
            if ctrl[0] >= 0:
                break
            it_done = it
            # ---------------- pass 1: warp, sample, statistics
            if tid == 0:
                warp[0] = F32(1.0 + P[0])
                warp[1] = F32(P[1])
                warp[2] = F32(P[2])
                warp[3] = F32(P[3])
                warp[4] = F32(1.0 + P[4])
                warp[5] = F32(P[5])
                warp[6] = F32(P[6])
                warp[7] = F32(P[7])
                warp[8] = F32(1.0 + P[8])
                warp[9] = F32(x0 + P[9])
                warp[10] = F32(y0 + P[10])
                warp[11] = F32(z0 + P[11])
                ctrl[1] = 0
            cuda.syncthreads()
            s1 = F32(0.0)
            s2 = F32(0.0)
            s1f = F32(0.0)
            s2f = F32(0.0)
            nv = F32(0.0)
            nref = F32(0.0)
            oob = F32(0.0)
            for v in range(tid, S, BLOCK):
                iz_ = v // (sy * sx)
                rem = v - iz_ * (sy * sx)
                iy_ = rem // sx
                ix_ = rem - iy_ * sx
                dx = -hx + stride * ix_
                dy = -hy + stride * iy_
                dz = -hz + stride * iz_
                zz = z0 + dz
                yy = y0 + dy
                xx = x0 + dx
                if mask[zz, yy, xx] == 0:
                    gbuf[blk, v] = math.nan
                    continue
                nref += F32(1.0)
                X = F32(dx)
                Y = F32(dy)
                Z = F32(dz)
                xw = warp[0] * X + warp[1] * Y + warp[2] * Z + warp[9]
                yw = warp[3] * X + warp[4] * Y + warp[5] * Z + warp[10]
                zw = warp[6] * X + warp[7] * Y + warp[8] * Z + warp[11]
                if not inside(zw, yw, xw, nz, ny, nx):
                    oob = F32(1.0)
                    gbuf[blk, v] = math.nan
                    continue
                val = sample(g, zw, yw, xw, mode)
                gbuf[blk, v] = val
                if math.isnan(val):
                    continue
                fv = f[zz, yy, xx]
                nv += F32(1.0)
                s1 += val
                s2 += val * val
                s1f += fv
                s2f += fv * fv
            red[0, tid] = s1
            red[1, tid] = s2
            red[2, tid] = s1f
            red[3, tid] = s2f
            red[4, tid] = nv
            red[5, tid] = nref
            red[6, tid] = oob
            block_reduce(red, 7)
            if tid == 0:
                n_valid = float64(red[4, 0])
                n_ref = float64(red[5, 0])
                if red[6, 0] > F32(0.0):
                    ctrl[0] = STATUS_OUT_OF_BOUNDS
                elif n_valid < MIN_SUBSET_VOXELS or n_valid < MIN_VALID_FRACTION * n_ref:
                    ctrl[0] = STATUS_INVALID_SUBSET
                else:
                    meang = float64(red[0, 0]) / n_valid
                    ssg = float64(red[1, 0]) - n_valid * meang * meang
                    if ssg < 0.0:
                        ssg = 0.0
                    bottomg = math.sqrt(max(ssg, 1e-30))
                    meanf_d = float64(red[2, 0]) / n_valid
                    ssf = float64(red[3, 0]) - n_valid * meanf_d * meanf_d
                    if ssf < 0.0:
                        ssf = 0.0
                    bottomf_d = math.sqrt(max(ssf, 1e-30))
                    st[0] = meanf_d
                    st[1] = bottomf_d
                    st[2] = meang
                    st[3] = bottomg
                    st[4] = 1.0 / bottomf_d
                    st[5] = 1.0 / bottomg
                    st[6] = n_valid
            cuda.syncthreads()
            if ctrl[0] >= 0:
                status = ctrl[0]
                zncc = math.nan
                break
            # ---------------- pass 2: residual, gradient, ZNCC numerator
            meanf_d32 = F32(st[0])
            meang32 = F32(st[2])
            inv_bf32 = F32(st[4])
            inv_bg32 = F32(st[5])
            b0 = F32(0.0)
            b1 = F32(0.0)
            b2 = F32(0.0)
            b3 = F32(0.0)
            b4 = F32(0.0)
            b5 = F32(0.0)
            b6 = F32(0.0)
            b7 = F32(0.0)
            b8 = F32(0.0)
            b9 = F32(0.0)
            b10 = F32(0.0)
            b11 = F32(0.0)
            scc = F32(0.0)
            for v in range(tid, S, BLOCK):
                gv = gbuf[blk, v]
                if math.isnan(gv):
                    continue
                iz_ = v // (sy * sx)
                rem = v - iz_ * (sy * sx)
                iy_ = rem // sx
                ix_ = rem - iy_ * sx
                dx = -hx + stride * ix_
                dy = -hy + stride * iy_
                dz = -hz + stride * iz_
                zz = z0 + dz
                yy = y0 + dy
                xx = x0 + dx
                fd = f[zz, yy, xx] - meanf_d32
                gd = gv - meang32
                scc += fd * gd
                res = fd * inv_bf32 - gd * inv_bg32
                dfx, dfy, dfz = grad_at(f, gx, gy, gz, stored, zz, yy, xx)
                gxv = dfx * res
                gyv = dfy * res
                gzv = dfz * res
                X = F32(dx)
                Y = F32(dy)
                Z = F32(dz)
                b0 += gxv * X
                b1 += gxv * Y
                b2 += gxv * Z
                b3 += gyv * X
                b4 += gyv * Y
                b5 += gyv * Z
                b6 += gzv * X
                b7 += gzv * Y
                b8 += gzv * Z
                b9 += gxv
                b10 += gyv
                b11 += gzv
            red[0, tid] = b0
            red[1, tid] = b1
            red[2, tid] = b2
            red[3, tid] = b3
            red[4, tid] = b4
            red[5, tid] = b5
            red[6, tid] = b6
            red[7, tid] = b7
            red[8, tid] = b8
            red[9, tid] = b9
            red[10, tid] = b10
            red[11, tid] = b11
            red[12, tid] = scc
            block_reduce(red, 13)
            # ---------------- control (thread 0)
            if tid == 0:
                bottomf_d = st[1]
                norm_abs = 0.0
                for k in range(12):
                    bvec[k] = float64(red[k, 0]) * bottomf_d
                    norm_abs += bvec[k] * bvec[k]
                norm_abs = math.sqrt(norm_abs)
                zncc_local = float64(red[12, 0]) * st[4] * st[5]
                if math.isnan(norm_abs) or math.isinf(norm_abs):
                    ctrl[0] = STATUS_NAN
                else:
                    if norm_init < 0.0:
                        norm_init = norm_abs
                    norm_rel = norm_abs / norm_init if norm_init > 1e-300 else 0.0
                    zncc = zncc_local
                    if norm_rel < tol or norm_abs < ABS_TOL:
                        ctrl[0] = STATUS_CONVERGED
                    else:
                        for k in range(12):
                            negb[k] = -bvec[k]
                        use_corr = False
                        if noise_gain > 0.0 and n_full > 0.0 and last_dp < NOISE_CORR_STEP:
                            s2n = max(1.0 - zncc, 0.0) * bottomf_d * bottomf_d / st[6]
                            corr = noise_gain * s2n * (st[6] / n_full)
                            use_corr = corrected_cholesky12(H_all[n], pattern, corr, H0, L0)
                        if use_corr:
                            chol_solve12(L0, negb, dP)
                        else:
                            chol_solve12(L_all[n], negb, dP)
                        dp_norm = 0.0
                        for k in range(9):
                            vv = dP[k] * half_scale
                            dp_norm += vv * vv
                        for k in range(9, 12):
                            dp_norm += dP[k] * dP[k]
                        dp_norm = math.sqrt(dp_norm)
                        if math.isnan(dp_norm) or math.isinf(dp_norm):
                            ctrl[0] = STATUS_NAN
                        else:
                            last_dp = dp_norm
                            if dp_norm < dp_tol:
                                ctrl[0] = STATUS_CONVERGED
                            elif (
                                predictive
                                and dp_prev > 0.0
                                and dp_norm < PREDICT_CONTRACTION * dp_prev
                                and dp_norm * dp_norm < dp_tol * dp_prev
                            ):
                                if compose_warp(P, dP, M):
                                    ctrl[0] = STATUS_CONVERGED
                                else:
                                    ctrl[0] = STATUS_SINGULAR
                            else:
                                dp_prev = dp_norm
                                improved = zncc > best_zncc + STALL_ZNCC_EPS
                                if improved:
                                    best_zncc = zncc
                                    for k in range(12):
                                        P_best[k] = P[k]
                                if dp_norm < best_dp * STALL_STEP_DECAY:
                                    best_dp = dp_norm
                                    improved = True
                                if improved:
                                    stall = 0
                                else:
                                    stall += 1
                                if patience > 0 and stall >= patience:
                                    for k in range(12):
                                        P[k] = P_best[k]
                                    zncc = best_zncc
                                    ctrl[0] = STATUS_STALLED
                                elif not compose_warp(P, dP, M):
                                    ctrl[0] = STATUS_SINGULAR
            cuda.syncthreads()
            if ctrl[0] >= 0:
                status = ctrl[0]
                break
        if tid == 0:
            if ctrl[0] >= 0:
                status = ctrl[0]
            for k in range(12):
                P_io[n, k] = P[k]
            n_iter_out[n] = it_done
            status_out[n] = status
            zncc_out[n] = (
                zncc if (status == STATUS_CONVERGED or status == STATUS_STALLED or status == STATUS_MAX_ITER) else math.nan
            )

    @cuda.jit(device=True)
    def build_h3(H, bf2, mu, corr, H3, Hd, Hinv):
        """``Hinv = (2 (H_tt - corr I) / bf2 + mu I + LM damping)^-1``; False when singular."""
        for i in range(3):
            for j in range(3):
                H3[i, j] = H[9 + i, 9 + j] * 2.0 / bf2
            limit_i = (1.0 - MIN_CORRECTED_FRACTION) * H[9 + i, 9 + i]
            c = corr if corr < limit_i else limit_i
            H3[i, i] -= c * 2.0 / bf2
            H3[i, i] += mu
        max_diag = max(H3[0, 0], max(H3[1, 1], H3[2, 2]))
        if not (max_diag > 0.0):
            return False
        for i in range(3):
            for j in range(3):
                Hd[i, j] = H3[i, j]
            Hd[i, i] += LM_DAMPING_3DOF * max_diag
        a00 = Hd[0, 0]
        a01 = Hd[0, 1]
        a02 = Hd[0, 2]
        a10 = Hd[1, 0]
        a11 = Hd[1, 1]
        a12 = Hd[1, 2]
        a20 = Hd[2, 0]
        a21 = Hd[2, 1]
        a22 = Hd[2, 2]
        det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
        if det == 0.0 or math.isnan(det):
            return False
        inv = 1.0 / det
        Hinv[0, 0] = (a11 * a22 - a12 * a21) * inv
        Hinv[0, 1] = (a02 * a21 - a01 * a22) * inv
        Hinv[0, 2] = (a01 * a12 - a02 * a11) * inv
        Hinv[1, 0] = (a12 * a20 - a10 * a22) * inv
        Hinv[1, 1] = (a00 * a22 - a02 * a20) * inv
        Hinv[1, 2] = (a02 * a10 - a00 * a12) * inv
        Hinv[2, 0] = (a10 * a21 - a11 * a20) * inv
        Hinv[2, 1] = (a01 * a20 - a00 * a21) * inv
        Hinv[2, 2] = (a00 * a11 - a01 * a10) * inv
        return True

    @cuda.jit
    def icgn3_kernel(
        idx,
        coords,
        U_io,
        F_fixed,
        vdual,
        hx,
        hy,
        hz,
        stride,
        f,
        gx,
        gy,
        gz,
        stored,
        mask,
        g,
        mode,
        H_all,
        meanf_all,
        bottomf_all,
        mu,
        n_full,
        noise_gain,
        tol,
        dp_tol,
        max_iter,
        patience,
        predictive,
        gbuf,
        n_iter_out,
        status_out,
        zncc_out,
    ):
        blk = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        n = idx[blk]
        red = cuda.shared.array((RED_Q, BLOCK), float32)
        P = cuda.shared.array(12, float64)
        st = cuda.shared.array(8, float64)
        ctrl = cuda.shared.array(4, int32)
        warp = cuda.shared.array(12, float32)
        H3 = cuda.shared.array((3, 3), float64)
        Hd = cuda.shared.array((3, 3), float64)
        Hinv = cuda.shared.array((3, 3), float64)
        tb = cuda.shared.array(3, float64)
        dt = cuda.shared.array(3, float64)
        if tid == 0:
            finite = True
            for i in range(3):
                for j in range(3):
                    P[3 * i + j] = F_fixed[n, i, j]
                    if math.isnan(P[3 * i + j]) or math.isinf(P[3 * i + j]):
                        finite = False
            for k in range(3):
                P[9 + k] = U_io[n, k]
                if math.isnan(P[9 + k]) or math.isinf(P[9 + k]):
                    finite = False
            ctrl[0] = -1 if finite else STATUS_NAN
            if finite:
                bf2 = bottomf_all[n] * bottomf_all[n]
                if bf2 < 1e-30:
                    bf2 = 1e-30
                st[7] = bf2
                if not build_h3(H_all[n], bf2, mu, 0.0, H3, Hd, Hinv):
                    ctrl[0] = STATUS_SINGULAR
        cuda.syncthreads()
        x0 = coords[n, 0]
        y0 = coords[n, 1]
        z0 = coords[n, 2]
        nz = g.shape[0]
        ny = g.shape[1]
        nx = g.shape[2]
        sx = (2 * hx) // stride + 1
        sy = (2 * hy) // stride + 1
        sz = (2 * hz) // stride + 1
        S = sx * sy * sz
        norm_init = -1.0
        best_dn = 1e300
        last_dn = 1e300
        dn_prev = -1.0
        stall = 0
        zncc = math.nan
        it_done = 0
        status = STATUS_MAX_ITER
        if ctrl[0] >= 0:
            status = ctrl[0]
        # the warp coefficients never change: F is fixed, only the translation moves
        if tid == 0:
            warp[0] = F32(1.0 + P[0])
            warp[1] = F32(P[1])
            warp[2] = F32(P[2])
            warp[3] = F32(P[3])
            warp[4] = F32(1.0 + P[4])
            warp[5] = F32(P[5])
            warp[6] = F32(P[6])
            warp[7] = F32(P[7])
            warp[8] = F32(1.0 + P[8])
        for it in range(1, max_iter + 1):
            if ctrl[0] >= 0:
                break
            it_done = it
            if tid == 0:
                warp[9] = F32(x0 + P[9])
                warp[10] = F32(y0 + P[10])
                warp[11] = F32(z0 + P[11])
            cuda.syncthreads()
            s1 = F32(0.0)
            s2 = F32(0.0)
            s1f = F32(0.0)
            s2f = F32(0.0)
            nv = F32(0.0)
            nref = F32(0.0)
            oob = F32(0.0)
            for v in range(tid, S, BLOCK):
                iz_ = v // (sy * sx)
                rem = v - iz_ * (sy * sx)
                iy_ = rem // sx
                ix_ = rem - iy_ * sx
                dx = -hx + stride * ix_
                dy = -hy + stride * iy_
                dz = -hz + stride * iz_
                zz = z0 + dz
                yy = y0 + dy
                xx = x0 + dx
                if mask[zz, yy, xx] == 0:
                    gbuf[blk, v] = math.nan
                    continue
                nref += F32(1.0)
                X = F32(dx)
                Y = F32(dy)
                Z = F32(dz)
                xw = warp[0] * X + warp[1] * Y + warp[2] * Z + warp[9]
                yw = warp[3] * X + warp[4] * Y + warp[5] * Z + warp[10]
                zw = warp[6] * X + warp[7] * Y + warp[8] * Z + warp[11]
                if not inside(zw, yw, xw, nz, ny, nx):
                    oob = F32(1.0)
                    gbuf[blk, v] = math.nan
                    continue
                val = sample(g, zw, yw, xw, mode)
                gbuf[blk, v] = val
                if math.isnan(val):
                    continue
                fv = f[zz, yy, xx]
                nv += F32(1.0)
                s1 += val
                s2 += val * val
                s1f += fv
                s2f += fv * fv
            red[0, tid] = s1
            red[1, tid] = s2
            red[2, tid] = s1f
            red[3, tid] = s2f
            red[4, tid] = nv
            red[5, tid] = nref
            red[6, tid] = oob
            block_reduce(red, 7)
            if tid == 0:
                n_valid = float64(red[4, 0])
                n_ref = float64(red[5, 0])
                if red[6, 0] > F32(0.0):
                    ctrl[0] = STATUS_OUT_OF_BOUNDS
                elif n_valid < MIN_SUBSET_VOXELS or n_valid < MIN_VALID_FRACTION * n_ref:
                    ctrl[0] = STATUS_INVALID_SUBSET
                else:
                    meang = float64(red[0, 0]) / n_valid
                    ssg = float64(red[1, 0]) - n_valid * meang * meang
                    if ssg < 0.0:
                        ssg = 0.0
                    bottomg = math.sqrt(max(ssg, 1e-30))
                    meanf_d = float64(red[2, 0]) / n_valid
                    ssf = float64(red[3, 0]) - n_valid * meanf_d * meanf_d
                    if ssf < 0.0:
                        ssf = 0.0
                    bottomf_d = math.sqrt(max(ssf, 1e-30))
                    st[0] = meanf_d
                    st[1] = bottomf_d
                    st[2] = meang
                    st[3] = bottomg
                    st[4] = 1.0 / bottomf_d
                    st[5] = 1.0 / bottomg
                    st[6] = n_valid
            cuda.syncthreads()
            if ctrl[0] >= 0:
                status = ctrl[0]
                zncc = math.nan
                break
            meanf_d32 = F32(st[0])
            meang32 = F32(st[2])
            inv_bf32 = F32(st[4])
            inv_bg32 = F32(st[5])
            b0 = F32(0.0)
            b1 = F32(0.0)
            b2 = F32(0.0)
            scc = F32(0.0)
            for v in range(tid, S, BLOCK):
                gv = gbuf[blk, v]
                if math.isnan(gv):
                    continue
                iz_ = v // (sy * sx)
                rem = v - iz_ * (sy * sx)
                iy_ = rem // sx
                ix_ = rem - iy_ * sx
                dx = -hx + stride * ix_
                dy = -hy + stride * iy_
                dz = -hz + stride * iz_
                zz = z0 + dz
                yy = y0 + dy
                xx = x0 + dx
                fd = f[zz, yy, xx] - meanf_d32
                gd = gv - meang32
                scc += fd * gd
                res = fd * inv_bf32 - gd * inv_bg32
                dfx, dfy, dfz = grad_at(f, gx, gy, gz, stored, zz, yy, xx)
                b0 += dfx * res
                b1 += dfy * res
                b2 += dfz * res
            red[0, tid] = b0
            red[1, tid] = b1
            red[2, tid] = b2
            red[3, tid] = scc
            block_reduce(red, 4)
            if tid == 0:
                bottomf_d = st[1]
                bf2 = st[7]
                norm_abs = 0.0
                for k in range(3):
                    bk = float64(red[k, 0]) * bottomf_d
                    tb[k] = bk * 2.0 / bf2 + mu * (P[9 + k] - U_io[n, k] - vdual[n, k])
                    norm_abs += tb[k] * tb[k]
                norm_abs = math.sqrt(norm_abs)
                zncc_local = float64(red[3, 0]) * st[4] * st[5]
                if math.isnan(norm_abs) or math.isinf(norm_abs):
                    ctrl[0] = STATUS_NAN
                else:
                    if norm_init < 0.0:
                        norm_init = norm_abs
                    norm_rel = norm_abs / norm_init if norm_init > 1e-300 else 0.0
                    zncc = zncc_local
                    if norm_rel < tol or norm_abs < mu * 1e-4:
                        ctrl[0] = STATUS_CONVERGED
                    else:
                        if noise_gain > 0.0 and last_dn < NOISE_CORR_STEP:
                            s2n = max(1.0 - zncc, 0.0) * bottomf_d * bottomf_d / st[6]
                            if not build_h3(H_all[n], bf2, mu, noise_gain * s2n * st[6], H3, Hd, Hinv):
                                build_h3(H_all[n], bf2, mu, 0.0, H3, Hd, Hinv)
                        for i in range(3):
                            s = 0.0
                            for j in range(3):
                                s += Hinv[i, j] * tb[j]
                            dt[i] = -s
                        dn = math.sqrt(dt[0] * dt[0] + dt[1] * dt[1] + dt[2] * dt[2])
                        if math.isnan(dn) or math.isinf(dn):
                            ctrl[0] = STATUS_NAN
                        else:
                            last_dn = dn
                            if dn < dp_tol:
                                ctrl[0] = STATUS_CONVERGED
                            else:
                                stop_after = False
                                if (
                                    predictive
                                    and dn_prev > 0.0
                                    and dn < PREDICT_CONTRACTION * dn_prev
                                    and dn * dn < dp_tol * dn_prev
                                ):
                                    stop_after = True
                                else:
                                    dn_prev = dn
                                    if dn < best_dn * STALL_STEP_DECAY:
                                        best_dn = dn
                                        stall = 0
                                    else:
                                        stall += 1
                                if patience > 0 and stall >= patience and not stop_after:
                                    ctrl[0] = STATUS_STALLED
                                else:
                                    a00 = 1.0 + P[0]
                                    a01 = P[1]
                                    a02 = P[2]
                                    a10 = P[3]
                                    a11 = 1.0 + P[4]
                                    a12 = P[5]
                                    a20 = P[6]
                                    a21 = P[7]
                                    a22 = 1.0 + P[8]
                                    P[9] -= a00 * dt[0] + a01 * dt[1] + a02 * dt[2]
                                    P[10] -= a10 * dt[0] + a11 * dt[1] + a12 * dt[2]
                                    P[11] -= a20 * dt[0] + a21 * dt[1] + a22 * dt[2]
                                    if stop_after:
                                        ctrl[0] = STATUS_CONVERGED
            cuda.syncthreads()
            if ctrl[0] >= 0:
                status = ctrl[0]
                break
        if tid == 0:
            if ctrl[0] >= 0:
                status = ctrl[0]
            for k in range(3):
                U_io[n, k] = P[9 + k]
            n_iter_out[n] = it_done
            status_out[n] = status
            zncc_out[n] = (
                zncc if (status == STATUS_CONVERGED or status == STATUS_STALLED or status == STATUS_MAX_ITER) else math.nan
            )

    _kernels["icgn3"] = icgn3_kernel

    @cuda.jit
    def precompute_kernel(coords, hx, hy, hz, stride, f, gx, gy, gz, stored, mask, H_out, sums_out):
        """Per node: the 78 upper-triangle entries of ``sum J J^T``, ``sum f``, ``sum f^2``, ``n_valid``, and a bounds flag.

        ``H_out[n, 12, 12]`` is filled symmetrically in float64 by thread 0 after a float32 block reduction
        of 78 + 3 partial sums (done in six rounds of at most 16 quantities to keep the shared memory small).
        """
        n = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        red = cuda.shared.array((RED_Q, BLOCK), float32)
        acc = cuda.local.array(81, float32)
        x0 = coords[n, 0]
        y0 = coords[n, 1]
        z0 = coords[n, 2]
        nz = f.shape[0]
        ny = f.shape[1]
        nx = f.shape[2]
        if x0 - hx < 0 or x0 + hx >= nx or y0 - hy < 0 or y0 + hy >= ny or z0 - hz < 0 or z0 + hz >= nz:
            if tid == 0:
                sums_out[n, 0] = 0.0
                sums_out[n, 1] = 0.0
                sums_out[n, 2] = 0.0
                sums_out[n, 3] = 1.0  # out of bounds
            return
        for q in range(81):
            acc[q] = F32(0.0)
        sx = (2 * hx) // stride + 1
        sy = (2 * hy) // stride + 1
        sz = (2 * hz) // stride + 1
        S = sx * sy * sz
        sd = cuda.local.array(12, float32)
        for v in range(tid, S, BLOCK):
            iz_ = v // (sy * sx)
            rem = v - iz_ * (sy * sx)
            iy_ = rem // sx
            ix_ = rem - iy_ * sx
            dx = -hx + stride * ix_
            dy = -hy + stride * iy_
            dz = -hz + stride * iz_
            zz = z0 + dz
            yy = y0 + dy
            xx = x0 + dx
            if mask[zz, yy, xx] == 0:
                continue
            fv = f[zz, yy, xx]
            gxv, gyv, gzv = grad_at(f, gx, gy, gz, stored, zz, yy, xx)
            X = F32(dx)
            Y = F32(dy)
            Z = F32(dz)
            sd[0] = gxv * X
            sd[1] = gxv * Y
            sd[2] = gxv * Z
            sd[3] = gyv * X
            sd[4] = gyv * Y
            sd[5] = gyv * Z
            sd[6] = gzv * X
            sd[7] = gzv * Y
            sd[8] = gzv * Z
            sd[9] = gxv
            sd[10] = gyv
            sd[11] = gzv
            q = 0
            for a in range(12):
                sa = sd[a]
                for b in range(a, 12):
                    acc[q] += sa * sd[b]
                    q += 1
            acc[78] += fv
            acc[79] += fv * fv
            acc[80] += F32(1.0)
        # reduce the 81 partial sums in rounds of RED_Q quantities
        for base in range(0, 81, RED_Q):
            nq = min(RED_Q, 81 - base)
            for q in range(nq):
                red[q, tid] = acc[base + q]
            block_reduce(red, nq)
            if tid == 0:
                for q in range(nq):
                    acc[base + q] = red[q, 0]
            cuda.syncthreads()
        if tid == 0:
            q = 0
            for a in range(12):
                for b in range(a, 12):
                    H_out[n, a, b] = float64(acc[q])
                    H_out[n, b, a] = float64(acc[q])
                    q += 1
            sums_out[n, 0] = float64(acc[78])
            sums_out[n, 1] = float64(acc[79])
            sums_out[n, 2] = float64(acc[80])
            sums_out[n, 3] = 0.0

    _kernels["precompute"] = precompute_kernel

    @cuda.jit
    def zncc_direct_kernel(f, g, coords, hx, hy, hz, rx, ry, rz, wx0, wy0, wz0, t_ok, out):
        """ZNCC maps ``out[n, dz, dy, dx]`` for a batch of nodes: one block per node, threads over offsets.

        The centred template lives in shared memory (``TPL_MAX`` floats); each offset's window
        sums (g, g^2, f g) are accumulated directly over the template voxels in float32.
        """
        n = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        tpl = cuda.shared.array(TPL_MAX, float32)
        st = cuda.shared.array(4, float32)
        sx = 2 * hx + 1
        sy = 2 * hy + 1
        sz = 2 * hz + 1
        n_tpl = sx * sy * sz
        vx = 2 * rx + 1
        vy = 2 * ry + 1
        vz = 2 * rz + 1
        n_off = vx * vy * vz
        x0 = coords[n, 0]
        y0 = coords[n, 1]
        z0 = coords[n, 2]
        if not t_ok[n]:
            for o in range(tid, n_off, BLOCK):
                a = o // (vy * vx)
                r = o - a * vy * vx
                b = r // vx
                c = r - b * vx
                out[n, a, b, c] = F32(-2.0)
            return
        # template sum (block reduction through the shared stats slot)
        red = cuda.shared.array((RED_Q, BLOCK), float32)
        s = F32(0.0)
        for v in range(tid, n_tpl, BLOCK):
            k = v // (sy * sx)
            r = v - k * sy * sx
            j = r // sx
            i = r - j * sx
            val = f[z0 - hz + k, y0 - hy + j, x0 - hx + i]
            tpl[v] = val
            s += val
        red[0, tid] = s
        block_reduce(red, 1)
        mf = red[0, 0] / F32(n_tpl)
        cuda.syncthreads()
        ss = F32(0.0)
        for v in range(tid, n_tpl, BLOCK):
            val = tpl[v] - mf
            tpl[v] = val
            ss += val * val
        red[0, tid] = ss
        block_reduce(red, 1)
        if tid == 0:
            st[0] = red[0, 0]
        cuda.syncthreads()
        ssf = st[0]
        if ssf < F32(1e-20):
            for o in range(tid, n_off, BLOCK):
                a = o // (vy * vx)
                r = o - a * vy * vx
                b = r // vx
                c = r - b * vx
                out[n, a, b, c] = F32(-2.0)
            return
        zb = wz0[n]
        yb = wy0[n]
        xb = wx0[n]
        for o in range(tid, n_off, BLOCK):
            dz = o // (vy * vx)
            r = o - dz * vy * vx
            dy = r // vx
            dx = r - dy * vx
            sg = F32(0.0)
            sgg = F32(0.0)
            sfg = F32(0.0)
            v = 0
            for k in range(sz):
                for j in range(sy):
                    for i in range(sx):
                        gv = g[zb + dz + k, yb + dy + j, xb + dx + i]
                        sg += gv
                        sgg += gv * gv
                        sfg += tpl[v] * gv
                        v += 1
            var = sgg - sg * sg / F32(n_tpl)
            if var < F32(1e-20):
                out[n, dz, dy, dx] = F32(0.0)
            else:
                val = sfg / math.sqrt(var * ssf)
                if val > F32(1.0):
                    val = F32(1.0)
                elif val < F32(-1.0):
                    val = F32(-1.0)
                out[n, dz, dy, dx] = val

    _kernels["zncc_direct"] = zncc_direct_kernel
    _kernels["icgn12"] = icgn12_kernel
    return _kernels


# --------------------------------------------------------------------------- Python wrappers
def icgn_12dof_cuda(
    coords,
    P0,
    hx,
    hy,
    hz,
    f,
    gx,
    gy,
    gz,
    mask,
    g,
    mode,
    L_all,
    meanf_all,
    bottomf_all,
    valid,
    tol,
    dp_tol,
    max_iter,
    patience,
    stride=1,
    H_all=None,
    pattern=None,
    noise_gain=0.0,
    predictive=True,
    chunk: int = CHUNK_NODES,
    progress_fn=None,
):
    """Same contract as :func:`al_dvc.solver.numba_kernels.icgn_12dof_parallel`, on the GPU."""
    from numba import cuda

    kern = _build_kernels()["icgn12"]
    N = coords.shape[0]
    P_out = np.array(P0, dtype=np.float64).copy()
    n_iter = np.zeros(N, dtype=np.int32)
    status = np.full(N, STATUS_SKIPPED, dtype=np.int8)
    zncc = np.full(N, np.nan)
    status[~np.asarray(valid, dtype=bool)] = STATUS_INVALID_SUBSET
    idx_all = np.flatnonzero(np.asarray(valid, dtype=bool)).astype(np.int64)
    if idx_all.size == 0:
        return P_out, n_iter, status, zncc
    stored = gx.shape == f.shape
    d_f = _cache.get(np.ascontiguousarray(f, dtype=np.float32))
    d_gx = _cache.get(np.ascontiguousarray(gx, dtype=np.float32))
    d_gy = _cache.get(np.ascontiguousarray(gy, dtype=np.float32))
    d_gz = _cache.get(np.ascontiguousarray(gz, dtype=np.float32))
    d_mask = _cache.get(np.ascontiguousarray(mask, dtype=np.uint8))
    d_g = _cache.get(np.ascontiguousarray(g, dtype=np.float32))
    d_coords = cuda.to_device(np.ascontiguousarray(coords, dtype=np.int64))
    d_P = cuda.to_device(P_out)
    d_L = cuda.to_device(np.ascontiguousarray(L_all, dtype=np.float64))
    if H_all is None or pattern is None or noise_gain <= 0.0:
        H_arr = np.zeros((1, 12, 12)) if H_all is None else np.ascontiguousarray(H_all, dtype=np.float64)
        if H_arr.shape[0] != N:
            H_arr = np.zeros((N, 12, 12))
        pattern = np.zeros((12, 12))
        noise_gain = 0.0
    else:
        H_arr = np.ascontiguousarray(H_all, dtype=np.float64)
    d_H = cuda.to_device(H_arr)
    d_pattern = cuda.to_device(np.ascontiguousarray(pattern, dtype=np.float64))
    d_meanf = cuda.to_device(np.ascontiguousarray(meanf_all, dtype=np.float64))
    d_bottomf = cuda.to_device(np.ascontiguousarray(bottomf_all, dtype=np.float64))
    d_niter = cuda.to_device(n_iter)
    d_status = cuda.to_device(np.zeros(N, dtype=np.int32))
    d_zncc = cuda.to_device(zncc)
    S = ((2 * hx) // stride + 1) * ((2 * hy) // stride + 1) * ((2 * hz) // stride + 1)
    chunk = max(1, int(chunk))
    gbuf = cuda.device_array((min(chunk, idx_all.size), S), np.float32)
    for start in range(0, idx_all.size, chunk):
        ids = idx_all[start : start + chunk]
        d_idx = cuda.to_device(ids)
        kern[ids.size, BLOCK](
            d_idx,
            d_coords,
            d_P,
            int(hx),
            int(hy),
            int(hz),
            int(stride),
            d_f,
            d_gx,
            d_gy,
            d_gz,
            bool(stored),
            d_mask,
            d_g,
            int(mode),
            d_L,
            d_H,
            d_meanf,
            d_bottomf,
            d_pattern,
            float(noise_gain),
            float(tol),
            float(dp_tol),
            int(max_iter),
            int(patience),
            bool(predictive),
            gbuf,
            d_niter,
            d_status,
            d_zncc,
        )
        cuda.synchronize()
        if progress_fn is not None:
            progress_fn(min(1.0, (start + ids.size) / idx_all.size))
    P_out = d_P.copy_to_host()
    n_iter = d_niter.copy_to_host()
    st = d_status.copy_to_host()
    zncc = d_zncc.copy_to_host()
    status[idx_all] = st[idx_all].astype(np.int8)
    return P_out, n_iter, status, zncc


def icgn_3dof_cuda(
    coords,
    U_old,
    F_fixed,
    vdual,
    hx,
    hy,
    hz,
    f,
    gx,
    gy,
    gz,
    mask,
    g,
    mode,
    H_all,
    meanf_all,
    bottomf_all,
    valid,
    mu,
    tol,
    dp_tol,
    max_iter,
    patience,
    stride=1,
    n_full=0.0,
    noise_gain=0.0,
    predictive=True,
    chunk: int = CHUNK_NODES,
    progress_fn=None,
):
    """Same contract as :func:`al_dvc.solver.numba_kernels.icgn_3dof_parallel`, on the GPU."""
    from numba import cuda

    kern = _build_kernels()["icgn3"]
    N = coords.shape[0]
    U_out = np.array(U_old, dtype=np.float64).copy()
    n_iter = np.zeros(N, dtype=np.int32)
    status = np.full(N, STATUS_SKIPPED, dtype=np.int8)
    zncc = np.full(N, np.nan)
    status[~np.asarray(valid, dtype=bool)] = STATUS_INVALID_SUBSET
    idx_all = np.flatnonzero(np.asarray(valid, dtype=bool)).astype(np.int64)
    if idx_all.size == 0:
        return U_out, n_iter, status, zncc
    stored = gx.shape == f.shape
    d_f = _cache.get(np.ascontiguousarray(f, dtype=np.float32))
    d_gx = _cache.get(np.ascontiguousarray(gx, dtype=np.float32))
    d_gy = _cache.get(np.ascontiguousarray(gy, dtype=np.float32))
    d_gz = _cache.get(np.ascontiguousarray(gz, dtype=np.float32))
    d_mask = _cache.get(np.ascontiguousarray(mask, dtype=np.uint8))
    d_g = _cache.get(np.ascontiguousarray(g, dtype=np.float32))
    d_coords = cuda.to_device(np.ascontiguousarray(coords, dtype=np.int64))
    d_U = cuda.to_device(U_out)
    d_F = cuda.to_device(np.ascontiguousarray(F_fixed, dtype=np.float64).reshape(N, 3, 3))
    d_v = cuda.to_device(np.ascontiguousarray(vdual, dtype=np.float64).reshape(N, 3))
    d_H = cuda.to_device(np.ascontiguousarray(H_all, dtype=np.float64))
    d_meanf = cuda.to_device(np.ascontiguousarray(meanf_all, dtype=np.float64))
    d_bottomf = cuda.to_device(np.ascontiguousarray(bottomf_all, dtype=np.float64))
    d_niter = cuda.to_device(n_iter)
    d_status = cuda.to_device(np.zeros(N, dtype=np.int32))
    d_zncc = cuda.to_device(zncc)
    S = ((2 * hx) // stride + 1) * ((2 * hy) // stride + 1) * ((2 * hz) // stride + 1)
    chunk = max(1, int(chunk))
    gbuf = cuda.device_array((min(chunk, idx_all.size), S), np.float32)
    for start in range(0, idx_all.size, chunk):
        ids = idx_all[start : start + chunk]
        d_idx = cuda.to_device(ids)
        kern[ids.size, BLOCK](
            d_idx,
            d_coords,
            d_U,
            d_F,
            d_v,
            int(hx),
            int(hy),
            int(hz),
            int(stride),
            d_f,
            d_gx,
            d_gy,
            d_gz,
            bool(stored),
            d_mask,
            d_g,
            int(mode),
            d_H,
            d_meanf,
            d_bottomf,
            float(mu),
            float(n_full),
            float(noise_gain if noise_gain > 0.0 else 0.0),
            float(tol),
            float(dp_tol),
            int(max_iter),
            int(patience),
            bool(predictive),
            gbuf,
            d_niter,
            d_status,
            d_zncc,
        )
        cuda.synchronize()
        if progress_fn is not None:
            progress_fn(min(1.0, (start + ids.size) / idx_all.size))
    U_out = d_U.copy_to_host()
    n_iter = d_niter.copy_to_host()
    st = d_status.copy_to_host()
    zncc = d_zncc.copy_to_host()
    status[idx_all] = st[idx_all].astype(np.int8)
    return U_out, n_iter, status, zncc


def precompute_nodes_cuda(coords, hx, hy, hz, f, gx, gy, gz, mask, min_valid_ratio, cond_max, stride=1, chunk: int = CHUNK_NODES):
    """Same contract as :func:`al_dvc.solver.numba_kernels.precompute_nodes`, on the GPU.

    The 12x12 sums are accumulated in float32 (relative error ~1e-6, i.e. far below the
    Hessian conditioning that matters); the Cholesky factors and the conditioning test run
    on the CPU in float64 exactly as in the Numba kernel.
    """
    from numba import cuda

    from .numba_kernels import _cholesky12_batch

    kern = _build_kernels()["precompute"]
    N = coords.shape[0]
    stored = gx.shape == f.shape
    d_f = _cache.get(np.ascontiguousarray(f, dtype=np.float32))
    d_gx = _cache.get(np.ascontiguousarray(gx, dtype=np.float32))
    d_gy = _cache.get(np.ascontiguousarray(gy, dtype=np.float32))
    d_gz = _cache.get(np.ascontiguousarray(gz, dtype=np.float32))
    d_mask = _cache.get(np.ascontiguousarray(mask, dtype=np.uint8))
    d_coords = cuda.to_device(np.ascontiguousarray(coords, dtype=np.int64))
    H_all = np.zeros((N, 12, 12))
    sums = np.zeros((N, 4))
    d_H = cuda.to_device(H_all)
    d_sums = cuda.to_device(sums)
    chunk = max(1, int(chunk))
    for start in range(0, N, chunk):
        stop = min(N, start + chunk)
        kern[stop - start, BLOCK](
            d_coords[start:stop],
            int(hx),
            int(hy),
            int(hz),
            int(stride),
            d_f,
            d_gx,
            d_gy,
            d_gz,
            bool(stored),
            d_mask,
            d_H[start:stop],
            d_sums[start:stop],
        )
        cuda.synchronize()
    H_all = d_H.copy_to_host()
    sums = d_sums.copy_to_host()
    total = ((2 * hx) // stride + 1) * ((2 * hy) // stride + 1) * ((2 * hz) // stride + 1)
    n_valid = sums[:, 2].astype(np.int64)
    meanf = np.where(n_valid > 0, sums[:, 0] / np.maximum(n_valid, 1), 0.0)
    ssf = sums[:, 1] - n_valid * meanf * meanf
    ssf = np.where(ssf < 0.0, 0.0, ssf)
    bottomf = np.sqrt(np.maximum(ssf, 1e-30))
    ok = (sums[:, 3] == 0.0) & (n_valid >= MIN_SUBSET_VOXELS) & (n_valid >= min_valid_ratio * total) & (bottomf >= 1e-10)
    meanf = np.where(ok | (n_valid > 0), meanf, 0.0)
    bottomf = np.where(ok | (n_valid > 0), bottomf, 1.0)
    L_all, chol_ok = _cholesky12_batch(H_all, ok, float(cond_max))
    valid = ok & chol_ok
    return H_all, L_all, meanf, bottomf, n_valid, valid


def zncc_direct_cuda(f, g, coords, hx, hy, hz, rx, ry, rz, wx0, wy0, wz0, t_ok, out) -> None:
    """Same contract as ``integer_search._zncc_direct`` (fills ``out`` in place), on the GPU.

    Templates larger than ``TPL_MAX`` voxels fall back to the CPU kernel.
    """
    from numba import cuda

    from .integer_search import _zncc_direct

    n_tpl = (2 * hx + 1) * (2 * hy + 1) * (2 * hz + 1)
    if n_tpl > TPL_MAX or coords.shape[0] == 0:
        _zncc_direct(f, g, coords, hx, hy, hz, rx, ry, rz, wx0, wy0, wz0, t_ok, out)
        return
    kern = _build_kernels()["zncc_direct"]
    d_f = _cache.get(np.ascontiguousarray(f, dtype=np.float32))
    d_g = _cache.get(np.ascontiguousarray(g, dtype=np.float32))
    d_coords = cuda.to_device(np.ascontiguousarray(coords, dtype=np.int64))
    d_wx0 = cuda.to_device(np.ascontiguousarray(wx0, dtype=np.int64))
    d_wy0 = cuda.to_device(np.ascontiguousarray(wy0, dtype=np.int64))
    d_wz0 = cuda.to_device(np.ascontiguousarray(wz0, dtype=np.int64))
    d_ok = cuda.to_device(np.ascontiguousarray(t_ok, dtype=np.bool_))
    d_out = cuda.device_array(out.shape, np.float32)
    N = coords.shape[0]
    for start in range(0, N, CHUNK_NODES):
        stop = min(N, start + CHUNK_NODES)
        kern[stop - start, BLOCK](
            d_f,
            d_g,
            d_coords[start:stop],
            int(hx),
            int(hy),
            int(hz),
            int(rx),
            int(ry),
            int(rz),
            d_wx0[start:stop],
            d_wy0[start:stop],
            d_wz0[start:stop],
            d_ok[start:stop],
            d_out[start:stop],
        )
        cuda.synchronize()
    out[...] = d_out.copy_to_host()
