"""Initial guess assembly and cleaning (MATLAB ``RemoveOutliers3.m`` + ``Init3.m``)."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ..core.config import DVCPara
from ..core.data_structures import DVCMesh
from ..utils.inpaint import fill_nan_grid
from ..utils.outlier_detection import universal_median_test
from .integer_search import (
    DEFAULT_INIT_SUBSET,
    _scatter_search,
    ncc_search_expanding,
    phase_correlation_shift,
    pyramid_search,
)

logger = logging.getLogger(__name__)


def _ncc_backend(para: DVCPara) -> str:
    """``'cuda'`` when the parameter set resolves to the GPU backend, else ``'numba'``."""
    from .local_icgn import resolve_backend

    try:
        return resolve_backend(para)
    except RuntimeError:
        return "numba"


def compute_initial_guess(
    f: NDArray[np.float32],
    g: NDArray[np.float32],
    mesh: DVCMesh,
    para: DVCPara,
    previous: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """Return ``U0`` (N, 3) and an info dict according to ``para.init_guess_method``."""
    N = mesh.n_nodes
    method = para.init_guess_method
    if method == "zero":
        return np.zeros((N, 3)), {"method": "zero"}
    if method == "previous":
        if previous is not None and previous.shape == (N, 3) and np.all(np.isfinite(previous)):
            return previous.copy(), {"method": "previous"}
        method = "pyramid"

    coords = np.round(mesh.coordinates).astype(np.int64)
    if para.init_subset is not None:
        subset = para.init_subset
    else:
        # the NCC only has to locate the integer peak; a 17^3 template is ample
        # for speckle and keeps the search cost independent of winsize
        subset = tuple(min(int(w), DEFAULT_INIT_SUBSET) for w in para.winsize)
    valid = mesh.node_valid if mesh.node_valid.size == N else None  # search only nodes with a usable reference subset
    shift = None
    if para.global_shift:
        shift = phase_correlation_shift(f, g, para.voi)
        logger.info("Global phase-correlation shift (dx, dy, dz) = %s voxels", np.array2string(shift, precision=1))

    if method == "pyramid":
        info = pyramid_search(
            f,
            g,
            coords,
            mesh.grid_shape,
            subset,
            para.search_radius,
            para.pyramid_levels,
            global_shift=shift,
            outlier_threshold=para.init_outlier_threshold,
            auto_expand=para.ncc_auto_expand,
            max_expand=para.ncc_max_expand,
            fine_radius=int(getattr(para, "pyramid_fine_radius", 2)),
            backend=_ncc_backend(para),
            valid=valid,
        )
        disp = info["disp"]
        ok = info.get("ok", np.ones(N, dtype=bool))
        pce = info.get("pce", np.full(N, np.nan))
        info["method"] = "pyramid"
    else:
        shift0 = None if shift is None else np.tile(np.round(shift).astype(np.int64), (N, 1))
        active = np.arange(N) if valid is None or not valid.any() else np.flatnonzero(valid)
        res = ncc_search_expanding(
            f,
            g,
            coords[active],
            subset,
            tuple(int(r) for r in para.search_radius),
            None if shift0 is None else shift0[active],
            para.ncc_auto_expand,
            para.ncc_max_expand,
            backend=_ncc_backend(para),
        )
        res = _scatter_search(res, active, N)
        disp = res["disp"]
        ok = res["ok"]
        pce = res["pce"]
        info = {
            "method": "ncc",
            "radius": res["radius"],
            "expansions": res["expansions"],
            "cc": res["cc"],
            "pce": pce,
            "ok": ok,
            "clipped": res["clipped"],
        }

    U0, bad = clean_initial_guess(disp, ok, pce, mesh, para)
    info["n_bad"] = int(bad.sum())
    info["global_shift"] = shift
    return U0, info


def clean_initial_guess(
    disp: NDArray[np.float64],
    ok: NDArray[np.bool_],
    pce: NDArray[np.float64],
    mesh: DVCMesh,
    para: DVCPara,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Reject failed / low-quality / outlying nodes and inpaint them."""
    N = mesh.n_nodes
    d = np.array(disp, dtype=np.float64).reshape(N, 3)
    bad = ~np.asarray(ok, dtype=bool) | ~np.all(np.isfinite(d), axis=1)
    if mesh.node_valid.size == N:
        bad |= ~mesh.node_valid
    if para.init_min_pce > 0 and pce is not None:
        p = np.asarray(pce, dtype=np.float64)
        bad |= np.isfinite(p) & (p < para.init_min_pce)
    d[bad] = np.nan
    grid = mesh.grid_shape
    if para.init_outlier_threshold > 0 and (~bad).sum() > 27:
        flag = universal_median_test(d.reshape(grid + (3,)), (~bad).reshape(grid), para.init_outlier_threshold)
        bad |= flag.ravel()
        d[bad] = np.nan
    if bad.all():
        logger.warning("Initial guess failed at every node; starting from zero displacement.")
        return np.zeros((N, 3)), bad
    out = np.empty_like(d)
    for c in range(3):
        out[:, c] = fill_nan_grid(d[:, c].reshape(grid)).ravel()
    logger.info("Initial guess: %d/%d nodes replaced by inpainting (%.1f%%)", int(bad.sum()), N, 100 * bad.mean())
    return out, bad
