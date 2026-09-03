"""Automatic selection of the ADMM penalty ``beta`` (L-curve, MATLAB Section 5).

For every candidate ``beta_k = beta_range[k] * mean(h)^2 * mu`` the global
step is solved with zero duals and ``Err1 = |u1 - u_hat|``,
``Err2 = |F1 - grad(u_hat)|`` are recorded. ``para.beta_criterion`` selects
the score:

* ``"matlab"`` (default): ``Err1 + mean(h)^2 * Err2`` (both terms in voxels),
  discrete minimum over the candidates -- exactly the MATLAB rule;
* ``"normalized"``: z-normalised ``Err1`` and ``Err2`` summed, with a
  quadratic refinement in ``log10(beta)`` around the discrete minimum.

Boundary nodes are excluded from the error norms.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ..core.config import DVCPara
from .global_operators import GlobalOperators, nodal_gradient
from .subpb2_solver import build_global_system, solve_subpb2

logger = logging.getLogger(__name__)


def beta_candidates(para: DVCPara, mu: float) -> NDArray[np.float64]:
    h2 = float(np.mean(para.winstepsize)) ** 2
    return np.asarray(para.beta_range, dtype=np.float64) * h2 * mu


def auto_tune_beta(
    ops: GlobalOperators,
    para: DVCPara,
    mu: float,
    U1: NDArray[np.float64],
    F1: NDArray[np.float64],
    exclude: NDArray[np.bool_] | None = None,
) -> tuple[float, dict]:
    """Return the tuned ``beta`` and a dict with the sweep for reporting."""
    betas = beta_candidates(para, mu)
    n = ops.n_nodes
    include = ops.active.copy()
    if exclude is not None and exclude.size == n:
        include &= ~exclude
    if include.sum() < 8:
        include = ops.active.copy()
    U1 = np.asarray(U1, dtype=np.float64).reshape(n, 3)
    F1 = np.asarray(F1, dtype=np.float64).reshape(n, 3, 3)
    W0 = np.zeros((n, 3, 3))
    v0 = np.zeros((n, 3))
    err1 = np.zeros(len(betas))
    err2 = np.zeros(len(betas))
    for k, bk in enumerate(betas):
        system = build_global_system(ops, float(bk), mu, para.alpha, para)
        U2, _ = solve_subpb2(system, U1, F1, W0, v0)
        F2 = nodal_gradient(ops, U2)
        F2 = np.where(np.isfinite(F2), F2, F1)
        err1[k] = np.linalg.norm((U1 - U2)[include])
        err2[k] = np.linalg.norm((F1 - F2)[include])
    h2 = float(np.mean(para.winstepsize)) ** 2
    criterion = getattr(para, "beta_criterion", "matlab")
    s1, s2 = np.std(err1), np.std(err2)
    if criterion == "normalized" and s1 > 1e-15 and s2 > 1e-15:
        score = (err1 - err1.mean()) / s1 + (err2 - err2.mean()) / s2
    else:
        score = err1 + err2 * h2
    k_best = int(np.argmin(score))
    beta = float(betas[k_best])
    if criterion == "normalized" and 0 < k_best < len(betas) - 1:
        x = np.log10(betas[k_best - 1 : k_best + 2])
        y = score[k_best - 1 : k_best + 2]
        p = np.polyfit(x, y, 2)
        if p[0] > 1e-15:
            cand = 10.0 ** (-p[1] / (2.0 * p[0]))
            if betas[k_best - 1] <= cand <= betas[k_best + 1]:
                beta = float(cand)
    logger.info("Auto-tuned beta = %.4e (candidates %s)", beta, np.array2string(betas, precision=3))
    return beta, {
        "betas": betas,
        "err1": err1,
        "err2": err2,
        "score": score,
        "k_best": k_best,
        "beta": beta,
        "criterion": criterion,
    }
