"""ADMM subproblem 2: global compatible displacement (MATLAB ``Subpb23.m`` / FD path).

Solves ``A u_i = b_i`` for the three components at once with a
Jacobi-preconditioned conjugate gradient (the system is SPD and very well
conditioned, see docs/design.md 2.3) or a direct sparse LU for small meshes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

from ..core.config import DVCPara
from .global_operators import GlobalOperators

logger = logging.getLogger(__name__)

# Sparse LU of a 3-D FEM/FD operator fills in heavily (~3 s per factorisation at
# 20k nodes); the system is so well conditioned that Jacobi-PCG converges in a
# few dozen iterations, so the direct path is reserved for small meshes.
DIRECT_MAX_ACTIVE = 5_000


@dataclass
class GlobalSystem:
    ops: GlobalOperators
    beta: float
    mu: float
    alpha: float
    A: sparse.csr_matrix              # active x active
    inv_diag: NDArray[np.float64]
    active_idx: NDArray[np.int64]
    lu: object | None
    solver: str
    pcg_tol: float
    pcg_max_iter: int


def build_global_system(ops: GlobalOperators, beta: float, mu: float, alpha: float, para: DVCPara) -> GlobalSystem:
    """Assemble ``A = (beta + alpha) Kg + mu M`` restricted to active nodes."""
    A_full = ((beta + alpha) * ops.Kg + mu * ops.M).tocsr()
    active_idx = np.flatnonzero(ops.active)
    A = A_full[active_idx][:, active_idx].tocsr()
    diag = A.diagonal()
    inv_diag = np.where(np.abs(diag) > 1e-300, 1.0 / diag, 1.0)
    solver = para.global_solver
    if solver == "auto":
        solver = "direct" if active_idx.size <= DIRECT_MAX_ACTIVE else "pcg"
    lu = None
    if solver == "direct":
        try:
            lu = splu(A.tocsc())
        except Exception as exc:  # pragma: no cover - numerical safety net
            logger.warning("Sparse LU failed (%s); falling back to PCG.", exc)
            solver = "pcg"
    return GlobalSystem(
        ops=ops, beta=beta, mu=mu, alpha=alpha, A=A, inv_diag=inv_diag, active_idx=active_idx,
        lu=lu, solver=solver, pcg_tol=para.pcg_tol, pcg_max_iter=para.pcg_max_iter,
    )


def pcg_multi(
    A: sparse.csr_matrix,
    B: NDArray[np.float64],
    inv_diag: NDArray[np.float64],
    tol: float,
    max_iter: int,
    X0: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], int]:
    """Jacobi-preconditioned CG for several right-hand sides ``B`` (n, k)."""
    X = np.zeros_like(B) if X0 is None else X0.copy()
    R = B - A @ X
    Z = inv_diag[:, None] * R
    P = Z.copy()
    rz = np.einsum("ij,ij->j", R, Z)
    bnorm = np.linalg.norm(B, axis=0)
    bnorm[bnorm == 0] = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        AP = A @ P
        denom = np.einsum("ij,ij->j", P, AP)
        denom[denom == 0] = np.inf
        alpha = rz / denom
        X += P * alpha
        R -= AP * alpha
        if np.all(np.linalg.norm(R, axis=0) / bnorm < tol):
            break
        Z = inv_diag[:, None] * R
        rz_new = np.einsum("ij,ij->j", R, Z)
        beta = rz_new / np.where(rz == 0, np.inf, rz)
        P = Z + P * beta
        rz = rz_new
    return X, it


def global_rhs(
    ops: GlobalOperators,
    beta: float,
    mu: float,
    alpha: float,
    U1: NDArray[np.float64],
    F1: NDArray[np.float64],
    W: NDArray[np.float64],
    v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``b_i = beta sum_j G_j (F1 - W)_ij + mu M (U1 - v)_i + alpha Kg U1_i`` as ``(N, 3)``."""
    n = ops.n_nodes
    FmW = np.asarray(F1, dtype=np.float64).reshape(n, 3, 3) - np.asarray(W, dtype=np.float64).reshape(n, 3, 3)
    Umv = np.asarray(U1, dtype=np.float64).reshape(n, 3) - np.asarray(v, dtype=np.float64).reshape(n, 3)
    b = mu * (ops.M @ Umv)
    for j in range(3):
        b += beta * (ops.G[j] @ FmW[:, :, j])
    if alpha != 0.0:
        b += alpha * (ops.Kg @ np.asarray(U1, dtype=np.float64).reshape(n, 3))
    return b


def solve_subpb2(
    system: GlobalSystem,
    U1: NDArray[np.float64],
    F1: NDArray[np.float64],
    W: NDArray[np.float64],
    v: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict]:
    """Return ``u_hat`` (N, 3); inactive nodes keep ``U1``."""
    t0 = time.perf_counter()
    ops = system.ops
    b = global_rhs(ops, system.beta, system.mu, system.alpha, U1, F1, W, v)
    U1 = np.asarray(U1, dtype=np.float64).reshape(ops.n_nodes, 3)
    b_act = b[system.active_idx]
    x0 = U1[system.active_idx]
    if system.lu is not None:
        X = system.lu.solve(b_act)
        n_it = 0
    else:
        X, n_it = pcg_multi(system.A, b_act, system.inv_diag, system.pcg_tol, system.pcg_max_iter, X0=x0)
    U2 = U1.copy()
    U2[system.active_idx] = X
    info = {"solver": system.solver, "iterations": int(n_it), "time": time.perf_counter() - t0}
    return U2, info
