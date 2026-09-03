"""Scalar global operators for ADMM subproblem 2 (FEM hex8 or finite difference).

The global step solves, for each displacement component ``i`` independently,

    (beta * Kg + mu * M + alpha * Kg) u_i = beta * sum_j G_j (F - W)_ij + mu * M (u1 - v)_i + alpha * Kg u1_i

where, for the FEM discretisation (MATLAB ``Subpb23.m``)

    Kg = int grad(N)^T grad(N),   M = int N^T N,   G_j[a, b] = int (dN_a/dx_j) N_b

and for the finite-difference discretisation (MATLAB ``funDerivativeOp3.m``)

    Kg = sum_j D_j^T D_j,         M = I,           G_j = D_j^T.

Both are assembled once per mesh; ``beta``/``mu`` enter as scalars, so the
L-curve sweep and every ADMM iteration reuse the same matrices.

The nodal gradient of a global solution is the lumped L2 projection
``F_ij = (G_j^T u_i) / m_L`` (FEM) or ``D_j u_i`` (FD).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from ..core.data_structures import DVCMesh
from ..mesh.grid_mesh import active_elements
from ..mesh.hex8 import hex8_box_matrices


@dataclass
class GlobalOperators:
    Kg: sparse.csr_matrix
    M: sparse.csr_matrix
    G: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]
    mL: NDArray[np.float64]         # lumped mass (row sums of M); ones for FD
    active: NDArray[np.bool_]       # nodes that carry equations
    method: str
    n_nodes: int

    @property
    def n_active(self) -> int:
        return int(self.active.sum())


def _scatter(elements: NDArray[np.int64], Ke: NDArray[np.float64], n_nodes: int) -> sparse.csr_matrix:
    E = elements.shape[0]
    I = np.broadcast_to(elements[:, :, None], (E, 8, 8)).ravel()
    J = np.broadcast_to(elements[:, None, :], (E, 8, 8)).ravel()
    V = np.broadcast_to(Ke[None, :, :], (E, 8, 8)).ravel()
    return sparse.coo_matrix((V, (I, J)), shape=(n_nodes, n_nodes)).tocsr()


def build_fem_operators(mesh: DVCMesh, gauss_pt_order: int = 2) -> GlobalOperators:
    """Assemble hex8 operators over the active elements of ``mesh``."""
    n = mesh.n_nodes
    elems = active_elements(mesh)
    if elems.shape[0] == 0:
        raise ValueError("Mesh has no active elements; cannot build FEM operators.")
    box = hex8_box_matrices(mesh.spacing, gauss_pt_order)
    Kg = _scatter(elems, box["K"], n)
    M = _scatter(elems, box["M"], n)
    G = tuple(_scatter(elems, box["G"][j], n) for j in range(3))
    mL = np.asarray(M.sum(axis=1)).ravel()
    active = np.zeros(n, dtype=bool)
    active[np.unique(elems)] = True
    return GlobalOperators(Kg=Kg, M=M, G=G, mL=mL, active=active, method="fem", n_nodes=n)  # type: ignore[arg-type]


def _difference_operator(grid_shape: tuple[int, int, int], axis_xyz: int, h: float,
                         valid: NDArray[np.bool_]) -> sparse.csr_matrix:
    """Central-difference operator along x (0), y (1) or z (2) on the node grid.

    Uses one-sided differences where only one neighbour is valid and a zero
    row where neither is (MATLAB ``funDerivativeOp3`` uses one-sided
    differences at the grid border).
    """
    nz, ny, nx = grid_shape
    n = nz * ny * nx
    idx = np.arange(n).reshape(nz, ny, nx)
    ax = {0: 2, 1: 1, 2: 0}[axis_xyz]  # array axis of the x/y/z direction
    stride = {2: 1, 1: nx, 0: nx * ny}[ax]
    coord = np.indices(grid_shape)[ax].ravel()
    size = grid_shape[ax]
    v = valid.ravel()
    has_prev = (coord > 0)
    has_prev[has_prev] &= v[idx.ravel()[has_prev] - stride]
    has_next = (coord < size - 1)
    has_next[has_next] &= v[idx.ravel()[has_next] + stride]
    rows: list[NDArray] = []
    cols: list[NDArray] = []
    vals: list[NDArray] = []
    i = np.arange(n)
    both = has_prev & has_next & v
    rows += [i[both], i[both]]
    cols += [i[both] - stride, i[both] + stride]
    vals += [np.full(both.sum(), -0.5 / h), np.full(both.sum(), 0.5 / h)]
    only_next = has_next & ~has_prev & v
    rows += [i[only_next], i[only_next]]
    cols += [i[only_next], i[only_next] + stride]
    vals += [np.full(only_next.sum(), -1.0 / h), np.full(only_next.sum(), 1.0 / h)]
    only_prev = has_prev & ~has_next & v
    rows += [i[only_prev], i[only_prev]]
    cols += [i[only_prev] - stride, i[only_prev]]
    vals += [np.full(only_prev.sum(), -1.0 / h), np.full(only_prev.sum(), 1.0 / h)]
    return sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n),
    ).tocsr()


def build_fd_operators(mesh: DVCMesh) -> GlobalOperators:
    """Finite-difference operators on the node grid (valid nodes only)."""
    n = mesh.n_nodes
    valid = np.asarray(mesh.node_valid, dtype=bool) if mesh.node_valid.size == n else np.ones(n, dtype=bool)
    D = tuple(_difference_operator(mesh.grid_shape, j, mesh.spacing[j], valid) for j in range(3))
    Kg = (D[0].T @ D[0] + D[1].T @ D[1] + D[2].T @ D[2]).tocsr()
    M = sparse.diags(valid.astype(np.float64), format="csr")
    G = tuple(Dj.T.tocsr() for Dj in D)
    mL = np.ones(n, dtype=np.float64)
    ops = GlobalOperators(Kg=Kg, M=M, G=G, mL=mL, active=valid.copy(), method="fd", n_nodes=n)  # type: ignore[arg-type]
    ops.D = D  # type: ignore[attr-defined]
    return ops


def build_global_operators(mesh: DVCMesh, method: str, gauss_pt_order: int = 2) -> GlobalOperators:
    if method == "fem":
        return build_fem_operators(mesh, gauss_pt_order)
    if method == "fd":
        return build_fd_operators(mesh)
    raise ValueError(f"unknown subpb2 method {method!r}")


def nodal_gradient(ops: GlobalOperators, U: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(N, 3, 3)`` displacement gradient of a nodal field via the operators.

    Inactive nodes get NaN (the caller decides what to substitute).
    """
    U = np.asarray(U, dtype=np.float64).reshape(ops.n_nodes, 3)
    F = np.empty((ops.n_nodes, 3, 3), dtype=np.float64)
    if ops.method == "fem":
        safe_mL = np.where(ops.mL > 1e-15, ops.mL, 1.0)
        for j in range(3):
            GT = ops.G[j].T
            for i in range(3):
                F[:, i, j] = (GT @ U[:, i]) / safe_mL
    else:
        D = ops.D  # type: ignore[attr-defined]
        for j in range(3):
            for i in range(3):
                F[:, i, j] = D[j] @ U[:, i]
    F[~ops.active] = np.nan
    return F
