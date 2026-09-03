import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.mesh.grid_mesh import apply_mask_to_mesh, mesh_setup
from al_dvc.solver.beta_tuning import auto_tune_beta, beta_candidates
from al_dvc.solver.global_operators import build_global_operators, nodal_gradient
from al_dvc.solver.subpb2_solver import build_global_system, pcg_multi, solve_subpb2


@pytest.fixture
def mesh():
    ax = np.arange(0.0, 41.0, 8.0)
    return mesh_setup(ax, ax, ax)


@pytest.mark.parametrize("method", ["fem", "fd"])
def test_nodal_gradient_exact_for_linear_field(mesh, method):
    ops = build_global_operators(mesh, method)
    A = np.array([[0.01, 0.02, -0.03], [0.04, -0.05, 0.06], [0.07, 0.08, 0.09]])
    U = mesh.coordinates @ A.T + np.array([1.0, 2.0, 3.0])
    F = nodal_gradient(ops, U)
    assert np.allclose(F, A, atol=1e-10)


@pytest.mark.parametrize("method", ["fem", "fd"])
def test_constant_field_has_zero_stiffness_energy(mesh, method):
    ops = build_global_operators(mesh, method)
    c = np.ones(mesh.n_nodes)
    assert np.allclose(ops.Kg @ c, 0.0, atol=1e-10)
    assert ops.active.all()


@pytest.mark.parametrize("method", ["fem", "fd"])
def test_subpb2_limits(mesh, method):
    """Large mu -> u_hat = u1; large beta with a compatible F -> grad(u_hat) = F."""
    ops = build_global_operators(mesh, method)
    para = dvcpara_default(global_solver="direct")
    rng = np.random.default_rng(0)
    A = np.array([[0.01, 0.0, 0.0], [0.0, -0.02, 0.0], [0.0, 0.0, 0.03]])
    U_lin = mesh.coordinates @ A.T
    U1 = U_lin + 0.05 * rng.standard_normal(U_lin.shape)
    F1 = np.tile(A, (mesh.n_nodes, 1, 1))
    W = np.zeros_like(F1)
    v = np.zeros_like(U1)
    sys_mu = build_global_system(ops, 1e-12, 1.0, 0.0, para)
    U2, _ = solve_subpb2(sys_mu, U1, F1, W, v)
    assert np.allclose(U2, U1, atol=1e-8)
    sys_beta = build_global_system(ops, 1e6, 1e-6, 0.0, para)
    U2, _ = solve_subpb2(sys_beta, U1, F1, W, v)
    F2 = nodal_gradient(ops, U2)
    assert np.allclose(F2, A, atol=1e-4)


def test_pcg_matches_direct(mesh):
    ops = build_global_operators(mesh, "fem")
    rng = np.random.default_rng(1)
    U1 = rng.standard_normal((mesh.n_nodes, 3))
    F1 = rng.standard_normal((mesh.n_nodes, 3, 3)) * 0.01
    W = np.zeros_like(F1)
    v = np.zeros_like(U1)
    beta, mu = 0.05, 1e-3
    direct = build_global_system(ops, beta, mu, 0.0, dvcpara_default(global_solver="direct"))
    pcg = build_global_system(ops, beta, mu, 0.0, dvcpara_default(global_solver="pcg", pcg_tol=1e-12))
    Ud, _ = solve_subpb2(direct, U1, F1, W, v)
    Up, info = solve_subpb2(pcg, U1, F1, W, v)
    assert info["solver"] == "pcg" and info["iterations"] < 60
    assert np.allclose(Ud, Up, atol=1e-8)


def test_pcg_multi_solves_spd():
    from scipy import sparse

    n = 50
    rng = np.random.default_rng(2)
    A = sparse.random(n, n, density=0.1, random_state=2)
    A = (A @ A.T + sparse.eye(n) * n).tocsr()
    B = rng.standard_normal((n, 3))
    X, it = pcg_multi(A, B, 1.0 / A.diagonal(), 1e-12, 500)
    assert np.allclose(A @ X, B, atol=1e-8)


def test_masked_mesh_operators(mesh):
    mask = np.ones((48, 48, 48), dtype=np.uint8)
    mask[:, :, 30:] = 0
    m = apply_mask_to_mesh(mesh, mask, (8, 8, 8), 0.5)
    ops = build_global_operators(m, "fem")
    assert ops.active.sum() < mesh.n_nodes
    A = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03]])
    U = mesh.coordinates @ A.T
    F = nodal_gradient(ops, U)
    assert np.allclose(F[ops.active], A, atol=1e-10)
    assert np.all(np.isnan(F[~ops.active]))


def test_auto_tune_beta_in_range(mesh):
    ops = build_global_operators(mesh, "fem")
    para = dvcpara_default(winstepsize=8)
    rng = np.random.default_rng(3)
    A = np.array([[0.01, 0.0, 0.0], [0.0, -0.02, 0.0], [0.0, 0.0, 0.03]])
    U1 = mesh.coordinates @ A.T + 0.02 * rng.standard_normal((mesh.n_nodes, 3))
    F1 = np.tile(A, (mesh.n_nodes, 1, 1)) + 0.002 * rng.standard_normal((mesh.n_nodes, 3, 3))
    beta, info = auto_tune_beta(ops, para, para.mu, U1, F1)
    cands = beta_candidates(para, para.mu)
    assert cands.min() <= beta <= cands.max()
    assert info["err1"].shape == cands.shape
