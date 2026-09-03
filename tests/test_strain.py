import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.mesh.grid_mesh import mesh_setup
from al_dvc.strain.compute_strain import compute_strain
from al_dvc.strain.gradient_methods import gradient_fd, gradient_plane_fit
from al_dvc.strain.strain_types import (
    det_deformation_gradient,
    polar_rotation_deg,
    principal_strains,
    scale_to_physical,
    strain_tensor,
    von_mises_strain,
)


@pytest.fixture
def mesh():
    ax = np.arange(0.0, 49.0, 8.0)
    return mesh_setup(ax, ax, ax)


A_LIN = np.array([[0.01, 0.02, -0.03], [0.04, -0.05, 0.06], [0.07, 0.08, 0.09]])


@pytest.mark.parametrize("fn", [gradient_plane_fit, gradient_fd])
def test_gradients_exact_for_linear(mesh, fn):
    U = mesh.coordinates @ A_LIN.T
    Ug = U.reshape(mesh.grid_shape + (3,))
    if fn is gradient_plane_fit:
        F, complete = fn(Ug, mesh.spacing, (1, 1, 1))
    else:
        F, complete = fn(Ug, mesh.spacing)
    assert np.allclose(F, A_LIN, atol=1e-9)
    assert complete[1:-1, 1:-1, 1:-1].all() and not complete[0].any()


def test_plane_fit_honours_mask(mesh):
    U = mesh.coordinates @ A_LIN.T
    Ug = U.reshape(mesh.grid_shape + (3,))
    valid = np.ones(mesh.grid_shape, dtype=bool)
    valid[:, :, 4:] = False
    Ug[~valid] = 1e6  # garbage must not leak in
    F, complete = gradient_plane_fit(Ug, mesh.spacing, (1, 1, 1), valid)
    assert np.allclose(F[valid], A_LIN, atol=1e-8)
    assert np.all(np.isnan(F[~valid]))
    assert not complete[:, :, 3].any()  # window truncated next to the invalid slab


def test_strain_types_uniaxial_stretch():
    lam = 1.05
    H = np.diag([lam - 1.0, 0.0, 0.0])[None]
    e = strain_tensor(H, "infinitesimal")[0]
    E = strain_tensor(H, "green_lagrange")[0]
    a = strain_tensor(H, "euler_almansi")[0]
    h = strain_tensor(H, "hencky")[0]
    assert np.isclose(e[0, 0], lam - 1)
    assert np.isclose(E[0, 0], 0.5 * (lam**2 - 1))
    assert np.isclose(a[0, 0], 0.5 * (1 - lam**-2))
    assert np.isclose(h[0, 0], np.log(lam))
    assert np.isclose(det_deformation_gradient(H)[0], lam)


def test_principal_von_mises_rotation():
    E = np.diag([0.03, 0.01, -0.02])[None]
    p = principal_strains(E)[0]
    assert np.allclose(p, [0.03, 0.01, -0.02])
    vm = von_mises_strain(E)[0]
    dev = E[0] - np.trace(E[0]) / 3 * np.eye(3)
    assert np.isclose(vm, np.sqrt(2 / 3 * np.sum(dev**2)))
    ang = np.radians(7.0)
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    assert np.isclose(polar_rotation_deg((R - np.eye(3))[None])[0], 7.0)
    assert np.isnan(principal_strains(np.full((1, 3, 3), np.nan))[0]).all()


def test_scale_to_physical():
    U = np.array([[1.0, 2.0, 3.0]])
    F = np.ones((1, 3, 3))
    Up, Fp = scale_to_physical(U, F, (2.0, 4.0, 8.0))
    assert np.allclose(Up, [[2.0, 8.0, 24.0]])
    assert np.isclose(Fp[0, 0, 1], 2.0 / 4.0) and np.isclose(Fp[0, 2, 0], 8.0 / 2.0)


@pytest.mark.parametrize("method", ["plane_fit", "fd", "fem", "direct"])
def test_compute_strain_linear_field(mesh, method):
    para = dvcpara_default(winstepsize=8, strain_method=method, voxel_size=(2.0, 2.0, 2.0))
    U = mesh.coordinates @ A_LIN.T
    F_direct = np.tile(A_LIN, (mesh.n_nodes, 1, 1))
    sr = compute_strain(mesh, para, U, F_direct=F_direct)
    sym = 0.5 * (A_LIN + A_LIN.T)
    v = sr.strain_valid
    assert v.sum() > 0
    assert np.allclose(sr.exx[v], sym[0, 0], atol=1e-8)
    assert np.allclose(sr.exy[v], sym[0, 1], atol=1e-8)
    assert np.allclose(sr.disp_u, 2.0 * U[:, 0])
    assert sr.method == method
    assert np.allclose(sr.field("e1")[v], principal_strains(sym[None])[0][0], atol=1e-8)
