import numpy as np
import pytest

from al_dvc.core.data_structures import VOIRange
from al_dvc.mesh.grid_mesh import (
    active_elements,
    apply_mask_to_mesh,
    build_grid_axes,
    grid_surface_nodes,
    mesh_setup,
    subset_valid_fraction,
)
from al_dvc.mesh.hex8 import hex8_box_matrices, hex8_dshape, hex8_gauss_points, hex8_shape


def test_build_grid_axes_fit_inside_volume():
    shape = (64, 72, 80)
    x0, y0, z0 = build_grid_axes(VOIRange(), shape, (16, 16, 16), (8, 8, 8))
    for ax, n in ((x0, 80), (y0, 72), (z0, 64)):
        assert ax[0] - 8 >= 5 and ax[-1] + 8 <= n - 1 - 5
        assert np.allclose(np.diff(ax), 8)
    with pytest.raises(ValueError):
        build_grid_axes(VOIRange(), (20, 20, 20), (16, 16, 16), (8, 8, 8))


def test_mesh_setup_ordering_and_elements():
    mesh = mesh_setup(np.array([0.0, 4.0, 8.0]), np.array([0.0, 4.0]), np.array([0.0, 4.0, 8.0, 12.0]))
    assert mesh.grid_shape == (4, 2, 3)
    assert mesh.n_nodes == 24
    # node n = iz*ny*nx + iy*nx + ix -> coordinates
    n = mesh.node_index(2, 1, 3)
    assert np.allclose(mesh.coordinates[n], [8.0, 4.0, 12.0])
    grid = mesh.to_grid(mesh.coordinates[:, 0])
    assert grid.shape == (4, 2, 3) and np.allclose(grid[0, 0, :], [0, 4, 8])
    assert mesh.elements.shape == (3 * 1 * 2, 8)
    e = mesh.elements[0]
    c = mesh.coordinates[e]
    # standard hex8 ordering: bottom face counter-clockwise then top
    assert np.allclose(c[:, 2][:4], 0) and np.allclose(c[:, 2][4:], 4)
    assert np.allclose(c[1] - c[0], [4, 0, 0]) and np.allclose(c[3] - c[0], [0, 4, 0])
    assert mesh.spacing == (4.0, 4.0, 4.0)
    assert set(grid_surface_nodes(mesh.grid_shape)) == set(range(24))  # every node is on the surface here


def test_hex8_shape_functions():
    pts, wts = hex8_gauss_points(2)
    assert pts.shape == (8, 3) and np.isclose(wts.sum(), 8.0)
    for ksi, eta, zeta in pts:
        N = hex8_shape(ksi, eta, zeta)
        assert np.isclose(N.sum(), 1.0)
        dN = hex8_dshape(ksi, eta, zeta)
        assert np.allclose(dN.sum(axis=0), 0.0)
    # nodal values: N_a(node_b) = delta_ab
    from al_dvc.mesh.hex8 import HEX8_SIGNS

    for b in range(8):
        N = hex8_shape(*HEX8_SIGNS[b])
        assert np.allclose(N, np.eye(8)[b])


def test_hex8_box_matrices_properties():
    box = hex8_box_matrices((2.0, 3.0, 4.0), 2)
    K, M, G = box["K"], box["M"], box["G"]
    assert np.allclose(K, K.T) and np.allclose(M, M.T)
    assert np.allclose(K.sum(axis=1), 0.0)            # constant field has zero gradient energy
    assert np.isclose(M.sum(), 24.0)                   # integral of 1 over the box
    assert np.allclose(G.sum(axis=1), 0.0)             # sum_a dN_a/dx_j = 0
    # linear field u = x: nodal gradient via lumped projection
    coords = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0], [0, 0, 4], [2, 0, 4], [2, 3, 4], [0, 3, 4]], float)
    u = coords[:, 0]
    mL = M.sum(axis=1)
    assert np.allclose((G[0].T @ u) / mL, 1.0)
    assert np.allclose((G[1].T @ u) / mL, 0.0)
    assert np.isclose(u @ K @ u, 24.0)                 # int |grad u|^2 = volume


def test_subset_valid_fraction_and_mask_trimming():
    x0 = y0 = z0 = np.array([8.0, 16.0, 24.0, 32.0])
    mesh = mesh_setup(x0, y0, z0)
    mask = np.ones((40, 40, 40), dtype=np.uint8)
    mask[:, :, 20:] = 0  # right half invalid
    frac = subset_valid_fraction(mask, mesh.coordinates, (8, 8, 8))
    x = mesh.coordinates[:, 0]
    assert np.allclose(frac[x == 8], 1.0)
    assert np.allclose(frac[x == 32], 0.0)
    assert np.all((frac[x == 16] > 0.6) & (frac[x == 16] < 1.0))
    m2 = apply_mask_to_mesh(mesh, mask, (8, 8, 8), 0.5)
    assert m2.node_valid[x == 8].all() and not m2.node_valid[x == 24].any()
    act = active_elements(m2)
    assert act.shape[0] < mesh.elements.shape[0]
    assert m2.node_valid[act].all()
    assert m2.boundary_nodes.size > 0
