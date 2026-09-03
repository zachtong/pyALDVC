import numpy as np

from al_dvc.io.volume_ops import normalize_volume
from al_dvc.mesh.grid_mesh import mesh_setup
from al_dvc.solver.integer_search import auto_pyramid_levels, ncc_search, ncc_search_expanding
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian


def _pair(shift, shape=(64, 64, 72), seed=21):
    vol = generate_speckle_volume(shape, sigma=2.0, seed=seed)
    g = warp_volume_lagrangian(vol, affine_displacement(None, shift))
    return normalize_volume(vol), normalize_volume(g)


def test_expansion_only_re_searches_clipped_nodes():
    f, g = _pair((6.3, -0.4, 0.2))
    ax = np.arange(20.0, 46.0, 8.0)
    mesh = mesh_setup(ax, ax, ax)
    coords = mesh.coordinates.astype(np.int64)
    small = ncc_search(f, g, coords, (16, 16, 16), (2, 2, 2))
    assert small["clipped"].all()
    res = ncc_search_expanding(f, g, coords, (16, 16, 16), (2, 2, 2), max_expand=3)
    assert res["expansions"] >= 2 and res["radius"][0] >= 8
    assert res["ok"].all() and not res["clipped"].any()
    assert np.all(np.abs(res["disp"] - np.array([6.3, -0.4, 0.2])) < 0.25)


def test_clamped_window_is_not_clipped():
    """Nodes whose window is pushed against the volume boundary must not trigger expansions."""
    f, g = _pair((3.0, 0.0, 0.0))
    coords = np.array([[62, 32, 32]], dtype=np.int64)  # template touches the right boundary of g
    res = ncc_search(f, g, coords, (16, 16, 16), (4, 4, 4))
    assert res["ok"][0] and not res["clipped"][0]
    res2 = ncc_search_expanding(f, g, coords, (16, 16, 16), (4, 4, 4))
    assert res2["expansions"] == 0


def test_expansion_keeps_previous_when_window_does_not_fit():
    f, g = _pair((1.2, 0.0, 0.0), shape=(40, 40, 44))
    coords = np.array([[22, 20, 20], [18, 20, 20]], dtype=np.int64)
    res = ncc_search_expanding(f, g, coords, (16, 16, 16), (9, 9, 9), max_expand=2)
    assert res["ok"].all()
    assert np.all(np.abs(res["disp"][:, 0] - 1.2) < 0.3)


def test_direct_and_fft_engines_agree(monkeypatch):
    from al_dvc.solver import integer_search as isearch

    f, g = _pair((2.4, -1.3, 0.7))
    ax = np.arange(20.0, 46.0, 8.0)
    mesh = mesh_setup(ax, ax, ax)
    coords = mesh.coordinates.astype(np.int64)
    monkeypatch.setattr(isearch, "DIRECT_NCC_MAX_OPS", 1e15)
    a = ncc_search(f, g, coords, (16, 16, 16), (4, 4, 4))
    monkeypatch.setattr(isearch, "DIRECT_NCC_MAX_OPS", 0.0)
    b = ncc_search(f, g, coords, (16, 16, 16), (4, 4, 4))
    assert a["ok"].all() and b["ok"].all()
    assert np.allclose(a["disp"], b["disp"], atol=1e-5)
    assert np.allclose(a["cc"], b["cc"], atol=1e-5)
    assert np.all(np.abs(a["disp"] - np.array([2.4, -1.3, 0.7])) < 0.25)


def test_auto_levels_respect_texture():
    fine = generate_speckle_volume((96, 96, 96), sigma=1.0, seed=3)
    coarse = generate_speckle_volume((96, 96, 96), sigma=6.0, seed=3)
    n_fine = auto_pyramid_levels(fine.shape, (16, 16, 16), (4, 4, 4), f=fine)
    n_coarse = auto_pyramid_levels(coarse.shape, (16, 16, 16), (4, 4, 4), f=coarse)
    assert n_coarse >= n_fine
    assert auto_pyramid_levels(fine.shape, (16, 16, 16), (4, 4, 4)) >= n_fine
