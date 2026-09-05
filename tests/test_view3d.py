"""3-D view: scene construction (pyvista, off-screen) and the panel's static backend (offscreen Qt)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.gui.view3d_scene import (  # noqa: E402
    CAMERAS,
    MODES,
    SceneOptions,
    auto_clim,
    available,
    build_scene,
    node_grid,
    render_image,
    render_png,
    volume_slice_planes,
)
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE = (40, 44, 48)


@pytest.fixture(scope="module")
def small_result():
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=5)
    dfm = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.02, -0.01, 0.01]), (0.6, -0.4, 0.3), centre))
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    return run_aldvc(para, [ref, dfm]), ref


def test_available():
    assert available()


def test_node_grid_matches_mesh_ordering(small_result):
    res, _ = small_result
    grid = node_grid(res, 0, ("disp_u", "disp_magnitude"))
    mesh = res.dvc_mesh
    nz, ny, nx = mesh.grid_shape
    assert tuple(grid.dimensions) == (nx, ny, nz)
    assert grid.n_points == mesh.n_nodes
    # lattice points reproduce the node coordinates in node order (x fastest)
    np.testing.assert_allclose(np.asarray(grid.points), mesh.coordinates, atol=1e-9)
    # the per-node field reshaped as (nz, ny, nx) equals mesh.to_grid
    u = np.asarray(grid.point_data["disp_u"])
    np.testing.assert_allclose(u.reshape(nz, ny, nx), mesh.to_grid(res.result_disp[0].U[:, 0]), equal_nan=True)
    vec = np.asarray(grid.point_data["displacement"])
    assert vec.shape == (mesh.n_nodes, 3)
    assert np.all(np.isfinite(vec))


def test_volume_slice_planes_geometry():
    vol = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    planes = volume_slice_planes(vol, {"z": 5, "y": 7, "x": 9}, voxel_size=(2.0, 1.0, 0.5))
    nz, ny, nx = SHAPE
    xy, xz, yz = planes["xy"], planes["xz"], planes["yz"]
    assert tuple(xy.dimensions) == (nx, ny, 1) and xy.origin[2] == pytest.approx(5 * 0.5)
    assert tuple(xz.dimensions) == (nx, 1, nz) and xz.origin[1] == pytest.approx(7.0)
    assert tuple(yz.dimensions) == (1, ny, nz) and yz.origin[0] == pytest.approx(9 * 2.0)
    # point ordering: intensity at lattice point (ix, iy) of the XY plane equals vol[5, iy, ix]
    inten = np.asarray(xy.point_data["intensity"]).reshape(ny, nx)
    np.testing.assert_array_equal(inten, vol[5])
    inten = np.asarray(xz.point_data["intensity"]).reshape(nz, nx)
    np.testing.assert_array_equal(inten, vol[:, 7, :])
    inten = np.asarray(yz.point_data["intensity"]).reshape(nz, ny)
    np.testing.assert_array_equal(inten, vol[:, :, 9])


def test_volume_slice_planes_subsample_large_slice():
    from al_dvc.gui import view3d_scene

    big = np.zeros((2, 2500, 2500), dtype=np.float32)
    planes = volume_slice_planes(big, {"z": 0, "y": 0, "x": 0})
    assert planes["xy"].n_points <= view3d_scene.VOLUME_SLICE_MAX_PIXELS
    assert planes["xy"].spacing[0] > 1.0  # coarser lattice keeps the physical extent
    assert planes["xy"].bounds[1] == pytest.approx(2499, abs=planes["xy"].spacing[0])


def test_auto_clim_and_option_validation():
    assert auto_clim(np.array([np.nan, np.nan])) == (0.0, 1.0)
    lo, hi = auto_clim(np.array([1.0, 1.0, 1.0]))
    assert hi > lo
    with pytest.raises(ValueError):
        SceneOptions(mode="nope")
    with pytest.raises(ValueError):
        SceneOptions(arrow_stride=0)
    with pytest.raises(ValueError):
        SceneOptions(opacity=1.5)


@pytest.mark.parametrize("mode", MODES)
def test_build_scene_modes(small_result, mode):
    res, ref = small_result
    pl = pv.Plotter(off_screen=True, window_size=(300, 240))
    opts = SceneOptions(
        mode=mode,
        show_arrows=True,
        arrow_stride=2,
        show_volume_slices=True,
        slice_index={"z": 20, "y": 22, "x": 24},
        title="Displacement magnitude",
    )
    info = build_scene(pl, res, opts, volume=ref)
    assert info.n_nodes == res.dvc_mesh.n_nodes
    if mode == "slices":  # the scalar bar carries the readable title, not the field key
        assert any(k.replace(chr(10), " ").startswith("Displacement magnitude") for k in pl.scalar_bars.keys())
    assert info.n_finite > 0
    assert "field" in info.actors
    assert info.n_arrows > 0
    assert {"volume_xy", "volume_xz", "volume_yz", "outline"} <= set(info.actors)
    if mode == "surface":
        lo, hi = info.clim
        assert lo <= info.actors["iso_level"] <= hi
    img = pl.screenshot(None, return_img=True)
    pl.close()
    assert img.shape[0] == 240 and img.std() > 1.0  # not a blank frame


def test_render_png_and_cameras(small_result, tmp_path):
    res, ref = small_result
    for cam in CAMERAS:
        info = render_png(res, tmp_path / f"{cam}.png", SceneOptions(field="disp_w"), camera=cam, window_size=(320, 240))
        assert (tmp_path / f"{cam}.png").stat().st_size > 1000
        assert info.field == "disp_w"
    img, _ = render_image(res, SceneOptions(mode="points"), window_size=(320, 240))
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    with pytest.raises(ValueError):
        render_image(res, camera="top", window_size=(64, 48))


def test_arrows_respect_the_cap(small_result):
    from al_dvc.gui import view3d_scene

    res, _ = small_result
    pl = pv.Plotter(off_screen=True, window_size=(120, 100))
    old = view3d_scene.MAX_ARROWS
    view3d_scene.MAX_ARROWS = 10
    try:
        info = build_scene(pl, res, SceneOptions(show_arrows=True, arrow_stride=1))
    finally:
        view3d_scene.MAX_ARROWS = old
    pl.close()
    assert 0 < info.n_arrows <= 10


# ----------------------------------------------------------------------------- panel (static backend, offscreen Qt)
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from al_dvc.gui.app import create_application

    return create_application(["pytest"])


def test_panel_static_backend_renders_and_follows_the_state(qapp, small_result, tmp_path):
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow

    res, ref = small_result
    window = MainWindow()
    window.show()
    panel = window.view3d
    assert panel.backend == "static"  # offscreen platform: no OpenGL context for QtInteractor
    assert not panel.mode.isEnabled()
    window.state.set_volume_arrays([ref, ref], ["a", "b"])
    window.state.set_results(res)
    window.center_tabs.setCurrentWidget(panel)
    for _ in range(20):
        QApplication.processEvents()
    assert panel.mode.isEnabled()
    assert panel._last_info is not None and panel._last_info.field == "disp_magnitude"
    assert panel._image.pixmap() is not None and not panel._image.pixmap().isNull()
    # display changes reach the panel through the state
    window.state.set_display(display_field="exx", colormap="coolwarm")
    assert panel._last_info.field == "exx"
    window.results_panel.select_field("disp_u")
    assert panel.options().field == "disp_u"
    # slider moves of the slice viewer move the 3-D slices
    window.viewer.sliders["z"].setValue(3)
    assert panel.options().slice_index["z"] == 3
    panel.mode.setCurrentIndex(MODES.index("warped"))
    assert panel.options().mode == "warped" and panel.warp_scale.isEnabled()
    panel.arrows.setChecked(True)
    assert panel._last_info.n_arrows > 0
    panel.volume_slices.setChecked(True)
    assert "volume_xy" in panel._last_info.actors
    panel.camera.setCurrentIndex(CAMERAS.index("xz"))
    assert panel._camera == "xz"
    out = panel.screenshot(tmp_path / "shot")
    assert out is not None and out.suffix == ".png" and out.stat().st_size > 1000
    window.close()


def test_panel_without_results_shows_hint(qapp):
    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    window.show()
    panel = window.view3d
    panel.refresh()
    assert panel._stack.currentWidget() is panel._hint
    assert panel.screenshot("nowhere.png") is None
    window.close()
