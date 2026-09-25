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
    MAX_ISO_LEVELS,
    MODES,
    CameraSpec,
    CameraState,
    SceneOptions,
    apply_camera,
    auto_clim,
    available,
    build_scene,
    camera_direction,
    facing_quadrant,
    iso_surface_levels,
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


ROTATION_STEP = 8.0


@pytest.fixture(scope="module")
def rotation_result():
    """A rigid rotation about the z axis through the centre of a 17 x 17 x 4 node grid, by the angle that makes
    ``|u| = 2 r sin(angle / 2) = r / 2`` exactly (``r``: distance from the axis): the iso-surfaces of ``|u|`` are
    nested cylinders about the axis, of radius twice their level."""
    from dataclasses import replace

    from al_dvc.core.data_structures import FrameResult, FrameSchedule, PipelineResult
    from al_dvc.mesh.grid_mesh import mesh_setup

    x0 = 16.0 + ROTATION_STEP * np.arange(17)
    z0 = 16.0 + ROTATION_STEP * np.arange(4)
    mesh = mesh_setup(x0, x0, z0)
    mesh = replace(mesh, node_valid=np.ones(mesh.n_nodes, dtype=bool))
    angle = 2.0 * np.arcsin(0.25)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = mesh.coordinates[:, :2] - x0.mean()
    U = np.zeros((mesh.n_nodes, 3))
    U[:, :2] = xy @ rot.T - xy
    frame = FrameResult(U=U, F=np.zeros((mesh.n_nodes, 3, 3)), U_accum=U)
    para = dvcpara_default(winsize=16, winstepsize=8, verbose=False)
    return PipelineResult(
        dvc_para=para,
        dvc_mesh=mesh,
        result_disp=[frame],
        result_strain=[],
        frame_schedule=FrameSchedule(ref_indices=(0,)),
        volume_shape=(64, 160, 160),
    )


def _drawn(actor) -> "pv.DataSet":
    """The dataset an actor draws (its mapper run first)."""
    actor.mapper.Update()
    return pv.wrap(actor.mapper.GetInputDataObject(0, 0))


def _surface_points(result, opts: SceneOptions):
    """``(points, field values, info)`` of the iso-surfaces ``build_scene`` draws for ``opts``."""
    pl = pv.Plotter(off_screen=True, window_size=(160, 120))
    try:
        info = build_scene(pl, result, opts, None)
        surf = _drawn(info.actors["field"])
        return np.asarray(surf.points, dtype=float), np.asarray(surf.point_data[opts.field], dtype=float), info
    finally:
        pl.close()


def _grid_centre(result) -> tuple[float, float]:
    xmin, xmax, ymin, ymax, _zmin, _zmax = node_grid(result, 0).bounds
    return 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)


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


def test_node_grid_reference_state_and_blend(small_result):
    res, _ref = small_result
    from al_dvc.gui.view3d_scene import DISPLACEMENT_ARRAY

    g0 = node_grid(res, 0, ("disp_magnitude",))
    ref = node_grid(res, -1, ("disp_magnitude",))
    valid = np.asarray(res.dvc_mesh.node_valid, dtype=bool)
    assert np.all(np.asarray(ref.point_data[DISPLACEMENT_ARRAY]) == 0.0)
    assert np.all(np.asarray(ref.point_data["disp_magnitude"])[valid] == 0.0)
    assert np.all(np.isnan(np.asarray(ref.point_data["disp_magnitude"])[~valid]))
    half = node_grid(res, -1, ("disp_magnitude",), blend=0.5)  # half-way from the reference state to frame 0
    half_u = np.asarray(half.point_data[DISPLACEMENT_ARRAY])
    np.testing.assert_allclose(half_u, 0.5 * np.asarray(g0.point_data[DISPLACEMENT_ARRAY]))
    # slices mode draws only the planes that are switched on
    img_all, info_all = render_image(res, SceneOptions(mode="slices"), None, window_size=(240, 180))
    one = SceneOptions(mode="slices", slice_visible={"y": False, "x": False})
    img_one, info_one = render_image(res, one, None, window_size=(240, 180))
    assert info_one.actors["field"].bounds[4] == info_one.actors["field"].bounds[5]  # a single XY plane: flat in z
    assert np.abs(img_all.astype(int) - img_one.astype(int)).mean() > 0.5


def test_volume_slice_planes_geometry():
    vol = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    planes = volume_slice_planes(vol, {"z": 5, "y": 7, "x": 9}, voxel_size=(2.0, 1.0, 0.5))
    nz, ny, nx = SHAPE
    (xy, t_xy), (xz, t_xz), (yz, t_yz) = planes["xy"], planes["xz"], planes["yz"]
    # four-vertex quads spanning the volume in world units, one texture pixel per voxel
    assert xy.n_points == 4 and xy.bounds == pytest.approx((0, (nx - 1) * 2.0, 0, ny - 1, 5 * 0.5, 5 * 0.5))
    assert xz.n_points == 4 and xz.bounds == pytest.approx((0, (nx - 1) * 2.0, 7.0, 7.0, 0, (nz - 1) * 0.5))
    assert yz.n_points == 4 and yz.bounds == pytest.approx((9 * 2.0, 9 * 2.0, 0, ny - 1, 0, (nz - 1) * 0.5))
    assert tuple(t_xy.dimensions) == (nx, ny) and tuple(t_xz.dimensions) == (nx, nz) and tuple(t_yz.dimensions) == (ny, nz)


def test_volume_slice_texture_orientation():
    """Image row 0 (y = 0) sits at the quad's origin: seen from +z with y up it is at the bottom."""
    import pyvista as pv

    nz, ny, nx = 4, 40, 60
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    vol[:, :8, :] = 1.0  # bright band at small y
    quad, texture = volume_slice_planes(vol, {"z": 2, "y": 20, "x": 30})["xy"]
    pl = pv.Plotter(off_screen=True, window_size=(300, 200))
    pl.add_mesh(quad, texture=texture, lighting=False)
    pl.camera_position = "xy"
    img = pl.screenshot(return_img=True)
    pl.close()
    h = img.shape[0]
    assert img[-h // 3 :].mean() > img[: h // 3].mean() + 30


def test_volume_slice_planes_subsample_large_slice():
    from al_dvc.gui import view3d_scene

    big = np.zeros((2, 2500, 2500), dtype=np.float32)
    planes = volume_slice_planes(big, {"z": 0, "y": 0, "x": 0})
    w, h = planes["xy"][1].dimensions
    assert w * h <= view3d_scene.VOLUME_SLICE_MAX_PIXELS
    assert planes["xy"][0].bounds == pytest.approx((0, 2499, 0, 2499, 0, 0))  # the quad still spans the slice


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


# ----------------------------------------------------------------------------- several iso-surfaces, cut-away, opacity
def test_iso_options_validation():
    opts = SceneOptions()
    assert opts.iso_levels == 1 and not opts.iso_cutaway and opts.cutaway_quadrant == (1, 1)
    assert SceneOptions(iso_levels=MAX_ISO_LEVELS).iso_levels == MAX_ISO_LEVELS == 10
    for bad in (0, -1, MAX_ISO_LEVELS + 1, 2.5, "3"):
        with pytest.raises(ValueError):
            SceneOptions(iso_levels=bad)
    for bad in ((0, 1), (1, 2), (1,), (1, 1, 1), (-1, 0)):
        with pytest.raises(ValueError):
            SceneOptions(cutaway_quadrant=bad)
    assert SceneOptions(cutaway_quadrant=[-1, 1]).cutaway_quadrant == (-1, 1)  # kept as a tuple, so options compare


def test_iso_surface_levels_are_spread_over_the_colour_range():
    assert iso_surface_levels((0.0, 120.0), 5) == (20.0, 40.0, 60.0, 80.0, 100.0)
    assert iso_surface_levels((0.0, 120.0), 1, 0.25) == (30.0,)  # one surface: at the fraction of the range
    assert iso_surface_levels((0.0, 120.0), 1) == (60.0,)
    assert iso_surface_levels((-1.0, 1.0), 3) == pytest.approx((-0.5, 0.0, 0.5))


def test_several_iso_surfaces_one_per_level(rotation_result):
    """Five levels give five nested cylinders, each carrying its level as its field value -- so each takes the colour
    of its level on the colour bar -- at the radius where ``|u| = r / 2`` equals the level."""
    res = rotation_result
    levels = (6.0, 12.0, 18.0, 24.0, 30.0)
    pts, values, info = _surface_points(res, SceneOptions(mode="surface", iso_levels=5, clim=(0.0, 36.0)))
    assert info.iso_levels == pytest.approx(levels) and "iso_level" not in info.actors
    nearest = np.abs(values[:, None] - np.asarray(levels)[None, :]).argmin(axis=1)
    np.testing.assert_allclose(values, np.asarray(levels)[nearest], rtol=1e-6)  # the contour's scalars are the levels
    assert sorted(set(nearest.tolist())) == [0, 1, 2, 3, 4]  # one surface per level
    cx, cy = _grid_centre(res)
    radius = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    for k, level in enumerate(levels):
        r = radius[nearest == k]
        assert np.abs(r - 2.0 * level).max() < 0.1 * ROTATION_STEP, f"level {level}: radius {r.min():.2f}..{r.max():.2f}"
    # a single surface still reports its level under the old key
    _pts, one, info = _surface_points(res, SceneOptions(mode="surface", clim=(0.0, 36.0), iso_fraction=0.5))
    assert info.actors["iso_level"] == pytest.approx(18.0) and info.iso_levels == pytest.approx((18.0,))
    np.testing.assert_allclose(one, 18.0, rtol=1e-6)


def test_cutaway_removes_one_quarter_and_keeps_the_rest(rotation_result):
    """The quarter ``sign(x - cx), sign(y - cy) == cutaway_quadrant`` about the node grid's centre is removed from every
    surface; everything outside it is kept point for point."""
    from dataclasses import replace

    res = rotation_result
    base = SceneOptions(mode="surface", iso_levels=5, clim=(0.0, 36.0))
    whole, _values, _info = _surface_points(res, base)
    cx, cy = _grid_centre(res)
    for quadrant in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        qx, qy = quadrant
        cut, values, info = _surface_points(res, replace(base, iso_cutaway=True, cutaway_quadrant=quadrant))
        depth_whole = np.minimum(qx * (whole[:, 0] - cx), qy * (whole[:, 1] - cy))  # > 0 inside the quarter
        depth_cut = np.minimum(qx * (cut[:, 0] - cx), qy * (cut[:, 1] - cy))
        assert (depth_whole > 1.0).any()  # the surfaces reach into this quarter
        assert (depth_cut <= 1e-6).all(), f"{quadrant}: points left in the removed quarter"
        outside = {tuple(p) for p in np.round(whole[depth_whole < -1e-6], 5)}
        assert outside <= {tuple(p) for p in np.round(cut, 5)}
        # the new points along the cut keep the levels' values (and colours)
        assert np.abs(values[:, None] - np.asarray(info.iso_levels)[None, :]).min(axis=1).max() < 1e-4


def test_facing_quadrant_follows_the_camera(small_result):
    """The direction worked out for a preset or a turned preset is where ``apply_camera`` puts the camera."""
    res, _ = small_result
    specs = [
        "iso",
        "xy",
        "xz",
        "yz",
        CameraSpec(azimuth=180.0),
        CameraSpec(azimuth=-95.0, elevation=-5.0, zoom=1.25),
        CameraSpec(preset="xz", azimuth=30.0),
        CameraSpec(preset="xy", elevation=30.0),
        CameraSpec(view_up="y", azimuth=40.0),
    ]
    pl = pv.Plotter(off_screen=True, window_size=(120, 90))
    build_scene(pl, res, SceneOptions(mode="points"), None)
    for spec in specs:
        apply_camera(pl, spec)
        actual = np.subtract(pl.camera.position, pl.camera.focal_point)
        np.testing.assert_allclose(camera_direction(spec), actual / np.linalg.norm(actual), atol=1e-6, err_msg=str(spec))
        assert facing_quadrant(CameraState.from_camera(pl.camera)) == facing_quadrant(spec)
    pl.close()
    assert facing_quadrant("iso") == (1, 1) == SceneOptions().cutaway_quadrant  # the default faces the default camera
    assert facing_quadrant(CameraSpec(azimuth=180.0)) == (-1, -1)
    assert facing_quadrant(CameraSpec(azimuth=-95.0, elevation=-5.0)) == (1, -1)
    assert facing_quadrant("xz") == (1, -1) and facing_quadrant("yz") == (1, 1)


@pytest.mark.parametrize(
    "mode, extra",
    [("surface", {}), ("surface", {"iso_levels": 5, "iso_cutaway": True}), ("warped", {}), ("points", {})],
)
def test_nan_free_fields_are_drawn_opaque(small_result, mode, extra):
    """The iso-surfaces, the deformed lattice and the points hold no NaN; drawn with a transparent NaN colour they were
    put into VTK's translucent pass at opacity 1, where nested or folded surfaces blend through the near ones."""
    res, _ = small_result
    pl = pv.Plotter(off_screen=True, window_size=(160, 120))
    info = build_scene(pl, res, SceneOptions(mode=mode, **extra), None)
    actor = info.actors["field"]
    drawn = _drawn(actor)  # VTK decides the pass from the mapped scalars, which exist once the mapper ran
    assert np.isfinite(np.asarray(drawn.point_data[info.field])).all()
    assert actor.mapper.lookup_table.GetNanColor()[3] == 1.0
    assert not actor.HasTranslucentPolygonalGeometry()
    translucent = build_scene(pl, res, SceneOptions(mode=mode, opacity=0.5, **extra), None).actors["field"]
    _drawn(translucent)
    assert translucent.HasTranslucentPolygonalGeometry()  # the opacity setting still applies
    pl.close()


def test_slices_keep_the_transparent_nan_colour(small_result):
    """Slices mode is drawn as before: unmeasured nodes transparent, in the translucent pass, with or without the grey
    volume slices it blends with."""
    res, ref = small_result
    for volume in (None, ref):
        pl = pv.Plotter(off_screen=True, window_size=(160, 120))
        info = build_scene(pl, res, SceneOptions(mode="slices", show_volume_slices=volume is not None), volume)
        actor = info.actors["field"]
        _drawn(actor)
        assert actor.mapper.lookup_table.GetNanColor()[3] == 0.0
        assert actor.HasTranslucentPolygonalGeometry()
        assert info.iso_levels == ()
        pl.close()


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


def test_panel_iso_surface_controls(qapp, small_result):
    """Surfaces and Cut away a quarter: shown in the iso-surface mode only, passed to the scene options, and the
    single-surface Iso level is disabled (with its reason in the tooltip) while several surfaces are drawn."""
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow
    from al_dvc.gui.names import select_key

    res, ref = small_result
    window = MainWindow()
    window.show()
    panel = window.view3d
    window.state.set_volume_arrays([ref, ref], ["a", "b"])
    window.state.set_results(res)
    window.center_tabs.setCurrentWidget(panel)
    for _ in range(20):
        QApplication.processEvents()
    assert (panel.iso_levels.minimum(), panel.iso_levels.value(), panel.iso_levels.maximum()) == (1, 1, MAX_ISO_LEVELS)
    assert not panel.iso_cutaway.isChecked() and panel.iso_cutaway.text() and panel.iso_levels.toolTip()
    opts = panel.options()
    assert opts.iso_levels == 1 and not opts.iso_cutaway
    assert not panel.iso_levels.isVisibleTo(panel) and not panel.iso_cutaway.isVisibleTo(panel)  # slices mode
    assert select_key(panel.mode, "surface")
    assert {"iso", "iso_levels", "iso_cutaway"} <= panel.visible_controls()
    one_surface = panel._last_image.copy()
    single_tip = panel.iso.toolTip()
    assert panel.iso.isEnabled() and single_tip and len(panel._last_info.iso_levels) == 1
    # several surfaces: the scene is rebuilt with one surface per level, and the single level is not used
    panel.iso_levels.setValue(3)
    assert panel.options().iso_levels == 3 and len(panel._last_info.iso_levels) == 3
    assert not panel.iso.isEnabled() and panel.iso.toolTip() != single_tip
    assert np.abs(panel._last_image.astype(int) - one_surface.astype(int)).mean() > 0.05
    # the cut-away quarter faces the camera the camera row describes
    assert panel.options().cutaway_quadrant == (1, 1)
    panel.iso_cutaway.setChecked(True)
    assert panel.options().iso_cutaway and panel.options().cutaway_quadrant == (1, 1)
    panel.azimuth.setValue(180)
    assert panel.options().cutaway_quadrant == (-1, -1)  # the isometric camera turned to -x, -y
    assert select_key(panel.camera, "xz")
    panel.azimuth.setValue(-150)
    assert panel.options().cutaway_quadrant == (-1, 1)  # xz (camera at -y) turned by -150 degrees: at -x, +y
    panel.reset_camera()
    assert panel.options().cutaway_quadrant == (1, -1)  # xz: at -y (x = 0 counts as +)
    # the interactive backend follows the mouse-turned camera, unless the camera row is about to replace it
    panel.backend, panel._camera_reset_pending = "interactive", False
    panel._live_state = CameraState((-500.0, -400.0, 300.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert panel.options().cutaway_quadrant == (-1, -1)
    panel._camera_reset_pending = True
    assert panel.options().cutaway_quadrant == facing_quadrant(panel.camera_spec()) == (1, -1)
    panel.backend, panel._live_state = "static", None
    panel.iso_levels.setValue(1)
    assert panel.iso.isEnabled() and panel.iso.toolTip() == single_tip
    select_key(panel.mode, "warped")
    assert not {"iso", "iso_levels", "iso_cutaway"} & panel.visible_controls()
    window.close()


def test_warped_lattice_drops_the_cells_the_mask_separates(small_result):
    """A mesh cut at a boundary must still render, and without the cells that span it."""
    import copy

    from al_dvc.gui.view3d_scene import _live_cells

    res, ref = small_result
    mesh = res.dvc_mesh
    n_cells = (mesh.grid_shape[0] - 1) * (mesh.grid_shape[1] - 1) * (mesh.grid_shape[2] - 1)
    assert _live_cells(res, n_cells) is None  # nothing cut: the fast path

    cut = copy.copy(res)
    cut_mesh_ = copy.copy(mesh)
    elements = np.array(mesh.elements, copy=True)
    elements[0] = -1  # as cut_mesh leaves an element the mask separates
    cut_mesh_.elements = elements
    object.__setattr__(cut, "dvc_mesh", cut_mesh_)
    live = _live_cells(cut, n_cells)
    assert live is not None and live.sum() == n_cells - 1

    shots = []
    for r in (res, cut):
        pl = pv.Plotter(off_screen=True, window_size=(300, 240))
        info = build_scene(pl, r, SceneOptions(mode="warped", show_volume_slices=False), volume=ref)
        assert "field" in info.actors and info.note != "nodes_only"
        img = pl.screenshot(None, return_img=True)
        pl.close()
        assert img.std() > 1.0  # the lattice is on screen, not a blank frame
        shots.append(img.astype(np.float64))
    assert np.abs(shots[0] - shots[1]).mean() > 0.1  # the dropped element is missing from the picture


def test_node_grid_is_in_physical_units(small_result):
    """The field lattice sits where the slices, the outline, the volume planes and the warp vectors are:
    in physical units. It was in voxels, so with a voxel size other than 1 the field floated off the scene."""
    from dataclasses import replace

    from al_dvc.gui.view3d_scene import _slice_positions

    res, _ = small_result
    vs = (2.0, 1.0, 0.5)
    scaled = replace(res, dvc_para=replace(res.dvc_para, voxel_size=vs, units="um"))
    grid = node_grid(scaled, 0, ("disp_u",))
    np.testing.assert_allclose(np.asarray(grid.points), res.dvc_mesh.coordinates * np.asarray(vs), atol=1e-9)
    x, y, z = _slice_positions(scaled, {"x": None, "y": None, "z": None})
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    assert xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax
