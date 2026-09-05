"""Mask drawing on the slice viewer: gestures, toolbar, sessions, and the pipeline using the drawn mask."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.gui.app import MainWindow, create_application  # noqa: E402
from al_dvc.gui.app_state import RunState  # noqa: E402
from al_dvc.gui.mask_editor import MaskOp  # noqa: E402
from al_dvc.gui.session import load_session  # noqa: E402
from al_dvc.io.volume_io import load_volume, save_volume  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE = (40, 48, 56)  # (nz, ny, nx)


@pytest.fixture(scope="module")
def qapp():
    return create_application(["pytest"])


@pytest.fixture(scope="module")
def pair():
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=8)
    dfm = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, -0.005, 0.005]), (0.5, -0.3, 0.2), centre))
    return ref, dfm


def _pump(n: int = 10) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _window(qapp, pair) -> MainWindow:
    window = MainWindow()
    window.show()
    window.state.set_volume_arrays(list(pair), ["ref", "def"])
    window.viewer.canvas.draw()  # transforms must exist before synthetic mouse events
    _pump()
    return window


def _event(viewer, plane_index: int, h: float, v: float, kind: str, button: int = 1, dblclick: bool = False) -> MouseEvent:
    ax = viewer.axes[plane_index]
    x, y = ax.transData.transform((h, v))
    return MouseEvent(kind, viewer.canvas, x, y, button=button, dblclick=dblclick)


def drag(viewer, plane_index: int, p0, p1, button: int = 1) -> None:
    viewer._on_press(_event(viewer, plane_index, *p0, "button_press_event", button))
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    viewer._on_motion(_event(viewer, plane_index, *mid, "motion_notify_event", button))
    viewer._on_motion(_event(viewer, plane_index, *p1, "motion_notify_event", button))
    viewer._on_release(_event(viewer, plane_index, *p1, "button_release_event", button))


def click(viewer, plane_index: int, p, button: int = 1) -> None:
    viewer._on_press(_event(viewer, plane_index, *p, "button_press_event", button))
    viewer._on_release(_event(viewer, plane_index, *p, "button_release_event", button))


def test_rectangle_drag_on_xy_extrudes_through_all_slices(qapp, pair):
    window = _window(qapp, pair)
    viewer, state = window.viewer, window.state
    assert state.mask_editor is None
    drag(viewer, 0, (10, 12), (30, 40))  # tool is off: nothing happens
    assert state.mask_editor is None
    viewer.mask_tools.set_tool("rectangle")
    drag(viewer, 0, (10, 12), (30, 40))
    ed = state.mask_editor
    assert ed is not None and len(ed.ops) == 1 and ed.ops[0].shape == "rectangle" and ed.ops[0].depth is None
    mask = state.volumes[0].mask
    assert mask is not None and mask.shape == SHAPE
    assert mask[:, 12:41, 10:31].all() and not mask[:, :12, :].any() and not mask[:, :, 31:].any()
    assert state.volumes[1].mask is None  # target: current frame only
    assert "material" in viewer.mask_tools._status.text()
    assert viewer.mask_tools._btn["undo"].isEnabled()
    # a preview does not linger after the gesture
    assert viewer._gesture is None
    window.close()


def test_depth_rules_and_cut_mode(qapp, pair):
    window = _window(qapp, pair)
    viewer, state = window.viewer, window.state
    viewer.mask_tools.set_tool("rectangle")
    state.apply_mask_op(MaskOp("fill"))
    # cut a box on the XZ plane (normal y) restricted to the current y slice
    viewer.mask_tools.mode.setCurrentIndex(1)  # cut
    viewer.mask_tools.set_depth("current")
    iy = viewer.slice_indices()[1]
    drag(viewer, 1, (5, 6), (20, 18))
    mask = state.current_mask()
    assert not mask[6:19, iy, 5:21].any()
    assert mask[6:19, iy + 1, 5:21].all() and mask[6:19, iy - 1, 5:21].all()
    # range on the YZ plane (normal x)
    viewer.mask_tools.set_depth("range", 3, 5)
    drag(viewer, 2, (0, 0), (47, 39))
    mask = state.current_mask()
    assert not mask[:, :, 3:6].any() and mask[0, :, 6].all()  # z = 0 was untouched by the first cut
    assert len(state.mask_editor.ops) == 3
    window.close()


def test_polygon_brush_ellipse_and_escape(qapp, pair):
    window = _window(qapp, pair)
    viewer, state = window.viewer, window.state
    viewer.mask_tools.set_tool("polygon")
    for p in [(5, 5), (40, 5), (40, 30)]:
        click(viewer, 0, p)
    assert viewer._gesture is not None and len(viewer._gesture["points"]) == 3
    click(viewer, 0, (5, 30), button=3)  # right-click closes (the point itself is not added)
    assert viewer._gesture is None
    mask = state.current_mask()
    assert mask[:, 6, 38].all() and not mask[:, 28, 6].any()
    # escape cancels a polygon in progress
    click(viewer, 0, (1, 1))
    viewer._on_key(KeyEvent("key_press_event", viewer.canvas, "escape"))
    assert viewer._gesture is None and len(state.mask_editor.ops) == 1
    # brush stroke with radius 3 on the YZ plane
    viewer.mask_tools.set_tool("brush")
    viewer.mask_tools.radius.setValue(3)
    drag(viewer, 2, (10, 20), (30, 20))
    mask = state.current_mask()
    assert mask[20, 10:31, :].all()  # the stroke spans every x (depth: all slices)
    assert mask[23, 20, 2] and not mask[24, 20, 2]  # x = 2 lies outside the polygon: only the stroke reaches it
    # ellipse
    viewer.mask_tools.set_tool("ellipse")
    drag(viewer, 1, (20, 10), (50, 30))
    assert state.current_mask()[20, :, 35].all()
    assert len(state.mask_editor.ops) == 3
    window.close()


def test_undo_redo_invert_and_target_all_frames(qapp, pair):
    window = _window(qapp, pair)
    viewer, state = window.viewer, window.state
    viewer.mask_tools.set_tool("rectangle")
    drag(viewer, 0, (0, 0), (27, 47))
    cov = state.mask_editor.coverage
    assert state.undo_mask() and state.mask_editor.coverage == 0.0
    assert state.redo_mask() and state.mask_editor.coverage == cov
    viewer.mask_tools._btn["invert"].click()
    assert state.mask_editor.coverage == pytest.approx(1.0 - cov)
    viewer.mask_tools.target.setCurrentIndex(1)  # all frames
    assert state.volumes[1].mask is not None
    np.testing.assert_array_equal(state.volumes[1].mask, state.volumes[0].mask)
    viewer.mask_tools.show_mask.setChecked(False)
    assert not state.show_mask
    viewer.redraw()
    viewer.mask_tools._btn["remove"].click()
    assert state.volumes[0].mask is None and state.mask_editor is None
    window.close()


def test_save_mask_and_session_roundtrip(qapp, pair, tmp_path):
    ref, dfm = pair
    p0, p1 = tmp_path / "ref.npy", tmp_path / "def.npy"
    save_volume(p0, ref)
    save_volume(p1, dfm)
    window = MainWindow()
    window.show()
    window.state.add_volume_paths([str(p0), str(p1)])
    window.viewer.canvas.draw()
    viewer, state = window.viewer, window.state
    viewer.mask_tools.set_tool("ellipse")
    drag(viewer, 0, (5, 5), (50, 42))
    viewer.mask_tools.set_tool("rectangle")
    viewer.mask_tools.mode.setCurrentIndex(1)
    drag(viewer, 0, (20, 20), (30, 30))
    drawn = state.current_mask().copy()
    out = state.save_mask(tmp_path / "mask.tif")
    assert out.exists() and state.volumes[0].mask_path == str(out)
    np.testing.assert_array_equal(load_volume(out) > 0, drawn)
    # the session stores the drawing operations and the mask file
    session = window.save_session_path(tmp_path / "masked.aldvc")
    doc = load_session(session)
    assert doc.volumes[0]["mask_ops"] is not None and len(doc.volumes[0]["mask_ops"]["ops"]) == 2
    window.close()
    window2 = MainWindow()
    window2.show()
    assert window2.open_session_path(str(session)) == []
    entry = window2.state.volumes[0]
    assert entry.mask_ops is not None
    np.testing.assert_array_equal(entry.mask, drawn)
    np.testing.assert_array_equal(window2.state.current_mask(), drawn)
    window2.close()


def test_pipeline_uses_the_drawn_mask(qapp, pair, tmp_path):
    window = _window(qapp, pair)
    viewer, state = window.viewer, window.state
    viewer.mask_tools.set_tool("rectangle")
    viewer.mask_tools.target.setCurrentIndex(1)  # both frames
    drag(viewer, 0, (0, 0), (27, 47))  # material: x <= 27
    state.set_params(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    state.set_output_dir(tmp_path / "out")
    state.write_checkpoints = False
    window.run_panel.start()
    assert window.run_panel.wait(300_000)
    _pump()
    assert state.run_state == RunState.DONE
    res = state.results
    mesh = res.dvc_mesh
    x = mesh.coordinates[:, 0]
    assert not mesh.node_valid[x > 27 + 8].any()  # subsets fully outside the material are dropped
    assert mesh.node_valid[x < 27 - 8].all()
    fr = res.result_disp[0]
    inside = mesh.node_valid & (x < 19)
    assert np.isfinite(fr.U[inside]).all()
    window.close()


def test_drag_beyond_the_image_edge_hugs_the_border(qapp, pair):
    window = _window(qapp, pair)
    viewer, state = window.viewer, window.state
    viewer.mask_tools.set_tool("rectangle")
    nz, ny, nx = pair[0].shape
    drag(viewer, 0, (10, 12), (nx + 40, ny + 30))  # the pointer leaves the axes: clamped to the far corner
    mask = state.current_mask()
    assert mask[:, 12:, 10:].all() and not mask[:, :12, :].any() and not mask[:, :, :10].any()
    viewer.mask_tools.set_mode("replace")
    drag(viewer, 1, (nx // 2, nz // 2), (-50, -50))  # ends beyond the origin: clamped to voxel 0
    mask = state.current_mask()
    assert mask[: nz // 2 + 1, :, : nx // 2 + 1].all() and not mask[nz // 2 + 1 :, :, :].any()
    window.close()
