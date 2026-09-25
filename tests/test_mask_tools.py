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
from al_dvc.io.session_bundle import read_session_document  # noqa: E402
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
    viewer.mask_tools.target.setCurrentIndex(1)  # all frames: where the next operations go, nothing copied yet
    assert state.volumes[1].mask is None
    assert state.copy_mask_to_all_frames()  # the explicit, reversible copy
    assert state.volumes[1].mask is not None and state.volumes[1].mask is not state.volumes[0].mask
    np.testing.assert_array_equal(state.volumes[1].mask, state.volumes[0].mask)
    assert state.undo_mask() and state.volumes[1].mask is None  # undo reverses the copy first
    assert state.copy_mask_to_all_frames()
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
    # the session holds the composed mask itself (and remembers the file it came from): nothing is replayed
    assert state.volumes[0].mask_ops is None
    session = window.save_session_path(tmp_path / "masked.aldvc")
    doc = load_session(session)
    assert doc.volumes[0]["mask_ops"] is None and doc.volumes[0]["mask_path"] is None
    np.testing.assert_array_equal(doc.volumes[0]["mask"], drawn)
    assert read_session_document(session)["volumes"][0]["mask_source"]["relative"] == "mask.tif"
    window.close()
    window2 = MainWindow()
    window2.show()
    assert window2.open_session_path(str(session)) == []
    entry = window2.state.volumes[0]
    assert entry.mask_ops is None
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


# --------------------------------------------------------------------------- the automatic mask, off the UI thread
def test_threshold_region_reports_its_stages_and_stops_between_them():
    from al_dvc.gui.mask_editor import ThresholdCancelled, threshold_region

    vol = generate_speckle_volume((24, 28, 32), sigma=2.0, seed=3)
    stages: list = []
    region = threshold_region(vol, progress=lambda f, s: stages.append((f, s)))
    assert [s for _, s in stages] == ["threshold", "fill holes", "largest component", "done"]
    assert [f for f, _ in stages] == sorted(f for f, _ in stages)
    assert region.dtype == bool and region.shape == vol.shape
    assert np.array_equal(region, threshold_region(vol))  # the callbacks change nothing

    seen: list = []
    with pytest.raises(ThresholdCancelled):  # asked to stop during the first stage: never reaches the second
        threshold_region(vol, progress=lambda f, s: seen.append(s), stop=lambda: bool(seen))
    assert seen == ["threshold"]


def test_apply_computed_equals_apply_and_the_replay_reuses_the_region(monkeypatch):
    from al_dvc.gui import mask_editor as me
    from al_dvc.gui.mask_editor import MaskEditor

    vol = generate_speckle_volume((24, 28, 32), sigma=2.0, seed=3)
    op = MaskOp("threshold", mode="replace")
    direct = MaskEditor(vol.shape, base=None, volume=vol)
    direct.apply(op)
    region = me.threshold_region(vol)

    calls: list = []
    real = me.threshold_region
    monkeypatch.setattr(me, "threshold_region", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ed = MaskEditor(vol.shape, base=None, volume=vol)
    ed.apply_computed(op, region)
    assert np.array_equal(ed.mask, direct.mask) and ed.ops == [op] and calls == []

    ed.apply(MaskOp("rectangle", "xy", ((2.0, 2.0), (10.0, 10.0)), mode="cut", depth=(3, 8)))
    ed.undo()  # replays the threshold: from the kept region, not a new computation on the UI thread
    assert np.array_equal(ed.mask, direct.mask) and calls == []
    ed.undo()  # the threshold itself moves to the redo stack; its region stays for a redo
    assert not ed.mask.any() and ed._threshold_cache
    ed.apply(MaskOp("fill"))  # a new operation clears the redo stack: the region has no operation left, so it goes
    assert ed._threshold_cache == {}
    with pytest.raises(ValueError):
        ed.apply_computed(MaskOp("fill"), region)


def test_the_automatic_mask_runs_off_the_ui_thread_and_lands_in_the_history(qapp, pair):
    """The button starts a worker, reads Cancel meanwhile, and the result is an ordinary undoable operation."""
    from al_dvc.gui.mask_editor import threshold_region

    window = _window(qapp, pair)
    tools, state = window.viewer.mask_tools, window.state
    states: list = []
    state.auto_mask_state.connect(states.append)
    assert not state.auto_mask_running()

    tools._on_auto()
    worker = state._auto_mask
    assert worker is not None and state.auto_mask_running()
    assert tools._btn["auto"].text() == "Cancel" and not tools._btn["invert"].isEnabled()
    assert worker.wait(120_000)
    _pump(40)
    assert states[0] == "started" and states[-1] == "finished"
    assert not state.auto_mask_running()
    mask = state.current_mask()
    assert mask is not None and mask.any() and not mask.all()
    assert np.array_equal(mask, threshold_region(state.volume_array(0)))  # what the UI thread used to compute
    ed = state.mask_editor
    assert ed.ops[-1].shape == "threshold" and ed.can_undo
    assert tools._btn["auto"].text() != "Cancel" and tools._btn["invert"].isEnabled()
    window.close()


def _held_threshold(monkeypatch, gate):
    """threshold_region that reports its first stage, then waits for the test before going on."""
    from al_dvc.gui import app_state as st
    from al_dvc.gui import mask_editor as me

    real = me.threshold_region

    def slow(vol, *args, progress=None, stop=None, **kw):
        if progress is not None:
            progress(0.0, "threshold")
        gate.wait(60)
        if stop is not None and stop():
            raise me.ThresholdCancelled()
        return real(vol, *args, progress=progress, stop=stop, **kw)

    monkeypatch.setattr(st, "threshold_region", slow)


def test_cancelling_the_automatic_mask_leaves_the_history_alone(qapp, pair, monkeypatch):
    import threading

    gate = threading.Event()
    _held_threshold(monkeypatch, gate)
    window = _window(qapp, pair)
    tools, state = window.viewer.mask_tools, window.state
    states: list = []
    state.auto_mask_state.connect(states.append)
    assert state.mask_editor is None and state.current_mask() is None

    tools._on_auto()
    worker = state._auto_mask
    assert state.auto_mask_running()
    tools._on_auto()  # the same button, now Cancel
    gate.set()
    assert worker.wait(60_000)
    _pump(40)
    assert states == ["started", "cancelled"]
    assert state.mask_editor is None and state.current_mask() is None  # nothing was applied, nothing created
    assert tools._btn["auto"].text() != "Cancel" and tools._btn["invert"].isEnabled()
    window.close()


def test_a_result_for_a_frame_that_is_no_longer_selected_is_discarded(qapp, pair, monkeypatch):
    import threading

    gate = threading.Event()
    _held_threshold(monkeypatch, gate)
    window = _window(qapp, pair)
    tools, state = window.viewer.mask_tools, window.state
    states: list = []
    state.auto_mask_state.connect(states.append)

    tools._on_auto()
    worker = state._auto_mask
    state.set_current_frame(1)  # the user moves on while the mask of frame 0 is being computed
    _pump()
    gate.set()
    assert worker.wait(120_000)
    _pump(40)
    assert states[-1] == "cancelled" and not state.auto_mask_running()
    assert state.volumes[0].mask is None and state.volumes[1].mask is None  # applied to neither
    window.close()


# --------------------------------------------------------------------------- saving the mask, off the UI thread
def test_write_mask_file_writes_uint8_without_copying_a_contiguous_mask(tmp_path, monkeypatch):
    from al_dvc.gui import app_state as st

    mask = np.zeros((12, 14, 16), dtype=bool)
    mask[2:8, 3:9, 4:10] = True
    seen = {}
    real = st.write_mask_file.__globals__  # noqa: F841 - keep the module alive for the patch below
    import al_dvc.io.volume_io as vio

    orig = vio.save_volume

    def spy(path, vol, **kw):
        seen["shares"] = np.shares_memory(vol, mask)
        seen["dtype"] = vol.dtype
        return orig(path, vol, **kw)

    monkeypatch.setattr(vio, "save_volume", spy)
    out = st.write_mask_file(tmp_path / "m.npy", mask)
    assert seen["dtype"] == np.uint8 and seen["shares"]  # a view of the boolean array, not a copy
    assert np.array_equal(load_volume(out).astype(bool), mask)


def test_saving_the_mask_runs_off_the_ui_thread_and_attaches_the_file(qapp, pair, tmp_path):
    window = _window(qapp, pair)
    tools, state = window.viewer.mask_tools, window.state
    state.apply_mask_op(MaskOp("rectangle", "xy", ((4.0, 4.0), (30.0, 28.0)), depth=(3, 20)))
    _pump()
    drawn = state.current_mask().copy()
    events: list = []
    state.save_mask_state.connect(lambda s, p: events.append(s))
    out = tmp_path / "saved_mask.tif"

    worker = state.start_save_mask(out)
    assert worker is not None and state.save_mask_running() and not tools._btn["save"].isEnabled()
    assert state.start_save_mask(out) is None  # one at a time
    assert worker.wait(60_000)
    _pump(40)
    assert events == ["started", "finished"] and not state.save_mask_running()
    assert np.array_equal(load_volume(out).astype(bool), drawn)
    assert state.volumes[0].mask_path == str(out) and state.volumes[0].mask_ops is None
    assert state.mask_editor is not None and not state.mask_editor.can_undo  # the file is the new starting point
    assert tools._btn["save"].isEnabled()
    window.close()


def test_a_drawing_made_while_the_mask_is_being_saved_is_kept(qapp, pair, tmp_path, monkeypatch):
    """The file holds the mask as it was; the editor is not reset over the newer drawing."""
    import threading

    from al_dvc.gui import app_state as st

    gate = threading.Event()
    real = st.write_mask_file

    def slow(path, mask):
        gate.wait(60)
        return real(path, mask)

    monkeypatch.setattr(st, "write_mask_file", slow)
    window = _window(qapp, pair)
    state = window.state
    state.apply_mask_op(MaskOp("rectangle", "xy", ((4.0, 4.0), (30.0, 28.0)), depth=(3, 20)))
    _pump()
    before = state.current_mask().copy()
    out = tmp_path / "saved_mask.npy"
    worker = state.start_save_mask(out)
    state.apply_mask_op(MaskOp("rectangle", "xy", ((6.0, 6.0), (12.0, 12.0)), mode="cut", depth=(3, 20)))  # meanwhile
    _pump()
    after = state.current_mask().copy()
    assert not np.array_equal(before, after)
    gate.set()
    assert worker.wait(60_000)
    _pump(40)
    assert np.array_equal(load_volume(out).astype(bool), before)  # what was there when the save started
    assert np.array_equal(state.current_mask(), after)  # the newer drawing survived
    assert state.mask_editor.can_undo  # and its history was not reset
    window.close()


def test_a_failed_save_attaches_nothing(qapp, pair, tmp_path):
    window = _window(qapp, pair)
    state = window.state
    state.apply_mask_op(MaskOp("fill"))
    _pump()
    events: list = []
    state.save_mask_state.connect(lambda s, p: events.append(s))
    worker = state.start_save_mask(tmp_path / "mask.unsupported")
    assert worker.wait(60_000)
    _pump(40)
    assert events == ["started", "failed"] and not state.save_mask_running()
    assert state.volumes[0].mask_path is None
    window.close()
