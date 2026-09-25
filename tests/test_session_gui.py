"""Sessions in the window: a completed analysis saved and reopened in a fresh window comes back as it was left
(results without a rerun, strain in the post-processing window, regions, display, 3-D view, texture analysis),
missing volumes, the folder the user points at, saving and opening on the worker thread, and the unsaved flag.

Every window a test creates is closed and deleted before it returns (an accumulation of windows once made a
single-process CI run hang)."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.gui import session_views  # noqa: E402
from al_dvc.gui.app import MainWindow, create_application  # noqa: E402
from al_dvc.gui.app_state import RunState  # noqa: E402
from al_dvc.gui.names import select_key  # noqa: E402
from al_dvc.io.session_bundle import read_session_document  # noqa: E402
from al_dvc.io.volume_io import save_volume  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402
from tests.test_session import assert_same  # noqa: E402

SHAPE = (40, 44, 48)


@pytest.fixture(scope="module")
def qapp():
    return create_application(["pytest"])


@pytest.fixture(scope="module")
def frames():
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=31)
    out = [ref]
    for k in (1, 2):
        disp = affine_displacement(np.diag([0.005 * k, -0.003 * k, 0.004 * k]), (0.4 * k, -0.3, 0.2), centre)
        out.append(warp_volume_lagrangian(ref, disp))
    return out


def _pump(n: int = 30) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _close(*windows) -> None:
    """Close and delete the windows and the independent windows they opened."""
    for w in windows:
        w.state.dirty = False
        subs = [getattr(w, name, None) for name in ("strain_window", "texture_window", "guide_window")]
        w.close()
        for sub in subs:
            if sub is not None:
                sub.close()
                sub.deleteLater()
        w.deleteLater()
    _pump(5)


def _files(root, frames) -> list:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, vol in enumerate(frames):
        save_volume(root / f"v{k}.npy", vol)
        paths.append(root / f"v{k}.npy")
    return paths


def _run(window, paths, out) -> None:
    window.state.add_volume_paths([str(p) for p in paths])
    assert window.state.wait_for_shapes(10_000)
    window.state.set_params(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    window.state.set_output_dir(out)
    window.state.write_checkpoints = False
    window.run_panel.start()
    assert window.run_panel.wait(300_000)
    _pump()
    assert window.state.run_state == RunState.DONE and window.state.results is not None


def _logged(state) -> list:
    seen: list = []
    state.log_message.connect(lambda message, level: seen.append((level, message)))
    return seen


def test_a_completed_analysis_comes_back_in_a_fresh_window(qapp, frames, tmp_path):
    paths = _files(tmp_path / "data", frames)
    window = MainWindow()
    window.show()
    _run(window, paths, tmp_path / "out")
    state = window.state
    # strain in the post-processing window, with settings of its own
    sw = window.open_strain_window()
    assert select_key(sw.method, "fd") and select_key(sw.measure, "green_lagrange")
    sw.fit_window.setValue(5)
    sw.disp_smoothing.setValue(1.0)
    sw.compute()
    assert sw.wait(120_000)
    _pump()
    assert state.results.result_strain and not sw.is_stale
    # statistics: a region, the motion fitted over it and shown in the main window
    sw.show_tab("analysis")
    tab = sw.analysis
    grip = tab.regions_panel.add("sphere")
    assert select_key(tab.motion, "rigid") and select_key(tab.fit_region_combo, grip.id)
    tab.apply_main.setChecked(True)
    assert tab.wait()
    _pump()
    assert state.display_correction is not None and state.display_correction.fit_region == grip
    sw.show_tab("strain")
    sw.frame_slider.setValue(1)
    assert select_key(sw.field, "exy") and select_key(sw.layout_combo, "row")
    sw.colormap.setCurrentText("viridis")
    # the display of the main window
    state.set_current_frame(2)
    assert window.results_panel.select_field("exx")
    state.set_display(colormap="magma", color_auto=False, color_min=-0.01, color_max=0.02, overlay_alpha=0.5)
    window.viewer.set_layout("row")
    window.viewer.show_mesh.setChecked(False)
    state.set_slice("z", 12)
    # the 3-D view
    v3 = window.view3d
    assert select_key(v3.mode, "surface")
    v3.iso_levels.setValue(3)
    v3.iso_cutaway.setChecked(True)
    v3.arrows.setChecked(True)
    v3.stride.setValue(3)
    i = v3.background.findData("white")
    v3.background.setCurrentIndex(i)
    v3.background.activated.emit(i)  # as the user's pick: the theme no longer decides
    assert select_key(v3.camera, "xy")
    v3.azimuth.setValue(30)
    v3.elevation.setValue(10)
    v3.zoom.setValue(1.5)
    assert select_key(v3.anim_kind, "frames")
    v3.anim_speed.setValue(3.0)
    v3.anim_smooth.setChecked(True)
    window.center_tabs.setCurrentIndex(1)
    # the texture analysis
    tw = window.open_texture_window()
    _pump()
    tw.region.set_centre((24, 22, 20))
    tw.go_to_step(tw.TAB_ACF)
    tw.cube_size.setValue(24)
    tw.factor.setValue(3.0)
    tw.analyse()
    assert tw.wait(120_000)
    _pump()
    assert tw.result is not None and tw.recommendation is not None and not tw.is_stale
    views = session_views.collect(window)
    expected = {
        "results": state.results,
        "uids": [v.uid for v in state.volumes],
        "result_uids": list(state.result_uids),
        "regions": list(state.regions),
        "correction": state.display_correction,
        "settings": dict(state.analysis_settings),
        "view3d": views["view3d"],
        "post": {k: views["post"][k] for k in ("tab", "frame", "controls", "display")},
        "texture": {k: views["texture"][k] for k in ("step", "centre", "cube_size", "factor", "sweep", "result_current")},
        "texture_result": tw.result,
        "recommendation": tw.recommendation,
    }
    path = window.save_session_path(tmp_path / "proj" / "s.aldvc")
    assert path is not None and not state.dirty
    doc = read_session_document(path)
    assert doc["results"]["strain"] and doc["views"]["post"]["open"] and doc["texture"]["archive"] == "texture.npz"
    _close(window)

    fresh = MainWindow()
    fresh.show()
    seen = _logged(fresh.state)
    assert fresh.open_session_path(str(path)) == []
    _pump()
    st = fresh.state
    assert st.run_state == RunState.DONE and not st.dirty  # nothing to rerun, nothing unsaved
    assert_same(expected["results"], st.results)
    assert [v.uid for v in st.volumes] == expected["uids"] and st.result_uids == expected["result_uids"]
    assert st.regions == expected["regions"] and st.display_correction == expected["correction"]
    assert st.analysis_settings == expected["settings"]
    assert st.display_field == "exx" and st.colormap == "magma" and not st.color_auto and st.overlay_alpha == 0.5
    assert st.current_frame == 2 and st.slice_index["z"] == 12
    assert fresh.viewer.layout_key == "row" and not fresh.viewer.show_mesh.isChecked()
    assert fresh.results_panel.field.currentData() == "exx" and fresh.results_panel._export_group.isEnabled()
    assert fresh.center_tabs.currentIndex() == 1
    assert session_views.view3d_state(fresh.view3d) == expected["view3d"]
    # the post-processing window was open: it is again, with the strain and its settings
    sw2 = fresh.strain_window
    assert sw2.isVisible() and not sw2.is_stale
    assert {k: session_views.post_state(sw2)[k] for k in ("tab", "frame", "controls", "display")} == expected["post"]
    assert "exx" in [sw2.field.itemData(k) for k in range(sw2.field.count())]
    assert "Strain:" in sw2._status.text()
    assert sw2.analysis.motion_kind() == "rigid" and sw2.analysis.fit_region() == expected["regions"][0]
    # the texture window too, with its analysis still describing the input
    tw2 = fresh.texture_window
    assert tw2.isVisible() and not tw2.is_stale and tw2.tabs.currentIndex() == tw2.TAB_ACF
    assert_same(expected["texture_result"], tw2.result)
    assert tw2.recommendation == expected["recommendation"] and tw2._btn_apply.isEnabled()
    assert {k: session_views.texture_state(tw2)[0][k] for k in expected["texture"]} == expected["texture"]
    assert not [m for level, m in seen if level == "error"]
    _pump()
    assert not st.dirty
    _close(fresh)


def test_missing_volumes_are_marked_and_the_results_still_show(qapp, frames, tmp_path):
    paths = _files(tmp_path / "data", frames)
    window = MainWindow()
    _run(window, paths, tmp_path / "out")
    path = window.save_session_path(tmp_path / "proj" / "s.aldvc")
    _close(window)
    shutil.rmtree(tmp_path / "data")
    fresh = MainWindow()
    fresh.show()
    seen = _logged(fresh.state)
    missing = fresh.open_session_path(str(path))
    _pump()
    assert missing == [str(p) for p in paths]  # asked for nothing (headless), kept in the list, marked
    st = fresh.state
    assert all(v.missing for v in st.volumes) and st.results is not None and st.run_state == RunState.DONE
    assert "not found" in fresh.viewer._empty.text() and fresh.viewer._empty.isVisible()
    assert "⚠" in fresh.volume_panel._list.item(0, 2).text()
    sw = fresh.open_strain_window()  # the post-processing works on the result alone
    _pump()
    assert sw._btn_compute.isEnabled()
    assert any(level == "warning" and "were not found" in m for level, m in seen)
    assert not [m for level, m in seen if level == "error"]
    _close(fresh)


def test_the_folder_the_user_points_at_is_searched(qapp, frames, tmp_path):
    paths = _files(tmp_path / "data", frames)
    window = MainWindow()
    _run(window, paths, tmp_path / "out")
    path = window.save_session_path(tmp_path / "proj" / "s.aldvc")
    _close(window)
    moved = tmp_path / "somewhere" / "else"
    moved.parent.mkdir()
    shutil.move(str(tmp_path / "data"), str(moved))
    fresh = MainWindow()
    asked = []
    fresh.locate_volumes_folder = lambda first: asked.append(first) or str(moved)  # the user's choice
    assert fresh.open_session_path(str(path)) == []
    assert asked == [str(paths[0])]
    assert [v.path for v in fresh.state.volumes] == [str(moved / p.name) for p in paths]
    assert fresh.viewer._volume is not None  # the grey values are back
    _close(fresh)


def test_saving_and_opening_run_on_a_worker(qapp, frames, tmp_path, monkeypatch):
    paths = _files(tmp_path / "data", frames)
    window = MainWindow()
    _run(window, paths, tmp_path / "out")
    seen = _logged(window.state)
    target = tmp_path / "proj" / "s.aldvc"
    assert window.save_session_path(target, wait=False) is None  # returns at once
    assert window.sessions.busy
    assert window.save_session_path(tmp_path / "other.aldvc") is None  # one session job at a time
    assert any("being saved or opened" in m for _level, m in seen)
    assert window.sessions.wait(60_000)
    _pump()
    assert target.is_file() and window.state.session_path == target and not window.sessions.busy
    assert not window.state.dirty
    assert window.recent_sessions()[0] == str(target)
    # a failed save keeps the changes marked as unsaved and says why
    from al_dvc.gui import session_ops
    from al_dvc.gui.session import SessionError

    def fail(*_a, **_k):
        raise SessionError("cannot write session: disk full")

    monkeypatch.setattr(session_ops, "write_session", fail)
    window.state.set_params(winsize=20)
    assert window.state.dirty
    assert window.save_session_path(target) is None
    assert window.state.dirty and any("disk full" in m for _level, m in seen)
    monkeypatch.undo()
    _close(window)
    # a session dropped on the window opens there, on the worker too
    fresh = MainWindow()
    assert fresh.volume_panel.add_dropped([str(target)]) == 0
    assert fresh.sessions.busy or fresh.state.results is not None
    assert fresh.sessions.wait(60_000)
    _pump()
    assert fresh.state.results is not None and len(fresh.state.volumes) == 3 and not fresh.state.dirty
    fresh.state.set_params(winsize=24)  # an edit after opening: unsaved again
    assert fresh.state.dirty
    _close(fresh)
