"""Offscreen tests of the PySide6 application (skipped when PySide6 is missing)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.core.data_structures import STATUS_CONVERGED  # noqa: E402
from al_dvc.gui.app import MainWindow, create_application  # noqa: E402
from al_dvc.gui.app_state import RunState  # noqa: E402
from al_dvc.gui.i18n import SUPPORTED_LANGUAGES, load_table  # noqa: E402
from al_dvc.gui.names import select_key  # noqa: E402
from al_dvc.gui.session import SessionError, load_session, save_session  # noqa: E402
from al_dvc.io.volume_io import save_volume  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return create_application(["pytest"])


@pytest.fixture(scope="module")
def small_pair():
    shape = (48, 52, 56)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=3)
    dfm = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, -0.005, 0.005]), (0.4, -0.3, 0.2), centre))
    return ref, dfm


def _pump(n=30):
    for _ in range(n):
        QApplication.processEvents()


def _run_window(window, tmp_path):
    window.state.set_params(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    window.state.set_output_dir(tmp_path / "out")
    window.state.write_checkpoints = False
    window.run_panel.start()
    assert window.run_panel.wait(300_000)
    _pump()


def test_window_builds_and_reacts_to_parameters(qapp, small_pair):
    window = MainWindow()
    window.show()
    assert window.state.results is None
    window.param_panel.winsize.setValue(24)
    assert window.state.para.winsize == (24, 24, 24)
    assert select_key(window.param_panel.interp, "bspline")
    assert window.state.para.interp_method == "bspline"
    # an invalid value is refused by the parameter validation and the state is unchanged
    before = window.state.para
    window.param_panel._set("winstepsize", 0)
    assert window.state.para == before
    window.state.set_volume_arrays(list(small_pair), ["ref", "def"])
    assert window.volume_panel._list.count() == 2
    assert "bytes/voxel" in window.param_panel._memory.text()
    assert window.viewer._volume is not None
    window.close()


def test_run_export_and_display(qapp, small_pair, tmp_path):
    window = MainWindow()
    window.state.set_volume_arrays(list(small_pair), ["ref", "def"])
    _run_window(window, tmp_path)
    assert window.state.run_state == RunState.DONE
    res = window.state.results
    assert res is not None and res.n_frames == 1
    assert np.mean(res.result_disp[0].status == STATUS_CONVERGED) > 0.9
    assert "ZNCC" in window.results_panel._summary.text()
    fields = window.results_panel.field_names()
    assert "exx" not in fields and "disp_std" in fields  # strain is a post-processing step now
    sw = window.open_strain_window()
    sw.compute()
    assert sw.wait(120_000)
    qapp.processEvents()
    fields = window.results_panel.field_names()
    assert "exx" in fields
    sw.close()
    assert window.results_panel.select_field("exx")
    assert window.state.display_field == "exx"
    window.viewer.redraw()
    assert window.viewer._cbar is not None
    for kind in ("npz", "vtk", "report"):
        path = window.results_panel.export(kind)
        assert path is not None and path.exists()
    assert window.results_panel.export("nope") is None
    window.close()


def test_stop_request_keeps_partial_results(qapp, small_pair, tmp_path):
    ref, dfm = small_pair
    window = MainWindow()
    window.state.set_volume_arrays([ref, dfm, dfm, dfm], ["a", "b", "c", "d"])
    window.state.set_params(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    window.state.set_output_dir(tmp_path / "out")
    window.state.write_checkpoints = False
    window.run_panel.start()
    window.run_panel.stop()
    assert window.run_panel.wait(300_000)
    _pump()
    assert window.state.run_state in (RunState.DONE, RunState.FAILED)
    if window.state.run_state == RunState.DONE:
        assert window.state.results.stopped_early or window.state.results.n_frames == 3
    window.close()


def test_session_roundtrip(qapp, small_pair, tmp_path):
    ref, dfm = small_pair
    p0, p1 = tmp_path / "ref.npy", tmp_path / "def.npy"
    save_volume(p0, ref)
    save_volume(p1, dfm)
    window = MainWindow()
    window.state.add_volume_paths([str(p0), str(p1)])
    window.state.set_params(winsize=20, winstepsize=10, voxel_size=(2.0, 2.0, 2.0), units="um")
    window.state.set_output_dir(tmp_path / "results")
    window.state.set_display(display_field="disp_u", colormap="magma")
    path = window.save_session_path(tmp_path / "test.aldvc")
    assert path is not None and path.exists()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["volumes"][0]["path"] == "ref.npy"  # relative to the session file
    data = load_session(path)
    assert data.para.winsize == (20, 20, 20) and data.para.units == "um"
    other = MainWindow()
    missing = other.open_session_path(str(path))
    assert missing == []
    assert [v.path for v in other.state.volumes] == [str(p0.resolve()), str(p1.resolve())]
    assert other.state.para.voxel_size == (2.0, 2.0, 2.0)
    assert other.state.colormap == "magma" and other.state.display_field == "disp_u"
    assert other.param_panel.winsize.value() == 21  # odd span of winsize 20
    # a missing file is reported, not fatal
    p1.unlink()
    third = MainWindow()
    assert third.open_session_path(str(path)) == [str(p1.resolve())]
    # in-memory volumes cannot be saved
    mem = MainWindow()
    mem.state.set_volume_arrays([ref, dfm])
    with pytest.raises(SessionError):
        save_session(mem.state, tmp_path / "mem.aldvc")
    window.close()
    other.close()
    third.close()
    mem.close()


def test_language_switch(qapp):
    mgr = qapp._pyaldvc_lang_mgr
    window = MainWindow()
    en = window.run_panel._btn_run.text()
    for code in SUPPORTED_LANGUAGES:
        assert code == "en" or len(load_table(code)) > 100
    mgr.load("zh_CN")
    _pump()
    zh = window.run_panel._btn_run.text()
    assert zh != en and "AL-DVC" in zh
    mgr.load("en")
    _pump()
    assert window.run_panel._btn_run.text() == en
    with pytest.raises(ValueError):
        mgr.load("xx")
    window.close()


def test_self_test_passes(qapp, tmp_path):
    from al_dvc.gui.self_test import run_self_test

    report = tmp_path / "self_test.txt"
    assert run_self_test(report) == 0
    assert "all checks passed" in report.read_text(encoding="utf-8")


def test_odd_subset_size_display(qapp):
    window = MainWindow()
    window.show()
    panel = window.param_panel
    assert panel.winsize.value() == window.state.para.winsize[0] + 1
    panel.winsize.setValue(33)
    assert window.state.para.winsize == (32, 32, 32)
    panel.winsize.setValue(24)  # an even value typed by hand rounds up
    assert panel.winsize.value() == 25 and window.state.para.winsize == (24, 24, 24)
    window.close()


def test_wheel_changes_values_only_when_focused(qapp):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    window = MainWindow()
    window.show()
    spin = window.param_panel.winstepsize
    before = spin.value()

    def wheel(widget):
        event = QWheelEvent(
            QPointF(5, 5),
            QPointF(5, 5),
            QPoint(0, 120),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        qapp.sendEvent(widget, event)

    window.activateWindow()
    qapp.setActiveWindow(window)
    window.param_panel.search_radius.setFocus()
    qapp.processEvents()
    assert not spin.hasFocus()
    wheel(spin)  # pointer over an unfocused box: nothing happens
    assert spin.value() == before
    spin.setFocus()
    qapp.processEvents()
    assert spin.hasFocus()
    wheel(spin)
    assert spin.value() == before + 1
    window.close()


def test_console_and_analysed_box_from_mask(qapp, small_pair):
    window = MainWindow()
    window.show()
    n0 = window.console.count
    window.state.log("hello", "warning")
    assert window.console.count == n0 + 1 and "hello" in window.console.text()
    window.state.set_volume_arrays(list(small_pair), ["ref", "def"])
    assert window.state.effective_voi() is None  # no region drawn: the whole volume
    nz, ny, nx = small_pair[0].shape
    mask = np.zeros(small_pair[0].shape, dtype=bool)
    mask[nz // 2 - 2 : nz // 2 + 2, ny // 2 - 2 : ny // 2 + 2, nx // 2 - 2 : nx // 2 + 2] = True
    window.state.set_mask(0, mask=mask)
    window.state.set_params(winsize=8, winstepsize=4, search_radius=2, admm_max_iter=1, verbose=False)
    voi = window.state.effective_voi()
    assert voi is not None and voi.x[0] > 0 and voi.x[1] < nx - 1
    assert "region of interest" in window.param_panel._memory.text()
    # a region too small for the subset is refused with a message, not a worker traceback
    window.state.set_params(winsize=40)
    messages = []
    window.state.log_message.connect(lambda m, level: messages.append((m, level)))
    window.run_panel.start()
    assert any("do not fit" in m for m, _ in messages)
    assert window.state.run_state == RunState.IDLE
    window.close()


def test_volume_table_region_column_and_reorder(qapp, small_pair):
    window = MainWindow()
    window.show()
    window.state.set_volume_arrays(list(small_pair), ["alpha", "beta"])
    table = window.volume_panel._list
    assert table.count() == 2 and table.item(0, 2).text() == "alpha"
    assert table.item(0, 4).text() == "whole volume" and table.item(1, 4).text() == "-"
    mask = np.zeros(small_pair[0].shape, dtype=bool)
    mask[10:30, 10:30, 10:30] = True
    window.state.set_mask(0, mask=mask)
    assert table.item(0, 4).text().startswith("ROI")
    assert table.cellWidget(0, 0).pixmap() is not None  # thumbnail of the middle slice
    window.state.move_volume(1, 0)
    assert table.item(0, 2).text() == "beta" and window.state.current_frame == 0
    assert table.item(1, 4).text() == "own mask"  # the mask travels with its frame
    window.state.move_volume(0, 5)  # out of range: ignored
    assert table.item(0, 2).text() == "beta"
    window.close()


def test_recent_sessions_menu(qapp, small_pair, tmp_path):
    from PySide6.QtCore import QSettings

    from al_dvc.gui.app import SETTINGS_APP, SETTINGS_ORG

    QSettings(SETTINGS_ORG, SETTINGS_APP).remove("recent_sessions")
    window = MainWindow()
    window.show()
    assert window.recent_sessions() == [] and not window._menus["recent"].isEnabled()
    p0, p1 = tmp_path / "ref.npy", tmp_path / "def.npy"
    save_volume(p0, small_pair[0])
    save_volume(p1, small_pair[1])
    window.state.add_volume_paths([str(p0), str(p1)])
    for k in range(3):
        assert window.save_session_path(tmp_path / f"s{k}.aldvc") is not None
    recent = window.recent_sessions()
    assert [Path(p).name for p in recent] == ["s2.aldvc", "s1.aldvc", "s0.aldvc"]
    assert window._menus["recent"].isEnabled() and len(window._menus["recent"].actions()) == 3
    QSettings(SETTINGS_ORG, SETTINGS_APP).remove("recent_sessions")
    window.close()


def test_sticky_headers_follow_the_scroll(qapp):
    window = MainWindow()
    window.resize(1200, 700)
    window.show()
    qapp.processEvents()
    overlay = window.sticky_headers
    assert overlay.pinned_titles() == []
    scroll = window._left_column
    bar = scroll.verticalScrollBar()
    assert bar.maximum() > 0  # the column is taller than the viewport
    bar.setValue(bar.maximum())
    qapp.processEvents()
    pinned = overlay.pinned_titles()
    assert pinned and pinned[0] == "Volumes"  # the sections above the viewport are pinned in order
    overlay.scroll_to(window._sections["volumes"])
    qapp.processEvents()
    assert bar.value() == 0 and overlay.pinned_titles() == []
    window.close()


def test_application_icon_is_shipped(qapp):
    from pathlib import Path as _Path

    import al_dvc.gui as gui_pkg

    icon = _Path(gui_pkg.__file__).parent / "assets" / "pyALDVC.png"
    assert icon.is_file() and icon.stat().st_size > 1000
    assert not qapp.windowIcon().isNull()
