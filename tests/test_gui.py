"""Offscreen tests of the PySide6 application (skipped when PySide6 is missing)."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.core.data_structures import STATUS_CONVERGED  # noqa: E402
from al_dvc.gui.app import MainWindow, create_application  # noqa: E402
from al_dvc.gui.app_state import RunState  # noqa: E402
from al_dvc.gui.i18n import SUPPORTED_LANGUAGES, load_table  # noqa: E402
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
    window.param_panel.interp.setCurrentText("bspline")
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
    fields = [window.results_panel.field.itemText(i) for i in range(window.results_panel.field.count())]
    assert "exx" in fields and "disp_std" in fields
    window.results_panel.field.setCurrentText("exx")
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
    assert other.param_panel.winsize.value() == 20
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
