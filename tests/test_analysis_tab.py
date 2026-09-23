"""The Analysis tab of the post-processing window, driven offscreen on results whose statistics are known."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from tests.test_analysis import make_result, rigid_in_physical  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application([])


@pytest.fixture
def window(qapp):
    from al_dvc.gui.app import MainWindow

    w = MainWindow()
    yield w
    w.state.dirty = False
    w.close()


def _select(combo, key) -> None:
    idx = combo.findData(key)
    assert idx >= 0, key
    combo.setCurrentIndex(idx)


def test_the_statistics_open_on_the_analysis_tab_with_the_current_frame(window, qapp):
    fns = [lambda x, y, z, a=a: (a + 0 * x, 0.002 * (y - 60.0), 0 * z) for a in (0.25, 0.5)]
    window.state.set_results(make_result(fns, voxel_size=(2.0, 2.0, 2.0)))
    assert window.results_panel._btn_statistics.isEnabled()
    assert window._actions["statistics"].shortcut().toString() == "Ctrl+Shift+T"
    post = window.open_statistics()
    tab = post.analysis
    assert post.tabs.currentWidget() is tab
    assert tab.wait()
    # the displacement group: u, v, w, |u| with the physical unit
    assert tab.table.rowCount() == 4
    assert "um" in tab.table.verticalHeaderItem(0).text()
    mean_u = float(tab.table.item(0, 1).text())
    assert mean_u == pytest.approx(0.5)  # 0.25 voxel x 2 um
    assert "of" in tab.nodes_info.text()
    assert tab.canvas.has_field()
    # another frame, then back through the strain tab: the two tabs show the same frame
    tab.frame_slider.setValue(1)
    assert tab.wait()
    assert float(tab.table.item(0, 1).text()) == pytest.approx(1.0)
    post.show_tab("strain")
    assert post.frame_slider.value() == 1
    post.show_tab("analysis")
    # the strain groups are offered because the result carries strain
    groups = [tab.group.itemData(i) for i in range(tab.group.count())]
    assert groups[:2] == ["displacement", "strain"] or "strain" in groups
    _select(tab.group, "strain")
    assert tab.wait()
    assert tab.table.rowCount() == 6
    assert window.settle_workers(5_000) == []


def test_removing_the_rigid_motion_in_the_tab(window, qapp):
    fn, _R = rigid_in_physical([0.0, 0.0, 4.0], [3.0, -2.0, 5.0], (60.0, 60.0, 70.0), (1.0, 1.0, 1.0))
    window.state.set_results(make_result([fn], strain_type="infinitesimal"))
    tab = window.open_statistics().analysis
    assert tab.wait()
    raw_std_v = float(tab.table.item(1, 2).text())
    assert raw_std_v > 0.5  # a 4-degree rotation spreads v over the grid
    _select(tab.motion, "rigid")
    assert tab.wait()
    assert "4" in tab.motion_info.text() and "deg" in tab.motion_info.text()
    for row in range(3):  # nothing left of u, v, w once the rigid motion is removed
        assert abs(float(tab.table.item(row, 1).text())) < 1e-6
    _select(tab.group, "strain")
    assert tab.wait()
    assert abs(float(tab.table.item(1, 3).text())) < 1e-6  # eyy median: the cos(4 deg) - 1 is gone
    _select(tab.shown, "nodes_used")
    assert tab.wait() and tab.canvas.has_field()


def test_series_noise_floor_and_exports(window, qapp, tmp_path):
    fns = [lambda x, y, z, a=a: (a + 0 * x, 0 * y, 0.001 * (z - 70.0)) for a in (0.1, 0.2, 0.3)]
    window.state.set_results(make_result(fns))
    window.state.set_output_dir(tmp_path)
    tab = window.open_statistics().analysis
    assert tab.wait()
    assert not tab._btn_csv.isEnabled()  # nothing to save before the series is computed
    tab.compute_series()
    assert tab.wait()
    assert tab.series_is_current() and len(tab.series[1][0][1]) == 3  # one region (every node), three frames
    out = tab.export_csv(tmp_path / "series.csv")
    text = out.read_text(encoding="utf-8")
    assert "disp_u_mean" in text and text.count("\n") > 3
    _select(tab.motion, "translation")  # the series no longer describes the controls
    assert tab.wait()
    assert not tab.series_is_current() and "Out of date" in tab.series_info.text()
    assert tab.export_csv(tmp_path / "stale.csv") is None and not (tmp_path / "stale.csv").exists()
    tab.compute_noise_floor()
    assert tab.wait()
    assert tab.noise is not None and tab.noise_table.rowCount() >= 4
    doc = json.loads(tab.export_json(tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert doc["meta"]["motion"] == "translation" and "noise_floor" in doc and doc["frame"]["stats"]["disp_u"]["n"] > 0
    tab.results_tabs.setCurrentIndex(1)
    assert tab.export_png(tmp_path / "hist.png").stat().st_size > 1000
    copied = tab.copy_table()
    assert copied.splitlines()[0].count("\t") == 12  # a blank corner, the ten statistics, n_eff and the CI
    assert window.settle_workers(5_000) == []


def test_a_correction_needing_more_nodes_than_are_left_says_so(window, qapp):
    n = 9 * 10 * 11
    status = np.ones(n, dtype=np.int8)
    status[:2] = 0  # two converged nodes: too few for a rigid fit
    window.state.set_results(make_result([lambda x, y, z: (0 * x, 0 * y, 0 * z)], status=status))
    tab = window.open_statistics().analysis
    assert tab.wait()
    _select(tab.motion, "rigid")
    assert tab.wait()
    assert "Too few nodes" in tab.nodes_info.text()


def test_the_histogram_of_a_constant_field():
    """A pure translation makes u constant: the histogram used to fail with 'too many bins for data range'."""
    from al_dvc.gui.analysis_tab import _histogram

    for value in (0.5, 0.0, -1234.5):
        counts, edges = _histogram(np.full(500, value))
        assert counts.sum() == 500 and edges[0] <= value <= edges[-1]
    assert _histogram(np.array([np.nan])) is None


def test_a_failed_correction_clears_what_the_previous_fit_showed(window, qapp):
    """Review: the rotation text, the histograms and the Homogeneous table kept the previous fit's numbers."""
    n = 9 * 10 * 11
    zncc = np.full(n, 0.5)
    zncc[:2] = 0.99
    fn, _R = rigid_in_physical([0.0, 0.0, 3.0], [1.0, 0.0, 0.0], (60.0, 60.0, 70.0), (1.0, 1.0, 1.0))
    window.state.set_results(make_result([fn], zncc=zncc))
    tab = window.open_statistics().analysis
    _select(tab.motion, "rigid")
    assert tab.wait()
    assert "deg" in tab.motion_info.text() and tab.hom_table.rowCount() > 0 and tab.hist_figure.axes
    tab.min_zncc.setValue(0.9)  # two nodes left: no rigid fit
    assert tab.wait()
    assert "Too few nodes" in tab.nodes_info.text()
    assert tab.motion_info.text() == "" and tab.hom_table.rowCount() == 0 and not tab.hist_figure.axes


def test_the_applied_displacement_outdates_the_noise_floor_only(window, qapp):
    """Review: editing the noise floor's applied displacement marked the series over frames out of date."""
    fns = [lambda x, y, z, a=a: (a + 0 * x, 0 * y, 0 * z) for a in (0.1, 0.2)]
    window.state.set_results(make_result(fns))
    tab = window.open_statistics().analysis
    assert tab.wait()
    tab.compute_series()
    assert tab.wait()
    tab.compute_noise_floor()
    assert tab.wait()
    assert tab.series_is_current() and tab.noise_is_current()
    tab.nominal[0].setValue(0.5)
    assert tab.series_is_current() and tab._btn_csv.isEnabled()
    assert not tab.noise_is_current() and "Out of date" in tab.noise_info.text()
