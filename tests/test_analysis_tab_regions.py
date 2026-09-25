"""The Statistics tab, phase 2: regions (typed and drawn), the comparison of regions, the motion fitted over a
region, confidence intervals, profiles, the line and the extensometer, the corrected field in the main window and
the exports, and the session. Driven offscreen on results whose numbers are known in closed form."""

import csv
import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from tests.test_analysis import make_result, rigid_in_physical  # noqa: E402

# node grid of make_result: x 20..84, y 24..96, z 28..108 (step 8), volume 200^3


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


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.reader(line for line in fh if not line.startswith("#")))


def _open(window, result):
    window.state.set_results(result)
    tab = window.open_statistics().analysis
    assert tab.wait()
    return tab


def _column(table, header: str) -> int:
    for c in range(table.columnCount()):
        if table.horizontalHeaderItem(c).text() == header:
            return c
    raise AssertionError(header)


def test_a_typed_region_restricts_the_statistics(window, qapp):
    # u = 0.01 x: the mean over a box is 0.01 x its centre
    tab = _open(window, make_result([lambda x, y, z: (0.01 * x, 0 * y, 0 * z)]))
    panel = tab.regions_panel
    box = panel.add("box")
    assert [r.id for r in window.state.regions] == [box.id] and panel.table.rowCount() == 1
    n_inside = int(box.node_mask(window.state.results).sum())
    assert panel.table.item(0, 2).text().replace(",", "") == str(n_inside) and n_inside > 0
    # edit its corners in the form: x from 20 to 36 -> nodes at x 20, 28, 36
    lo = [20.0, 24.0, 28.0]
    hi = [36.0, 96.0, 108.0]
    for i in range(3):
        panel._edits[f"lo{i}"].setValue(lo[i])
        panel._edits[f"hi{i}"].setValue(hi[i])
    assert panel.apply_editor()
    region = window.state.regions[0]
    assert region.params["lo"] == lo and region.params["hi"] == hi
    _select(tab.stats_region_combo, region.id)
    assert tab.wait()
    assert int(tab.table.item(0, 0).text().replace(",", "")) == 3 * 10 * 11
    assert float(tab.table.item(0, 1).text()) == pytest.approx(0.28)  # 0.01 x mean(20, 28, 36)
    assert region.name in tab.nodes_info.text()
    # the regions tab compares every node with the box
    tab.results_tabs.setCurrentIndex(3)
    assert tab.regions_table.rowCount() == 2
    assert float(tab.regions_table.item(1, 1).text()) == pytest.approx(0.28)
    assert float(tab.regions_table.item(0, 1).text()) == pytest.approx(0.52)  # every node: 0.01 x 52
    # an invalid edit is refused and says why
    panel._edits["hi0"].setValue(10.0)
    assert not panel.apply_editor() and "Not applied" in panel.hint.text()
    assert window.state.regions[0].params["hi"][0] == 36.0
    panel.delete_selected()
    assert window.state.regions == [] and tab.stats_region_combo.count() == 1
    assert window.settle_workers(5_000) == []


def test_drawing_regions_on_the_slices(window, qapp):
    tab = _open(window, make_result([lambda x, y, z: (0 * x, 0 * y, 0 * z)]))
    tab._start_drawing("rect")
    assert tab.drawer.active and "Esc" in tab.canvas_hint.text()
    tab.drawer.press("xy", 18.0, 22.0)
    tab.drawer.move("xy", 40.0, 50.0)
    tab.drawer.release(40.0, 50.0)
    assert not tab.drawer.active
    (rect,) = window.state.regions
    assert rect.shape == "prism" and rect.params["plane"] == "xy" and rect.params["outline"] == "rect"
    assert (rect.params["lo"], rect.params["hi"]) == (28.0, 108.0)  # through the whole node grid along z
    assert int(rect.node_mask(window.state.results).sum()) == 3 * 4 * 11  # x 20..36, y 24..48, every z
    # outlined on the slices: a contour on the XY plane and the region's name
    tab.canvas.set_slices(iz=60)
    texts = [t.get_text().strip() for t in tab.canvas.axes[0].texts]
    assert rect.name in texts
    # a click is not a rectangle; Esc cancels
    tab._start_drawing("ellipse")
    tab.drawer.press("xz", 50.0, 50.0)
    tab.drawer.release(50.2, 50.1)
    assert tab.drawer.active and len(window.state.regions) == 1
    tab.drawer.key("escape")
    assert not tab.drawer.active and "cancelled" in tab.canvas_hint.text()
    # a polygon on YZ: three corners and Enter
    tab._start_drawing("polygon")
    for a, b in ((20.0, 20.0), (100.0, 20.0), (60.0, 110.0)):
        tab.drawer.press("yz", a, b)
    tab.drawer.press("xy", 5.0, 5.0)  # a click on another plane is not a corner
    tab.drawer.key("enter")
    poly = window.state.regions[-1]
    assert poly.params["outline"] == "polygon" and len(poly.params["points"]) == 3 and poly.params["plane"] == "yz"
    assert poly.id == rect.id + 1 and poly.color != rect.color
    assert tab.stats_region_combo.count() == 3  # every node and the two regions
    assert tab.wait()
    assert window.settle_workers(5_000) == []


def test_the_motion_fitted_over_a_fixture_region(window, qapp):
    """A grip (x <= 36) only translates; the rest stretches. Fitting over the grip removes exactly the grip's motion;
    fitting over every node would not."""
    t = 0.7

    def fn(x, y, z):
        return (t + 0.01 * np.maximum(x - 36.0, 0.0), 0 * y + 0.2, 0 * z)

    tab = _open(window, make_result([fn]))
    grip = tab.regions_panel.add("box")
    tab.regions_panel._edits["lo0"].setValue(0.0)
    tab.regions_panel._edits["hi0"].setValue(36.0)
    for i, (lo, hi) in enumerate(((0.0, 200.0), (0.0, 200.0)), start=1):
        tab.regions_panel._edits[f"lo{i}"].setValue(lo)
        tab.regions_panel._edits[f"hi{i}"].setValue(hi)
    assert tab.regions_panel.apply_editor()
    _select(tab.motion, "translation")
    assert tab.wait()
    all_nodes_u = float(tab.table.item(0, 1).text())
    assert abs(all_nodes_u) < 1e-9  # fitted over every node: the mean is removed
    _select(tab.fit_region_combo, grip.id)
    assert tab.wait()
    assert "of " + window.state.regions[0].name in tab.motion_info.text()
    assert tab.current.fit.translation[0] == pytest.approx(t) and tab.current.fit.translation[1] == pytest.approx(0.2)
    grip_row = next(i for i, (r, _s) in enumerate(tab.current.region_rows) if r is not None)
    assert tab.current.region_rows[grip_row][1].mean == pytest.approx(0.0, abs=1e-12)  # the grip is at rest
    # every node: the mean of the stretch 0.01 (x - 36) for x > 36
    x = np.arange(20.0, 85.0, 8.0)
    assert float(tab.table.item(0, 1).text()) == pytest.approx(np.mean(0.01 * np.maximum(x - 36.0, 0.0)), rel=1e-3)
    assert window.settle_workers(5_000) == []


def test_the_confidence_interval_columns(window, qapp):
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.05, 9 * 10 * 11)
    tab = _open(window, make_result([lambda x, y, z: (noise, 0 * y, 0 * z)]))
    c_neff = _column(tab.table, "Eff. nodes")
    c_ci = _column(tab.table, "95 % CI \u00b1")
    n_eff = float(tab.table.item(0, c_neff).text())
    half = float(tab.table.item(0, c_ci).text())
    n = 9 * 10 * 11
    assert 0.5 * n < n_eff <= n  # white noise: nearly every node is independent
    assert half == pytest.approx(1.96 * noise.std() / np.sqrt(n_eff), rel=0.02)
    assert window.settle_workers(5_000) == []


def test_profiles_the_line_and_the_extensometer(window, qapp, tmp_path):
    # frame k stretches x by e_k: u = e_k (x - 52); the extensometer along x measures e_k exactly
    strains = (0.0, 0.01, 0.02)
    fns = [lambda x, y, z, e=e: (e * (x - 52.0), 0 * y, 0 * z) for e in strains]
    tab = _open(window, make_result(fns, voxel_size=(2.0, 2.0, 2.0)))
    window.state.set_output_dir(tmp_path)
    tab.frame_slider.setValue(2)
    # profile along x: the layer mean of u is 0.02 (x - 52) voxels, x 2 um
    tab.results_tabs.setCurrentIndex(4)
    _select(tab.shown, "disp_u")
    tab.profile_axis.setCurrentText("x")
    assert tab.wait()
    prof = tab.current.profile
    np.testing.assert_allclose(prof.positions, 2.0 * np.arange(20.0, 85.0, 8.0))
    np.testing.assert_allclose(prof.mean, 2.0 * 0.02 * (np.arange(20.0, 85.0, 8.0) - 52.0), atol=1e-12)
    assert (prof.n == 10 * 11).all()
    tab.compute_profiles()
    assert tab.wait() and tab.long_is_current("profiles") and len(tab.long["profiles"][1]) == 3
    out = tab.export_csv(tmp_path / "profile.csv")
    rows = _rows(out)
    assert rows[0] == ["frame", "axis", "position", "mean", "std", "n"] and len(rows) == 1 + 3 * 9
    # the line: default along x through the middle of the grid; values along it follow u
    tab.results_tabs.setCurrentIndex(5)
    tab._start_drawing("line")
    assert tab.results_tabs.currentIndex() == 5
    tab.drawer.press("xy", 28.0, 60.0)  # at the slice shown (z = 100)
    tab.drawer.press("xz", 76.0, 68.0)  # at y = 100
    assert not tab.drawer.active
    p0, p1 = tab.line_points()
    assert p0 == (28.0, 60.0, 100.0) and p1 == (76.0, 100.0, 68.0)
    tab.set_line((28.0, 60.0, 68.0), (76.0, 60.0, 68.0))
    assert tab.wait()
    distance, values = tab.current.line
    assert distance[-1] == pytest.approx(2.0 * 48.0)
    np.testing.assert_allclose(values, 2.0 * 0.02 * (28.0 + distance / 2.0 - 52.0), atol=1e-9)
    tab.compute_line()
    assert tab.wait() and tab.long_is_current("line")
    series = tab.long["line"][1]
    ext = series.extensometer
    np.testing.assert_allclose(ext.strain, strains, atol=1e-12)
    assert ext.L0 == pytest.approx(2.0 * 48.0)
    # the line in every frame, and the field at its ends: u = e (x - 52) x 2 um at x = 28 and 76
    assert series.values.shape == (3, distance.size)
    np.testing.assert_allclose(series.ends, [[2.0 * e * (28.0 - 52.0), 2.0 * e * (76.0 - 52.0)] for e in strains], atol=1e-9)
    assert len(tab.line_figure.axes) == 3
    out = tab.export_csv(tmp_path / "line.csv")
    assert _rows(out)[0] == ["frame", "distance", "disp_u"]
    assert len(_rows(out)) == 1 + 3 * distance.size  # every frame
    ext_rows = _rows(tmp_path / "line_extensometer.csv")
    assert ext_rows[0] == ["frame", "length", "strain", "disp_u_first_point", "disp_u_second_point"]
    assert float(ext_rows[3][2]) == pytest.approx(0.02) and float(ext_rows[3][4]) == pytest.approx(2.0 * 0.02 * 24.0)
    assert tab.export_png(tmp_path / "line.png").stat().st_size > 1000
    doc = json.loads(tab.export_json(tmp_path / "s.json").read_text(encoding="utf-8"))
    assert doc["line_over_frames"]["extensometer"]["strain"][2] == pytest.approx(0.02) and len(doc["profiles"]) == 3
    # moving an end point outdates the extensometer
    tab.line_spins[0].setValue(36.0)
    assert tab.wait()
    assert not tab.long_is_current("line") and "Out of date" in tab.line_info.text()
    assert window.settle_workers(5_000) == []


def test_comparing_regions_over_frames(window, qapp, tmp_path):
    fns = [lambda x, y, z, a=a: (a * x, 0 * y, 0 * z) for a in (0.001, 0.002)]
    tab = _open(window, make_result(fns))
    tab.regions_panel.add("box")
    tab.regions_panel.add("slab")
    tab.compare_regions.setChecked(True)
    tab.compute_series()
    assert tab.wait() and tab.series_is_current()
    runs = tab.series[1]
    assert [r is None for r, _s in runs] == [True, False, False] and all(len(s) == 2 for _r, s in runs)
    assert len(tab.series_figure.axes[0].lines) == 3  # one curve per region and one for every node
    tab.results_tabs.setCurrentIndex(2)
    rows = _rows(tab.export_csv(tmp_path / "compare.csv"))
    assert rows[0][:2] == ["region", "frame"] and len(rows) == 1 + 3 * 2
    assert "disp_u_ci95" in rows[0]
    assert {r[0] for r in rows[1:]} == {"All nodes", "R1", "R2"}
    tab.compare_regions.setChecked(False)
    assert not tab.series_is_current()
    assert window.settle_workers(5_000) == []


def test_the_corrected_field_in_the_main_window_and_the_exports(window, qapp, tmp_path):
    from al_dvc.analysis import CorrectedResult
    from al_dvc.gui.dialogs.export_dialog import ExportConfig, export_formats

    fn, _R = rigid_in_physical([0.0, 0.0, 5.0], [2.0, -1.0, 0.5], (52.0, 60.0, 68.0), (1.0, 1.0, 1.0))
    tab = _open(window, make_result([fn]))
    state = window.state
    _select(tab.motion, "rigid")
    assert tab.wait()
    assert state.display_correction is None  # only the Statistics window shows it until asked
    tab.apply_main.setChecked(True)
    assert state.display_correction is not None and state.display_correction.motion == "rigid"
    shown = state.display_result()
    assert isinstance(shown, CorrectedResult) and shown.source is state.results
    np.testing.assert_allclose(shown.result_disp[0].U_accum, 0.0, atol=1e-9)
    # the main window: a badge, a colour-bar note, the corrected field on the slices
    panel = window.results_panel
    assert not panel._correction_row.isHidden() and "rigid" in panel._correction_badge.text()
    grid = window.viewer._field_grid()
    if grid is not None:
        assert np.nanmax(np.abs(grid[0])) < 1e-6
    # the exports: fields corrected, archives as measured, a note next to them
    cfg = ExportConfig(out_dir=tmp_path, npz=True, csv=True, fields=["disp_u"])
    outcome = export_formats(state.results, cfg, shown=shown)
    assert outcome.ok, outcome.errors
    note = json.loads((tmp_path / "aldvc_correction.json").read_text(encoding="utf-8"))
    assert note["correction"]["motion"] == "rigid" and note["uncorrected_frames"] == []
    rows = _rows(next((tmp_path / "csv").glob("*.csv")))
    col = rows[0].index("disp_u")
    assert max(abs(float(r[col])) for r in rows[1:] if r[col]) < 1e-6
    with np.load(tmp_path / "aldvc.npz") as npz:
        np.testing.assert_allclose(npz["U_accum_1"], state.results.result_disp[0].U_accum)  # as measured
    # changing the motion follows; "As measured" in the main window turns it off everywhere
    _select(tab.motion, "translation")
    assert state.display_correction.motion == "translation"
    panel._btn_as_measured.click()
    assert state.display_correction is None and not tab.apply_main.isChecked()
    assert panel._correction_row.isHidden() and state.display_result() is state.results
    assert tab.wait()
    assert window.settle_workers(5_000) == []


def test_regions_correction_and_settings_come_back_with_the_session(window, qapp, tmp_path):
    from al_dvc.gui.session import apply_session, load_session, save_session
    from al_dvc.io.session_bundle import read_session_document

    vol = tmp_path / "ref.npy"
    np.save(vol, np.zeros((8, 8, 8), dtype=np.uint8))
    state = window.state
    state.add_volume_paths([str(vol), str(vol)])
    tab = _open(window, make_result([lambda x, y, z: (0.01 * x, 0 * y, 0 * z)]))
    grip = tab.regions_panel.add("sphere")
    tab.regions_panel.add("slab")
    _select(tab.motion, "rigid")
    _select(tab.fit_region_combo, grip.id)
    _select(tab.stats_region_combo, grip.id)
    _select(tab.group, "strain")
    _select(tab.shown, "exx")
    tab.profile_axis.setCurrentText("y")
    tab.apply_main.setChecked(True)
    assert tab.wait()
    saved_regions = list(state.regions)
    saved_corr = state.display_correction
    path = save_session(state, tmp_path / "s.aldvc")
    doc = read_session_document(path)
    assert len(doc["analysis"]["regions"]) == 2 and doc["analysis"]["correction"]["fit_region"]["id"] == grip.id
    # a fresh state, then the session back
    state.set_regions([])
    state.set_display_correction(None)
    tab.profile_axis.setCurrentText("x")
    apply_session(load_session(path), state, path)
    assert state.regions == saved_regions and state.display_correction == saved_corr
    assert state.results is not None and state.results.n_frames == 1  # the result comes back with the session
    state.set_results(make_result([lambda x, y, z: (0.01 * x, 0 * y, 0 * z)]))  # and a new one keeps the settings
    assert tab.wait()
    assert tab.motion_kind() == "rigid" and tab.fit_region() == grip and tab.stats_region() == grip
    assert tab.profile_axis.currentText() == "y" and tab.apply_main.isChecked()
    # review: the field lists are empty while the session loads without results; the choice must survive that
    assert tab.group.currentData() == "strain" and tab.shown.currentData() == "exx"
    assert window.settle_workers(5_000) == []


def test_a_bad_analysis_entry_makes_the_session_unreadable(tmp_path):
    from al_dvc.core.config import dvcpara_default, para_to_dict
    from al_dvc.gui.session import SessionError, load_session

    doc = {
        "format": 2,
        "volumes": [],
        "para": para_to_dict(dvcpara_default()),
        "analysis": {"regions": [{"id": 1, "shape": "cube"}]},
    }
    p = tmp_path / "bad.aldvc"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SessionError, match="region"):
        load_session(p)
    doc["analysis"] = None  # an older session: no analysis at all
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert load_session(p).analysis["regions"] == []


def test_a_session_loaded_before_the_statistics_window_opens_sets_its_controls(window, qapp, tmp_path):
    """The tab saved its default controls into the state while starting, over the session's settings."""
    from al_dvc.analysis import Correction, Region
    from al_dvc.gui.session import apply_session, load_session, save_session

    vol = tmp_path / "ref.npy"
    np.save(vol, np.zeros((8, 8, 8), dtype=np.uint8))
    state = window.state
    state.add_volume_paths([str(vol), str(vol)])
    grip = Region(1, "grip", "sphere", {"centre": [52, 60, 68], "radius": 16.0})
    state.set_regions([grip])
    state.set_display_correction(Correction("rigid", fit_region=grip))
    state.analysis_settings = {"motion": "rigid", "fit_region": 1, "stats_region": 1, "profile_axis": "x", "min_share": 30}
    path = save_session(state, tmp_path / "s.aldvc")
    state.set_regions([])
    state.set_display_correction(None)
    state.analysis_settings = {}
    apply_session(load_session(path), state, path)
    state.set_results(make_result([lambda x, y, z: (0.01 * x, 0 * y, 0 * z)]))
    tab = window.open_statistics().analysis  # the tab is created now, after the session
    assert tab.wait()
    assert tab.motion_kind() == "rigid" and tab.fit_region() == grip and tab.stats_region() == grip
    assert tab.profile_axis.currentText() == "x" and tab.min_share.value() == 30 and tab.apply_main.isChecked()
    assert state.display_correction == Correction("rigid", fit_region=grip)
    assert window.settle_workers(5_000) == []
