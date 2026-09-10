"""The texture analysis window and the ``al-dvc texture`` command."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.gui.app import MainWindow, create_application  # noqa: E402
from al_dvc.gui.mask_editor import MaskOp  # noqa: E402
from al_dvc.io.volume_io import save_volume  # noqa: E402
from al_dvc.texture import THRESHOLDS  # noqa: E402

SHAPE = (48, 56, 64)


@pytest.fixture(scope="module")
def qapp():
    return create_application(["pytest"])


@pytest.fixture(scope="module")
def aniso():
    rng = np.random.default_rng(5)
    return gaussian_filter(rng.normal(size=SHAPE), sigma=(3.5, 1.2, 1.2)).astype(np.float32)  # long along z


def _pump(n=20):
    for _ in range(n):
        QApplication.processEvents()


def test_window_analyses_applies_and_exports(qapp, aniso, tmp_path):
    window = MainWindow()
    window.show()
    tw = window.open_texture_window()
    assert tw.isVisible() and window.open_texture_window() is tw
    assert not tw._btn_analyse.isEnabled()  # no volume yet
    window.state.set_volume_arrays([aniso, aniso], ["ref", "def"])
    window.state.set_output_dir(tmp_path)
    _pump()
    assert tw._btn_analyse.isEnabled()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[4:-4, 6:-6, 8:-8] = True
    window.state.set_mask(0, mask=mask)
    _pump()
    tw.use_dvc_roi()  # the DVC region of interest becomes the texture region; its bounding box is analysed
    assert tw.range_box() == ((8, 56), (6, 50), (4, 44))
    tw.region.apply(MaskOp("rectangle", plane="xy", points=((10.0, 12.0), (30.0, 20.0)), mode="replace", depth=(4, 20)))
    assert tw.range_box() == ((10, 31), (12, 21), (4, 21))  # a drawn rectangle: x, y from the shape, z from its depth
    tw.region.undo()
    assert tw.range_box() == ((8, 56), (6, 50), (4, 44))
    # the centre follows the region only when it would fall outside it: the drawn rectangle moved it,
    # and undoing back to the larger region leaves it where it was
    assert tw.centre() == (20, 16, 12)
    tw.region.set_centre((30, 26, 22))  # as a click on a slice would: the other two panes follow
    assert tw.region.slice_indices() == (22, 26, 30) and tw.centre_spin["x"].value() == 30
    tw.centre_on_region()
    assert tw.centre() == (32, 28, 24)
    tw.go_to_step(tw.TAB_ACF)
    assert tw.tabs.currentIndex() == tw.TAB_ACF and tw.pages.currentIndex() == tw.TAB_ACF  # tab, page and strip follow
    tw.cube_size.setValue(32)
    tw.analyse()
    assert tw.wait(120_000)
    _pump()
    res = tw.result
    assert res is not None and res.status == "ok"
    assert res.settings["box"] == ((16, 48), (12, 44), (8, 40)) and res.settings["centre"] == (32, 28, 24)
    assert res.settings["size"] == (32, 32, 32) and res.settings["max_lag"] == (8, 8, 8)  # a quarter of the edge
    assert res.settings["estimator"] == "overlap" and res.settings["fill"] == 1.0
    assert all(sl.stop - sl.start == 32 for sl in res.window)
    assert res.length("z") > 2 * res.length("x")
    assert tw.table.item(2, 0).text() != "-" and tw.table.item(3, 0).text() != "-"
    assert tw.recommendation is not None and tw._btn_apply.isEnabled()
    before = window.state.para.winsize
    tw.apply_recommendation()
    _pump()
    ws = window.state.para.winsize
    assert ws == tw.recommendation.subset and ws[2] > ws[0] and ws != before
    assert not window.param_panel.winsize_lock.isChecked()  # a non-cubic subset unlocks the axes
    assert window.state.para.winstepsize == tw.recommendation.step
    # exports
    tw.save_csv(tmp_path / "p.csv")
    tw.save_json(tmp_path / "s.json")
    lines = (tmp_path / "p.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("axis,lag_voxel") and sum(ln.startswith("radial,") for ln in lines) > 5
    summary = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok" and summary["recommendation"]["subset"] == list(ws)
    assert summary["lengths_voxel"]["z"]["1/e"]["value"] == pytest.approx(res.length("z"))
    tw._on_save_png()  # headless: writes the default path
    assert (tmp_path / "texture_profiles.png").is_file()
    # the RVE analysis runs on its own, next to the autocorrelation analysis
    tw.go_to_step(tw.TAB_SWEEP)
    tw.sweep_start.setValue(16)
    tw.sweep_step.setValue(8)
    tw.sweep_count.setValue(4)
    assert len(tw.cube_sizes()) == 4 and tw.cube_sizes()[0] == (16, 16, 16)
    assert len(tw.region._cubes) == 5  # the four sizes plus the cube step 3 would analyse
    # the slice viewer follows the step: step 2 picks the centre on it, so it moves into that tab
    assert tw.region.isAncestorOf(tw.region.canvas) and tw.tabs.widget(tw.TAB_SWEEP).isAncestorOf(tw.region)
    tw.go_to_step(tw.TAB_REGION)
    assert tw.tabs.widget(tw.TAB_REGION).isAncestorOf(tw.region) and not tw.region._cubes
    tw.go_to_step(tw.TAB_SWEEP)
    assert len(tw.region._cubes) == 5
    tw.run_sweep_analysis()
    assert tw.wait(300_000)
    _pump()
    assert tw.sweep is not None and len(tw.sweep.levels) >= 3 and tw.tabs.currentIndex() == tw.TAB_SWEEP
    assert all(lvl.radial is not None for lvl in tw.sweep.levels)  # every size keeps its radial curve for the plot
    assert all(len(lvl.samples) == 1 for lvl in tw.sweep.levels)  # concentric cubes, one per size
    assert tw.sweep.settings["centre"] == (32, 28, 24)
    assert tw.result is not None  # the autocorrelation result is untouched
    stable = tw.sweep_size()
    if stable is not None:
        assert tw.cube_size.value() >= stable  # step 3 already carries what step 2 found
        tw.use_sweep_size()
        assert tw.cube_size.value() >= stable and tw.tabs.currentIndex() == tw.TAB_ACF
    # plot controls: curves off, log scale, background, reset view
    tw.curve_checks["x"].setChecked(False)
    tw.plot_scale.setCurrentIndex(1)
    tw.plot_background.setCurrentIndex(1)
    tw.reset_view()
    _pump()
    assert set(tw.sweep.decisions) == {float(t) for t in THRESHOLDS}
    tw.save_json(tmp_path / "s2.json")
    assert "sweep" in json.loads((tmp_path / "s2.json").read_text(encoding="utf-8"))
    # translations follow the application language
    mgr = qapp._pyaldvc_lang_mgr
    mgr.load("zh_CN")
    _pump()
    assert tw.windowTitle() != "Texture analysis"
    mgr.load("en")
    _pump()
    assert tw.windowTitle() == "Texture analysis"
    tw.close()
    window.close()


def test_window_reports_no_texture(qapp):
    window = MainWindow()
    tw = window.open_texture_window()
    flat = np.full((32, 32, 32), 3.0, dtype=np.float32)
    window.state.set_volume_arrays([flat], ["flat"])
    _pump()
    tw.analyse()
    assert tw.wait(60_000)
    _pump()
    assert tw.result is not None and tw.result.status == "no_texture" and tw.recommendation is None
    assert not tw._btn_apply.isEnabled() and "No texture" in tw._status.text()
    tw.close()
    window.close()


def test_cli_texture_writes_the_files(aniso, tmp_path, capsys):
    from al_dvc.cli import main

    path = tmp_path / "vol.h5"
    save_volume(path, aniso)
    out = tmp_path / "tex"
    args = ["texture", str(path), "--size", "32", "--sweep", "--sweep-start", "16", "--sweep-step", "8", "--sweep-count", "4"]
    assert main([*args, "-o", str(out)]) == 0
    assert (out / "texture_profiles.csv").is_file() and (out / "texture_profiles.png").is_file()
    summary = json.loads((out / "texture_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok" and "recommendation" in summary and "sweep" in summary
    text = capsys.readouterr().out
    assert "suggested subset" in text and "sweep 1/e" in text
    roi = np.zeros(SHAPE, dtype=np.uint8)
    roi[8:-8, 8:-8, 8:-8] = 1
    save_volume(tmp_path / "roi.h5", roi)
    assert main(["texture", str(path), "--roi", str(tmp_path / "roi.h5"), "--size", "16", "-o", str(out / "roi")]) == 0
    summary = json.loads((out / "roi" / "texture_summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["centre"] == [32, 28, 24]  # the centre of the region's bounding box (x, y, z)
    assert summary["settings"]["size"] == [16, 16, 16] and summary["settings"]["estimator"] == "overlap"
    # an explicit centre near the -x face: the cube is capped there, per axis, by twice the distance to it
    assert main(["texture", str(path), "--centre", "12", "28", "24", "--size", "64", "-o", str(out / "edge")]) == 0
    summary = json.loads((out / "edge" / "texture_summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["size"] == [24, 56, 48] and summary["settings"]["centre"] == [12, 28, 24]


# --------------------------------------------------------------------------- the 2026-09-10 UI review
def test_the_region_the_worker_was_given_survives_being_edited(qapp, aniso):
    """Scenario 1: a sweep runs on the region as it was at dispatch, whatever the user draws next.

    The job used to carry the mask editor's own array, so a shape added mid-sweep changed the input of
    the cube sizes that had not been analysed yet, and no stale check could reconstruct what had
    actually been correlated.
    """
    window = MainWindow()
    tw = window.open_texture_window()
    window.state.set_volume_arrays([aniso, aniso], ["ref", "def"])
    _pump()
    tw.region.apply(MaskOp("rectangle", plane="xy", points=((8.0, 8.0), (52.0, 44.0)), mode="replace", depth=(6, 40)))
    _pump()
    given = tw.region.snapshot()  # what _start hands the worker
    before = np.array(given, copy=True)
    assert not given.flags.writeable

    tw.region.apply(MaskOp("rectangle", plane="xy", points=((10.0, 10.0), (20.0, 20.0)), mode="cut", depth=(6, 40)))
    _pump()
    assert np.array_equal(given, before), "the worker's region changed under it"
    assert not np.array_equal(tw.region.mask, before), "the edit did not reach the editor"
    assert not np.shares_memory(given, tw.region.mask)

    # and the job _start builds must carry that snapshot, not the editor's array
    from al_dvc.gui import texture_window as tw_mod

    started: list = []

    class _NoRun(tw_mod._TextureWorker):
        def start(self, *a, **k):  # this test inspects the payload; it does not need the thread
            started.append(self)

    tw.region.set_centre((32, 24, 20))
    keep, tw_mod._TextureWorker = tw_mod._TextureWorker, _NoRun
    try:
        tw.run_sweep_analysis()
    finally:
        tw_mod._TextureWorker = keep
    assert started, f"the sweep did not dispatch: {tw._sweep_status.text()}"
    payload = started[0]._job["mask"]
    assert payload is not None and not payload.flags.writeable  # a snapshot, not the live array
    handed = np.array(payload, copy=True)

    tw.region.apply(MaskOp("rectangle", plane="xy", points=((30.0, 30.0), (40.0, 40.0)), mode="cut", depth=(6, 40)))
    _pump()
    assert np.array_equal(payload, handed), "the dispatched region changed under the worker"
    assert not np.shares_memory(payload, tw.region.mask), "the job still aliases the editor"
    tw._worker = None  # no thread was started
    window.close()


def test_a_drag_on_another_step_browses_instead_of_drawing(qapp, aniso):
    """Scenario 3: with Pick off, dragging on the step-2 slices must not touch the region.

    The drawing tools live on step 1's page, but the canvas kept the last tool and mode, so a
    navigation-like drag on step 2 with Replace selected threw the drawn region away.
    """
    window = MainWindow()
    tw = window.open_texture_window()
    window.state.set_volume_arrays([aniso, aniso], ["ref", "def"])
    _pump()
    tw.region.apply(MaskOp("rectangle", plane="xy", points=((8.0, 8.0), (52.0, 44.0)), mode="replace", depth=(6, 40)))
    tw.region.tools.set_mode("replace")  # the mode that makes a stray drag destructive
    tw.region.tools.set_tool("rectangle")
    _pump()

    tw.go_to_step(tw.TAB_REGION)
    _pump()
    assert tw.region.editable
    tw.go_to_step(tw.TAB_SWEEP)
    _pump()
    assert not tw.region.editable and not tw.region.picking_centre

    kept = tw.region.mask.copy()
    revision = tw.region.revision
    ax = tw.region.axes[0]
    press = SimpleNamespace(inaxes=ax, button=1, xdata=30.0, ydata=25.0, dblclick=False)
    move = SimpleNamespace(inaxes=ax, button=1, xdata=40.0, ydata=35.0, dblclick=False)
    tw.region._on_press(press)
    tw.region._on_motion(move)
    tw.region._on_release(move)
    _pump()
    assert np.array_equal(tw.region.mask, kept) and tw.region.revision == revision

    tw.go_to_step(tw.TAB_REGION)  # step 1 draws again
    _pump()
    tw.region._on_press(press)
    tw.region._on_motion(move)
    tw.region._on_release(move)
    _pump()
    assert not np.array_equal(tw.region.mask, kept)
    window.close()


def test_whole_volume_and_a_typed_box_can_be_undone(qapp, aniso):
    """Scenario 5: replacing the region is an operation, not a reset of the history.

    Both went through ``MaskEditor.reset``, which drops the operations and the redo stack, so the Undo
    button next to them could not bring back a region that had taken real work to draw.
    """
    window = MainWindow()
    tw = window.open_texture_window()
    window.state.set_volume_arrays([aniso, aniso], ["ref", "def"])
    _pump()
    tri = ((10.0, 10.0), (40.0, 12.0), (30.0, 38.0))
    tw.region.apply(MaskOp("polygon", plane="xy", points=tri, mode="replace", depth=(8, 30)))
    _pump()
    drawn = tw.region.mask.copy()
    assert drawn.any()

    tw.set_range_whole()  # the Whole volume button
    _pump()
    assert tw.region.fill_fraction() == 1.0
    tw.region.undo()
    _pump()
    assert np.array_equal(tw.region.mask, drawn), "Undo did not restore the region after Whole volume"

    tw.set_range(((4, 30), (5, 25), (6, 20)))  # typing the bounding box
    _pump()
    assert tw.range_box() == ((4, 30), (5, 25), (6, 20))
    tw.region.undo()
    _pump()
    assert np.array_equal(tw.region.mask, drawn), "Undo did not restore the region after a typed box"
    window.close()


def test_a_typed_box_is_exactly_the_box(qapp, aniso):
    """The box became a rectangle extruded through z: MaskOp corners are inclusive, a box is half-open."""
    window = MainWindow()
    tw = window.open_texture_window()
    window.state.set_volume_arrays([aniso, aniso], ["ref", "def"])
    _pump()
    for box in (((4, 30), (5, 25), (6, 20)), ((0, 64), (0, 56), (0, 48)), ((10, 12), (11, 13), (12, 14))):
        tw.set_range(box)
        _pump()
        want = np.zeros(SHAPE, dtype=bool)
        (x0, x1), (y0, y1), (z0, z1) = box
        want[z0:z1, y0:y1, x0:x1] = True
        assert np.array_equal(tw.region.mask, want), box
    window.close()


@pytest.fixture(scope="module")
def real_sweep(aniso):
    """A genuine sweep result, computed once through the analysis API rather than the window.

    The completion path reads several of its fields, so a stub would be attribute whack-a-mole; this
    lets the tests below drive ``_on_sweep_finished`` with a real object and vary only the source.
    """
    from al_dvc.texture.concentric import sweep_concentric

    return sweep_concentric(aniso, (32, 24, 22), ((6, 57), (6, 49), (4, 43)), start=16, step=8, count=4)


def _ready(window, aniso):
    """A texture window with a volume, a drawn region and a centre."""
    tw = window.open_texture_window()
    window.state.set_volume_arrays([aniso, aniso], ["ref", "def"])
    _pump()
    tw.region.apply(MaskOp("rectangle", plane="xy", points=((6.0, 6.0), (56.0, 48.0)), mode="replace", depth=(4, 42)))
    tw.region.set_centre((32, 24, 22))
    _pump()
    return tw


def test_a_finished_sweep_does_not_overwrite_a_size_it_no_longer_describes(qapp, aniso, real_sweep, monkeypatch):
    """Scenario 2: start sweep A, move to B, and A's completion must not write A's size into B.

    ``_on_sweep_finished`` used to call ``_write_size`` whenever a stable size existed, before any
    staleness check, and to force the view to the RVE tab -- so a sweep of the region the user had
    already replaced quietly became the cube of step 3, labelled as coming from the RVE analysis.
    The manual *Use size* action had the check; automatic completion bypassed it.

    The stable size is stubbed so the decision under test is the only variable: whether a completion
    writes it. Whether this particular volume stabilises is the subject of other tests.
    """
    window = MainWindow()
    tw = _ready(window, aniso)
    monkeypatch.setattr(type(tw), "sweep_size", lambda self: 40)
    monkeypatch.setattr(type(tw), "_draw_sweep", lambda self: None)

    # fresh input, untouched size: the completion is welcome to write it
    tw.go_to_step(tw.TAB_SWEEP)
    tw._job_source = tw._sweep_input()
    tw._size_edited = False
    tw._on_sweep_finished(real_sweep)
    _pump()
    assert tw.cube_size.value() == 40 and tw._size_from_rve == 40
    assert tw.tabs.currentIndex() == tw.TAB_SWEEP

    # the region moved while it ran: keep the sweep, keep the user's controls, stay where we are
    tw.go_to_step(tw.TAB_ACF)
    tw.cube_size.setValue(24)
    _pump()
    tw._size_edited = False
    tw._job_source = dict(tw._sweep_input() or {}, revision=-1)  # a source that cannot match
    tw._on_sweep_finished(real_sweep)
    _pump()
    assert tw.is_sweep_stale
    assert tw.cube_size.value() == 24, "a stale sweep overwrote the current size"
    assert tw.tabs.currentIndex() == tw.TAB_ACF, "a stale sweep dragged the view away"

    # fresh input, but the user typed a cube after dispatch: their value wins
    tw._job_source = tw._sweep_input()
    tw.cube_size.setValue(32)
    _pump()
    assert tw._size_edited
    tw._on_sweep_finished(real_sweep)
    _pump()
    assert not tw.is_sweep_stale
    assert tw.cube_size.value() == 32, "the completion overwrote a size typed after dispatch"
    window.close()


def test_the_buttons_settle_whichever_signal_arrives_first(qapp, aniso, real_sweep, monkeypatch):
    """Scenario 9: the result can reach the UI before the thread terminates.

    The result, cancel and failure signals are emitted from inside ``QThread.run``, so a slot can run
    while ``isRunning()`` is still true -- and ``_refresh_validity`` declines to act then, leaving
    *Apply* and *Use size* disabled until some later input change happened to refresh them. The
    native ``finished`` signal is now what guarantees the final state.
    """
    window = MainWindow()
    tw = _ready(window, aniso)
    monkeypatch.setattr(type(tw), "sweep_size", lambda self: 40)
    monkeypatch.setattr(type(tw), "_draw_sweep", lambda self: None)

    class _Alive:
        """A worker that is still running when its terminal signal is delivered."""

        def __init__(self):
            self.running = True

        def isRunning(self):  # noqa: N802 - the Qt name
            return self.running

    worker = _Alive()
    tw._worker = worker
    tw._job_source = tw._sweep_input()
    tw._size_edited = False
    tw._on_sweep_finished(real_sweep)
    _pump()
    assert not tw._btn_use_size.isEnabled()  # the early arrival cannot settle it, and must not claim to

    worker.running = False  # the thread ends
    tw._on_worker_finished(worker)
    _pump()
    assert tw._btn_use_size.isEnabled(), "Use size stayed disabled after the thread ended"

    # a termination from a worker that has since been replaced must not touch the current job
    stale_worker, tw._worker = worker, _Alive()
    tw._btn_use_size.setEnabled(False)
    tw._on_worker_finished(stale_worker)
    _pump()
    assert not tw._btn_use_size.isEnabled()
    tw._worker = None
    window.close()
