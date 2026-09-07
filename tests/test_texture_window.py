"""The texture analysis window and the ``al-dvc texture`` command."""

import json
import os

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
    tw.go_to_step(tw.TAB_ACF)
    assert tw.tabs.currentIndex() == tw.TAB_ACF and tw.pages.currentIndex() == tw.TAB_ACF  # tab, page and strip follow
    tw.window_size.setValue(16)
    tw.analyse()
    assert tw.wait(120_000)
    _pump()
    res = tw.result
    assert res is not None and res.status == "ok"
    assert res.settings["range"] == ((8, 56), (6, 50), (4, 44)) and res.settings["max_lag"] == (16, 14, 12)
    assert all(s.stop - s.start == 16 for s in res.window)  # the 16-voxel window centred in the range
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
    # the window size analysis runs on its own, next to the autocorrelation analysis
    tw.sweep_start.setValue(8)
    tw.sweep_step.setValue(8)
    tw.run_sweep_analysis()
    assert tw.wait(300_000)
    _pump()
    assert tw.sweep is not None and len(tw.sweep.levels) >= 3 and tw.tabs.currentIndex() == tw.TAB_SWEEP
    assert all(lvl.radial is not None for lvl in tw.sweep.levels)  # every size keeps its radial curve for the plot
    assert all(len(lvl.samples) == 1 for lvl in tw.sweep.levels)  # concentric windows, one per size
    assert tw.result is not None and tw._btn_apply.isEnabled()  # the autocorrelation result is untouched
    if tw.sweep_size() is not None:
        tw.use_sweep_size()
        assert tw.window_size.value() >= tw.sweep_size()
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
    flat = np.full((24, 24, 24), 3.0, dtype=np.float32)
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
    assert (
        main(["texture", str(path), "--window", "16", "--sweep", "--sweep-start", "8", "--sweep-step", "8", "-o", str(out)]) == 0
    )
    assert (out / "texture_profiles.csv").is_file() and (out / "texture_profiles.png").is_file()
    summary = json.loads((out / "texture_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok" and "recommendation" in summary and "sweep" in summary
    text = capsys.readouterr().out
    assert "suggested subset" in text and "sweep 1/e" in text
    roi = np.zeros(SHAPE, dtype=np.uint8)
    roi[8:-8, 8:-8, 8:-8] = 1
    save_volume(tmp_path / "roi.h5", roi)
    assert main(["texture", str(path), "--roi", str(tmp_path / "roi.h5"), "--window", "16", "-o", str(out / "roi")]) == 0
    summary = json.loads((out / "roi" / "texture_summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["range"] == [[8, 56], [8, 48], [8, 40]]  # the region's bounding box (x, y, z)
