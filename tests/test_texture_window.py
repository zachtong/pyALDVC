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
    tw.max_lag.setValue(16)
    tw.analyse()
    assert tw.wait(120_000)
    _pump()
    res = tw.result
    assert res is not None and res.status == "ok"
    assert res.window == (slice(4, 44), slice(6, 50), slice(8, 56))  # the region's box
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
    # the size sweep through the worker
    tw.run_sweep.setChecked(True)
    tw.sweep_start.setValue(16)
    tw.sweep_step.setValue(16)
    tw.sweep_count.setValue(3)
    tw.sweep_samples.setValue(2)
    tw.analyse()
    assert tw.wait(300_000)
    _pump()
    assert tw.sweep is not None and len(tw.sweep.levels) == 3 and tw.tabs.currentIndex() == 1
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
    assert main(["texture", str(path), "--max-lag", "12", "--sweep", "--sweep-count", "2", "--samples", "2", "-o", str(out)]) == 0
    assert (out / "texture_profiles.csv").is_file() and (out / "texture_profiles.png").is_file()
    summary = json.loads((out / "texture_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok" and "recommendation" in summary and "sweep" in summary
    text = capsys.readouterr().out
    assert "suggested subset" in text and "sweep 1/e" in text
    roi = np.zeros(SHAPE, dtype=np.uint8)
    roi[8:-8, 8:-8, 8:-8] = 1
    save_volume(tmp_path / "roi.h5", roi)
    assert main(["texture", str(path), "--roi", str(tmp_path / "roi.h5"), "--max-lag", "8", "-o", str(out / "roi")]) == 0
    summary = json.loads((out / "roi" / "texture_summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["window"] == [[8, 40], [8, 48], [8, 56]]
