"""Short checks of the export, batch and display fixes from the UI audit."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from al_dvc.export.slice_plots import ordered_limits  # noqa: E402
from al_dvc.gui.dialogs.export_dialog import (  # noqa: E402
    ExportConfig,
    ExportError,
    existing_outputs,
    export_formats,
    run_export,
    validate_basename,
)
from al_dvc.gui.panels.volume_panel import natural_key  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application([])


def test_basename_validation_natural_sort_and_limits():
    assert validate_basename("run1") is None
    assert validate_basename("  ") == "empty" and validate_basename("..") == "dots"
    assert validate_basename("../x") == "characters" and validate_basename(r"C:\x") == "characters"
    assert validate_basename("a:b") == "characters"
    names = ["frame10.tif", "frame2.tif", "Frame1.tif"]
    assert sorted(names, key=natural_key) == ["Frame1.tif", "frame2.tif", "frame10.tif"]
    assert ordered_limits(3.0, 1.0) == (1.0, 3.0) and ordered_limits(2.0, 2.0)[1] > 2.0


def test_export_goes_on_after_a_failed_format(tmp_path, monkeypatch):
    import al_dvc.export as ex

    def fake_npz(result, path):
        Path(path).write_text("npz", encoding="utf-8")
        return Path(path)

    def fake_mat(result, path):
        raise OSError("disk full")

    monkeypatch.setattr(ex, "export_npz", fake_npz)
    monkeypatch.setattr(ex, "export_mat", fake_mat)
    result = SimpleNamespace(result_strain=None)
    cfg = ExportConfig(out_dir=tmp_path, basename="case", npz=True, mat=True)
    outcome = export_formats(result, cfg)
    assert outcome.written["npz"] == [tmp_path / "case.npz"] and "disk full" in outcome.errors["mat"]
    with pytest.raises(ExportError) as info:
        run_export(result, cfg)
    assert info.value.outcome.paths == [tmp_path / "case.npz"]
    assert existing_outputs(cfg) == [tmp_path / "case.npz"]
    with pytest.raises(ValueError):  # a format that takes fields is refused without any, before anything is written
        export_formats(result, ExportConfig(out_dir=tmp_path / "none", npz=False, csv=True, fields=[]))
    assert not (tmp_path / "none").exists()


def test_step_per_axis_and_colour_limits(qapp):
    from al_dvc.gui.app_state import AppState
    from al_dvc.gui.panels.param_panel import ParamPanel
    from al_dvc.gui.panels.results_panel import ResultsPanel

    state = AppState()
    panel = ParamPanel(state)
    state.set_params(winstepsize=(4, 4, 10))
    assert [w.value() for w in panel.winstepsize_axes] == [4, 4, 10] and not panel.winstepsize_lock.isChecked()
    panel.winstepsize_axes[1].setValue(6)
    assert state.para.winstepsize == (4, 6, 10)  # editing one axis leaves the others alone
    panel.winstepsize_lock.setChecked(True)
    assert state.para.winstepsize == (4, 4, 4)
    results = ResultsPanel(state)
    results.auto_range.setChecked(False)
    results.vmax.setValue(1.0)
    results.vmin.setValue(2.0)  # min above max pushes max up: the pair stays ordered
    assert state.color_min == 2.0 and state.color_max > 2.0 and results.vmax.value() > 2.0
    state.set_display(color_auto=True, overlay_alpha=0.4, show_overlay=False)
    assert results.auto_range.isChecked() and results.alpha.value() == 40 and not results.show_overlay.isChecked()
