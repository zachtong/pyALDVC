"""The length unit belongs to the voxel size: a voxel size with the unit still "voxel" mislabels every displacement."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from al_dvc.core.config import dvcpara_default, length_unit, units_problem  # noqa: E402


def test_the_label_follows_the_voxel_size():
    assert length_unit(dvcpara_default()) == "voxel"
    assert length_unit(dvcpara_default(voxel_size=(6.5, 6.5, 6.5), units="um")) == "um"
    assert length_unit(dvcpara_default(units="mm")) == "mm"  # 1-mm voxels are a real case
    for label in ("voxel", "", "  ", "VX"):
        para = dvcpara_default(voxel_size=(6.5, 6.5, 6.5), units=label)
        assert length_unit(para) == "?"  # scaled values in an unnamed unit: never "voxel"
        assert "6.5" in units_problem(para) and "unit" in units_problem(para)
    assert units_problem(dvcpara_default()) == "" and units_problem(dvcpara_default(voxel_size=(2, 2, 2), units="um")) == ""


def test_the_window_asks_for_the_unit_and_does_not_run_without_it():
    pytest.importorskip("PySide6")
    import numpy as np

    from al_dvc.gui.app import MainWindow, create_application
    from al_dvc.gui.app_state import RunState

    app = create_application(["pytest"])
    window = MainWindow()
    panel, state = window.param_panel, window.state
    seen = []
    state.log_message.connect(lambda message, level: seen.append((level, message)))
    assert not panel.unit_hint.isVisibleTo(panel)
    panel.voxel[0].setValue(6.5)
    app.processEvents()
    assert panel.unit_hint.isVisibleTo(panel)  # the voxel size is scaled, the unit is still "voxel"
    state.set_volume_arrays([np.zeros((24, 24, 24), np.float32)] * 2, ["a", "b"])
    window.run_panel.start()
    assert window.run_panel._worker is None and state.run_state is RunState.IDLE
    assert any(level == "error" and "unit" in m for level, m in seen)
    panel.units.setText("um")
    panel.units.editingFinished.emit()
    app.processEvents()
    assert state.para.units == "um" and not panel.unit_hint.isVisibleTo(panel)
    window.close()
