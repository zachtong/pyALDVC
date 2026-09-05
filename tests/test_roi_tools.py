"""Region-of-interest round 2: threshold masks, replace mode, icon toolbar, deformed lattice, 3-D sliders."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from al_dvc.gui.mask_editor import MaskEditor, MaskOp, otsu_threshold, threshold_region  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE = (40, 48, 56)


def _object_volume():
    rng = np.random.default_rng(3)
    zz, yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    obj = ((yy - 24) ** 2 + (xx - 28) ** 2) < 16**2
    cyl = obj.copy()
    obj[10:30, 22:26, 26:30] = False  # an enclosed cavity in the cylinder: filled by the clean-up
    noise = rng.random(SHAPE).astype(np.float32)
    vol = np.where(obj, 2.0 + 0.3 * noise, 0.3 * noise).astype(np.float32)
    vol[2:5, 2:5, 2:5] = 2.5  # a small bright speck: dropped by "largest component"
    return vol, cyl


def test_otsu_and_threshold_region():
    vol, obj = _object_volume()
    level = otsu_threshold(vol)
    assert 0.4 < level < 1.9  # in the gap between background (< 0.3) and material (> 2)
    region = threshold_region(vol)
    assert region.shape == SHAPE and region[2, 2, 2] is not None
    assert not region[3, 3, 3]  # the speck is gone
    assert region[20, 24, 28]  # the hole is filled
    assert np.mean(region == obj) > 0.995  # the filled cylinder
    raw = threshold_region(vol, level=level, keep_largest=False, fill_holes=False)
    assert raw[3, 3, 3] and not raw[20, 24, 28]
    with pytest.raises(ValueError):
        threshold_region(np.zeros((4, 4)))
    assert otsu_threshold(np.array([np.nan, np.nan])) == 0.0
    assert otsu_threshold(np.ones(10)) == 1.0


def test_editor_threshold_and_replace_modes():
    vol, obj = _object_volume()
    ed = MaskEditor(SHAPE, volume=vol)
    ed.apply(MaskOp("rectangle", "xy", ((0, 0), (10, 10))))
    ed.apply(MaskOp("threshold", mode="replace"))
    assert ed.mask[20, 24, 28] and not ed.mask[5, 5, 5]  # replace: the rectangle is gone
    ed.apply(MaskOp("rectangle", "xy", ((0, 0), (5, 5)), mode="replace"))
    assert ed.mask[:, :6, :6].all() and ed.mask.sum() == SHAPE[0] * 36
    ed.undo()
    assert ed.mask[20, 24, 28]
    d = ed.to_dict()
    again = MaskEditor.from_dict(d, volume=vol)
    assert np.array_equal(again.mask, ed.mask)
    op = MaskOp("threshold", level=1.0, keep_largest=False, fill_holes=False)
    assert MaskOp.from_dict(op.to_dict()) == op
    assert "level" not in MaskOp("rectangle", "xy", ((0, 0), (1, 1))).to_dict()
    with pytest.raises(ValueError, match="volume intensities"):
        MaskEditor(SHAPE).apply(MaskOp("threshold"))


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application([])


def test_toolbar_buttons_and_auto_mask(qapp):
    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    window.show()
    tools = window.viewer.mask_tools
    vol, obj = _object_volume()
    window.state.set_volume_arrays([vol, vol], ["a", "b"])
    qapp.processEvents()
    assert all(not b.isChecked() for b in tools.tool_buttons.values())
    tools.tool_buttons["rectangle"].click()
    assert tools.settings().tool == "rectangle" and tools.tool_buttons["rectangle"].isChecked()
    tools.tool_buttons["rectangle"].click()  # click again: drawing off
    assert tools.settings().tool == "none"
    tools.set_tool("brush")
    assert tools.tool_buttons["brush"].isChecked() and tools.radius.isVisible()
    tools.mode_buttons["cut"].click()
    assert tools.settings().mode == "cut" and tools.mode.currentIndex() == 1
    tools.set_mode("replace")
    assert tools.mode_buttons["replace"].isChecked()
    tools._btn["auto"].click()
    qapp.processEvents()
    mask = window.state.current_mask()
    assert mask is not None and 0.2 < mask.mean() < 0.5
    assert abs(mask.mean() - obj.mean()) < 0.03
    assert "material" in tools._status.text()
    assert window.volume_panel._list.item(0, 4).text().startswith("ROI")
    window.close()


def test_deformed_lattice_shows_only_valid_cells(qapp):
    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    window.show()
    ref = generate_speckle_volume(SHAPE, seed=9)
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    dfm = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, 0.0, 0.0]), (0.5, 0.0, 0.0), centre))
    window.state.set_volume_arrays([ref, dfm], ["ref", "def"])
    mask = np.zeros(SHAPE, dtype=bool)
    mask[8:32, 10:38, 12:44] = True
    window.state.set_mask(0, mask=mask)
    window.state.set_params(winsize=8, winstepsize=4, search_radius=2, admm_max_iter=1, verbose=False)
    window.run_panel.start()
    assert window.run_panel.wait(300_000)
    qapp.processEvents()
    panel = window.view3d
    if panel.backend == "unavailable":
        pytest.skip("pyvista not installed")
    window.center_tabs.setCurrentIndex(1)
    panel.mode.setCurrentIndex(3)  # deformed lattice
    qapp.processEvents()
    panel.refresh()
    info = panel._last_info
    assert info is not None and "field" in info.actors and info.n_finite < info.n_nodes
    # slices mode: the sliders drive the shared slice positions
    panel.mode.setCurrentIndex(0)
    qapp.processEvents()
    assert panel._slider_box.isVisible()
    panel.slice_sliders["y"].setValue(7)
    assert window.state.slice_index["y"] == 7 and panel.slice_spins["y"].value() == 7
    window.viewer.sliders["x"].setValue(9)
    qapp.processEvents()
    assert panel.slice_sliders["x"].value() == 9
    window.close()
