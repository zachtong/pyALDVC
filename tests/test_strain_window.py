"""Strain post-processing window, the shared slice-plot drawing and the image export."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.export.slice_plots import build_axes, draw_field_planes, export_field_images, field_label  # noqa: E402
from al_dvc.gui.names import select_key  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application([])


@pytest.fixture(scope="module")
def pair():
    shape = (48, 52, 56)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=5)
    dfm = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, -0.005, 0.005]), (0.4, -0.3, 0.2), centre))
    return ref, dfm


@pytest.fixture(scope="module")
def result_no_strain(pair):
    para = dvcpara_default(winsize=12, winstepsize=6, search_radius=3, admm_max_iter=1, verbose=False)
    return run_aldvc(para, list(pair), compute_strain=False)


def test_slice_plots_draw_and_export(result_no_strain, tmp_path):
    from matplotlib.figure import Figure

    res = result_no_strain
    fig = Figure(figsize=(9, 3))
    axes, cax = build_axes(fig, "grid")
    info = draw_field_planes(axes, cax, res, 0, "disp_u", {"z": 10, "y": None, "x": 5}, background=None)
    assert cax.get_visible() and info["indices"] == (10, res.volume_shape[1] // 2, 5)
    lo, hi = info["clim"]
    assert hi > lo
    with pytest.raises(ValueError):
        build_axes(fig, "diagonal")
    with pytest.raises(ValueError):
        draw_field_planes(axes, cax, res, 0, "disp_u", background=np.zeros((2, 2, 2)))
    files = export_field_images(res, tmp_path / "img", ["disp_u", "disp_magnitude"], frames=None, layout="row", dpi=60)
    assert len(files) == 2 and all(p.is_file() and p.stat().st_size > 1000 for p in files)
    assert field_label("disp_u", "mm") == "u [mm]" and field_label("exx") == "exx"


def test_strain_window_computes_and_publishes(qapp, pair):
    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    window.show()
    window.state.set_volume_arrays(list(pair), ["ref", "def"])
    window.state.set_params(winsize=12, winstepsize=6, search_radius=3, admm_max_iter=1, verbose=False)
    window.run_panel.start()
    assert window.run_panel.wait(300_000)
    qapp.processEvents()
    res = window.state.results
    assert res is not None and not res.result_strain  # the GUI run leaves strain to the post-processing window
    sw = window.open_strain_window()
    qapp.processEvents()
    assert sw.isVisible() and window.open_strain_window() is sw
    fields = [sw.field.itemData(i) for i in range(sw.field.count())]
    assert "disp_magnitude" in fields and "exx" not in fields
    assert sw.canvas.last_clim is not None  # displacement is drawn before any strain exists
    assert select_key(sw.method, "plane_fit") and select_key(sw.measure, "infinitesimal")
    sw.compute()
    assert sw.wait(120_000)
    qapp.processEvents()
    res = window.state.results
    assert len(res.result_strain) == 1 and sw.field.currentData() == "exx"
    exx = res.result_strain[0].field("exx")
    assert abs(np.nanmedian(exx) - 0.01) < 2e-3  # the synthetic stretch is recovered
    assert window.results_panel.field.count() > len(fields)  # the main window sees the strain too
    assert not sw.is_stale
    sw.halfwidth.setValue(2)
    assert sw.is_stale
    sw.frame_slider.setValue(0)
    sw.layout_combo.setCurrentIndex(2)
    qapp.processEvents()
    assert sw.canvas.layout_key == "grid"
    sw.auto_range.setChecked(False)
    assert sw.vmin.isEnabled() and sw.vmax.value() > sw.vmin.value()
    sw.close()
    window.close()


def test_strain_window_without_results(qapp):
    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    sw = window.open_strain_window()
    assert not sw._btn_compute.isEnabled() and sw.field.count() == 0
    sw.compute()  # no-op without results
    assert sw.wait(1000)
    sw.close()
    window.close()
