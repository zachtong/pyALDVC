"""The selected volume drives the displayed result; the mask hides behind a field; the 3-D lattice never vanishes."""

import copy
import os
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pv = pytest.importorskip("pyvista")

from al_dvc.gui.view3d_scene import SceneOptions, build_scene  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application(["pytest"])


@pytest.fixture(scope="module")
def triple():
    shape = (40, 44, 48)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=4)
    f1 = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, 0.0, 0.0]), (0.3, 0.0, 0.0), centre))
    f2 = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.02, 0.0, 0.0]), (0.6, 0.0, 0.0), centre))
    return ref, f1, f2


def test_selected_volume_drives_the_result_frame_and_hides_the_mask(qapp, triple):
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    window.show()
    state = window.state
    state.set_volume_arrays(list(triple), ["ref", "d1", "d2"])
    mask = np.zeros(triple[0].shape, dtype=bool)
    mask[4:-4, 4:-4, 4:-4] = True
    state.set_mask(0, mask=mask)
    assert state.show_mask
    state.set_params(winsize=12, winstepsize=6, search_radius=3, admm_max_iter=1, verbose=False)
    state.write_checkpoints = False
    window.run_panel.start()
    assert window.run_panel.wait(300_000)
    QApplication.processEvents()
    res = state.results
    assert res is not None and len(res.result_disp) == 2
    # results select the first deformed volume, whose result is frame 0; the mask tint steps aside
    assert state.current_frame == 1 and state.result_frame() == 0 and state.display_frame == 0
    assert not state.show_mask and not window.viewer.mask_tools.show_mask.isChecked()
    assert window.results_panel.frame.value() == 1
    # the frame box in the results panel selects the volume, and the list selection sets the frame box
    window.results_panel.frame.setValue(2)
    QApplication.processEvents()
    assert state.current_frame == 2 and state.result_frame() == 1
    assert window.viewer._field_grid() is not None
    state.set_current_frame(1)
    QApplication.processEvents()
    assert window.results_panel.frame.value() == 1 and state.result_frame() == 0
    # the reference volume has no result: no overlay, the frame box stays at 1
    state.set_current_frame(0)
    QApplication.processEvents()
    assert state.result_frame() is None and window.viewer._field_grid() is None
    assert window.results_panel.frame.value() == 1
    # the user can bring the mask back
    window.viewer.mask_tools.show_mask.setChecked(True)
    assert state.show_mask
    window.close()


def test_deformed_lattice_falls_back_to_nodes_when_no_cell_is_complete(triple):
    from al_dvc.core.config import dvcpara_default
    from al_dvc.core.pipeline import run_aldvc

    ref, f1, _f2 = triple
    para = dvcpara_default(winsize=12, winstepsize=6, search_radius=3, admm_max_iter=1, verbose=False)
    res = run_aldvc(para, [ref, f1], compute_strain=False)
    pl = pv.Plotter(off_screen=True, window_size=(240, 200))
    info = build_scene(pl, res, SceneOptions(mode="warped", warp_scale=3.0), None)
    assert "field" in info.actors and info.note == ""
    b = np.asarray(info.actors["field"].bounds)
    assert np.isfinite(b).all() and (b[1::2] - b[0::2] > 5).all(), "the warped lattice fills the region"
    img = pl.screenshot(None, return_img=True)
    pl.close()
    assert img.std() > 1.0
    # a single valid layer of nodes: no hex cell has 8 valid corners, the nodes are drawn instead of nothing
    thin = copy.deepcopy(res)
    fr = thin.result_disp[0]
    nz, ny, nx = thin.dvc_mesh.grid_shape
    keep = np.zeros((nz, ny, nx), dtype=bool)
    keep[nz // 2] = True
    U = np.array(fr.U, dtype=float)
    U[~keep.ravel()] = np.nan
    thin.result_disp[0] = replace(fr, U=U, U_accum=None if fr.U_accum is None else U)  # FrameResult is frozen
    pl = pv.Plotter(off_screen=True, window_size=(240, 200))
    info = build_scene(pl, thin, SceneOptions(mode="warped", warp_scale=3.0), None)
    assert info.note == "nodes_only" and "field" in info.actors
    b = np.asarray(info.actors["field"].bounds)
    assert np.isfinite(b).all() and b[1] - b[0] > 5 and b[3] - b[2] > 5, "the valid layer of nodes is drawn"
    img = pl.screenshot(None, return_img=True)
    pl.close()
    assert img.std() > 1.0


def test_scalar_bar_is_plain_and_inside_the_viewport(triple):
    """The bar lives in its own narrow renderer: it never overlaps the scene, whatever the title length."""
    from al_dvc.core.config import dvcpara_default
    from al_dvc.core.pipeline import run_aldvc

    ref, f1, _f2 = triple
    para = dvcpara_default(winsize=12, winstepsize=6, search_radius=3, admm_max_iter=1, verbose=False)
    res = run_aldvc(para, [ref, f1], compute_strain=False)
    for title in ("Displacement magnitude", "u"):
        pl = pv.Plotter(off_screen=True, window_size=(640, 480), shape=(1, 2), col_weights=[6, 1], border=False)
        pl.subplot(0, 0)
        build_scene(pl, res, SceneOptions(mode="slices", title=title), None)
        (name, bar), *_ = pl.scalar_bars.items()
        assert name.replace(chr(10), " ").strip() == title
        title_prop = bar.GetTitleTextProperty()
        assert not title_prop.GetBold() and not title_prop.GetItalic()
        assert len(name.splitlines()[0]) <= 13  # wrapped to the bar's width
        x, _y = bar.GetPosition()
        assert x == pytest.approx(0.18)  # inside the bar's renderer, left of its own viewport
        # the scene renderer ends where the bar renderer starts
        assert pl.renderers[0].GetViewport()[2] == pytest.approx(6 / 7)
        pl.close()
