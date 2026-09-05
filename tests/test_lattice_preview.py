"""The planned node lattice: the pure helpers and their preview in the slice viewer."""

import os

import numpy as np
import pytest

from al_dvc.core.data_structures import VOIRange
from al_dvc.gui.lattice_preview import (
    HOVER_RADIUS,
    describe,
    layer_grid,
    layer_index,
    layer_nodes,
    layer_segments,
    nearest_node,
    plan_lattice,
    subset_rect,
)
from al_dvc.mesh.grid_mesh import build_grid_axes, mesh_setup

SHAPE = (48, 56, 64)  # (nz, ny, nx)


def test_plan_matches_the_pipeline_mesh():
    plan = plan_lattice(SHAPE, (16, 16, 16), (8, 8, 8))
    x0, y0, z0 = build_grid_axes(VOIRange(), SHAPE, (16, 16, 16), (8, 8, 8))
    mesh = mesh_setup(x0, y0, z0)
    assert np.array_equal(plan.x0, x0) and np.array_equal(plan.y0, y0) and np.array_equal(plan.z0, z0)
    assert plan.n_nodes == mesh.n_nodes and plan.grid_shape == mesh.grid_shape
    assert plan.half == (8, 8, 8) and plan.centre_valid is None and plan.n_valid == plan.n_nodes
    assert plan.overlap == pytest.approx((1 - 8 / 17,) * 3)


def test_plan_respects_the_voi_and_the_mask():
    voi = VOIRange(x=(10, 50), y=(8, 40), z=(4, 44))
    plan = plan_lattice(SHAPE, (12, 12, 12), (6, 6, 6), voi=voi)
    assert plan.x0.min() >= 10 + 6 and plan.x0.max() <= 50 - 6
    assert plan.z0.min() >= 4 + 6 and plan.z0.max() <= 44 - 6
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:, :, : SHAPE[2] // 2] = True  # only the left half is analysed
    plan = plan_lattice(SHAPE, (12, 12, 12), (6, 6, 6), mask=mask)
    assert plan.centre_valid is not None and plan.centre_valid.shape == plan.grid_shape
    assert 0 < plan.n_valid < plan.n_nodes
    assert np.array_equal(plan.centre_valid.any(axis=(0, 1)), plan.x0 < SHAPE[2] // 2)
    with pytest.raises(ValueError):
        plan_lattice(SHAPE, (12, 12, 12), (6, 6, 6), mask=np.zeros((2, 2, 2), dtype=bool))
    with pytest.raises(ValueError):  # the pipeline's own message for a subset that does not fit
        plan_lattice((20, 20, 20), (32, 32, 32), (8, 8, 8))


def test_layers_nodes_and_subset_geometry():
    plan = plan_lattice(SHAPE, (16, 24, 8), (8, 8, 8))  # a non-cubic subset: x 17, y 25, z 9 voxels
    k = layer_index(plan, "xy", int(plan.z0[2]))
    assert k == 2
    H, V, valid, dist = layer_nodes(plan, "xy", int(plan.z0[2]) + 1)
    assert dist == 1.0 and H.size == len(plan.x0) * len(plan.y0) and valid.all()
    assert set(np.unique(H)) == set(plan.x0) and set(np.unique(V)) == set(plan.y0)
    H, V, _v, _d = layer_nodes(plan, "yz", int(plan.x0[0]))
    assert set(np.unique(H)) == set(plan.y0) and set(np.unique(V)) == set(plan.z0)
    node = (float(plan.x0[1]), float(plan.y0[1]))
    assert nearest_node(plan, "xy", node[0] + 2.0, node[1] - 3.0, int(plan.z0[0])) == node
    far = node[0] + (HOVER_RADIUS + 1) * 8 * 10
    assert nearest_node(plan, "xy", far, node[1], int(plan.z0[0])) is None
    left, bottom, w, h = subset_rect(plan, "xy", node)
    assert (w, h) == (17, 25) and left == node[0] - 8.5 and bottom == node[1] - 12.5
    left, bottom, w, h = subset_rect(plan, "xz", (node[0], float(plan.z0[0])))
    assert (w, h) == (17, 9)
    text = describe(plan)
    assert f"{plan.n_nodes:,} nodes" in text and "17 x 25 x 9" in text


def test_every_layer_of_every_plane_indexes_the_mask_correctly():
    """Regression: the xz / yz layers were indexed with the other plane's layer count and went out of bounds."""
    mask = np.zeros(SHAPE, dtype=bool)
    mask[4:-4, 6:-6, 8 : SHAPE[2] // 2] = True  # the right half of x is outside the region
    plan = plan_lattice(SHAPE, (12, 12, 12), (6, 6, 6), mask=mask)
    nz, ny, nx = plan.grid_shape
    assert len({nz, ny, nx}) == 3  # unequal node counts, so a wrong axis shows
    for plane, axis in (("xy", plan.z0), ("xz", plan.y0), ("yz", plan.x0)):
        for index in axis:
            H, V, valid, dist = layer_grid(plan, plane, int(index))
            assert H.shape == V.shape == valid.shape and dist == 0.0
            segments, _d = layer_segments(plan, plane, int(index))
            assert segments.shape[1:] == (2, 2)
    # the grid stops at the region's edge: no segment touches a node outside the mask
    H, V, valid, _d = layer_grid(plan, "xy", int(plan.z0[nz // 2]))
    segments, _d = layer_segments(plan, "xy", int(plan.z0[nz // 2]))
    outside = {(float(h), float(v)) for h, v, ok in zip(H.ravel(), V.ravel(), valid.ravel()) if not ok}
    assert outside and not any(tuple(p) in outside for seg in segments for p in seg)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    assert np.all(lengths == 6.0)


# --------------------------------------------------------------------------- the viewer
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.patches import Rectangle  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application(["pytest"])


def _rects(ax):
    return [p for p in ax.patches if isinstance(p, Rectangle)]


def _grids(ax):
    from matplotlib.collections import LineCollection

    return [c for c in ax.collections if isinstance(c, LineCollection)]


class _Event:
    def __init__(self, ax, x, y):
        self.inaxes, self.xdata, self.ydata = ax, x, y
        self.button, self.x, self.y = None, 0, 0


def test_viewer_previews_the_lattice(qapp):
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow
    from al_dvc.synthetic import generate_speckle_volume

    window = MainWindow()
    window.show()
    vol = generate_speckle_volume(SHAPE, sigma=2.0, seed=1)
    window.state.set_volume_arrays([vol, vol], ["a", "b"])
    window.state.set_params(winsize=16, winstepsize=8, search_radius=4)
    QApplication.processEvents()
    viewer = window.viewer
    assert viewer.show_lattice.isChecked()
    assert "nodes" in viewer._lattice_label.text() and "17 x 17 x 17" in viewer._lattice_label.text()
    assert viewer._plan is None and all(not _grids(ax) for ax in viewer.axes)  # no region yet: label only
    mask = np.zeros(SHAPE, dtype=bool)
    mask[4:-4, 6:-6, 8:-8] = True
    window.state.set_mask(0, mask=mask)
    QApplication.processEvents()
    assert viewer._plan is not None
    plan = viewer._plan
    for ax in viewer.axes:
        assert _grids(ax), "every plane shows the grid of its nearest layer"
        assert len(_rects(ax)) >= 1, "the subset of the crosshair node is outlined"
    # hovering a node outlines its subset on that plane only
    ax = viewer.axes[0]
    n_before = len(_rects(ax))
    viewer._on_motion(_Event(ax, float(plan.x0[1]) + 1.0, float(plan.y0[1]) - 1.0))
    assert viewer._hover_artist is not None and len(_rects(ax)) == n_before + 1
    left, bottom, w, h = subset_rect(plan, "xy", (float(plan.x0[1]), float(plan.y0[1])))
    assert (viewer._hover_artist.get_x(), viewer._hover_artist.get_y()) == (left, bottom)
    viewer._on_motion(_Event(None, None, None))  # pointer off the planes
    assert viewer._hover_artist is None and len(_rects(ax)) == n_before
    # the subset size feeds straight through
    window.param_panel.winsize.setValue(25)
    QApplication.processEvents()
    assert "25 x 25 x 25" in viewer._lattice_label.text()
    assert viewer._plan.half == (12, 12, 12)
    # a subset that does not fit reports the pipeline's message instead of crashing
    window.state.set_params(winsize=200)
    QApplication.processEvents()
    assert viewer._plan is None and "does not fit" in viewer._lattice_label.text()
    window.state.set_params(winsize=16)
    # the checkbox hides everything and is remembered in the state
    viewer.show_lattice.setChecked(False)
    QApplication.processEvents()
    assert not window.state.show_lattice and viewer._plan is None and viewer._lattice_label.text() == ""
    assert all(not _rects(a) for a in viewer.axes)
    viewer.show_lattice.setChecked(True)
    QApplication.processEvents()
    assert viewer._plan is not None
    window.close()


def test_lattice_flag_round_trips_through_the_session(qapp, tmp_path):
    from al_dvc.gui.app_state import AppState
    from al_dvc.gui.session import apply_session, load_session, save_session

    state = AppState()
    state.show_lattice = False
    path = save_session(state, tmp_path / "s.json")
    fresh = AppState()
    assert fresh.show_lattice
    apply_session(load_session(path), fresh)
    assert not fresh.show_lattice
