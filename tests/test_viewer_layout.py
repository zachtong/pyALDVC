"""Slice viewer layouts and colorbar stability, ROI-trimmed displacement fields, 3-D view controls per mode."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from al_dvc.export.export_utils import field_array  # noqa: E402
from al_dvc.gui.panels.viewer import LAYOUTS  # noqa: E402
from al_dvc.gui.view3d_scene import BACKGROUNDS, SceneOptions, foreground_for  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application([])


@pytest.fixture(scope="module")
def small_pair():
    shape = (48, 52, 56)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=3)
    dfm = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, -0.005, 0.005]), (0.4, -0.3, 0.2), centre))
    return ref, dfm


@pytest.fixture(scope="module")
def window_with_result(qapp, small_pair):
    """A window that ran a tiny analysis with a mask covering part of the volume."""
    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    window.show()
    ref = small_pair[0]
    window.state.set_volume_arrays(list(small_pair), ["ref", "def"])
    mask = np.zeros(ref.shape, dtype=bool)
    nz, ny, nx = ref.shape
    mask[nz // 4 : 3 * nz // 4, ny // 4 : 3 * ny // 4, nx // 4 : 3 * nx // 4] = True  # central block: the grid is cropped to it
    window.state.set_mask(0, mask=mask)
    window.state.set_params(winsize=8, winstepsize=4, search_radius=2, admm_max_iter=1, verbose=False)
    window.run_panel.start()
    assert window.run_panel.wait(300_000)
    qapp.processEvents()
    assert window.state.results is not None
    yield window
    window.close()


def _bounds(viewer):
    viewer.canvas.draw()  # positions are final only after the aspect ratio has been applied at draw time
    return [tuple(round(v, 5) for v in ax.get_position().bounds) for ax in viewer.axes]


def test_axes_keep_their_size_when_slices_change(qapp, window_with_result):
    viewer = window_with_result.viewer
    assert viewer.cax.get_visible()  # a field is overlaid: the colorbar is shown
    before = _bounds(viewer)
    for k in range(5):
        viewer.sliders["z"].setValue(2 + k)
        viewer.sliders["x"].setValue(3 + k)
        qapp.processEvents()
    assert _bounds(viewer) == before  # the colorbar has its own axes: redraws never shrink the images


def test_layouts_switch_and_persist(qapp, window_with_result):
    window = window_with_result
    viewer = window.viewer
    assert viewer.layout_key == "row" and window.state.slice_layout == "row"
    for key in ("column", "grid", "row"):
        viewer.set_layout(key)
        qapp.processEvents()
        assert viewer.layout_key == key and window.state.slice_layout == key
        assert len(viewer.axes) == 3 and viewer.cax.get_visible()
        bounds = _bounds(viewer)
        assert len({b[:2] for b in bounds}) == 3  # three distinct positions
    viewer.layout_combo.setCurrentIndex(LAYOUTS.index("grid"))
    assert viewer.layout_key == "grid"
    xy, xz, yz = _bounds(viewer)
    assert xy[1] > xz[1] and abs(xy[0] - xz[0]) < 0.15  # XY above XZ in the left column
    assert yz[0] > xy[0] + 0.3 and yz[1] > xz[1]  # YZ in the right column, top row
    with pytest.raises(ValueError):
        viewer.set_layout("diagonal")
    viewer.set_layout("row")


def test_displacement_is_trimmed_to_valid_nodes(window_with_result):
    res = window_with_result.state.results
    valid = np.asarray(res.dvc_mesh.node_valid, dtype=bool)
    assert 0 < valid.sum() < valid.size  # the mask left some nodes invalid
    trimmed = field_array(res, 0, "disp_magnitude")
    full = field_array(res, 0, "disp_magnitude", trimmed=False)
    assert np.isnan(trimmed[~valid]).all() and np.isfinite(trimmed[valid]).all()
    assert np.isfinite(full).all()
    strain = field_array(res, 0, "exx")
    assert np.isnan(strain[~valid]).all()  # displacement now behaves like strain


def test_view3d_controls_follow_the_mode(qapp, window_with_result):
    panel = window_with_result.view3d
    expected = {"slices": {"slices"}, "points": set(), "surface": {"iso"}, "warped": {"warp_scale"}}
    for i, (mode, controls) in enumerate(expected.items()):
        panel.mode.setCurrentIndex(i)
        qapp.processEvents()
        assert panel.mode_key() == mode and panel.visible_controls() == controls
    panel.arrows.setChecked(True)
    assert "arrows" in panel.visible_controls()
    panel.arrows.setChecked(False)
    panel.mode.setCurrentIndex(0)


def test_view3d_slice_spins_share_the_state(qapp, window_with_result):
    window = window_with_result
    panel, viewer = window.view3d, window.viewer
    panel.slice_spins["z"].setValue(5)
    assert window.state.slice_index["z"] == 5 and viewer.slice_indices()[0] == 5
    viewer.sliders["y"].setValue(7)
    qapp.processEvents()
    assert panel.slice_spins["y"].value() == 7


def test_background_option_and_contrast():
    assert foreground_for(BACKGROUNDS["white"]) == "black"
    assert foreground_for(BACKGROUNDS["dark"]) == "white"
    assert foreground_for("nonsense") == "white"
    opts = SceneOptions(background=BACKGROUNDS["grey"])
    assert opts.background == "#808080" and opts.colormap == "turbo"


@pytest.mark.skipif(not pytest.importorskip("al_dvc.gui.view3d_scene").available(), reason="pyvista not installed")
def test_view3d_renders_on_a_white_background(qapp, window_with_result):
    panel = window_with_result.view3d
    panel.background.setCurrentIndex(list(BACKGROUNDS).index("white"))
    window_with_result.center_tabs.setCurrentIndex(1)
    qapp.processEvents()
    panel.refresh()
    assert panel.options().background == "#ffffff"
    assert panel._last_image is not None
    corner = panel._last_image[2, 2, :3]
    assert (corner > 240).all()  # the corner pixel shows the white background
    panel.background.setCurrentIndex(0)
    window_with_result.center_tabs.setCurrentIndex(0)
