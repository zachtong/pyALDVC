"""The background image has its own frame, so a reference-position field can be seen over the reference."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.gui.app import MainWindow, create_application  # noqa: E402

SHAPE = (16, 20, 24)


@pytest.fixture(scope="module")
def qapp():
    return create_application(["pytest"])


def _pump(n=15):
    for _ in range(n):
        QApplication.processEvents()


@pytest.fixture
def window(qapp):
    """Four frames whose grey level is the frame number, so the drawn one is identifiable."""
    w = MainWindow()
    w.state.set_volume_arrays([np.full(SHAPE, 100 * k, dtype=np.uint16) for k in range(4)], [f"f{k}" for k in range(4)])
    _pump()
    yield w
    w.close()


def _grey(window) -> int:
    return int(window.viewer._volume.flat[0])


def test_the_choices_name_the_reference(window):
    combo = window.viewer.background_frame
    assert [combo.itemData(i) for i in range(combo.count())] == [None, 0, 1, 2, 3]
    assert combo.itemText(0) == "Selected frame" and combo.itemText(1) == "Reference (frame 0)"
    assert combo.itemText(3) == "Frame 2"


def test_by_default_the_background_follows_the_selected_frame(window):
    assert window.state.background_frame is None
    for k in (0, 2, 3):
        window.state.set_current_frame(k)
        _pump()
        assert window.state.background_index() == k
        assert window.viewer._volume_index == k and _grey(window) == 100 * k


def test_pinning_the_reference_keeps_it_while_the_selection_moves(window):
    combo = window.viewer.background_frame
    window.state.set_current_frame(2)
    _pump()
    combo.setCurrentIndex(combo.findData(0))
    _pump()
    assert window.state.background_frame == 0
    assert window.state.background_index() == 0 and _grey(window) == 0
    window.state.set_current_frame(3)  # the field moves to frame 3, the image stays on the reference
    _pump()
    assert window.state.current_frame == 3
    assert window.state.background_index() == 0 and _grey(window) == 0
    combo.setCurrentIndex(0)  # back to following
    _pump()
    assert window.state.background_frame is None and _grey(window) == 300


def test_the_three_d_volume_slices_use_the_same_frame(window):
    combo = window.viewer.background_frame
    window.state.set_current_frame(3)
    combo.setCurrentIndex(combo.findData(1))
    _pump()
    assert window.state.background_index() == 1
    vol = window.view3d._volume_for_scene()
    if vol is not None:  # only when the volume-slice toggle is on
        assert int(np.asarray(vol).flat[0]) == 100


def test_a_pin_that_no_longer_points_at_a_frame_is_forgotten(window):
    combo = window.viewer.background_frame
    combo.setCurrentIndex(combo.findData(3))
    _pump()
    assert window.state.background_frame == 3
    window.state.set_volume_arrays([np.zeros(SHAPE, dtype=np.uint16)] * 2, ["a", "b"])
    _pump()
    assert window.state.background_frame is None  # not left dangling for a later sequence to re-apply
    assert combo.currentData() is None
    assert window.state.background_index() == 0


def test_the_label_says_which_configuration_is_on_screen(window, monkeypatch):
    viewer = window.viewer
    monkeypatch.setattr(type(window.state), "result_frame", lambda self: 0)  # pretend a result is loaded
    window.state.show_overlay = True
    window.state.set_current_frame(2)
    _pump()
    viewer._update_config_label()
    assert "image deformed" in viewer._config_label.text()
    viewer.background_frame.setCurrentIndex(viewer.background_frame.findData(0))
    _pump()
    viewer._update_config_label()
    assert "reference configuration" in viewer._config_label.text()
    window.state.show_overlay = False
    viewer._update_config_label()
    assert viewer._config_label.text() == ""  # nothing to say when no field is drawn


def test_the_background_frame_survives_a_session_roundtrip(window, tmp_path):
    from al_dvc.gui.session import apply_session, load_session, save_session
    from al_dvc.io.volume_io import save_volume

    for k in range(4):  # a session stores paths, so the frames need files
        save_volume(tmp_path / f"v{k}.npy", np.full(SHAPE, 100 * k, dtype=np.uint16))
    window.state.volumes = []
    window.state.add_volume_paths([str(tmp_path / f"v{k}.npy") for k in range(4)])
    _pump()
    combo = window.viewer.background_frame
    combo.setCurrentIndex(combo.findData(2))
    _pump()
    path = save_session(window.state, tmp_path / "s.aldvc")
    window.state.set_display(background_frame=None)
    apply_session(load_session(path), window.state, path)
    _pump()
    assert window.state.background_frame == 2
    assert window.viewer.background_frame.currentData() == 2
