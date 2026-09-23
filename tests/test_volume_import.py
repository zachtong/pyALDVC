"""Volume import rules: one file type per import, natural or character order, a folder replaces the sequence."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from al_dvc.gui.app_state import lexical_key, natural_key  # noqa: E402
from al_dvc.gui.panels.volume_panel import select_single_type  # noqa: E402
from al_dvc.io.volume_io import save_volume  # noqa: E402


def _pump():
    from PySide6.QtWidgets import QApplication

    for _ in range(5):
        QApplication.processEvents()


def test_select_single_type_keeps_the_majority_volume_type(tmp_path):
    names = ["a1.npy", "a2.npy", "a10.npy", "notes.txt", "preview.png", "mask.tif", "x.nii.gz"]
    paths = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"0")
        paths.append(str(p))
    kept, skipped = select_single_type(paths)
    assert [os.path.basename(k) for k in kept] == ["a1.npy", "a2.npy", "a10.npy"]
    assert skipped == {".txt": 1, ".png": 1, ".tif": 1, ".nii.gz": 1}
    assert select_single_type([str(tmp_path / "notes.txt")]) == ([], {".txt": 1})


def test_sort_keys():
    names = ["frame10.tif", "Frame2.tif", "frame1.tif"]
    assert sorted(names, key=natural_key) == ["frame1.tif", "Frame2.tif", "frame10.tif"]
    assert sorted(names, key=lexical_key) == ["frame1.tif", "frame10.tif", "Frame2.tif"]


def test_folder_import_replaces_filters_and_sorts(tmp_path):
    from al_dvc.gui.app import MainWindow, create_application

    create_application(["pytest"])
    vol = np.zeros((4, 5, 6), dtype=np.float32)
    first = tmp_path / "first"
    first.mkdir()
    for n in ("s1.npy", "s2.npy"):
        save_volume(first / n, vol)
    second = tmp_path / "second"
    second.mkdir()
    for n in ("img10.npy", "img2.npy", "img1.npy"):
        save_volume(second / n, vol)
    (second / "readme.txt").write_text("not a volume")
    (second / "other.tif").write_bytes(b"0")
    window = MainWindow()
    panel, state = window.volume_panel, window.state
    assert panel.import_folder(str(first)) == 2 and [v.name for v in state.volumes] == ["s1.npy", "s2.npy"]
    panel._natural_sort.setChecked(True)
    assert panel.import_folder(str(second)) == 3  # the previous sequence is replaced, other types are skipped
    assert [v.name for v in state.volumes] == ["img1.npy", "img2.npy", "img10.npy"]
    panel._natural_sort.setChecked(False)  # character order re-sorts the list
    assert [v.name for v in state.volumes] == ["img1.npy", "img10.npy", "img2.npy"]
    panel._natural_sort.setChecked(True)
    assert [v.name for v in state.volumes] == ["img1.npy", "img2.npy", "img10.npy"]
    # Add volumes appends
    extra = tmp_path / "img3.npy"
    save_volume(extra, vol)
    assert panel.import_files([str(extra)], replace=False) == 1 and len(state.volumes) == 4
    _pump()
    window.close()


# ------------------------------------------------------------------ volumes of different sizes
SEP = " × "  # the multiplication sign of the Shape column
FLAG = "⚠"  # warning sign in front of a size that differs from the reference


def _messages(state) -> list:
    seen: list = []
    state.log_message.connect(lambda message, level: seen.append((level, message)))
    return seen


def _shape_cells(panel) -> list[str]:
    return [panel._list.item(i, 3).text() for i in range(panel._list.rowCount())]


def test_volumes_of_another_size_are_flagged_as_soon_as_they_are_added(tmp_path):
    """A micro-CT sequence with scans of 1856, 1857 and 1851 slices: the run used to find out at frame 2."""
    from al_dvc.gui.app import MainWindow, create_application
    from al_dvc.gui.app_state import RunState

    create_application(["pytest"])
    shapes = {"a0.npy": (8, 9, 10), "a1.npy": (8, 9, 10), "a2.npy": (9, 9, 10), "a3.npy": (8, 9, 11)}
    for name, shape in shapes.items():
        save_volume(tmp_path / name, np.zeros(shape, np.float32))
    window = MainWindow()
    panel, state = window.volume_panel, window.state
    seen = _messages(state)
    panel.import_files([str(tmp_path / "a0.npy")])
    panel.import_files([str(tmp_path / n) for n in ("a1.npy", "a2.npy", "a3.npy")])
    assert state.wait_for_shapes(10_000)
    _pump()
    # every row shows its size, x by y by z, without the deformed volumes being read
    sizes = [text.lstrip(FLAG + " ") for text in _shape_cells(panel)]
    assert sizes == [
        SEP.join(["10", "9", "8"]),
        SEP.join(["10", "9", "8"]),
        SEP.join(["10", "9", "9"]),
        SEP.join(["11", "9", "8"]),
    ]
    assert not any(v.resident for v in state.volumes[1:])
    # the two that differ are marked, with the reference's size in the tooltip
    assert [text.startswith(FLAG) for text in _shape_cells(panel)] == [False, False, True, True]
    assert SEP.join(["10", "9", "8"]) in panel._list.item(2, 3).toolTip()
    assert state.shape_mismatches() == [(2, (9, 9, 10)), (3, (8, 9, 11))]
    # the user is told at once, once, and about the two that differ only
    reports = [m for level, m in seen if level == "error" and "a2.npy" in m]
    assert len(reports) == 1 and "a3.npy" in reports[0] and "a1.npy" not in reports[0]
    # the run does not start on them
    seen.clear()
    window.run_panel.start()
    assert window.run_panel._worker is None and state.run_state is RunState.IDLE
    assert any(level == "error" and "a2.npy" in m and "a3.npy" in m for level, m in seen)
    # the reference decides: made the reference, the odd volume is the one the others differ from
    state.move_volume(2, 0)
    _pump()
    assert [i for i, _shape in state.shape_mismatches()] == [1, 2, 3]
    # without the odd ones the marks are gone
    state.move_volume(0, 2)
    state.remove_volume(3)
    state.remove_volume(2)
    _pump()
    assert state.shape_mismatches() == []
    assert not any(text.startswith(FLAG) for text in _shape_cells(panel))
    window.close()


def test_the_sizes_are_read_off_the_ui_thread(tmp_path, monkeypatch):
    """A cloud drive (Box, OneDrive) downloads a file it has not synced yet on the first read, even of the
    header: 26-45 s per 3.7-GB scan on Box Drive. Adding such files must not freeze the window."""
    import threading
    import time

    from al_dvc.gui.app import MainWindow, create_application
    from al_dvc.io import volume_io

    create_application(["pytest"])
    for k in range(2):
        save_volume(tmp_path / f"v{k}.npy", np.zeros((4, 5, 6), np.float32))
    release = threading.Event()
    real = volume_io.read_volume_shape

    def slow(path, **kw):
        release.wait(10)
        return real(path, **kw)

    monkeypatch.setattr(volume_io, "read_volume_shape", slow)
    window = MainWindow()
    panel, state = window.volume_panel, window.state
    t0 = time.perf_counter()
    panel.import_files([str(tmp_path / "v0.npy"), str(tmp_path / "v1.npy")])
    _pump()
    assert time.perf_counter() - t0 < 5.0
    assert _shape_cells(panel)[1] == "…"  # being read
    release.set()
    assert state.wait_for_shapes(10_000)
    _pump()
    assert _shape_cells(panel)[1] == SEP.join(["6", "5", "4"])
    window.close()


def test_a_mask_file_of_another_size_is_refused(tmp_path):
    from al_dvc.gui.app import MainWindow, create_application

    create_application(["pytest"])
    for k in range(2):
        save_volume(tmp_path / f"v{k}.npy", np.zeros((4, 5, 6), np.float32))
    save_volume(tmp_path / "mask_ok.npy", np.ones((4, 5, 6), np.uint8))
    save_volume(tmp_path / "mask_bad.npy", np.ones((4, 5, 7), np.uint8))
    window = MainWindow()
    state = window.state
    state.add_volume_paths([str(tmp_path / "v0.npy"), str(tmp_path / "v1.npy")])
    assert state.wait_for_shapes(10_000)
    seen = _messages(state)
    assert state.set_mask(1, path=str(tmp_path / "mask_bad.npy")) is False
    assert state.volumes[1].mask_path is None
    assert any(level == "error" and "mask_bad.npy" in m and SEP.join(["7", "5", "4"]) in m for level, m in seen)
    assert state.set_mask(1, path=str(tmp_path / "mask_ok.npy")) is True
    assert state.volumes[1].mask_path == str(tmp_path / "mask_ok.npy")
    window.close()


def test_a_session_with_volumes_of_different_sizes_warns_when_opened(tmp_path):
    """Through the session functions, not the window's menu: those also edit the recent-sessions list that
    test_gui.py checks, and a parallel run would race on it."""
    from al_dvc.gui.app import MainWindow, create_application
    from al_dvc.gui.session import apply_session, load_session, save_session

    create_application(["pytest"])
    save_volume(tmp_path / "a0.npy", np.zeros((4, 5, 6), np.float32))
    save_volume(tmp_path / "a1.npy", np.zeros((4, 5, 7), np.float32))
    window = MainWindow()
    window.state.add_volume_paths([str(tmp_path / "a0.npy"), str(tmp_path / "a1.npy")])
    assert window.state.wait_for_shapes(10_000)
    path = save_session(window.state, tmp_path / "mixed.aldvc")
    other = MainWindow()
    seen = _messages(other.state)
    apply_session(load_session(path), other.state, path)
    assert other.state.wait_for_shapes(10_000)
    _pump()
    assert any(level == "error" and "a1.npy" in m for level, m in seen)
    assert other.state.shape_mismatches() == [(1, (4, 5, 7))]
    window.close()
    other.close()


def test_a_failure_in_the_reading_thread_still_ends_the_wait(tmp_path, monkeypatch):
    """No row may stay at "..." for ever, whatever goes wrong while the sizes are read."""
    from al_dvc.gui.app import MainWindow, create_application

    create_application(["pytest"])
    for k in range(2):
        save_volume(tmp_path / f"v{k}.npy", np.zeros((4, 5, 6), np.float32))
    window = MainWindow()
    state = window.state

    class Broken:
        def emit(self, *_args):
            raise ValueError("the result could not be delivered")

    monkeypatch.setattr(state, "_shape_read", Broken())
    state.add_volume_paths([str(tmp_path / "v0.npy"), str(tmp_path / "v1.npy")])
    assert state.wait_for_shapes(10_000)
    assert not any(v.shape_pending for v in state.volumes)
    window.close()


def test_a_slow_size_read_is_explained_in_the_console(tmp_path, monkeypatch):
    from al_dvc.gui.app import MainWindow, create_application
    from al_dvc.io import volume_io

    create_application(["pytest"])
    for k in range(2):
        save_volume(tmp_path / f"v{k}.npy", np.zeros((4, 5, 6), np.float32))
    window = MainWindow()
    seen = _messages(window.state)
    monkeypatch.setattr(volume_io, "SLOW_HEADER_READ_S", 0.0)
    window.state.add_volume_paths([str(tmp_path / "v0.npy"), str(tmp_path / "v1.npy")])
    assert window.state.wait_for_shapes(10_000)
    assert any("v1.npy" in m and "downloaded the whole file" in m for _level, m in seen)
    window.close()
