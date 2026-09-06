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
