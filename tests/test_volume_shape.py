"""The size of a volume file from its header, and a sequence of different sizes refused before any frame is solved.

A run used to find out that a deformed volume had a different size only when it reached that frame: on a
micro-CT sequence of 1856-slice scans with one of 1857, the error came after the first frame had taken a minute.
"""

import time

import numpy as np
import pytest

from al_dvc.io import volume_io
from al_dvc.io.volume_io import (
    FileVolumeProvider,
    VolumeShapeError,
    _squeeze_to_3d,
    _squeezed_shape,
    load_volume,
    read_volume_shape,
    save_volume,
    shape_text,
)

tifffile = pytest.importorskip("tifffile")


def _vol(shape=(6, 7, 8), dtype=np.uint16, seed=0):
    return np.random.default_rng(seed).integers(0, 1000, shape).astype(dtype)


def _write_mat73(path, datasets: dict) -> None:
    """A MATLAB v7.3 file as MATLAB writes it: HDF5 behind a 512-byte header that marks version 7.3.

    MATLAB stores its column-major (x, y, z) array as a C-order (z, y, x) dataset.
    """
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "w", userblock_size=512) as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=data)
    text = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Mon Sep 22 12:00:00 2026 HDF5 schema 1.00 ."
    header = text.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM"
    with open(path, "r+b") as fh:
        fh.write(header)


def _tiff_cases(tmp_path):
    vol = _vol((9, 7, 5), np.uint8)
    rgb = np.random.default_rng(1).integers(0, 255, (4, 7, 5, 3), dtype=np.uint8)
    cases = {
        "shaped.tif": (vol, {}),
        "imagej.tif": (vol, {"imagej": True}),
        "generic.tif": (vol, {"metadata": None}),
        "bigtiff.tif": (vol, {"bigtiff": True}),
        "zlib.tif": (vol, {"compression": "zlib"}),
        "single_page.tif": (vol[0], {}),
        "rgb_stack.tif": (rgb, {"photometric": "rgb"}),
    }
    paths = []
    for name, (arr, kw) in cases.items():
        p = tmp_path / name
        tifffile.imwrite(str(p), arr, **kw)
        paths.append(p)
    appended = tmp_path / "appended_pages.tif"  # one IFD per slice, spread through the file
    for k in range(vol.shape[0]):
        tifffile.imwrite(str(appended), vol[k], append=True, metadata=None)
    paths.append(appended)
    return paths


def _other_cases(tmp_path):
    from scipy.io import savemat

    vol = _vol((6, 7, 8))
    paths = []
    p = tmp_path / "c_order.npy"
    np.save(p, vol)
    paths.append(p)
    p = tmp_path / "fortran.npy"
    np.save(p, np.asfortranarray(vol))
    paths.append(p)
    p = tmp_path / "two_d.npy"
    np.save(p, vol[0])
    paths.append(p)
    p = tmp_path / "plain.npz"
    np.savez(p, first=vol, second=vol[:2])
    paths.append(p)
    p = tmp_path / "packed.npz"
    np.savez_compressed(p, vol=vol)
    paths.append(p)
    p = tmp_path / "vol.h5"
    save_volume(p, vol)
    paths.append(p)
    p = tmp_path / "v5.mat"
    savemat(str(p), {"note": np.zeros((2, 2)), "vol": np.transpose(vol, (2, 1, 0))})
    paths.append(p)
    p = tmp_path / "v5_unnamed.mat"
    savemat(str(p), {"small": np.zeros((2, 3)), "scan": np.transpose(vol, (2, 1, 0))}, do_compression=True)
    paths.append(p)
    p = tmp_path / "v73.mat"
    _write_mat73(p, {"meta": np.zeros((3, 3)), "vol": vol})
    paths.append(p)
    folder = tmp_path / "tiff_slices"
    folder.mkdir()
    for k in range(4):
        tifffile.imwrite(str(folder / f"s{k:03d}.tif"), vol[k])
    paths.append(folder)
    return paths


def test_the_header_gives_the_shape_load_volume_returns(tmp_path):
    for p in _tiff_cases(tmp_path) + _other_cases(tmp_path):
        assert read_volume_shape(p) == load_volume(p).shape, p.name


def test_the_loader_options_reach_the_header_reader(tmp_path):
    from scipy.io import savemat

    vol = _vol((6, 7, 8))
    savemat(str(tmp_path / "m.mat"), {"a": np.zeros((4, 4, 4)), "b": np.transpose(vol, (2, 1, 0))})
    for kw in ({}, {"mat_key": "b"}, {"matlab_order": False}, {"mat_key": "b", "matlab_order": False}):
        assert read_volume_shape(tmp_path / "m.mat", **kw) == load_volume(tmp_path / "m.mat", **kw).shape, kw
    np.savez(tmp_path / "z.npz", a=vol[:3], b=vol)
    assert read_volume_shape(tmp_path / "z.npz", mat_key="b") == (6, 7, 8)
    save_volume(tmp_path / "d.h5", vol)
    assert read_volume_shape(tmp_path / "d.h5", mat_key="volume", dtype=np.float32) == (6, 7, 8)  # dtype is ignored
    folder = tmp_path / "mixed"
    folder.mkdir()
    for k in range(3):
        tifffile.imwrite(str(folder / f"a{k}.tif"), vol[k])
        tifffile.imwrite(str(folder / f"b{k}.tif"), vol[k, :5])
    kw = {"slice_pattern": "b*"}
    assert read_volume_shape(folder, **kw) == load_volume(folder, **kw).shape == (3, 5, 8)


def test_only_the_header_is_read(tmp_path, monkeypatch):
    """Whatever would read voxels fails here: the shape still comes back, from the headers alone."""
    import h5py
    import scipy.io

    paths = _tiff_cases(tmp_path) + _other_cases(tmp_path)

    def no_voxels(*_a, **_k):
        raise AssertionError("the voxels were read")

    monkeypatch.setattr(volume_io, "load_volume", no_voxels)
    monkeypatch.setattr(tifffile.TiffFile, "asarray", no_voxels)
    monkeypatch.setattr(tifffile.TiffPage, "asarray", no_voxels)
    monkeypatch.setattr(tifffile, "imread", no_voxels)
    monkeypatch.setattr(np, "load", no_voxels)
    monkeypatch.setattr(scipy.io, "loadmat", no_voxels)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_voxels)
    monkeypatch.setattr(h5py.Dataset, "read_direct", no_voxels)
    shapes = {p.name: read_volume_shape(p) for p in paths}
    assert None not in shapes.values(), shapes


def test_a_size_the_header_cannot_tell_is_left_to_the_load(tmp_path):
    from scipy.io import savemat

    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = np.zeros((4, 5, 6))
    savemat(str(tmp_path / "cell.mat"), {"vol": cell})  # a MATLAB cell: its content is only known once read
    assert read_volume_shape(tmp_path / "cell.mat") is None
    assert load_volume(tmp_path / "cell.mat").shape == (6, 5, 4)
    (tmp_path / "notes.xyz").write_text("not a volume")
    assert read_volume_shape(tmp_path / "notes.xyz") is None
    (tmp_path / "broken.tif").write_bytes(b"II*\x00garbage")
    assert read_volume_shape(tmp_path / "broken.tif") is None
    with pytest.raises(FileNotFoundError):
        read_volume_shape(tmp_path / "missing.tif")


def test_a_dicom_folder_is_left_to_the_load(tmp_path):
    folder = tmp_path / "dicom"
    folder.mkdir()
    (folder / "slice_000.dcm").write_bytes(b"\x00" * 128 + b"DICM")
    assert read_volume_shape(folder) is None


@pytest.mark.parametrize(
    "shape",
    [
        (5, 6),
        (4, 5, 6),
        (4, 5, 6, 1),
        (4, 5, 6, 3),
        (4, 5, 6, 4),
        (1, 4, 5, 6),
        (3, 4, 5, 6),
        (2, 4, 5, 6),
        (7,),
        (2, 2, 4, 5, 6),
    ],
)
def test_the_shape_rule_mirrors_the_array_rule(shape):
    try:
        expected = _squeeze_to_3d(np.zeros(shape, np.uint8), "x").shape
    except ValueError:
        expected = None
    assert _squeezed_shape(shape) == expected


def test_shape_text_reads_x_y_z():
    assert shape_text((1857, 1009, 987)) == "987 x 1009 x 1857"
    assert shape_text((1857, 1009, 987), sep=" × ") == "987 × 1009 × 1857"


def test_reading_the_shape_of_a_large_stack_takes_milliseconds(tmp_path):
    """A 1856-slice stack written slice by slice (one IFD per page, the slowest TIFF layout to walk)."""
    p = tmp_path / "tall.tif"
    page = np.zeros((64, 64), np.uint8)
    with tifffile.TiffWriter(str(p)) as tw:
        for _ in range(1856):
            tw.write(page, metadata=None, contiguous=False)
    t0 = time.perf_counter()
    assert read_volume_shape(p) == (1856, 64, 64)
    assert time.perf_counter() - t0 < 2.0


# ------------------------------------------------------------------ the streaming provider refuses a mixed sequence
def _sequence(tmp_path, shapes):
    paths = []
    for k, shape in enumerate(shapes):
        p = tmp_path / f"frame{k}.npy"
        np.save(p, _vol(shape, np.float32, seed=k))
        paths.append(p)
    return paths


def test_a_sequence_of_different_sizes_is_refused_before_any_voxel_is_read(tmp_path, monkeypatch):
    paths = _sequence(tmp_path, [(8, 9, 10), (8, 9, 10), (9, 9, 10), (8, 9, 11)])
    loads = []
    real = volume_io.load_volume
    monkeypatch.setattr(volume_io, "load_volume", lambda *a, **k: loads.append(a[0]) or real(*a, **k))
    with pytest.raises(VolumeShapeError) as err:
        FileVolumeProvider(paths)
    text = str(err.value)
    # every volume that differs is named at once, with its size and the reference's, in x, y, z order
    assert "frame2.npy" in text and "10 x 9 x 9" in text
    assert "frame3.npy" in text and "11 x 9 x 8" in text
    assert "frame0.npy" in text and "10 x 9 x 8" in text and "frame1.npy" not in text
    assert loads == []  # not even the reference was read: the headers were enough
    assert isinstance(err.value, ValueError)  # callers that catch ValueError keep working


def test_mask_files_and_drawn_masks_of_another_size_are_refused_up_front(tmp_path):
    paths = _sequence(tmp_path, [(8, 9, 10), (8, 9, 10)])
    good = tmp_path / "good_mask.npy"
    np.save(good, np.ones((8, 9, 10), np.uint8))
    bad = tmp_path / "bad_mask.npy"
    np.save(bad, np.ones((8, 9, 11), np.uint8))
    FileVolumeProvider(paths, mask_paths=[good, good])
    with pytest.raises(VolumeShapeError, match="bad_mask.npy"):
        FileVolumeProvider(paths, mask_paths=[good, bad])
    with pytest.raises(VolumeShapeError, match="mask of frame 1"):
        FileVolumeProvider(paths, masks=[None, np.ones((8, 9, 11), bool)])
    # a drawn mask replaces the file of its frame: the file's size no longer matters
    FileVolumeProvider(paths, mask_paths=[good, bad], masks=[None, np.ones((8, 9, 10), bool)])


def test_a_reference_the_header_cannot_size_is_read_first_then_checked(tmp_path):
    from scipy.io import savemat

    ref = _vol((4, 5, 6), np.float32)
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = np.transpose(ref, (2, 1, 0))
    savemat(str(tmp_path / "ref.mat"), {"vol": cell})
    np.save(tmp_path / "same.npy", np.transpose(ref, (0, 1, 2)))
    np.save(tmp_path / "other.npy", _vol((4, 5, 7), np.float32))
    FileVolumeProvider([tmp_path / "ref.mat", tmp_path / "same.npy"])
    with pytest.raises(VolumeShapeError, match="other.npy"):
        FileVolumeProvider([tmp_path / "ref.mat", tmp_path / "same.npy", tmp_path / "other.npy"])


def test_a_file_that_changes_after_the_check_is_still_caught_when_read(tmp_path):
    paths = _sequence(tmp_path, [(8, 9, 10), (8, 9, 10)])
    prov = FileVolumeProvider(paths)
    np.save(paths[1], _vol((8, 9, 12), np.float32))
    with pytest.raises(VolumeShapeError, match="12 x 9 x 8"):
        prov.get_normalized(1)


def test_a_missing_file_is_reported_before_the_run_starts(tmp_path, monkeypatch):
    """Deliberately at once, like a missing reference always was: not after the frames before it are solved."""
    paths = _sequence(tmp_path, [(8, 9, 10), (8, 9, 10)])
    loads = []
    real = volume_io.load_volume
    monkeypatch.setattr(volume_io, "load_volume", lambda *a, **k: loads.append(a[0]) or real(*a, **k))
    with pytest.raises(FileNotFoundError, match="gone.npy"):
        FileVolumeProvider([*paths, tmp_path / "gone.npy"])
    with pytest.raises(FileNotFoundError, match="gone_mask.npy"):
        FileVolumeProvider(paths, mask_paths=[None, tmp_path / "gone_mask.npy"])
    assert loads == []


def test_a_slow_header_read_says_why_the_run_waits(tmp_path, monkeypatch, caplog):
    """A cloud drive downloads an unsynced file on the first read; the log says so instead of a silent wait."""
    import logging

    paths = _sequence(tmp_path, [(8, 9, 10), (8, 9, 10)])
    with caplog.at_level(logging.INFO, logger="al_dvc.io.volume_io"):
        FileVolumeProvider(paths)
    assert "downloaded" not in caplog.text
    monkeypatch.setattr(volume_io, "SLOW_HEADER_READ_S", 0.0)
    with caplog.at_level(logging.INFO, logger="al_dvc.io.volume_io"):
        FileVolumeProvider(paths)
    assert "frame1.npy" in caplog.text and "downloaded the whole file" in caplog.text
