"""Volume formats: HDF5, colour slices, folder resolution, DICOM detection and the optional readers."""

import importlib

import numpy as np
import pytest

from al_dvc.io import volume_io
from al_dvc.io.volume_io import VOLUME_EXT, load_volume, resolve_volume_paths, save_volume


@pytest.fixture
def vol():
    rng = np.random.default_rng(0)
    return (rng.random((6, 7, 8)) * 1000).astype(np.uint16)


def test_hdf5_round_trip_and_dataset_key(vol, tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "scan.h5"
    save_volume(path, vol)
    back = load_volume(path)
    assert back.shape == vol.shape and np.array_equal(back, vol)
    with h5py.File(tmp_path / "multi.h5", "w") as f:
        f.create_dataset("meta/table", data=np.zeros((3, 3)))
        f.create_group("images").create_dataset("ct", data=vol)
        f.create_dataset("other", data=vol[::-1])
    assert np.array_equal(load_volume(tmp_path / "multi.h5"), vol)  # the first 3-D dataset, wherever it is
    assert np.array_equal(load_volume(tmp_path / "multi.h5", mat_key="other"), vol[::-1])
    with pytest.raises(KeyError):
        load_volume(tmp_path / "multi.h5", mat_key="nope")
    with h5py.File(tmp_path / "flat.h5", "w") as f:
        f.create_dataset("table", data=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="no 3-D dataset"):
        load_volume(tmp_path / "flat.h5")


def test_colour_slices_become_luminance(tmp_path):
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    folder = tmp_path / "rgb"
    folder.mkdir()
    for k in range(3):
        rgb = np.zeros((5, 6, 3), dtype=np.uint8)
        rgb[..., 1] = 100 + k  # green only
        Image.fromarray(rgb).save(folder / f"slice_{k:02d}.png")
    arr = load_volume(folder)
    assert arr.shape == (3, 5, 6)
    assert abs(float(arr[0, 0, 0]) - 0.587 * 100) < 1e-3  # luminance, not the red channel (which would be 0)
    assert arr[2, 0, 0] > arr[0, 0, 0]
    del PIL


def test_folder_resolution_includes_hdf5_and_nifti(vol, tmp_path):
    pytest.importorskip("h5py")
    save_volume(tmp_path / "a.h5", vol)
    save_volume(tmp_path / "b.npy", vol)
    (tmp_path / "c.nii.gz").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    names = [p.name for p in resolve_volume_paths(tmp_path)]
    assert names == ["a.h5", "b.npy", "c.nii.gz"]
    assert {".h5", ".nii", ".nrrd", ".tif", ".mat", ".npy", ".npz"} <= VOLUME_EXT


def test_dicom_folder_detection(tmp_path):
    folder = tmp_path / "dcm"
    folder.mkdir()
    assert not volume_io._is_dicom_folder(folder)  # empty
    (folder / "IM0001").write_bytes(b"\0" * 128 + b"DICM" + b"\0" * 16)
    assert volume_io._is_dicom_folder(folder)  # magic number without an extension
    png_folder = tmp_path / "png"
    png_folder.mkdir()
    (png_folder / "s.png").write_bytes(b"")
    assert not volume_io._is_dicom_folder(png_folder)


@pytest.mark.parametrize("kind, module", [("nifti", "nibabel"), ("nrrd", "nrrd"), ("dicom", "pydicom")])
def test_optional_reader_message_when_missing(kind, module, tmp_path, monkeypatch):
    """Without the optional package the error names it; with it installed the reader is exercised elsewhere."""
    if importlib.util.find_spec(module) is not None:
        pytest.skip(f"{module} is installed")
    if kind == "nifti":
        path = tmp_path / "x.nii"
        path.write_bytes(b"")
    elif kind == "nrrd":
        path = tmp_path / "x.nrrd"
        path.write_bytes(b"")
    else:
        path = tmp_path / "dcm"
        path.mkdir()
        (path / "a.dcm").write_bytes(b"\0" * 132)
    with pytest.raises(ImportError, match="pip install"):
        load_volume(path)


def test_unsupported_suffix_message(tmp_path):
    (tmp_path / "x.raw").write_bytes(b"\0" * 8)
    with pytest.raises(ValueError, match="Unsupported volume format"):
        load_volume(tmp_path / "x.raw")
