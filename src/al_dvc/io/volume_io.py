"""Volume file I/O: TIFF stacks, MATLAB ``.mat``, NumPy, HDF5, slice folders, and (optional readers)
NIfTI, NRRD and DICOM series.

All loaders return ``(nz, ny, nx)`` arrays (``vol[z, y, x]``).

MATLAB ALDVC keeps volumes as ``vol(x, y, z)``. ``load_volume`` therefore
transposes ``.mat`` arrays with ``matlab_order=True`` (default for ``.mat``)
so the *same physical voxel* is addressed as ``vol[z, y, x]`` here.

Unicode-safe: every reader goes through ``pathlib`` / ``open`` rather than
C libraries that choke on non-ASCII Windows paths.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import VOIRange
from .volume_ops import normalize_volume

_TIFF_EXT = {".tif", ".tiff"}
_SLICE_EXT = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}
_HDF5_EXT = {".h5", ".hdf5", ".hdf"}
_NIFTI_EXT = {".nii", ".nii.gz"}
_NRRD_EXT = {".nrrd", ".nhdr"}
_DICOM_EXT = {".dcm", ".dicom", ""}
VOLUME_EXT = _TIFF_EXT | {".mat", ".npy", ".npz"} | _HDF5_EXT | _NIFTI_EXT | _NRRD_EXT
OPTIONAL_READERS = {"nifti": "nibabel", "nrrd": "pynrrd", "dicom": "pydicom"}
_MAT_NAMES = ("vol", "Img", "img", "volume", "V", "data")  # the variables a .mat is searched for, in this order
_MAT_NUMERIC = {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "logical"}
# A header read slower than this was a download (cloud drive) or a slow share: worth saying why the wait
SLOW_HEADER_READ_S = 2.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _read_tiff(path: Path) -> NDArray:
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        arr = tf.asarray()
    return _squeeze_to_3d(arr, path)


def _squeeze_to_3d(arr: NDArray, path: Path) -> NDArray:
    arr = np.asarray(arr)
    if arr.ndim == 4:
        # (Z, Y, X, C) or (C, Z, Y, X): drop the channel axis
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[..., 0]
        elif arr.shape[0] in (1, 3, 4):
            arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D volume, got shape {arr.shape}")
    return arr


def _read_mat(path: Path, key: str | None) -> NDArray:
    """Read the first 3-D array (or ``key``) from a v5/v7 or v7.3 ``.mat``."""
    try:
        from scipy.io import loadmat

        data = loadmat(str(path))
        candidates = {k: v for k, v in data.items() if not k.startswith("__")}
        arr = _pick_mat_array(candidates, key, path)
    except NotImplementedError:
        import h5py

        with h5py.File(str(path), "r") as h5:
            candidates = {}
            for k in h5.keys():
                obj = h5[k]
                if isinstance(obj, h5py.Dataset):
                    candidates[k] = obj[()]
                elif isinstance(obj, h5py.Group):
                    for k2 in obj.keys():
                        if isinstance(obj[k2], h5py.Dataset):
                            candidates[f"{k}/{k2}"] = obj[k2][()]
            arr = _pick_mat_array(candidates, key, path)
            # HDF5 stores MATLAB arrays transposed (C order of the MATLAB
            # column-major layout): a MATLAB (x, y, z) array reads as (z, y, x).
            # Bring it back to MATLAB (x, y, z) so the caller's transpose
            # below applies uniformly.
            arr = np.transpose(arr, (2, 1, 0))
    return arr


def _pick_mat_array(candidates: dict, key: str | None, path: Path) -> NDArray:
    if key is not None:
        if key not in candidates:
            raise KeyError(f"{path}: variable '{key}' not found (available: {sorted(candidates)})")
        arr = candidates[key]
    else:
        arr = None
        for k in _MAT_NAMES:
            if k in candidates:
                arr = candidates[k]
                break
        if arr is None:
            three_d = [v for v in candidates.values() if isinstance(v, np.ndarray) and v.ndim == 3]
            if not three_d:
                raise ValueError(f"{path}: no 3-D array found (variables: {sorted(candidates)})")
            arr = three_d[0]
    # MATLAB cell arrays (vol{1}) come back as object arrays
    while isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr.ravel()[0]
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D array, got shape {arr.shape}")
    return arr


def _read_hdf5(path: Path, key: str | None) -> NDArray:
    """First 3-D dataset of an HDF5 file (or the dataset at ``key``, a path inside the file)."""
    import h5py

    with h5py.File(str(path), "r") as f:
        if key is not None:
            if key not in f:
                raise KeyError(f"{path}: no dataset '{key}'")
            return np.asarray(f[key])
        found: list[str] = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and not found:
                found.append(name)

        f.visititems(visit)
        if not found:
            raise ValueError(f"{path}: no 3-D dataset found (pass mat_key=<dataset path>)")
        return np.asarray(f[found[0]])


def _optional(module: str, kind: str):
    try:
        return __import__(module)
    except ImportError as exc:
        pkg = OPTIONAL_READERS[kind]
        raise ImportError(f"reading {kind} volumes needs the optional package '{pkg}': pip install {pkg}") from exc


def _read_nifti(path: Path) -> NDArray:
    """NIfTI (``.nii`` / ``.nii.gz``): stored (x, y, z) -> ``(z, y, x)``."""
    nib = _optional("nibabel", "nifti")
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    return np.transpose(_squeeze_to_3d(arr, path), (2, 1, 0))


def _read_nrrd(path: Path) -> NDArray:
    """NRRD: pynrrd returns Fortran-ordered (x, y, z) -> ``(z, y, x)``."""
    nrrd = _optional("nrrd", "nrrd")
    arr, _header = nrrd.read(str(path))
    return np.transpose(_squeeze_to_3d(np.asarray(arr), path), (2, 1, 0))


def _read_dicom_folder(folder: Path) -> NDArray:
    """A folder of single-slice DICOM files, stacked by InstanceNumber (else by slice position, else by name)."""
    pydicom = _optional("pydicom", "dicom")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _DICOM_EXT)
    datasets = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), force=True)
        except Exception:
            continue
        if hasattr(ds, "pixel_array"):
            datasets.append((p, ds))
    if not datasets:
        raise FileNotFoundError(f"No DICOM slices in {folder}")

    def order(item):
        p, ds = item
        num = getattr(ds, "InstanceNumber", None)
        pos = getattr(ds, "ImagePositionPatient", None)
        return (0, float(num), "") if num is not None else ((1, float(pos[2]), "") if pos is not None else (2, 0.0, p.name))

    datasets.sort(key=order)
    slices = [np.asarray(ds.pixel_array) for _p, ds in datasets]
    shapes = {s.shape for s in slices}
    if len(shapes) != 1:
        raise ValueError(f"{folder}: DICOM slices have inconsistent shapes {sorted(shapes)}")
    return np.stack(slices, axis=0)


def _is_dicom_folder(folder: Path) -> bool:
    files = [p for p in folder.iterdir() if p.is_file()]
    if not files:
        return False
    if any(p.suffix.lower() in _SLICE_EXT for p in files):
        return False
    if any(p.suffix.lower() in {".dcm", ".dicom"} for p in files):
        return True
    try:
        with open(files[0], "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def _read_slices(folder: Path, pattern: str | None) -> NDArray:
    pat = pattern or "*"
    files = sorted(p for p in folder.glob(pat) if p.suffix.lower() in _SLICE_EXT)
    if not files:
        raise FileNotFoundError(f"No slice images matching '{pat}' in {folder}")
    slices = [_read_slice(p) for p in files]
    shapes = {s.shape for s in slices}
    if len(shapes) != 1:
        raise ValueError(f"{folder}: slices have inconsistent shapes {sorted(shapes)}")
    return np.stack(slices, axis=0)


def _read_slice(path: Path) -> NDArray:
    if path.suffix.lower() in _TIFF_EXT:
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        from PIL import Image

        with Image.open(path) as im:
            arr = np.asarray(im)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):  # colour slice: ITU-R 601 luminance, alpha ignored
            rgb = arr[..., :3].astype(np.float32)
            arr = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D slice, got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_volume(
    path: str | os.PathLike,
    *,
    mat_key: str | None = None,
    matlab_order: bool | None = None,
    slice_pattern: str | None = None,
    dtype: np.dtype | None = None,
) -> NDArray:
    """Load one volume as a ``(nz, ny, nx)`` array.

    Args:
        path: ``.tif/.tiff`` stack, ``.mat``, ``.npy``, ``.npz`` (first array),
            ``.h5/.hdf5`` (first 3-D dataset or ``mat_key``), ``.nii/.nii.gz``
            (nibabel), ``.nrrd`` (pynrrd), a folder of 2-D slices (TIFF, PNG,
            BMP, JPEG; colour slices become luminance) or a folder of DICOM
            files (pydicom, stacked by InstanceNumber)
            or a folder of 2-D slices.
        mat_key: variable name inside a ``.mat`` file (default: ``vol`` or the
            first 3-D array).
        matlab_order: if True, the file stores ``(x, y, z)`` and is transposed
            to ``(z, y, x)``. Defaults to True for ``.mat``, False otherwise.
        slice_pattern: glob for slice folders (default ``*``).
        dtype: optional cast (e.g. ``np.float32``) applied at the end.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Volume not found: {p}")
    suffix = p.suffix.lower()
    name = p.name.lower()
    if p.is_dir():
        arr = _read_dicom_folder(p) if _is_dicom_folder(p) else _read_slices(p, slice_pattern)
        if matlab_order is None:
            matlab_order = False
    elif suffix in _HDF5_EXT:
        arr = _read_hdf5(p, mat_key)
        if matlab_order is None:
            matlab_order = False
    elif name.endswith(".nii.gz") or suffix == ".nii":
        arr = _read_nifti(p)
        if matlab_order is None:
            matlab_order = False
    elif suffix in _NRRD_EXT:
        arr = _read_nrrd(p)
        if matlab_order is None:
            matlab_order = False
    elif suffix in _TIFF_EXT:
        arr = _read_tiff(p)
        if matlab_order is None:
            matlab_order = False
    elif suffix == ".mat":
        arr = _read_mat(p, mat_key)
        if matlab_order is None:
            matlab_order = True
    elif suffix == ".npy":
        arr = np.load(str(p), mmap_mode=None)
        if matlab_order is None:
            matlab_order = False
    elif suffix == ".npz":
        with np.load(str(p)) as z:
            key = mat_key or (z.files[0] if z.files else None)
            if key is None:
                raise ValueError(f"{p}: empty .npz archive")
            arr = z[key]
        if matlab_order is None:
            matlab_order = False
    else:
        raise ValueError(f"Unsupported volume format '{suffix}' ({p}).")

    arr = _squeeze_to_3d(np.asarray(arr), p)
    if matlab_order:
        arr = np.transpose(arr, (2, 1, 0))
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def resolve_volume_paths(spec: str | os.PathLike | Sequence[str | os.PathLike]) -> list[Path]:
    """Expand a glob / directory / list into a sorted list of volume paths."""
    if isinstance(spec, (str, os.PathLike)):
        s = str(spec)
        p = Path(s)
        if p.is_dir():
            files = sorted(q for q in p.iterdir() if q.suffix.lower() in VOLUME_EXT or q.name.lower().endswith(".nii.gz"))
            if not files:
                # a directory of directories (one slice folder per frame)?
                subdirs = sorted(q for q in p.iterdir() if q.is_dir())
                if subdirs:
                    return subdirs
                raise FileNotFoundError(f"No volumes found in {p}")
            return files
        if any(ch in s for ch in "*?["):
            files = sorted(Path(f) for f in glob.glob(s))
            if not files:
                raise FileNotFoundError(f"No files match '{s}'")
            return files
        return [p]
    return [Path(x) for x in spec]


def load_volumes(
    spec: str | os.PathLike | Sequence[str | os.PathLike],
    **kwargs,
) -> list[NDArray]:
    """Load every volume matched by ``spec`` (glob, folder or list), sorted."""
    return [load_volume(p, **kwargs) for p in resolve_volume_paths(spec)]


def save_volume(path: str | os.PathLike, vol: NDArray, *, matlab_order: bool = False) -> None:
    """Write a ``(nz, ny, nx)`` volume as TIFF stack, ``.npy`` or ``.mat``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(vol)
    if matlab_order:
        arr = np.transpose(arr, (2, 1, 0))
    suffix = p.suffix.lower()
    if suffix in _TIFF_EXT:
        import tifffile

        tifffile.imwrite(str(p), np.ascontiguousarray(arr))
    elif suffix == ".npy":
        np.save(str(p), arr)
    elif suffix == ".mat":
        from scipy.io import savemat

        savemat(str(p), {"vol": arr}, do_compression=True)
    elif suffix in _HDF5_EXT:
        import h5py

        with h5py.File(str(p), "w") as f:
            f.create_dataset("volume", data=np.ascontiguousarray(arr), compression="gzip")
    else:
        raise ValueError(f"Unsupported output format '{suffix}'")


# ---------------------------------------------------------------------------
# Size from the header
# ---------------------------------------------------------------------------


class VolumeShapeError(ValueError):
    """The volumes (or masks) of a sequence do not all have the reference volume's shape."""


def shape_text(shape: Sequence[int], sep: str = " x ") -> str:
    """A ``(nz, ny, nx)`` shape the way a volume's size is read: ``nx x ny x nz`` (x, y, z, as ImageJ shows it)."""
    nz, ny, nx = (int(s) for s in shape)
    return f"{nx}{sep}{ny}{sep}{nz}"


def _mismatch_message(reference: Path, ref_shape: Sequence[int], problems: list[str]) -> str:
    return (
        f"volumes of different sizes: every volume and mask of a sequence must have the size of the reference "
        f"{reference.name}, {shape_text(ref_shape)} voxels (x, y, z); these differ:\n  " + "\n  ".join(problems)
    )


def _squeezed_shape(shape: Sequence[int]) -> tuple[int, int, int] | None:
    """The shape :func:`_squeeze_to_3d` gives an array of ``shape``; ``None`` where it refuses the array."""
    s = tuple(int(v) for v in shape)
    if len(s) == 4:
        if s[-1] in (1, 3, 4):
            s = s[:-1]
        elif s[0] in (1, 3, 4):
            s = s[1:]
    if len(s) == 2:
        s = (1, *s)
    return s if len(s) == 3 else None  # type: ignore[return-value]


def _npy_header_shape(fh) -> tuple[int, ...] | None:
    """The shape in a ``.npy`` header, ``fh`` at its magic string; ``None`` for a format version without a public reader."""
    from numpy.lib import format as npy_format

    version = npy_format.read_magic(fh)
    readers = {(1, 0): npy_format.read_array_header_1_0, (2, 0): npy_format.read_array_header_2_0}
    if version not in readers:
        return None
    shape, _fortran, _dtype = readers[version](fh)
    return tuple(int(s) for s in shape)


def _npz_header_shape(path: Path, key: str | None) -> tuple[int, ...] | None:
    """The member ``load_volume`` reads (``key``, else the first), resolved as ``np.load``'s archive does."""
    import zipfile

    with zipfile.ZipFile(str(path)) as zf:
        names = zf.namelist()
        files = [n[:-4] if n.endswith(".npy") else n for n in names]
        if not files:
            return None
        key = key or files[0]
        member = key if key in names else (key + ".npy" if key in files else None)
        if member is None:
            return None
        with zf.open(member) as fh:
            return _npy_header_shape(fh)


def _hdf5_header_shape(path: Path, key: str | None) -> tuple[int, ...] | None:
    import h5py

    with h5py.File(str(path), "r") as f:
        if key is not None:
            obj = f.get(key)
            return tuple(int(s) for s in obj.shape) if isinstance(obj, h5py.Dataset) else None
        found: list[tuple[int, ...]] = []

        def visit(_name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and not found:
                found.append(tuple(int(s) for s in obj.shape))

        f.visititems(visit)  # the same walk as _read_hdf5: the first 3-D dataset
        return found[0] if found else None


def _pick_mat_name(ndims: dict[str, int], key: str | None) -> str | None:
    """The variable :func:`_pick_mat_array` picks, from the variables' dimension counts."""
    if key is not None:
        return key if key in ndims else None
    for name in _MAT_NAMES:
        if name in ndims:
            return name
    return next((name for name, nd in ndims.items() if nd == 3), None)


def _mat_header_shape(path: Path, key: str | None) -> tuple[int, ...] | None:
    """Shape of the array :func:`_read_mat` returns (MATLAB ``(x, y, z)`` order); ``None`` for a cell, a struct
    or anything else whose content is only known once read."""
    try:
        from scipy.io import whosmat

        listing = whosmat(str(path))
    except NotImplementedError:  # v7.3 is HDF5
        return _mat73_header_shape(path, key)
    variables = {name: (tuple(int(s) for s in shape), mclass) for name, shape, mclass in listing}
    name = _pick_mat_name({k: len(v[0]) for k, v in variables.items()}, key)
    if name is None:
        return None
    shape, mclass = variables[name]
    return shape if mclass in _MAT_NUMERIC and len(shape) == 3 else None


def _mat73_header_shape(path: Path, key: str | None) -> tuple[int, ...] | None:
    import h5py

    with h5py.File(str(path), "r") as h5:
        datasets = {}
        for k in h5.keys():  # the same variables, in the same order, as _read_mat reads them
            obj = h5[k]
            if isinstance(obj, h5py.Dataset):
                datasets[k] = obj
            elif isinstance(obj, h5py.Group):
                for k2 in obj.keys():
                    if isinstance(obj[k2], h5py.Dataset):
                        datasets[f"{k}/{k2}"] = obj[k2]
        name = _pick_mat_name({k: d.ndim for k, d in datasets.items()}, key)
        if name is None:
            return None
        ds = datasets[name]
        if ds.ndim != 3 or ds.dtype.kind not in "biuf":  # references (cells), structs, strings
            return None
        return tuple(int(s) for s in ds.shape[::-1])  # _read_mat transposes the (z, y, x) dataset to MATLAB's (x, y, z)


def _tiff_header_shape(path: Path) -> tuple[int, ...] | None:
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:  # the first series is what TiffFile.asarray() returns
        return tuple(int(s) for s in tf.series[0].shape) if tf.series else None


def _slices_header_shape(folder: Path, pattern: str | None) -> tuple[int, ...] | None:
    """Slice count by the first slice's size: :func:`_read_slices` insists that every slice has that size."""
    files = sorted(p for p in folder.glob(pattern or "*") if p.suffix.lower() in _SLICE_EXT)
    if not files:
        return None
    first = files[0]
    if first.suffix.lower() in _TIFF_EXT:
        import tifffile

        with tifffile.TiffFile(str(first)) as tf:
            s = tuple(int(v) for v in tf.series[0].shape) if tf.series else ()
    else:
        from PIL import Image

        with Image.open(first) as im:
            width, height = im.size
            s = (height, width)
    if len(s) == 3:
        s = s[:2]  # colour becomes luminance, another third axis loses all but its first plane (_read_slice)
    return (len(files), *s) if len(s) == 2 else None


def _header_shape(p: Path, key: str | None, pattern: str | None) -> tuple[tuple[int, ...] | None, bool]:
    """``(shape the reader returns, default matlab_order)``, dispatched exactly as :func:`load_volume` does."""
    suffix = p.suffix.lower()
    name = p.name.lower()
    if p.is_dir():
        return (None if _is_dicom_folder(p) else _slices_header_shape(p, pattern)), False
    if suffix in _HDF5_EXT:
        return _hdf5_header_shape(p, key), False
    if name.endswith(".nii.gz") or suffix == ".nii":
        img = _optional("nibabel", "nifti").load(str(p))
        s = _squeezed_shape(img.shape)
        return (None if s is None else s[::-1]), False  # _read_nifti: (x, y, z) -> (z, y, x)
    if suffix in _NRRD_EXT:
        header = _optional("nrrd", "nrrd").read_header(str(p))
        s = _squeezed_shape(tuple(int(v) for v in header["sizes"]))
        return (None if s is None else s[::-1]), False  # _read_nrrd: Fortran (x, y, z) -> (z, y, x)
    if suffix in _TIFF_EXT:
        return _tiff_header_shape(p), False
    if suffix == ".mat":
        return _mat_header_shape(p, key), True
    if suffix == ".npy":
        with open(p, "rb") as fh:
            return _npy_header_shape(fh), False
    if suffix == ".npz":
        return _npz_header_shape(p, key), False
    return None, False


def read_volume_shape(
    path: str | os.PathLike,
    *,
    mat_key: str | None = None,
    matlab_order: bool | None = None,
    slice_pattern: str | None = None,
    dtype: np.dtype | None = None,
) -> tuple[int, int, int] | None:
    """The ``(nz, ny, nx)`` shape :func:`load_volume` returns for ``path``, read from the file's header alone.

    Takes :func:`load_volume`'s options (``dtype`` does not change a shape and is ignored). No voxel is read,
    so a 4-GB scan is sized in milliseconds -- unless it sits on a cloud drive (Box Drive, OneDrive) that has
    not synced it yet: those download the whole file on the first read of any byte, 26-45 s for a 3.7-GB scan
    on Box Drive. Callers that must stay responsive read sizes off their main thread.

    Returns ``None`` when the header cannot tell: a DICOM folder, a MATLAB cell or struct, a missing optional
    reader, a header this reader does not understand. :func:`load_volume` then decides and reports.
    Raises ``FileNotFoundError`` when ``path`` does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Volume not found: {p}")
    try:
        raw, default_order = _header_shape(p, mat_key, slice_pattern)
    except Exception as exc:  # an unreadable or unusual header: the load decides, and says why
        logger.debug("no size from the header of %s: %s", p, exc)
        return None
    shape = None if raw is None else _squeezed_shape(raw)
    if shape is None:
        return None
    if matlab_order if matlab_order is not None else default_order:
        shape = shape[::-1]
    return shape  # type: ignore[return-value]


def volume_info(path: str | os.PathLike) -> dict:
    """Shape/dtype/intensity summary of a volume file (loads it)."""
    arr = load_volume(path)
    return {
        "path": str(path),
        "shape_zyx": tuple(int(s) for s in arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr, dtype=np.float64)),
        "std": float(np.std(arr, dtype=np.float64)),
        "nbytes": int(arr.nbytes),
    }


# ---------------------------------------------------------------------------
# Lazy (streaming) provider
# ---------------------------------------------------------------------------


class FileVolumeProvider:
    """Streams volumes from disk, normalising on demand with a bounded cache.

    Only ``cache_size`` normalised frames are resident at once (2 is enough
    for both accumulative and incremental tracking), and the raw arrays are
    never kept at all -- which is the whole point next to handing the pipeline a
    list of frames that something else is holding.

    ``mask_paths`` streams the masks the same way; ``masks`` passes arrays for
    the frames whose mask was drawn rather than read from a file, and wins over
    ``mask_paths`` where both are given.

    Every volume and mask must have the reference's shape. The sizes are read from
    the file headers when the provider is made (:func:`read_volume_shape`), so a
    sequence of different sizes is refused with :class:`VolumeShapeError` before
    anything is solved; it used to fail when the run reached the odd frame. A
    missing file raises ``FileNotFoundError`` at once for the same reason -- as a
    missing reference always did. On a cloud drive that has not synced the files,
    reading the headers downloads them before the first frame starts (the run
    reads them all anyway); a slow read is logged with the reason.
    """

    def __init__(
        self,
        paths: Iterable[str | os.PathLike],
        voi: VOIRange | None = None,
        mask_paths: Iterable[str | os.PathLike | None] | None = None,
        cache_size: int = 2,
        load_kwargs: dict | None = None,
        masks: list | None = None,
    ) -> None:
        self._paths = [Path(p) for p in paths]
        if not self._paths:
            raise ValueError("no volume paths given")
        self._mask_paths = [Path(p) if p is not None else None for p in mask_paths] if mask_paths else None
        if self._mask_paths is not None and len(self._mask_paths) != len(self._paths):
            raise ValueError("mask_paths must match the number of volumes")
        # masks drawn in the application have no file; they are held as arrays and take precedence
        self._masks = list(masks) if masks is not None else None
        if self._masks is not None and len(self._masks) != len(self._paths):
            raise ValueError("masks must match the number of volumes")
        self._load_kwargs = dict(load_kwargs or {})
        self._cache: OrderedDict[int, NDArray[np.float32]] = OrderedDict()
        self._mask_cache: OrderedDict[int, NDArray[np.bool_] | None] = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        # the headers first: a mismatch is refused before even the reference is read
        ref_header = read_volume_shape(self._paths[0], **self._load_kwargs)
        if ref_header is not None:
            self._check_shapes(ref_header)
        first = load_volume(self._paths[0], **self._load_kwargs)
        self._shape: tuple[int, int, int] = tuple(int(s) for s in first.shape)  # type: ignore[assignment]
        if ref_header is None:
            self._check_shapes(self._shape)
        elif ref_header != self._shape:  # the header misled; the others are checked when they are read
            logger.warning(
                "%s: the header says %s, the volume is %s; sizes are checked as the frames are read",
                self._paths[0].name,
                shape_text(ref_header),
                shape_text(self._shape),
            )
        self._voi = (voi or VOIRange()).clamp(self._shape)
        self._cache[0] = normalize_volume(first, self._voi)

    def _check_shapes(self, ref_shape: tuple[int, int, int]) -> None:
        """Raise :class:`VolumeShapeError` naming every volume and mask whose size is not ``ref_shape``.

        File sizes come from the headers; a file whose header cannot tell is checked when it is read.
        A drawn mask replaces the mask file of its frame, so it is the one checked there.
        """
        problems: list[str] = []
        for p in self._paths[1:]:
            shape = self._header_shape(p)
            if shape is not None and shape != ref_shape:
                problems.append(f"{p.name}: {shape_text(shape)}")
        read: dict[Path, tuple[int, int, int] | None] = {}
        for i in range(len(self._paths)):
            drawn = self._masks[i] if self._masks is not None else None
            if drawn is not None:
                shape = tuple(int(s) for s in np.shape(drawn))
                if shape != ref_shape:
                    problems.append(f"mask of frame {i}: {shape_text(shape)}")
                continue
            path = self._mask_paths[i] if self._mask_paths is not None else None
            if path is None or path in read:
                continue
            read[path] = shape = self._header_shape(path)
            if shape is not None and shape != ref_shape:
                problems.append(f"mask {path.name}: {shape_text(shape)}")
        if problems:
            raise VolumeShapeError(_mismatch_message(self._paths[0], ref_shape, problems))

    def _header_shape(self, path: Path) -> tuple[int, int, int] | None:
        """:func:`read_volume_shape` with the provider's options; a slow read says why the run has not started."""
        t0 = time.perf_counter()
        shape = read_volume_shape(path, **self._load_kwargs)
        elapsed = time.perf_counter() - t0
        if elapsed >= SLOW_HEADER_READ_S:
            logger.info(
                "%s: its size took %.0f s to read; a cloud or network drive probably downloaded the whole file",
                path.name,
                elapsed,
            )
        return shape

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def clamped_voi(self) -> VOIRange:
        return self._voi

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    def get_normalized(self, idx: int) -> NDArray[np.float32]:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        raw = load_volume(self._paths[idx], **self._load_kwargs)
        if tuple(raw.shape) != self._shape:  # a file the header could not size, or one changed since
            raise VolumeShapeError(
                _mismatch_message(self._paths[0], self._shape, [f"{self._paths[idx].name}: {shape_text(raw.shape)}"])
            )
        vol = normalize_volume(raw, self._voi)
        self._cache[idx] = vol
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vol

    @property
    def has_masks(self) -> bool:
        """True when at least one frame has a mask, from paths or arrays, without reading any of them."""
        if self._masks is not None and any(m is not None for m in self._masks):
            return True
        return self._mask_paths is not None and any(p is not None for p in self._mask_paths)

    def get_mask(self, idx: int) -> NDArray[np.bool_] | None:
        if self._masks is not None and self._masks[idx] is not None:
            mask = np.asarray(self._masks[idx], dtype=bool)
            if mask.shape != self._shape:
                raise VolumeShapeError(
                    _mismatch_message(self._paths[0], self._shape, [f"mask of frame {idx}: {shape_text(mask.shape)}"])
                )
            return mask
        if self._mask_paths is None or self._mask_paths[idx] is None:
            return None
        if idx in self._mask_cache:
            self._mask_cache.move_to_end(idx)
            return self._mask_cache[idx]
        m = load_volume(self._mask_paths[idx], **self._load_kwargs)
        mask = np.asarray(m) > (127 if np.issubdtype(m.dtype, np.integer) and m.max() > 1 else 0)
        if mask.shape != self._shape:
            raise VolumeShapeError(
                _mismatch_message(self._paths[0], self._shape, [f"mask {self._mask_paths[idx].name}: {shape_text(mask.shape)}"])
            )
        self._mask_cache[idx] = mask
        while len(self._mask_cache) > self._cache_size:
            self._mask_cache.popitem(last=False)
        return mask
