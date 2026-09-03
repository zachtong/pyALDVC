"""Volume file I/O: TIFF stacks, MATLAB ``.mat``, NumPy, slice folders.

All loaders return ``(nz, ny, nx)`` arrays (``vol[z, y, x]``).

MATLAB ALDVC keeps volumes as ``vol(x, y, z)``. ``load_volume`` therefore
transposes ``.mat`` arrays with ``matlab_order=True`` (default for ``.mat``)
so the *same physical voxel* is addressed as ``vol[z, y, x]`` here.

Unicode-safe: every reader goes through ``pathlib`` / ``open`` rather than
C libraries that choke on non-ASCII Windows paths.
"""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.data_structures import VOIRange
from .volume_ops import normalize_volume

_TIFF_EXT = {".tif", ".tiff"}
_SLICE_EXT = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}


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
        for k in ("vol", "Img", "img", "volume", "V", "data"):
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
        path: ``.tif/.tiff`` stack, ``.mat``, ``.npy``, ``.npz`` (first array)
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
    if p.is_dir():
        arr = _read_slices(p, slice_pattern)
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
            files = sorted(q for q in p.iterdir() if q.suffix.lower() in (_TIFF_EXT | {".mat", ".npy", ".npz"}))
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
    else:
        raise ValueError(f"Unsupported output format '{suffix}'")


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
    for both accumulative and incremental tracking).
    """

    def __init__(
        self,
        paths: Iterable[str | os.PathLike],
        voi: VOIRange | None = None,
        mask_paths: Iterable[str | os.PathLike | None] | None = None,
        cache_size: int = 2,
        load_kwargs: dict | None = None,
    ) -> None:
        self._paths = [Path(p) for p in paths]
        if not self._paths:
            raise ValueError("no volume paths given")
        self._mask_paths = [Path(p) if p is not None else None for p in mask_paths] if mask_paths else None
        if self._mask_paths is not None and len(self._mask_paths) != len(self._paths):
            raise ValueError("mask_paths must match the number of volumes")
        self._load_kwargs = dict(load_kwargs or {})
        self._cache: OrderedDict[int, NDArray[np.float32]] = OrderedDict()
        self._mask_cache: OrderedDict[int, NDArray[np.bool_] | None] = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        first = load_volume(self._paths[0], **self._load_kwargs)
        self._shape: tuple[int, int, int] = tuple(int(s) for s in first.shape)  # type: ignore[assignment]
        self._voi = (voi or VOIRange()).clamp(self._shape)
        self._cache[0] = normalize_volume(first, self._voi)

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
        if tuple(raw.shape) != self._shape:
            raise ValueError(f"{self._paths[idx]}: shape {raw.shape} != {self._shape}")
        vol = normalize_volume(raw, self._voi)
        self._cache[idx] = vol
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vol

    def get_mask(self, idx: int) -> NDArray[np.bool_] | None:
        if self._mask_paths is None or self._mask_paths[idx] is None:
            return None
        if idx in self._mask_cache:
            self._mask_cache.move_to_end(idx)
            return self._mask_cache[idx]
        m = load_volume(self._mask_paths[idx], **self._load_kwargs)
        mask = np.asarray(m) > (127 if np.issubdtype(m.dtype, np.integer) and m.max() > 1 else 0)
        if mask.shape != self._shape:
            raise ValueError(f"mask {self._mask_paths[idx]}: shape {mask.shape} != {self._shape}")
        self._mask_cache[idx] = mask
        while len(self._mask_cache) > self._cache_size:
            self._mask_cache.popitem(last=False)
        return mask
