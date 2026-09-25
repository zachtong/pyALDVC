"""Faithful (de)serialisation of result objects to a NumPy ``.npz`` archive (ported from pyALDIC).

A generic recursive encoder walks a dataclass tree (:class:`~al_dvc.core.data_structures.PipelineResult`
and every record nested in it, or a texture analysis result), routing arrays into the archive and
everything else into a compact JSON manifest stored in the archive under ``__manifest__``.

* Arrays keep their dtype, shape and memory order bit for bit, and are **deduplicated by content**:
  the mesh shared by every frame, or ``U_accum`` when it is ``U``, is stored once.
* NumPy scalars keep their dtype, Python floats their exact value (``repr`` round-trips), non-finite
  floats are spelled out so the manifest stays strict JSON, dictionaries keep non-string keys (the
  texture results key on thresholds), tuples stay tuples and slices stay slices.
* Only the classes of :func:`registry` can be rebuilt, and only from their own field names.

Security: archives are opened with ``allow_pickle=False`` (data only) and the manifest is plain
JSON, so reading an untrusted file never executes code; anything unexpected raises
:class:`SerializeError`.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
import zlib
from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np
from numpy.lib.format import write_array

SERIALIZE_VERSION = 1
MANIFEST_KEY = "__manifest__"
MAX_DEPTH = 64  # nesting a real result never reaches; a hostile manifest cannot recurse without bound
# Members are compressed only when it pays: integer and boolean arrays (status codes, flags, packed masks) always
# (level 6: they are small and shrink many times), floating-point ones when a sample of them shrinks by
# FLOAT_GAIN (NaN-heavy fields); measured-noise floats barely compress (4 % at level 1 for 3.4 x the time), so
# they are stored as they are.
INT_LEVEL = 6
FLOAT_LEVEL = 1
FLOAT_GAIN = 0.85
SAMPLE_BYTES = 1 << 20
_SCALAR_KINDS = "biuf"  # numpy scalars the manifest may name: bool, signed and unsigned integers, floats
_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}

ProgressFn = Callable[[float], None]


class SerializeError(ValueError):
    """An archive or manifest that this build cannot (or must not) read."""


@lru_cache(maxsize=1)
def registry() -> dict[str, type]:
    """Dataclasses that may be rebuilt from an archive, by class name (imported lazily: no import cycles)."""
    from al_dvc.core.config import DVCPara
    from al_dvc.core.data_structures import (
        ADMMInfo,
        DVCMesh,
        FrameResult,
        FrameSchedule,
        LocalSolveInfo,
        PipelineResult,
        StrainResult,
        VOIRange,
    )
    from al_dvc.texture.acf import Autocorrelation
    from al_dvc.texture.analysis import TextureResult
    from al_dvc.texture.crossing import Crossing
    from al_dvc.texture.profiles import Profile
    from al_dvc.texture.rve import PlateauDecision, SizeLevel, SizeSweep, SubVolume

    classes = (
        PipelineResult,
        DVCPara,
        DVCMesh,
        FrameResult,
        StrainResult,
        FrameSchedule,
        VOIRange,
        ADMMInfo,
        LocalSolveInfo,
        TextureResult,
        Autocorrelation,
        Profile,
        Crossing,
        SizeSweep,
        SizeLevel,
        SubVolume,
        PlateauDecision,
    )
    return {c.__name__: c for c in classes}


# ---------------------------------------------------------------------------- encoding
class _Encoder:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self._by_hash: dict[bytes, str] = {}
        self._by_id: dict[int, str] = {}
        self._alive: list = []  # keeps every array seen alive, so an ``id`` is never reused during the walk

    def array(self, arr: np.ndarray) -> str:
        key = self._by_id.get(id(arr))
        if key is not None:
            return key
        if arr.dtype.hasobject:
            raise TypeError("arrays of Python objects cannot be stored in a session")
        h = hashlib.sha256()
        h.update(arr.dtype.str.encode("ascii"))
        h.update(repr(arr.shape).encode("ascii"))
        h.update(b"F" if (arr.flags.f_contiguous and not arr.flags.c_contiguous) else b"C")
        h.update(np.ascontiguousarray(arr).data if arr.size else b"")
        digest = h.digest()
        key = self._by_hash.get(digest)
        if key is None:
            key = f"a{len(self.arrays):05d}"
            self.arrays[key] = arr
            self._by_hash[digest] = key
        self._by_id[id(arr)] = key
        self._alive.append(arr)
        return key

    def encode(self, obj: Any, depth: int = 0) -> Any:
        if depth > MAX_DEPTH:
            raise TypeError("object nested too deeply for a session")
        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": self.array(obj)}
        if isinstance(obj, np.generic):
            return _encode_scalar(obj)
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, float):
            return float(obj) if math.isfinite(obj) else {"__float__": _nonfinite_name(obj)}
        if isinstance(obj, tuple):
            return {"__tuple__": [self.encode(v, depth + 1) for v in obj]}
        if isinstance(obj, list):
            return [self.encode(v, depth + 1) for v in obj]
        if isinstance(obj, slice):
            return {"__slice__": [self.encode(v, depth + 1) for v in (obj.start, obj.stop, obj.step)]}
        if isinstance(obj, dict):
            if all(isinstance(k, str) for k in obj):
                return {"__dict__": {k: self.encode(v, depth + 1) for k, v in obj.items()}}
            return {"__items__": [[self.encode(k, depth + 1), self.encode(v, depth + 1)] for k, v in obj.items()]}
        if is_dataclass(obj) and not isinstance(obj, type):
            name = type(obj).__name__
            if registry().get(name) is not type(obj):
                raise TypeError(f"{name} is not a record a session can hold")
            return {
                "__dataclass__": name,
                "fields": {f.name: self.encode(getattr(obj, f.name), depth + 1) for f in fields(obj) if f.init},
            }
        raise TypeError(f"cannot store an object of type {type(obj).__name__} in a session")


def _nonfinite_name(value: float) -> str:
    return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")


def _encode_scalar(obj: np.generic) -> dict:
    dt = obj.dtype
    if dt.kind not in _SCALAR_KINDS:
        raise TypeError(f"cannot store a NumPy scalar of type {dt} in a session")
    value = obj.item()
    if isinstance(value, float) and not math.isfinite(value):
        value = _nonfinite_name(value)
    return {"__npscalar__": dt.str, "value": value}


def encode(obj: Any) -> tuple[dict[str, np.ndarray], Any]:
    """``(arrays, root)``: the deduplicated arrays by key and the JSON-able description of ``obj``."""
    enc = _Encoder()
    root = enc.encode(obj)
    return enc.arrays, root


# ---------------------------------------------------------------------------- decoding
def _decode_scalar(val: dict) -> np.generic:
    try:
        dt = np.dtype(str(val["__npscalar__"]))
    except (TypeError, ValueError, KeyError) as exc:
        raise SerializeError(f"invalid NumPy scalar in the manifest: {exc}") from exc
    if dt.kind not in _SCALAR_KINDS or set(val) != {"__npscalar__", "value"}:
        raise SerializeError(f"a NumPy scalar of type {dt} is not allowed in a session")
    raw = val["value"]
    if isinstance(raw, str):
        if dt.kind != "f" or raw not in _NONFINITE:
            raise SerializeError(f"invalid NumPy scalar value {raw!r}")
        raw = _NONFINITE[raw]
    if not isinstance(raw, (bool, int, float)):
        raise SerializeError("invalid NumPy scalar value")
    try:
        return np.array(raw, dtype=dt)[()]
    except (OverflowError, ValueError) as exc:
        raise SerializeError(f"invalid NumPy scalar value: {exc}") from exc


def _only(val: dict, key: str, extra: tuple[str, ...] = ()) -> None:
    if set(val) != {key, *extra}:
        raise SerializeError(f"unexpected keys {sorted(val)} in the manifest")


def decode(val: Any, arrays: dict[str, np.ndarray], depth: int = 0) -> Any:
    """Inverse of :func:`encode`; :class:`SerializeError` on anything it did not write."""
    if depth > MAX_DEPTH:
        raise SerializeError("the manifest is nested too deeply")
    if val is None or isinstance(val, (bool, int, str)):
        return val
    if isinstance(val, float):
        if not math.isfinite(val):
            raise SerializeError("non-finite number in the manifest")
        return val
    if isinstance(val, list):
        return [decode(v, arrays, depth + 1) for v in val]
    if not isinstance(val, dict):
        raise SerializeError(f"cannot decode a value of type {type(val).__name__}")
    if "__ndarray__" in val:
        _only(val, "__ndarray__")
        key = val["__ndarray__"]
        if not isinstance(key, str) or key not in arrays:
            raise SerializeError(f"the manifest names an array that is not in the archive: {key!r}")
        return arrays[key]
    if "__npscalar__" in val:
        return _decode_scalar(val)
    if "__float__" in val:
        _only(val, "__float__")
        if val["__float__"] not in _NONFINITE:
            raise SerializeError(f"invalid number {val['__float__']!r}")
        return _NONFINITE[val["__float__"]]
    if "__tuple__" in val:
        _only(val, "__tuple__")
        return tuple(decode(v, arrays, depth + 1) for v in _as_list(val["__tuple__"]))
    if "__slice__" in val:
        _only(val, "__slice__")
        parts = [decode(v, arrays, depth + 1) for v in _as_list(val["__slice__"])]
        if len(parts) != 3 or not all(p is None or isinstance(p, int) for p in parts):
            raise SerializeError("invalid slice in the manifest")
        return slice(*parts)
    if "__dict__" in val:
        _only(val, "__dict__")
        items = val["__dict__"]
        if not isinstance(items, dict):
            raise SerializeError("invalid mapping in the manifest")
        return {k: decode(v, arrays, depth + 1) for k, v in items.items()}
    if "__items__" in val:
        _only(val, "__items__")
        out = {}
        for pair in _as_list(val["__items__"]):
            if not isinstance(pair, list) or len(pair) != 2:
                raise SerializeError("invalid mapping entry in the manifest")
            key = decode(pair[0], arrays, depth + 1)
            try:
                hash(key)
            except TypeError as exc:
                raise SerializeError("unhashable mapping key in the manifest") from exc
            out[key] = decode(pair[1], arrays, depth + 1)
        return out
    if "__dataclass__" in val:
        return _decode_dataclass(val, arrays, depth)
    raise SerializeError(f"unrecognised entry with keys {sorted(val)[:5]} in the manifest")


def _as_list(value: Any) -> list:
    if not isinstance(value, list):
        raise SerializeError("invalid sequence in the manifest")
    return value


def _decode_dataclass(val: dict, arrays: dict[str, np.ndarray], depth: int) -> Any:
    _only(val, "__dataclass__", ("fields",))
    name = val["__dataclass__"]
    cls = registry().get(name) if isinstance(name, str) else None
    if cls is None:
        raise SerializeError(f"unknown record {name!r} in the archive")
    raw = val["fields"]
    if not isinstance(raw, dict):
        raise SerializeError(f"invalid fields of {name}")
    known = {f.name for f in fields(cls) if f.init}
    unknown = set(raw) - known
    if unknown:
        raise SerializeError(f"{name} has no field(s) {sorted(unknown)}; the archive comes from another version")
    kwargs = {k: decode(v, arrays, depth + 1) for k, v in raw.items()}
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise SerializeError(f"invalid {name} in the archive: {exc}") from exc


# ---------------------------------------------------------------------------- archives
def member_level(arr: np.ndarray) -> int:
    """The zlib level a member is written with (0: stored); see :data:`INT_LEVEL`."""
    if arr.size == 0 or arr.dtype.kind in "biuSU":
        return INT_LEVEL
    if arr.dtype.kind not in "fc":
        return 0
    # the first elements only (flat indexing copies no more than it returns, whatever the memory order)
    sample = np.asarray(arr.flat[: max(1, SAMPLE_BYTES // arr.itemsize)]).tobytes()
    return FLOAT_LEVEL if len(zlib.compress(sample, FLOAT_LEVEL)) < FLOAT_GAIN * len(sample) else 0


def write_npz(
    file: Any,
    arrays: dict[str, np.ndarray],
    manifest: dict,
    *,
    compress: bool = True,
    progress: ProgressFn | None = None,
) -> None:
    """Write ``arrays`` and the JSON ``manifest`` as a ``.npz`` archive to ``file`` (a path or a writable stream,
    which need not be seekable). Arrays are written one at a time, so ``progress`` reports a real fraction
    (by bytes) on a gigabyte-scale save; ``compress`` picks the level of every member (:func:`member_level`)."""
    if MANIFEST_KEY in arrays:
        raise ValueError(f"{MANIFEST_KEY} is reserved")
    text = json.dumps(manifest, allow_nan=False, separators=(",", ":")).encode("utf-8")
    payload = {MANIFEST_KEY: np.frombuffer(text, dtype=np.uint8), **arrays}
    total = float(sum(max(1, int(a.nbytes)) for a in payload.values()))
    done = 0.0
    with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for key, arr in payload.items():
            arr = np.asanyarray(arr)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            level = member_level(arr) if compress else 0
            info.compress_type = zipfile.ZIP_DEFLATED if level else zipfile.ZIP_STORED
            if level:
                info._compresslevel = level  # the per-member level ZipFile.open honours (Python >= 3.7)
            # np.savez names every member "<key>.npy"; the same naming lets np.load read the archive back
            with zf.open(info, "w", force_zip64=True) as fid:
                write_array(fid, arr, allow_pickle=False)
            done += max(1, int(arr.nbytes))
            if progress is not None:
                progress(done / total)


def _check_member(npz: np.lib.npyio.NpzFile, key: str) -> None:
    """Refuse a member whose header declares more data than the member holds: numpy allocates the declared
    shape before it reads, so a small hostile file could otherwise ask for any amount of memory."""
    from numpy.lib import format as npy_format

    info = npz.zip.getinfo(key + ".npy")
    with npz.zip.open(info) as fid:
        version = npy_format.read_magic(fid)
        if version == (1, 0):
            shape, _fortran, dtype = npy_format.read_array_header_1_0(fid)
        elif version == (2, 0):
            shape, _fortran, dtype = npy_format.read_array_header_2_0(fid)
        else:  # the writer never uses another version
            raise SerializeError(f"member {key!r} has an unsupported .npy version {version}")
    if dtype.hasobject:
        raise SerializeError(f"member {key!r} holds Python objects")
    declared = math.prod(int(n) for n in shape) * int(dtype.itemsize)
    if declared > info.file_size:
        raise SerializeError(f"member {key!r} declares {declared} bytes but holds {info.file_size}")


def read_npz(file: Any, progress: ProgressFn | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """``(arrays, manifest)`` of an archive written by :func:`write_npz` (a path or a seekable binary stream)."""
    try:
        npz = np.load(file, allow_pickle=False)
        if not isinstance(npz, np.lib.npyio.NpzFile):  # a bare .npy array
            raise SerializeError("not a NumPy archive")
        try:
            names = list(npz.files)
            if MANIFEST_KEY not in names:
                raise SerializeError("not a pyALDVC archive (no manifest)")
            _check_member(npz, MANIFEST_KEY)
            raw = npz[MANIFEST_KEY]
            if raw.dtype != np.uint8 or raw.ndim != 1:
                raise SerializeError("invalid manifest")
            manifest = json.loads(raw.tobytes().decode("utf-8"))
            arrays = {}
            keys = [k for k in names if k != MANIFEST_KEY]
            for i, key in enumerate(keys):
                _check_member(npz, key)
                arrays[key] = npz[key]
                if progress is not None:
                    progress((i + 1) / max(1, len(keys)))
        finally:
            npz.close()
    except SerializeError:
        raise
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, UnicodeDecodeError, MemoryError) as exc:
        raise SerializeError(f"unreadable archive: {type(exc).__name__}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SerializeError("invalid manifest")
    return arrays, manifest


def save_object_npz(file: Any, obj: Any, kind: str, *, compress: bool = True, progress: ProgressFn | None = None) -> None:
    """Encode ``obj`` and write it as an archive whose manifest says it holds a ``kind``."""
    arrays, root = encode(obj)
    manifest = {"serialize_version": SERIALIZE_VERSION, "kind": kind, "root": root}
    write_npz(file, arrays, manifest, compress=compress, progress=progress)


def load_object_npz(file: Any, kind: str, expected: type | tuple[type, ...], progress: ProgressFn | None = None) -> Any:
    """Read an archive of ``save_object_npz``; the object must be a ``kind`` and an instance of ``expected``."""
    arrays, manifest = read_npz(file, progress)
    version = manifest.get("serialize_version")
    if version != SERIALIZE_VERSION:
        raise SerializeError(f"archive encoding {version!r} is not supported (this build reads {SERIALIZE_VERSION})")
    if manifest.get("kind") != kind:
        raise SerializeError(f"the archive holds {manifest.get('kind')!r}, not {kind!r}")
    obj = decode(manifest.get("root"), arrays)
    if not isinstance(obj, expected):
        raise SerializeError(f"the archive does not hold a {kind}")
    return obj


def save_result_npz(file: Any, result, *, compress: bool = True, progress: ProgressFn | None = None) -> None:
    """Write a :class:`PipelineResult` faithfully (displacement, gradients, uncertainty, strain, flags, mesh, parameters)."""
    save_object_npz(file, result, "PipelineResult", compress=compress, progress=progress)


def load_result_npz(file: Any, progress: ProgressFn | None = None):
    """Read a :class:`PipelineResult` written by :func:`save_result_npz`."""
    from al_dvc.core.data_structures import PipelineResult

    return load_object_npz(file, "PipelineResult", PipelineResult, progress)


def array_nbytes(obj: Any) -> int:
    """Bytes of the distinct arrays inside ``obj`` (by identity: no hashing, no copies) -- a cheap estimate of what
    an archive of it holds, for a progress bar."""
    seen: set[int] = set()
    total = 0
    stack = [obj]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            total += int(o.nbytes)
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
        elif is_dataclass(o) and not isinstance(o, type):
            stack.extend(getattr(o, f.name) for f in fields(o))
    return total


def estimated_nbytes(obj: Any) -> int:
    """Uncompressed, deduplicated size of the arrays of ``obj`` (no I/O): what a save has to write at most."""
    arrays, _root = encode(obj)
    return int(sum(int(a.nbytes) for a in arrays.values()))


def to_bytes(obj: Any, kind: str) -> bytes:
    """``obj`` as the bytes of an archive (tests, small objects)."""
    buf = io.BytesIO()
    save_object_npz(buf, obj, kind)
    return buf.getvalue()
