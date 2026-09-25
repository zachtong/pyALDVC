"""The ``.aldvc`` session bundle: one zip file holding a readable document and NumPy archives.

Layout (session format 3)::

    session.json   what the user set up and saw: volumes (referenced, never embedded), parameters,
                   output folder, display, statistics, window settings -- human-readable JSON
    results.npz    the computed result, faithfully (al_dvc.io.session_serialize), when there is one
    masks.npz      every distinct composed mask, bit-packed and compressed
    texture.npz    the texture analysis (autocorrelation and RVE sweep), when there is one

The archives are stored uncompressed in the bundle (they compress their own members), so each can be
read in place: :meth:`SessionBundle.member` hands ``np.load`` a seekable window onto the bundle file
instead of extracting a copy. A bundle is written to a temporary file next to the target and moved over
it only once complete, so a failed save leaves the previous file untouched.

Older sessions (formats 1 and 2) were plain JSON; :func:`is_bundle` tells the two apart by content.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable

import numpy as np

from .session_serialize import SerializeError, read_npz, write_npz

DOCUMENT = "session.json"
RESULTS = "results.npz"
MASKS = "masks.npz"
TEXTURE = "texture.npz"
MAX_DOCUMENT_BYTES = 64 * 1024**2  # a session document is kilobytes; anything near this is not one
SPOOL_BYTES = 256 * 1024**2  # a compressed member (a repacked bundle) is extracted in memory up to this size

ProgressFn = Callable[[float], None]
Writer = Callable[[BinaryIO, ProgressFn], None]


class BundleError(ValueError):
    """A file that is not a readable session bundle."""


# ---------------------------------------------------------------------------- writing
def _plain(value: Any) -> Any:
    """JSON for the few non-JSON values a settings entry may carry (a NumPy number, a path); others are refused."""
    if isinstance(value, np.generic) and value.dtype.kind in "biuf":
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"a value of type {type(value).__name__} cannot be written in a session document")


def _finite(value: Any) -> Any:
    """``value`` with every non-finite number left out: a dict loses the key (reading it back gives the
    default), a list gets None. A NaN in one setting must not fail the save of a whole session."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(item, float) and not np.isfinite(item):
                continue
            out[key] = _finite(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def write_bundle(
    path: str | os.PathLike,
    members: Iterable[tuple[str, Writer, float]],
    document: Callable[[], dict],
    progress: ProgressFn | None = None,
) -> Path:
    """Write the bundle atomically: every ``(name, writer, weight)`` of ``members`` streams its archive, then
    ``document()`` (called last, so it can describe what the writers stored) becomes ``session.json``.

    ``OSError`` (disk full, no permission) and any exception of a writer propagate, and the previous file at
    ``path`` is left as it was.
    """
    target = Path(path)
    members = list(members)
    total = float(sum(max(w, 0.0) for _n, _w, w in members)) or 1.0
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    stamp = time.localtime(time.time())[:6]
    try:
        done = 0.0
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name, writer, weight in members:
                info = zipfile.ZipInfo(name, date_time=stamp)
                info.compress_type = zipfile.ZIP_STORED  # the archive compresses its own members

                def sub(fraction: float, _start=done, _w=max(weight, 0.0)) -> None:
                    if progress is not None:
                        progress(min(1.0, (_start + _w * float(fraction)) / total))

                with zf.open(info, "w", force_zip64=True) as out:
                    writer(out, sub)
                done += max(weight, 0.0)
            # through JSON once (NumPy numbers become floats), then without the non-finite numbers
            plain = json.loads(json.dumps(document(), allow_nan=True, default=_plain))
            text = json.dumps(_finite(plain), indent=1, allow_nan=False, ensure_ascii=False)
            doc_info = zipfile.ZipInfo(DOCUMENT, date_time=stamp)
            doc_info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(doc_info, text.encode("utf-8"))
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    if progress is not None:
        progress(1.0)
    return target


# ---------------------------------------------------------------------------- reading
def is_bundle(path: str | os.PathLike) -> bool:
    """True for a zip file (a session bundle), False for anything else (a JSON session of format 1 or 2)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"PK\x03\x04"
    except OSError as exc:
        raise BundleError(f"cannot read {path}: {exc}") from exc


class _Window(io.RawIOBase):
    """A read-only, seekable view of ``size`` bytes of a file from ``start``: an uncompressed zip member, read in
    place (``np.load`` opens the archive inside it without a copy)."""

    def __init__(self, path: Path, start: int, size: int) -> None:
        super().__init__()
        self._fh = open(path, "rb")  # noqa: SIM115 - closed with the window
        self._start, self._size, self._pos = int(start), int(size), 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._size}.get(whence)
        if base is None:
            raise ValueError(f"invalid whence {whence}")
        pos = base + int(offset)
        if pos < 0:
            raise OSError("seek before the start of the member")
        self._pos = pos
        return pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), max(0, self._size - self._pos))
        if n <= 0:
            return 0
        self._fh.seek(self._start + self._pos)
        data = self._fh.read(n)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


class SessionBundle:
    """An open bundle: its document and its archives. Use as a context manager."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        try:
            self._zip = zipfile.ZipFile(self.path, "r")
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            raise BundleError(f"{self.path.name} is not a readable session file: {exc}") from exc
        self.names = set(self._zip.namelist())

    def __enter__(self) -> "SessionBundle":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self._zip.close()

    def document(self) -> dict:
        if DOCUMENT not in self.names:
            raise BundleError(f"{self.path.name} is not a pyALDVC session (no {DOCUMENT})")
        info = self._zip.getinfo(DOCUMENT)
        if info.file_size > MAX_DOCUMENT_BYTES:
            raise BundleError(f"{self.path.name}: the session document is implausibly large")
        try:
            doc = json.loads(self._zip.read(DOCUMENT).decode("utf-8"))
        except (OSError, zipfile.BadZipFile, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BundleError(f"{self.path.name}: unreadable session document: {exc}") from exc
        if not isinstance(doc, dict):
            raise BundleError(f"{self.path.name}: the session document is not a JSON object")
        return doc

    def member(self, name: str) -> BinaryIO:
        """A seekable binary stream of the member ``name`` (close it after use)."""
        if name not in self.names:
            raise BundleError(f"{self.path.name} has no {name}")
        info = self._zip.getinfo(name)
        if info.flag_bits & 0x1:
            raise BundleError(f"{self.path.name}: {name} is encrypted")
        if info.compress_type == zipfile.ZIP_STORED:
            return _Window(self.path, self._data_offset(info), info.file_size)
        spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_BYTES)  # a repacked bundle: extract the member
        try:
            with self._zip.open(info) as src:
                shutil.copyfileobj(src, spool, 1 << 20)
        except (OSError, zipfile.BadZipFile, RuntimeError, EOFError) as exc:
            spool.close()
            raise BundleError(f"{self.path.name}: cannot read {name}: {exc}") from exc
        spool.seek(0)
        return spool  # type: ignore[return-value]

    def _data_offset(self, info: zipfile.ZipInfo) -> int:
        """Where the data of a member starts: after its local header, whose name and extra lengths are its own."""
        with open(self.path, "rb") as fh:
            fh.seek(info.header_offset)
            header = fh.read(zipfile.sizeFileHeader)
        if len(header) != zipfile.sizeFileHeader or header[:4] != zipfile.stringFileHeader:
            raise BundleError(f"{self.path.name}: corrupt entry {info.filename}")
        fields = struct.unpack(zipfile.structFileHeader, header)
        return info.header_offset + zipfile.sizeFileHeader + fields[10] + fields[11]

    def arrays(self, name: str, progress: ProgressFn | None = None) -> tuple[dict[str, np.ndarray], dict]:
        """``(arrays, manifest)`` of the archive ``name``."""
        stream = self.member(name)
        try:
            magic = stream.read(4)
            stream.seek(0)
            if magic != b"PK\x03\x04":
                raise BundleError(f"{self.path.name}: {name} is not a NumPy archive")
            return read_npz(stream, progress)
        except SerializeError as exc:
            raise BundleError(f"{self.path.name}: {name}: {exc}") from exc
        finally:
            stream.close()


def read_session_document(path: str | os.PathLike) -> dict:
    """The JSON document of a session of any format (the command line's ``--regions``, tests, tools)."""
    p = Path(path)
    if is_bundle(p):
        with SessionBundle(p) as bundle:
            return bundle.document()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read session {p}: {exc}") from exc
    if not isinstance(doc, dict):
        raise BundleError(f"{p} is not a pyALDVC session file")
    return doc


# ---------------------------------------------------------------------------- masks
def pack_mask(mask) -> tuple[np.ndarray, dict]:
    """A boolean volume as bit-packed bytes (one eighth of its size before compression) and its shape."""
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        m = m != 0
    return np.packbits(m.reshape(-1)), {"shape": [int(s) for s in m.shape], "count": int(np.count_nonzero(m))}


def unpack_mask(packed: np.ndarray, meta: Any) -> np.ndarray:
    """Inverse of :func:`pack_mask`, checked against the shape it declares."""
    if not isinstance(meta, dict) or not isinstance(meta.get("shape"), list):
        raise BundleError("invalid mask description")
    shape = meta["shape"]
    if len(shape) != 3 or not all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in shape):
        raise BundleError(f"invalid mask shape {shape}")
    n = int(np.prod(shape, dtype=np.int64))
    if packed.dtype != np.uint8 or packed.ndim != 1 or packed.size != (n + 7) // 8:
        raise BundleError(f"a mask of shape {tuple(shape)} does not fit its packed data")
    return np.unpackbits(packed, count=n).reshape(shape).view(np.bool_)


def write_masks(stream: BinaryIO, masks: dict[str, np.ndarray], progress: ProgressFn | None = None) -> None:
    """``masks`` (key -> boolean volume) as a compressed archive of bit-packed arrays."""
    arrays, meta = {}, {}
    for key, mask in masks.items():
        arrays[key], meta[key] = pack_mask(mask)
    write_npz(stream, arrays, {"kind": "masks", "masks": meta}, progress=progress)


def read_masks(bundle: SessionBundle, progress: ProgressFn | None = None) -> dict[str, np.ndarray]:
    """Every mask of the bundle by key (an empty mapping without a mask archive)."""
    if MASKS not in bundle.names:
        return {}
    arrays, manifest = bundle.arrays(MASKS, progress)
    meta = manifest.get("masks")
    if manifest.get("kind") != "masks" or not isinstance(meta, dict) or set(meta) != set(arrays):
        raise BundleError(f"{bundle.path.name}: invalid mask archive")
    return {key: unpack_mask(arrays[key], meta[key]) for key in arrays}
