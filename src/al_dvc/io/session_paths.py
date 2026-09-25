"""Where the files a session refers to are now: portable references and their relocation.

A session never embeds the volumes (they are gigabytes); it refers to each by its absolute path, its path
relative to the session file (always, even with ``..``) and a fingerprint (file name, size in bytes, and the
shape and dtype when they are known). Opening a session resolves every reference in this order:

1. the saved absolute path (accepted even when the fingerprint no longer matches: the user put that file
   there; the difference is reported);
2. the saved relative path, from the session file's folder (the project moved as a whole);
3. a folder next to the session with the original parent folder's name, then the session's own folder;
4. the relocation of any volume already found elsewhere, applied to the others (the common part of the old
   and the new path is kept, the prefix is swapped);
5. (the application) a folder the user points at, searched by name.

A candidate other than the saved path is accepted only when its name matches and its size (or, without a
size, its shape) matches the fingerprint, so a different file that happens to share the name is never
adopted silently. References of older sessions carry no fingerprint and are matched by name.

No Qt here: the batch runner and the command line resolve sessions the same way as the window.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HOW_SAVED = "saved"
HOW_RELATIVE = "relative"
HOW_BESIDE = "beside"  # a folder next to the session with the original parent folder's name
HOW_SESSION_FOLDER = "session_folder"
HOW_MOVED = "moved"  # the same relocation as another file of the session
HOW_LOCATED = "located"  # in the folder the user pointed at
HOW_MISSING = "missing"


@dataclass(frozen=True)
class Found:
    """Where one reference resolved: ``path`` (None when missing), how, and what differs from the fingerprint."""

    path: str | None
    how: str
    changed: str = ""  # the saved file is there but no longer matches its fingerprint (what differs)

    @property
    def missing(self) -> bool:
        return self.path is None

    @property
    def relocated(self) -> bool:
        return self.path is not None and self.how != HOW_SAVED


def _relative(path: str, base: Path) -> str | None:
    try:
        return Path(os.path.relpath(path, os.path.abspath(base))).as_posix()
    except ValueError:  # another drive (Windows): there is no relative path
        return None


def fingerprint(path: str | os.PathLike, shape=None, dtype=None) -> dict:
    """What identifies a volume file (or a folder of slices) besides its place: name, size or file count, and the
    shape and dtype when the caller knows them. Cheap: one ``stat`` (a folder: one listing), no file content."""
    p = Path(path)
    fp: dict = {"name": p.name}
    try:
        if p.is_dir():
            fp["kind"] = "folder"
            fp["n_files"] = sum(1 for q in p.iterdir() if q.is_file())
        elif p.exists():
            fp["kind"] = "file"
            fp["size"] = int(p.stat().st_size)
    except OSError:
        pass
    if shape is not None:
        fp["shape"] = [int(s) for s in shape]
    if dtype is not None:
        fp["dtype"] = str(dtype)
    return fp


def reference(path: str | os.PathLike, base: str | os.PathLike, *, identify: bool = True, shape=None, dtype=None) -> dict:
    """A portable reference to ``path`` for a session saved in the folder ``base``."""
    absolute = os.path.abspath(os.fspath(path))
    ref: dict = {"path": absolute, "relative": _relative(absolute, Path(base))}
    if identify:
        ref["fingerprint"] = fingerprint(absolute, shape, dtype)
    return ref


def check_reference(ref, what: str) -> dict:
    """``ref`` validated (``ValueError`` naming ``what``): a mapping with a string ``path`` and optional
    ``relative`` and ``fingerprint``."""
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not ref["path"]:
        raise ValueError(f"{what}: invalid file reference")
    rel = ref.get("relative")
    if rel is not None and not isinstance(rel, str):
        raise ValueError(f"{what}: invalid relative path")
    fp = ref.get("fingerprint")
    if fp is not None:
        if not isinstance(fp, dict) or not isinstance(fp.get("name", ""), str):
            raise ValueError(f"{what}: invalid fingerprint")
        for key in ("size", "n_files"):
            if key in fp and (not isinstance(fp[key], int) or isinstance(fp[key], bool) or fp[key] < 0):
                raise ValueError(f"{what}: invalid fingerprint {key}")
        shape = fp.get("shape")
        if shape is not None and (not isinstance(shape, list) or not all(isinstance(s, int) and s > 0 for s in shape)):
            raise ValueError(f"{what}: invalid fingerprint shape")
    return {"path": ref["path"], "relative": rel, "fingerprint": fp}


def _name(ref: dict) -> str:
    fp = ref.get("fingerprint") or {}
    return fp.get("name") or Path(ref["path"]).name


def _header_shape(path: Path):
    from .volume_io import read_volume_shape

    try:
        return read_volume_shape(path)
    except Exception:  # unreadable: no match
        return None


def difference(candidate: str | os.PathLike, fp: dict | None) -> str:
    """What makes ``candidate`` another file than the one ``fp`` describes ("" when nothing does)."""
    if not fp:
        return ""
    p = Path(candidate)
    try:
        if fp.get("kind") == "folder" or ("n_files" in fp and "size" not in fp):
            if not p.is_dir():
                return "not a folder"
            if "n_files" in fp:
                n = sum(1 for q in p.iterdir() if q.is_file())
                if n != fp["n_files"]:
                    return f"{n} files instead of {fp['n_files']}"
            return ""
        if not p.is_file():
            return "not a file"
        if "size" in fp:
            size = int(p.stat().st_size)
            return "" if size == fp["size"] else f"{size} bytes instead of {fp['size']}"
    except OSError as exc:
        return str(exc)
    if "shape" in fp:
        shape = _header_shape(p)
        if shape is None or [int(s) for s in shape] != list(fp["shape"]):
            return f"shape {shape} instead of {tuple(fp['shape'])}"
    return ""


def matches(candidate: str | os.PathLike, ref: dict) -> bool:
    """``candidate`` exists, carries the reference's name and fits its fingerprint (by name alone without one)."""
    p = Path(candidate)
    try:
        if not p.exists():
            return False
    except OSError:
        return False
    if os.path.normcase(p.name) != os.path.normcase(_name(ref)):
        return False
    return difference(p, ref.get("fingerprint")) == ""


def _parts(path: str | os.PathLike) -> tuple[str, ...]:
    return Path(os.path.abspath(os.fspath(path))).parts


def _shift(old: str, new: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """``(old prefix, new prefix)``: the paths with their longest common tail removed (the part that moved)."""
    a, b = _parts(old), _parts(new)
    k = 0
    while k < min(len(a), len(b)) - 1 and os.path.normcase(a[-1 - k]) == os.path.normcase(b[-1 - k]):
        k += 1
    if k == 0:
        return None
    return a[: len(a) - k], b[: len(b) - k]


def _apply_shift(path: str, shift) -> Path | None:
    old_prefix, new_prefix = shift
    parts = _parts(path)
    n = len(old_prefix)
    if len(parts) <= n or any(os.path.normcase(x) != os.path.normcase(y) for x, y in zip(parts[:n], old_prefix)):
        return None
    return Path(*new_prefix, *parts[n:])


def _direct(ref: dict, base: Path) -> Found | None:
    """Rules 1 to 3 for one reference."""
    saved = ref["path"]
    try:
        if Path(saved).exists():
            return Found(saved, HOW_SAVED, difference(saved, ref.get("fingerprint")))
    except OSError:
        pass
    name = _name(ref)
    candidates: list[tuple[Path, str]] = []
    if ref.get("relative"):
        candidates.append((base / ref["relative"], HOW_RELATIVE))
    parent = Path(saved).parent.name
    if parent:
        candidates.append((base / parent / name, HOW_BESIDE))
    candidates.append((base / name, HOW_SESSION_FOLDER))
    for cand, how in candidates:
        if matches(cand, ref):
            return Found(os.path.abspath(cand), how)
    return None


def resolve(refs: list[dict], session_path: str | os.PathLike | None) -> list[Found]:
    """Resolve every reference of a session opened from ``session_path`` (rules 1 to 4 of the module docstring)."""
    base = Path(os.path.abspath(session_path)).parent if session_path else Path.cwd()
    found: list[Found | None] = [_direct(ref, base) for ref in refs]
    shifts = []
    for ref, f in zip(refs, found):
        if f is not None and f.relocated:
            s = _shift(ref["path"], f.path)
            if s is not None and s not in shifts:
                shifts.append(s)
    return [f if f is not None else _by_shifts(ref, shifts) for ref, f in zip(refs, found)]


def _by_shifts(ref: dict, shifts, how: str = HOW_MOVED) -> Found:
    for s in shifts:
        cand = _apply_shift(ref["path"], s)
        if cand is not None and matches(cand, ref):
            return Found(os.path.abspath(cand), how)
    return Found(None, HOW_MISSING)


def relocate_into(refs: list[dict], found: list[Found], folder: str | os.PathLike) -> list[Found]:
    """The missing references searched for in ``folder`` (the user's choice): by name in it, then with the move
    from the first missing file's old folder to ``folder`` applied to the others (sibling folders)."""
    folder = Path(folder)
    missing = [i for i, f in enumerate(found) if f.missing]
    if not missing:
        return list(found)
    first = refs[missing[0]]["path"]
    shifts = [(_parts(first)[:-1], folder.parts)]
    shifts += [(_parts(first)[:-2], folder.parent.parts)] if len(_parts(first)) > 2 else []
    out = list(found)
    for i in missing:
        ref = refs[i]
        cand = folder / _name(ref)
        if matches(cand, ref):
            out[i] = Found(os.path.abspath(cand), HOW_LOCATED)
        else:
            out[i] = _by_shifts(ref, shifts, HOW_LOCATED)
    return out


def resolve_folder(ref: dict | None, session_path: str | os.PathLike | None, default: str) -> str:
    """A folder the session writes to (the output folder): the saved one when it exists, else the saved relative
    one when it exists, else -- for a folder that lived inside the session's folder -- the same place next to the
    session (the project moved; the folder is created when needed), else the saved one."""
    if not ref:
        return default
    saved, rel = ref["path"], ref.get("relative")
    if Path(saved).exists() or not session_path or not rel:
        return saved
    base = Path(os.path.abspath(session_path)).parent
    cand = base / rel
    if cand.exists() or not rel.startswith(".."):
        return os.path.abspath(cand)
    return saved
