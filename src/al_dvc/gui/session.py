"""Session files (``.aldvc``): everything the application holds after a computation, restored exactly.

Format 3 is a zip bundle (:mod:`al_dvc.io.session_bundle`):

* ``session.json`` -- the volumes (referenced, never embedded: absolute path, path relative to the session
  and a fingerprint), their order, labels and identities, the parameters, the output folder, the display,
  the statistics (regions, displayed correction, settings) and the settings of the windows (3-D view,
  post-processing, texture analysis). Human-readable.
* ``results.npz`` -- the computed result, faithfully (:mod:`al_dvc.io.session_serialize`): displacement,
  gradients, uncertainty, the strain of the post-processing window and its parameters, outlier flags,
  status, the mesh and the parameters of the run; the dtypes are kept bit for bit.
* ``masks.npz`` -- every distinct composed mask (and the texture region), bit-packed and compressed.
* ``texture.npz`` -- the texture analysis (autocorrelation and RVE sweep).

Opening a session restores the result without recomputing it (the run state is *done*), and a project
whose files moved is found again (:mod:`al_dvc.io.session_paths`); a volume that stays missing is kept in
the list, marked, and everything else is still restored. Formats 1 and 2 (plain JSON, drawn masks in a
``<name>_masks`` folder next to it) still open; saving always writes format 3.

``prepare_save`` runs on the UI thread and takes a snapshot; ``write_session`` does the heavy part and can run
on a worker thread; ``load_session`` too; ``apply_session`` runs on the UI thread.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from al_dvc import __version__
from al_dvc.core.config import DVCPara, para_to_dict
from al_dvc.io import session_paths
from al_dvc.io.session_bundle import (
    MASKS,
    RESULTS,
    TEXTURE,
    BundleError,
    SessionBundle,
    is_bundle,
    pack_mask,
    read_masks,
    write_bundle,
)
from al_dvc.io.session_serialize import (
    SerializeError,
    array_nbytes,
    load_object_npz,
    load_result_npz,
    save_object_npz,
    save_result_npz,
    write_npz,
)

from .app_state import AppState, RunState, VolumeEntry
from .session_doc import (
    PARA_REVISION,
    SessionError,
    analysis_of,
    background_frame,
    check_analysis,
    check_display,
    check_para,
    check_uid,
    check_uids,
    check_views,
    display_of,
)

__all__ = [
    "FORMAT_VERSION",
    "SESSION_SUFFIX",
    "SessionData",
    "SessionError",
    "apply_session",
    "load_session",
    "locate_missing",
    "prepare_save",
    "save_session",
    "write_session",
]

SESSION_SUFFIX = ".aldvc"
FORMAT_VERSION = 3  # 1, 2: JSON (2: drawn masks as files); 3: zip bundle with the results, masks and window settings
TEXTURE_REGION = "texture_region"  # key of the texture region in the mask archive
DEFAULT_OUTPUT = "aldvc_results"
ProgressFn = Callable[[float, str], None]


def _noop(_fraction: float, _message: str = "") -> None:
    return None


@dataclass
class SessionData:
    """A session as read from its file: checked values, nothing applied yet.

    Every entry of ``volumes`` is a mapping: ``path`` (where the file is now, the saved path while it is
    missing), ``saved``, ``label``, ``uid``, ``mask`` (the composed mask, an array, or None), ``mask_path`` (a mask
    file to read, formats 1 and 2), ``mask_ops`` (format 1 drawings), ``missing``, ``how`` it was found
    (:mod:`al_dvc.io.session_paths`), ``changed`` (how the file at the saved place differs from the saved one),
    ``shape`` (the saved size, when the file is the saved one) and ``ref`` (the saved reference).
    """

    volumes: list[dict[str, Any]]
    para: DVCPara
    output_dir: str
    display: dict[str, Any] = field(default_factory=dict)
    results_path: str | None = None
    version: str = __version__
    analysis: dict[str, Any] = field(default_factory=dict)  # {"regions", "correction", "settings"}
    notes: list[str] = field(default_factory=list)  # what loading changed (older parameter defaults)
    format: int = FORMAT_VERSION
    results: Any = None  # PipelineResult
    result_uids: list[str] = field(default_factory=list)
    views: dict[str, dict] = field(default_factory=dict)  # {"main", "view3d", "post", "texture"}
    texture: dict | None = None  # {"result", "sweep", "region"}
    write_checkpoints: bool | None = None
    path: str | None = None
    has_results: bool = False  # the file holds a result (read or not)

    @property
    def missing(self) -> list[str]:
        """The saved paths of the volumes that were not found."""
        return [v["saved"] for v in self.volumes if v["missing"]]

    @property
    def relocated(self) -> list[dict]:
        return [v for v in self.volumes if not v["missing"] and v["how"] != session_paths.HOW_SAVED]


@dataclass
class SavePlan:
    """What a save writes, taken on the UI thread (``prepare_save``) and written anywhere (``write_session``)."""

    path: Path
    document: dict
    results: Any
    masks: list[tuple[str, Any] | None]  # per volume: ("array", mask) or ("file", path) or None
    texture: dict | None  # {"result", "sweep", "region"}


# ---------------------------------------------------------------------------- saving
def _mask_source(state: AppState, index: int, entry: VolumeEntry) -> tuple[str, Any] | None:
    if index == state.current_frame and state.mask_editor is not None:
        return ("array", state.mask_editor.snapshot())  # the editor copies on its next edit: the save keeps this one
    if entry.mask is not None:
        return ("array", entry.mask)
    if entry.mask_path:
        return ("file", entry.mask_path)
    return None


def _file_ref(path: str | None, base: Path) -> dict | None:
    return None if not path else session_paths.reference(path, base, identify=False)


def prepare_save(
    state: AppState, path: str | os.PathLike, results_path: str | None = None, views: dict | None = None
) -> SavePlan:
    """Snapshot everything a save writes (UI thread; cheap: references, no copies of large arrays).

    ``views`` are the window settings (``session_views.collect``; default: the ones the last session brought),
    with the texture analysis under ``texture_data``. ``SessionError`` when the session cannot be saved."""
    p = Path(path)
    if p.suffix != SESSION_SUFFIX:
        p = p.with_suffix(SESSION_SUFFIX)
    p = Path(os.path.abspath(p))
    if any(v.path is None for v in state.volumes):
        raise SessionError("in-memory volumes cannot be saved in a session; save them to files first")
    base = p.parent
    views = dict(state.ui_state if views is None else views)
    texture = views.pop("texture_data", None)
    results = state.results
    results_path = results_path if results_path is not None else getattr(state, "results_path", None)
    volumes = []
    for entry in state.volumes:
        dtype = entry.array.dtype if entry.array is not None else None
        ref = session_paths.reference(entry.path, base, shape=entry.shape, dtype=dtype)
        volumes.append(
            {**ref, "label": entry.label, "uid": entry.uid, "mask": None, "mask_source": _file_ref(entry.mask_path, base)}
        )
    document = {
        "format": FORMAT_VERSION,
        "pyaldvc": __version__,
        "para_revision": PARA_REVISION,
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "volumes": volumes,
        "para": para_to_dict(state.para),
        "write_checkpoints": bool(state.write_checkpoints),
        "output_dir": session_paths.reference(state.output_dir, base, identify=False),
        "results": None,
        "results_path": _file_ref(results_path, base),
        "display": display_of(state),
        "analysis": analysis_of(state),
        "views": {k: v for k, v in views.items() if isinstance(v, dict)},
        "texture": None,
    }
    if results is not None:
        mesh = results.dvc_mesh
        document["results"] = {
            "archive": RESULTS,
            "result_uids": list(state.result_uids),
            "n_frames": int(results.n_frames),
            "n_nodes": int(mesh.n_nodes),
            "strain": bool(results.result_strain),
            "stopped_early": bool(results.stopped_early),
        }
    masks = [_mask_source(state, i, entry) for i, entry in enumerate(state.volumes)]
    return SavePlan(p, document, results, masks, texture if isinstance(texture, dict) else None)


def _read_mask_file(path: str) -> np.ndarray:
    from al_dvc.io.volume_io import load_volume

    return np.asarray(load_volume(path)) > 0


def _pack_masks(plan: SavePlan, report: ProgressFn) -> tuple[dict[str, tuple[np.ndarray, dict]], list, list[str]]:
    """Every distinct mask packed once (identical masks of several frames share one entry): ``(packed by key,
    the document's mask entry per volume, notes)``. Mask files are read here, one at a time."""
    base = plan.path.parent
    packed: dict[str, tuple[np.ndarray, dict]] = {}
    by_digest: dict[bytes, str] = {}
    by_id: dict[int, str] = {}
    by_file: dict[str, dict] = {}
    entries: list = []
    notes: list[str] = []

    def store(mask, remember: bool = True) -> str:
        """The key of ``mask``, packing it once; ``remember`` its identity only for arrays that stay alive (a mask
        read from a file is dropped after packing, and its ``id`` could come back for another array)."""
        key = by_id.get(id(mask)) if remember else None
        if key is not None:
            return key
        bits, meta = pack_mask(mask)
        digest = hashlib.sha256(repr(meta["shape"]).encode() + bits.tobytes()).digest()
        key = by_digest.get(digest)
        if key is None:
            key = f"m{len(by_digest)}"
            by_digest[digest] = key
            packed[key] = (bits, meta)
        if remember:
            by_id[id(mask)] = key
        return key

    n = max(1, len(plan.masks))
    for i, source in enumerate(plan.masks):
        report(0.1 * i / n, "masks")
        if source is None:
            entries.append(None)
            continue
        kind, value = source
        if kind == "array":  # the plan holds it: its identity is stable for the whole save
            entries.append({"embedded": store(value)})
            continue
        norm = os.path.normcase(os.path.abspath(value))
        if norm not in by_file:
            try:
                by_file[norm] = {"embedded": store(_read_mask_file(value), remember=False)}
            except Exception as exc:  # unreadable or gone: the session keeps pointing at the file
                by_file[norm] = {"file": session_paths.reference(value, base, identify=False)}
                notes.append(f"the mask file {value} could not be read ({exc}); the session refers to it by its path")
        entries.append(by_file[norm])
    region = (plan.texture or {}).get("region")
    if region is not None:
        bits, meta = pack_mask(region)
        packed[TEXTURE_REGION] = (bits, meta)
    return packed, entries, notes


def write_session(plan: SavePlan, progress: ProgressFn | None = None) -> Path:
    """Write the bundle of ``plan`` atomically (any thread). ``SessionError`` on any failure; a previous file at the
    same path is then left untouched."""
    report = progress or _noop
    try:
        plan.path.parent.mkdir(parents=True, exist_ok=True)
        packed, mask_entries, notes = _pack_masks(plan, report)
        members = []
        if packed:
            arrays = {k: v[0] for k, v in packed.items()}
            meta = {"kind": "masks", "masks": {k: v[1] for k, v in packed.items()}}
            members.append(
                (MASKS, lambda out, pr: write_npz(out, arrays, meta, progress=pr), float(sum(a.nbytes for a in arrays.values())))
            )
        if plan.results is not None:
            results = plan.results
            members.append((RESULTS, lambda out, pr: save_result_npz(out, results, progress=pr), float(array_nbytes(results))))
        texture = plan.texture or {}
        tex_objects = {k: texture.get(k) for k in ("result", "sweep")}
        if any(v is not None for v in tex_objects.values()):
            members.append(
                (
                    TEXTURE,
                    lambda out, pr: save_object_npz(out, tex_objects, "texture", progress=pr),
                    float(array_nbytes(tex_objects)),
                )
            )

        def document() -> dict:
            doc = copy.deepcopy(plan.document)
            for entry, mask in zip(doc["volumes"], mask_entries):
                entry["mask"] = mask
            if any(v is not None for v in tex_objects.values()) or TEXTURE_REGION in packed:
                doc["texture"] = {
                    "archive": TEXTURE if any(v is not None for v in tex_objects.values()) else None,
                    "region": TEXTURE_REGION if TEXTURE_REGION in packed else None,
                }
            if notes:
                doc["notes"] = notes
            return doc

        write_bundle(plan.path, members, document, progress=lambda f: report(0.1 + 0.9 * f, "writing"))
    except SessionError:
        raise
    except (OSError, ValueError, TypeError, MemoryError, BundleError, SerializeError) as exc:
        raise SessionError(f"cannot write session {plan.path}: {exc}") from exc
    report(1.0, "done")
    return plan.path


def save_session(
    state: AppState,
    path: str | os.PathLike,
    results_path: str | None = None,
    views: dict | None = None,
    progress: ProgressFn | None = None,
) -> Path:
    """Snapshot, write and mark the session clean in one call (scripts, tests; the window saves on a worker)."""
    out = write_session(prepare_save(state, path, results_path, views), progress)
    state.session_path = out
    state.mark_clean()
    return out


# ---------------------------------------------------------------------------- loading
def _legacy_ref(saved: str, base: Path) -> dict:
    """A reference of a format 1 or 2 session: a path relative to the session file or an absolute one."""
    p = Path(saved)
    return {
        "path": str(p if p.is_absolute() else (base / p).resolve()),
        "relative": None if p.is_absolute() else p.as_posix(),
        "fingerprint": None,
    }


def _volume(ref: dict, label: Any, uid: Any = None, mask=None, mask_ref=None, mask_ops=None) -> dict:
    return {
        "ref": ref,
        "saved": ref["path"],
        "path": ref["path"],
        "label": str(label or ""),
        "uid": uid,
        "mask": mask,
        "mask_ref": mask_ref,
        "mask_path": None,
        "mask_ops": mask_ops,
        "missing": False,
        "how": session_paths.HOW_SAVED,
        "changed": "",
        "shape": None,
    }


def _read_legacy(p: Path) -> SessionData:
    """A format 1 or 2 session (plain JSON)."""
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionError(f"cannot read session {p}: {exc}") from exc
    if not isinstance(doc, dict) or "para" not in doc or "volumes" not in doc:
        raise SessionError(f"{p} is not a pyALDVC session file")
    fmt = doc.get("format", 1)
    if not isinstance(fmt, int) or isinstance(fmt, bool) or fmt > FORMAT_VERSION:
        raise SessionError(f"{p} was written by a newer pyALDVC (session format {fmt}); update pyALDVC to open it")
    if fmt >= 3:
        raise SessionError(f"{p} is the document of a session bundle; open the .aldvc file itself")
    base = p.parent
    para, notes = check_para(doc, str(p))
    if not isinstance(doc["volumes"], list) or not all(isinstance(v, dict) for v in doc["volumes"]):
        raise SessionError(f"invalid volume list in {p}")
    volumes = []
    for k, v in enumerate(doc["volumes"]):
        path, mask, ops = v.get("path"), v.get("mask"), v.get("mask_ops")
        if not isinstance(path, str) or not path:
            raise SessionError(f"volume {k} of {p} has no file path")
        if mask is not None and not isinstance(mask, str):
            raise SessionError(f"volume {k} of {p}: invalid mask entry")
        if ops is not None and not (isinstance(ops, dict) and isinstance(ops.get("ops", []), list)):
            raise SessionError(f"volume {k} of {p}: invalid drawing operations")
        mask_ref = None if mask is None else _legacy_ref(mask, base)
        volumes.append(_volume(_legacy_ref(path, base), v.get("label"), mask_ref=mask_ref, mask_ops=ops))
    output_dir = doc.get("output_dir", DEFAULT_OUTPUT)
    results_path = doc.get("results_path")
    if (output_dir is not None and not isinstance(output_dir, str)) or (
        results_path is not None and not isinstance(results_path, str)
    ):
        raise SessionError(f"invalid output folder or results path in {p}")
    out_ref = _legacy_ref(output_dir or DEFAULT_OUTPUT, base)
    return SessionData(
        volumes=volumes,
        para=para,
        output_dir=session_paths.resolve_folder(out_ref, p, out_ref["path"]),
        display=check_display(doc.get("display", {}), str(p)),
        results_path=None if results_path is None else _legacy_ref(results_path, base)["path"],
        version=str(doc.get("pyaldvc", "")),
        analysis=check_analysis(doc, str(p)),
        notes=notes,
        format=fmt,
    )


def _check_mask_entry(entry: Any, k: int, where: str) -> tuple[str | None, dict | None]:
    """``(embedded key, file reference)`` of a volume's mask entry in a format 3 document."""
    if entry is None:
        return None, None
    if isinstance(entry, dict) and set(entry) == {"embedded"} and isinstance(entry["embedded"], str):
        return entry["embedded"], None
    if isinstance(entry, dict) and set(entry) == {"file"}:
        try:
            return None, session_paths.check_reference(entry["file"], f"volume {k} mask")
        except ValueError as exc:
            raise SessionError(f"{where}: {exc}") from exc
    raise SessionError(f"{where}: invalid mask entry of volume {k}")


def _bundle_volumes(doc: dict, where: str) -> tuple[list[dict], list[str | None]]:
    raw = doc.get("volumes")
    if not isinstance(raw, list) or not all(isinstance(v, dict) for v in raw):
        raise SessionError(f"invalid volume list in {where}")
    volumes, keys = [], []
    for k, v in enumerate(raw):
        try:
            ref = session_paths.check_reference(v, f"volume {k}")
        except ValueError as exc:
            raise SessionError(f"{where}: {exc}") from exc
        key, mask_ref = _check_mask_entry(v.get("mask"), k, where)
        volumes.append(_volume(ref, v.get("label"), uid=v.get("uid"), mask_ref=mask_ref))
        keys.append(key)
    return volumes, keys


def _optional_ref(value: Any, what: str, where: str) -> dict | None:
    if value is None:
        return None
    try:
        return session_paths.check_reference(value, what)
    except ValueError as exc:
        raise SessionError(f"{where}: {exc}") from exc


def _read_texture(bundle: SessionBundle, doc: dict, masks: dict, where: str) -> dict | None:
    from al_dvc.texture import SizeSweep, TextureResult

    entry = doc.get("texture")
    if entry is None:
        return None
    if not isinstance(entry, dict):
        raise SessionError(f"invalid texture analysis entry in {where}")
    out: dict = {"result": None, "sweep": None, "region": masks.get(entry.get("region")) if entry.get("region") else None}
    if entry.get("archive") is not None:
        if entry["archive"] != TEXTURE or TEXTURE not in bundle.names:
            raise SessionError(f"{where}: the texture analysis archive is missing")
        with bundle.member(TEXTURE) as stream:
            objects = load_object_npz(stream, "texture", dict)
        if set(objects) - {"result", "sweep"}:
            raise SessionError(f"{where}: invalid texture analysis archive")
        if objects.get("result") is not None and not isinstance(objects["result"], TextureResult):
            raise SessionError(f"{where}: invalid texture analysis result")
        if objects.get("sweep") is not None and not isinstance(objects["sweep"], SizeSweep):
            raise SessionError(f"{where}: invalid RVE analysis result")
        out.update(result=objects.get("result"), sweep=objects.get("sweep"))
    return out


def _read_bundle(p: Path, want_results: bool, report: ProgressFn) -> SessionData:
    where = p.name
    with SessionBundle(p) as bundle:
        doc = bundle.document()
        fmt = doc.get("format")
        if not isinstance(fmt, int) or isinstance(fmt, bool):
            raise SessionError(f"{where} is not a pyALDVC session (no format)")
        if fmt > FORMAT_VERSION:
            raise SessionError(f"{where} was written by a newer pyALDVC (session format {fmt}); update pyALDVC to open it")
        if fmt < 3:
            raise SessionError(f"{where}: invalid session bundle (format {fmt})")
        para, notes = check_para(doc, where)
        volumes, keys = _bundle_volumes(doc, where)
        display = check_display(doc.get("display"), where)
        analysis = check_analysis(doc, where)
        views = check_views(doc.get("views"), where)
        out_ref = _optional_ref(doc.get("output_dir"), "output folder", where)
        res_ref = _optional_ref(doc.get("results_path"), "results archive", where)
        res_doc = doc.get("results")
        if res_doc is not None and not isinstance(res_doc, dict):
            raise SessionError(f"invalid results entry in {where}")
        report(0.05, "masks")
        masks = read_masks(bundle, progress=lambda f: report(0.05 + 0.2 * f, "masks"))
        for v, key in zip(volumes, keys):
            if key is not None:
                if key not in masks:
                    raise SessionError(f"{where}: the mask {key} of a volume is not in the archive")
                v["mask"] = masks[key]
        results, uids = None, []
        if res_doc is not None:
            uids = check_uids(res_doc.get("result_uids"), where)
            if want_results:
                if RESULTS not in bundle.names:
                    raise SessionError(f"{where}: the results archive is missing")
                report(0.25, "results")
                with bundle.member(RESULTS) as stream:
                    results = load_result_npz(stream, progress=lambda f: report(0.25 + 0.65 * f, "results"))
        texture = _read_texture(bundle, doc, masks, where) if want_results else None
    report(0.95, "files")
    output_dir = session_paths.resolve_folder(out_ref, p, os.path.abspath(p.parent / DEFAULT_OUTPUT))
    results_path = None if res_ref is None else session_paths.resolve_folder(res_ref, p, res_ref["path"])
    checkpoints = doc.get("write_checkpoints")
    return SessionData(
        volumes=volumes,
        para=para,
        output_dir=output_dir,
        display=display,
        results_path=results_path,
        version=str(doc.get("pyaldvc", "")),
        analysis=analysis,
        notes=notes,
        format=fmt,
        results=results,
        result_uids=uids,
        views=views,
        texture=texture,
        write_checkpoints=checkpoints if isinstance(checkpoints, bool) else None,
        has_results=res_doc is not None,
    )


def _bind(volume: dict, found: session_paths.Found) -> None:
    volume["path"] = found.path if found.path else volume["saved"]
    volume["missing"] = found.missing
    volume["how"] = found.how
    volume["changed"] = found.changed
    fp = volume["ref"].get("fingerprint") or {}
    # the saved size describes the file when it is the saved one (checked by size) and is the best there is for a
    # missing one (the list shows it); a file that changed is read again
    same = not found.changed and (found.missing or "size" in fp or "n_files" in fp)
    shape = fp.get("shape")
    volume["shape"] = tuple(int(s) for s in shape) if same and isinstance(shape, list) and len(shape) == 3 else None


def _resolve(data: SessionData, session_path: Path) -> None:
    """Where every volume and mask file is now (the masks move with the volumes: one resolution for all)."""
    mask_owners = [v for v in data.volumes if v["mask_ref"] is not None]
    refs = [v["ref"] for v in data.volumes] + [v["mask_ref"] for v in mask_owners]
    found = session_paths.resolve(refs, session_path)
    for v, f in zip(data.volumes, found):
        _bind(v, f)
    for v, f in zip(mask_owners, found[len(data.volumes) :]):
        v["mask_path"] = f.path or v["mask_ref"]["path"]
        if f.missing:
            data.notes.append(f"the mask file {v['mask_ref']['path']} of {Path(v['saved']).name} was not found")


def load_session(path: str | os.PathLike, *, results: bool = True, progress: ProgressFn | None = None) -> SessionData:
    """Read and check a session of any format, and find its files (any thread; nothing is applied).

    ``results=False`` skips the result and the texture analysis (a batch run only needs the inputs).
    ``SessionError`` for anything unreadable, malformed or from a newer pyALDVC."""
    p = Path(os.path.abspath(path))
    report = progress or _noop
    try:
        bundle = is_bundle(p)
        data = _read_bundle(p, results, report) if bundle else _read_legacy(p)
    except SessionError:
        raise
    except (BundleError, SerializeError, OSError, ValueError, KeyError, TypeError, MemoryError, EOFError, RuntimeError) as exc:
        raise SessionError(f"cannot read session {p}: {type(exc).__name__}: {exc}") from exc
    data.path = str(p)
    _resolve(data, p)
    report(1.0, "done")
    return data


def locate_missing(data: SessionData, folder: str | os.PathLike) -> list[str]:
    """Look for the missing volumes of ``data`` in ``folder`` (the user's choice), by name and fingerprint;
    returns the saved paths still missing."""
    refs = [v["ref"] for v in data.volumes]
    found = [session_paths.Found(None if v["missing"] else v["path"], v["how"], v["changed"]) for v in data.volumes]
    for v, f in zip(data.volumes, session_paths.relocate_into(refs, found, folder)):
        if v["missing"] and not f.missing:
            _bind(v, f)
    return data.missing


# ---------------------------------------------------------------------------- applying
def _rebuild_drawn_masks(state: AppState) -> None:
    """Re-apply the drawing operations of a format 1 session (on top of its mask file, when there is one)."""
    from .mask_editor import MaskEditor

    for entry in state.volumes:
        if not entry.mask_ops:
            continue
        try:
            base = entry.load_mask() if entry.mask_path else None
            threshold = any(o.get("shape") == "threshold" for o in (entry.mask_ops or {}).get("ops", []))
            entry.mask = MaskEditor.from_dict(entry.mask_ops, base=base, volume=entry.load() if threshold else None).mask
        except Exception as exc:
            state.log(f"{entry.name}: drawn mask could not be rebuilt ({exc})", "warning")
            entry.mask_ops = None


def _entries(data: SessionData) -> list[VolumeEntry]:
    seen: set[str] = set()
    out = []
    for v in data.volumes:
        uid = check_uid(v.get("uid"), seen) or uuid.uuid4().hex
        out.append(
            VolumeEntry(
                path=v["path"],
                mask_path=v.get("mask_path"),
                mask=v.get("mask"),
                label=v.get("label") or "",
                mask_ops=v.get("mask_ops"),
                uid=uid,
                header_shape=v.get("shape"),
                missing=bool(v["missing"]),
            )
        )
    return out


def _display_values(d: dict, state: AppState, n: int) -> dict[str, Any]:
    slices = d.get("slice_index")
    return {
        "display_field": str(d.get("field", state.display_field)),
        "current_frame": max(0, min(int(d.get("frame", 0)), max(0, n - 1))),
        "colormap": str(d.get("colormap", state.colormap)),
        "slice_layout": str(d.get("slice_layout", state.slice_layout)),
        "slice_equal_scale": bool(d.get("slice_equal_scale", state.slice_equal_scale)),
        "show_mesh": bool(d.get("show_mesh", d.get("show_lattice", state.show_mesh))),
        "show_subset_window": bool(d.get("show_subset_window", state.show_subset_window)),
        "color_auto": bool(d.get("color_auto", True)),
        "color_min": float(d.get("color_min", 0.0)),
        "color_max": float(d.get("color_max", 1.0)),
        "overlay_alpha": float(d.get("overlay_alpha", 0.75)),
        "show_overlay": bool(d.get("show_overlay", True)),
        "background_frame": background_frame(d.get("background_frame"), n),
        "slice_index": dict(slices) if isinstance(slices, dict) else {"z": None, "y": None, "x": None},
        "show_mask": bool(d.get("show_mask", True)),
        "mask_alpha": float(d.get("mask_alpha", state.mask_alpha)),
        "mask_target": str(d.get("mask_target", "current")),
    }


def _report_files(data: SessionData, state: AppState) -> None:
    moved = data.relocated
    if moved:
        state.log(
            state.tr("{n} volume file(s) were not at their saved place and were found again: {files}").format(
                n=len(moved), files="; ".join(v["path"] for v in moved[:6]) + (" ..." if len(moved) > 6 else "")
            )
        )
    for v in data.volumes:
        if v["changed"] and not v["missing"]:
            state.log(
                state.tr("{name} is not the file the session was saved with ({why}); check that it is the right volume.").format(
                    name=Path(v["path"]).name, why=v["changed"]
                ),
                "warning",
            )
    missing = data.missing
    if missing:
        state.log(
            state.tr(
                "{n} volume file(s) of the session were not found: {files}. Results, masks and settings were restored; "
                "the views show no grey values for those frames. Move the files back, or open the session again and "
                "choose the folder that holds them."
            ).format(n=len(missing), files="; ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")),
            "warning",
        )


def apply_session(
    data: SessionData,
    state: AppState,
    path: str | os.PathLike | None = None,
    locate_folder_cb: Callable[[str], str | None] | None = None,
) -> list[str]:
    """Replace the document of ``state`` by the session ``data`` (UI thread); returns the saved paths of the volumes
    that are still missing (they are kept in the list, marked).

    ``locate_folder_cb(first missing path) -> folder or None`` is asked once when volumes are missing; the others are
    then searched for in that folder too. Every value was checked by ``load_session``, so a session replaces the
    document completely or not at all."""
    if data.missing and locate_folder_cb is not None:
        folder = locate_folder_cb(data.missing[0])
        if folder:
            before = len(data.missing)
            locate_missing(data, folder)
            state.log(
                state.tr("{n} of {m} missing volume file(s) found in {folder}").format(
                    n=before - len(data.missing), m=before, folder=folder
                )
            )
    volumes = _entries(data)
    values = _display_values(data.display, state, len(volumes))
    analysis = data.analysis or {}
    results = data.results
    # ---- commit
    state.volumes = volumes
    state.mask_editor = None
    state._mask_copy_backup = None
    _rebuild_drawn_masks(state)
    state.results = results
    state.result_uids = list(data.result_uids) if results is not None else []
    if results is not None and not state.result_uids:
        state.result_uids = [v.uid for v in volumes]
    state.results_path = data.results_path
    state.para = data.para
    state.output_dir = Path(data.output_dir)
    if data.write_checkpoints is not None:
        state.write_checkpoints = data.write_checkpoints
    for key, val in values.items():
        setattr(state, key, val)
    state.regions = list(analysis.get("regions") or [])
    state.display_correction = analysis.get("correction")
    state.analysis_settings = dict(analysis.get("settings") or {})
    state.ui_state = dict(data.views)
    if data.texture is not None:
        state.ui_state["texture_data"] = dict(data.texture)
    state._display_cache = None
    state.session_path = Path(path) if path else None
    state.session_generation += 1
    state.mask_revision += 1
    # the statistics controls first: every later signal makes the Statistics tab save its controls, which
    # must by then be the session's
    state.analysis_restored.emit()
    state.volumes_changed.emit()
    state.params_changed.emit()
    state.results_changed.emit()
    state.regions_changed.emit()
    state.correction_changed.emit()
    state.display_changed.emit()
    state.output_dir_changed.emit(str(state.output_dir))
    state.set_run_state(RunState.DONE if results is not None else RunState.IDLE)
    if results is not None:
        state.set_progress(1.0, state.tr("Results restored from the session"))
    state.start_shape_check()  # sizes the session did not know (older formats, files that changed)
    state.mask_changed.emit()
    for note in data.notes:
        state.log(note, "warning")
    _report_files(data, state)
    if results is not None:
        state.log(
            state.tr("Results restored from the session: {n} frame(s), {nodes} nodes{strain}").format(
                n=results.n_frames,
                nodes=f"{results.dvc_mesh.n_nodes:,}",
                strain=state.tr(", with strain") if results.result_strain else "",
            )
        )
    elif data.has_results:
        state.log(state.tr("The session holds results; they were not read."), "warning")
    if data.results_path and not Path(data.results_path).exists():
        state.log(
            state.tr("The results archive exported from this session is missing: {path}").format(path=data.results_path),
            "warning",
        )
    state.mark_clean()
    return data.missing
