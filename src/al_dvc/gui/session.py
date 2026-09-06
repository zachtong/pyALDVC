"""Session files (``.aldvc``): volumes, masks, parameters, output folder, display state.

A session is a small JSON document. Volume and mask files are stored as
paths relative to the session file when possible. A mask drawn in the
application is written next to the session as a composed mask file
(``<name>_masks/mask_<k>.npy``) so what was on screen is exactly what comes
back; the drawing operations are only an in-memory undo history (older
sessions that stored operations are still replayed). Results are not
embedded: the session remembers the archive the user exported.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from al_dvc import __version__
from al_dvc.core.config import DVCPara, para_from_dict, para_to_dict

from .app_state import AppState, VolumeEntry

SESSION_SUFFIX = ".aldvc"
FORMAT_VERSION = 2  # 2: drawn masks are stored as composed files
MASK_DIR_SUFFIX = "_masks"
DISPLAY_FLOATS = ("color_min", "color_max", "overlay_alpha")
DISPLAY_BOOLS = ("slice_equal_scale", "show_mesh", "show_subset_window", "color_auto")


class SessionError(Exception):
    """A session file could not be read or applied."""


@dataclass
class SessionData:
    volumes: list[dict[str, Any]]
    para: DVCPara
    output_dir: str
    display: dict[str, Any] = field(default_factory=dict)
    results_path: str | None = None
    version: str = __version__


def _relative(path: str | None, base: Path) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(p)


def _absolute(path: str | None, base: Path) -> str | None:
    if path is None:
        return None
    p = Path(path)
    return str(p if p.is_absolute() else (base / p).resolve())


def build_session(state: AppState, results_path: str | None = None) -> SessionData:
    if results_path is None:
        results_path = getattr(state, "results_path", None)
    return SessionData(
        volumes=[{"path": v.path, "mask": v.mask_path, "label": v.label, "mask_ops": v.mask_ops} for v in state.volumes],
        para=state.para,
        output_dir=str(state.output_dir),
        display={
            "field": state.display_field,
            "frame": state.current_frame,
            "colormap": state.colormap,
            "slice_layout": state.slice_layout,
            "slice_equal_scale": bool(state.slice_equal_scale),
            "show_mesh": bool(state.show_mesh),
            "show_subset_window": bool(state.show_subset_window),
            "color_auto": state.color_auto,
            "color_min": state.color_min,
            "color_max": state.color_max,
            "overlay_alpha": state.overlay_alpha,
            "current_frame": state.current_frame,
        },
        results_path=results_path,
    )


def _drawn_mask_files(state: AppState, session: Path) -> list[str | None]:
    """Write every drawn (or edited) mask as a composed file next to the session; returns one path per volume
    (the frame's own file when nothing was drawn on top of it)."""
    from al_dvc.io.volume_io import save_volume

    out: list[str | None] = []
    folder = session.parent / f"{session.stem}{MASK_DIR_SUFFIX}"
    for i, entry in enumerate(state.volumes):
        mask = entry.mask
        if i == state.current_frame and state.mask_editor is not None:
            mask = state.mask_editor.mask
        drawn = mask is not None and (entry.mask_ops is not None or entry.mask_path is None)
        if not drawn:
            out.append(entry.mask_path)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"mask_{i:02d}.npy"
        save_volume(target, np.asarray(mask, dtype=np.uint8))
        out.append(str(target))
    return out


def save_session(state: AppState, path: str | Path, results_path: str | None = None) -> Path:
    """Write the session atomically (a temporary sibling replaces the file); ``SessionError`` on any failure,
    leaving a previous save untouched."""
    p = Path(path)
    if p.suffix != SESSION_SUFFIX:
        p = p.with_suffix(SESSION_SUFFIX)
    base = p.parent
    data = build_session(state, results_path)
    if any(v["path"] is None for v in data.volumes):
        raise SessionError("in-memory volumes cannot be saved in a session; save them to files first")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mask_files = _drawn_mask_files(state, p)
    except OSError as exc:
        raise SessionError(f"cannot write the mask files of {p}: {exc}") from exc
    doc = {
        "format": FORMAT_VERSION,
        "pyaldvc": data.version,
        "volumes": [
            {
                "path": _relative(v["path"], base),
                "mask": _relative(mask_files[i], base),
                "label": v["label"],
                "mask_ops": None,  # the mask file above is the composed mask; nothing is replayed on top of it
            }
            for i, v in enumerate(data.volumes)
        ],
        "para": para_to_dict(data.para),
        "output_dir": _relative(data.output_dir, base),
        "display": data.display,
        "results_path": _relative(data.results_path, base),
    }
    tmp = p.with_name(p.name + ".tmp")
    try:
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise SessionError(f"cannot write session {p}: {exc}") from exc
    state.session_path = p
    state.mark_clean()
    return p


def load_session(path: str | Path) -> SessionData:
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionError(f"cannot read session {p}: {exc}") from exc
    if not isinstance(doc, dict) or "para" not in doc or "volumes" not in doc:
        raise SessionError(f"{p} is not a pyALDVC session file")
    fmt = doc.get("format", 1)
    if not isinstance(fmt, int) or fmt > FORMAT_VERSION:
        raise SessionError(f"{p} was written by a newer pyALDVC (session format {fmt}); update pyALDVC to open it")
    base = p.parent
    try:
        para = para_from_dict(doc["para"])
    except (TypeError, ValueError) as exc:
        raise SessionError(f"invalid parameters in {p}: {exc}") from exc
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
        volumes.append(
            {"path": _absolute(path, base), "mask": _absolute(mask, base), "label": str(v.get("label") or ""), "mask_ops": ops}
        )
    display = doc.get("display", {})
    if not isinstance(display, dict):
        raise SessionError(f"invalid display settings in {p}")
    try:
        for key in DISPLAY_FLOATS:
            if key in display:
                display[key] = float(display[key])
        for key in DISPLAY_BOOLS:
            if key in display:
                display[key] = bool(display[key])
        if "frame" in display:
            display["frame"] = int(display["frame"])
    except (TypeError, ValueError) as exc:
        raise SessionError(f"invalid display settings in {p}: {exc}") from exc
    if display.get("color_auto") is False and display.get("color_min", 0.0) >= display.get("color_max", 1.0):
        display["color_auto"] = True  # a reversed manual range is not restored
    output_dir = doc.get("output_dir", "aldvc_results")
    if output_dir is not None and not isinstance(output_dir, str):
        raise SessionError(f"invalid output folder in {p}")
    results_path = doc.get("results_path")
    if results_path is not None and not isinstance(results_path, str):
        raise SessionError(f"invalid results path in {p}")
    return SessionData(
        volumes=volumes,
        para=para,
        output_dir=_absolute(output_dir or "aldvc_results", base) or "aldvc_results",
        display=dict(display),
        results_path=_absolute(results_path, base),
        version=str(doc.get("pyaldvc", "")),
    )


def _rebuild_drawn_masks(state: AppState) -> None:
    """Re-apply stored drawing operations (on top of the mask file, when there is one)."""
    from .mask_editor import MaskEditor

    for entry in state.volumes:
        if not entry.mask_ops:
            continue
        try:
            base = entry.load_mask() if entry.mask_path else None
            entry.mask = MaskEditor.from_dict(
                entry.mask_ops,
                base=base,
                volume=entry.load()
                if any(o.get("shape") == "threshold" for o in (entry.mask_ops or {}).get("ops", []))
                else None,
            ).mask
        except Exception as exc:
            state.log(f"{entry.name}: drawn mask could not be rebuilt ({exc})", "warning")
            entry.mask_ops = None


def apply_session(data: SessionData, state: AppState, path: str | Path | None = None) -> list[str]:
    """Load a session into ``state``; returns the paths that do not exist (volumes are kept).

    Every value is converted before the state is touched (``load_session`` validated the document), so a
    session either replaces the document completely or not at all."""
    volumes = [
        VolumeEntry(path=v["path"], mask_path=v["mask"], label=v.get("label") or "", mask_ops=v.get("mask_ops"))
        for v in data.volumes
    ]
    missing = [v.path for v in volumes if v.path and not Path(v.path).exists()]
    d = data.display
    values = {
        "display_field": str(d.get("field", state.display_field)),
        "current_frame": max(0, min(int(d.get("frame", 0)), max(0, len(volumes) - 1))),
        "colormap": str(d.get("colormap", state.colormap)),
        "slice_layout": str(d.get("slice_layout", state.slice_layout)),
        "slice_equal_scale": bool(d.get("slice_equal_scale", state.slice_equal_scale)),
        "show_mesh": bool(d.get("show_mesh", d.get("show_lattice", state.show_mesh))),
        "show_subset_window": bool(d.get("show_subset_window", state.show_subset_window)),
        "color_auto": bool(d.get("color_auto", True)),
        "color_min": float(d.get("color_min", 0.0)),
        "color_max": float(d.get("color_max", 1.0)),
        "overlay_alpha": float(d.get("overlay_alpha", 0.75)),
    }
    output_dir = Path(data.output_dir)
    # ---- commit
    state.volumes = volumes
    state.mask_editor = None
    state._mask_copy_backup = None
    _rebuild_drawn_masks(state)
    state.results = None
    state.result_uids = []
    state.results_path = data.results_path
    state.para = data.para
    state.output_dir = output_dir
    for key, val in values.items():
        setattr(state, key, val)
    state.session_path = Path(path) if path else None
    state.session_generation += 1
    state.mask_revision += 1
    state.mark_clean()
    state.volumes_changed.emit()
    state.params_changed.emit()
    state.results_changed.emit()
    state.display_changed.emit()
    state.output_dir_changed.emit(str(state.output_dir))
    state.mask_changed.emit()
    if data.results_path:
        if Path(data.results_path).exists():
            state.log(
                f"results are not restored with a session; the archive exported from this session is {data.results_path} "
                "(open it with the tools of your choice, or run the analysis again)"
            )
        else:
            state.log(f"the results archive of this session is missing: {data.results_path}", "warning")
    return missing
