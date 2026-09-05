"""Session files (``.aldvc``): volumes, masks, parameters, output folder, display state.

A session is a small JSON document. Volume and mask files are stored as
paths relative to the session file when possible; results are not embedded
(they live in the output folder as ``.npz`` and can be reloaded from there).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from al_dvc import __version__
from al_dvc.core.config import DVCPara, para_from_dict, para_to_dict

from .app_state import AppState, VolumeEntry

SESSION_SUFFIX = ".aldvc"
FORMAT_VERSION = 1


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
    return SessionData(
        volumes=[{"path": v.path, "mask": v.mask_path, "label": v.label, "mask_ops": v.mask_ops} for v in state.volumes],
        para=state.para,
        output_dir=str(state.output_dir),
        display={
            "field": state.display_field,
            "frame": state.display_frame,
            "colormap": state.colormap,
            "slice_layout": state.slice_layout,
            "slice_equal_scale": bool(state.slice_equal_scale),
            "color_auto": state.color_auto,
            "color_min": state.color_min,
            "color_max": state.color_max,
            "overlay_alpha": state.overlay_alpha,
            "current_frame": state.current_frame,
        },
        results_path=results_path,
    )


def save_session(state: AppState, path: str | Path, results_path: str | None = None) -> Path:
    p = Path(path)
    if p.suffix != SESSION_SUFFIX:
        p = p.with_suffix(SESSION_SUFFIX)
    base = p.parent
    data = build_session(state, results_path)
    if any(v["path"] is None for v in data.volumes):
        raise SessionError("in-memory volumes cannot be saved in a session; save them to files first")
    doc = {
        "format": FORMAT_VERSION,
        "pyaldvc": data.version,
        "volumes": [
            {
                "path": _relative(v["path"], base),
                "mask": _relative(v["mask"], base),
                "label": v["label"],
                "mask_ops": v.get("mask_ops"),
            }
            for v in data.volumes
        ],
        "para": para_to_dict(data.para),
        "output_dir": _relative(data.output_dir, base),
        "display": data.display,
        "results_path": _relative(data.results_path, base),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    state.session_path = p
    return p


def load_session(path: str | Path) -> SessionData:
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionError(f"cannot read session {p}: {exc}") from exc
    if not isinstance(doc, dict) or "para" not in doc or "volumes" not in doc:
        raise SessionError(f"{p} is not a pyALDVC session file")
    base = p.parent
    try:
        para = para_from_dict(doc["para"])
    except (TypeError, ValueError) as exc:
        raise SessionError(f"invalid parameters in {p}: {exc}") from exc
    volumes = [
        {
            "path": _absolute(v.get("path"), base),
            "mask": _absolute(v.get("mask"), base),
            "label": v.get("label", ""),
            "mask_ops": v.get("mask_ops"),
        }
        for v in doc["volumes"]
    ]
    return SessionData(
        volumes=volumes,
        para=para,
        output_dir=_absolute(doc.get("output_dir", "aldvc_results"), base) or "aldvc_results",
        display=dict(doc.get("display", {})),
        results_path=_absolute(doc.get("results_path"), base),
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
    """Load a session into ``state``; returns the paths that do not exist (volumes are kept)."""
    missing = [v["path"] for v in data.volumes if v["path"] and not Path(v["path"]).exists()]
    state.volumes = [
        VolumeEntry(path=v["path"], mask_path=v["mask"], label=v.get("label") or "", mask_ops=v.get("mask_ops"))
        for v in data.volumes
    ]
    state.mask_editor = None
    _rebuild_drawn_masks(state)
    state.current_frame = 0
    state.results = None
    state.para = data.para
    state.output_dir = Path(data.output_dir)
    d = data.display
    state.display_field = d.get("field", state.display_field)
    state.display_frame = int(d.get("frame", 0))
    state.colormap = d.get("colormap", state.colormap)
    state.slice_layout = d.get("slice_layout", state.slice_layout)
    state.slice_equal_scale = bool(d.get("slice_equal_scale", state.slice_equal_scale))
    state.color_auto = bool(d.get("color_auto", True))
    state.color_min = float(d.get("color_min", 0.0))
    state.color_max = float(d.get("color_max", 1.0))
    state.overlay_alpha = float(d.get("overlay_alpha", 0.75))
    state.session_path = Path(path) if path else None
    state.volumes_changed.emit()
    state.params_changed.emit()
    state.results_changed.emit()
    state.display_changed.emit()
    state.output_dir_changed.emit(str(state.output_dir))
    state.mask_changed.emit()
    if data.results_path and Path(data.results_path).exists():
        try:
            from al_dvc.export.export_npz import load_npz_result

            state.log(f"results file available: {data.results_path} ({len(load_npz_result(data.results_path))} arrays)")
        except Exception as exc:
            state.log(f"could not read {data.results_path}: {exc}", "warning")
    return missing
