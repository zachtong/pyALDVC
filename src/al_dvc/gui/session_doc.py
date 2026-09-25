"""The session document (``session.json``): the sections that are plain settings, written and checked.

Everything here is data only: :mod:`al_dvc.gui.session` builds a document from the application state,
and reads documents of every format (the JSON files of formats 1 and 2, the ``session.json`` of a
format 3 bundle) back into checked values before anything is applied. A malformed value raises
:class:`SessionError` naming what is wrong; nothing in a document is ever executed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from al_dvc.core.config import DVCPara, para_from_dict

SLICE_AXES = ("z", "y", "x")
MASK_TARGETS = ("current", "all")
DISPLAY_FLOATS = ("color_min", "color_max", "overlay_alpha", "mask_alpha")
DISPLAY_BOOLS = ("slice_equal_scale", "show_mesh", "show_subset_window", "color_auto", "show_overlay", "show_mask")
VIEW_SECTIONS = ("main", "view3d", "post", "texture")

# Revision of the parameter defaults a session was saved with, written into every session. Sessions without it
# (1) come from the time when the noise-corrected IC-GN steps were on by default and could not be switched in the
# application: their "true" is the old default, not a choice.
PARA_REVISION = 2
NOISE_HESSIAN_NOTE = (
    "This session was saved when the noise-corrected steps were on by default; they are off now. Tick Advanced > "
    "Noise-corrected steps to reproduce the earlier results (and to resume checkpoints written then)."
)


class SessionError(Exception):
    """A session file could not be read, written or applied."""


# ---------------------------------------------------------------------------- display
def display_of(state) -> dict[str, Any]:
    """The display settings of ``state`` as the document stores them."""
    return {
        "field": state.display_field,
        "frame": int(state.current_frame),
        "colormap": state.colormap,
        "color_auto": bool(state.color_auto),
        "color_min": float(state.color_min),
        "color_max": float(state.color_max),
        "overlay_alpha": float(state.overlay_alpha),
        "show_overlay": bool(state.show_overlay),
        "background_frame": state.background_frame,
        "slice_index": {
            axis: (None if state.slice_index.get(axis) is None else int(state.slice_index[axis])) for axis in SLICE_AXES
        },
        "slice_layout": state.slice_layout,
        "slice_equal_scale": bool(state.slice_equal_scale),
        "show_mesh": bool(state.show_mesh),
        "show_subset_window": bool(state.show_subset_window),
        "show_mask": bool(state.show_mask),
        "mask_alpha": float(state.mask_alpha),
        "mask_target": state.mask_target,
    }


def _known_colormap(name: str) -> bool:
    import matplotlib

    return name in matplotlib.colormaps


def check_display(raw: Any, where: str) -> dict[str, Any]:
    """The display section, converted and checked (values the viewer cannot draw fall back to its defaults)."""
    from al_dvc.export.slice_plots import LAYOUTS

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError(f"invalid display settings in {where}")
    d = dict(raw)
    try:
        for key in DISPLAY_FLOATS:
            if key in d:
                d[key] = float(d[key])
        for key in DISPLAY_BOOLS:
            if key in d:
                d[key] = bool(d[key])
        if "frame" in d:
            d["frame"] = int(d["frame"])
        if d.get("background_frame") is not None:
            d["background_frame"] = int(d["background_frame"])
        if "field" in d:
            d["field"] = str(d["field"])
        slices = d.get("slice_index")
        if slices is not None:
            if not isinstance(slices, dict):
                raise TypeError("slice positions must be a mapping")
            d["slice_index"] = {a: (None if slices.get(a) is None else max(0, int(slices[a]))) for a in SLICE_AXES}
    except (TypeError, ValueError) as exc:
        raise SessionError(f"invalid display settings in {where}: {exc}") from exc
    if d.get("color_auto") is False and d.get("color_min", 0.0) >= d.get("color_max", 1.0):
        d["color_auto"] = True  # a reversed manual range is not restored
    if d.get("slice_layout") not in LAYOUTS:
        d.pop("slice_layout", None)
    if not isinstance(d.get("colormap"), str) or not _known_colormap(d["colormap"]):
        d.pop("colormap", None)
    if d.get("mask_target") not in MASK_TARGETS:
        d.pop("mask_target", None)
    for key in ("overlay_alpha", "mask_alpha"):
        if key in d:
            d[key] = min(1.0, max(0.0, d[key]))
    return d


def background_frame(value, n_volumes: int) -> int | None:
    """The saved background frame, or None (follow the selected one) when it no longer points at a frame."""
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if 0 <= index < n_volumes else None


# ---------------------------------------------------------------------------- statistics
def analysis_of(state) -> dict[str, Any]:
    return {
        "regions": [r.as_dict() for r in state.regions],
        "correction": None if state.display_correction is None else state.display_correction.as_dict(),
        "settings": dict(state.analysis_settings),
    }


def check_analysis(doc: dict, where: str) -> dict[str, Any]:
    """The ``analysis`` entry, checked: regions and the correction as objects, settings a mapping."""
    from al_dvc.analysis.corrected import Correction
    from al_dvc.analysis.regions import regions_from_dicts

    raw = doc.get("analysis") or {}
    if not isinstance(raw, dict):
        raise SessionError(f"invalid statistics settings in {where}")
    try:
        regions = regions_from_dicts(raw.get("regions") or [])
        corr = raw.get("correction")
        correction = None if corr is None else Correction.from_dict(corr)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise SessionError(f"invalid regions or motion correction in {where}: {exc}") from exc
    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        raise SessionError(f"invalid statistics settings in {where}")
    return {"regions": regions, "correction": correction, "settings": dict(settings)}


# ---------------------------------------------------------------------------- parameters
def _migrate_para(para: DVCPara, revision: int) -> tuple[DVCPara, list[str]]:
    """``para`` as the current defaults would read it, and a note for every change made."""
    if para.icgn_noise_hessian and revision < 2:
        return replace(para, icgn_noise_hessian=False), [NOISE_HESSIAN_NOTE]
    return para, []


def check_para(doc: dict, where: str) -> tuple[DVCPara, list[str]]:
    if not isinstance(doc.get("para"), dict):
        raise SessionError(f"{where} has no parameters")
    try:
        para = para_from_dict(doc["para"])
    except (TypeError, ValueError, KeyError) as exc:
        raise SessionError(f"invalid parameters in {where}: {exc}") from exc
    try:
        revision = int(doc.get("para_revision", 1))
    except (TypeError, ValueError) as exc:
        raise SessionError(f"invalid parameter revision in {where}") from exc
    return _migrate_para(para, revision)


# ---------------------------------------------------------------------------- window settings
def check_views(raw: Any, where: str) -> dict[str, dict]:
    """The window settings: a mapping of mappings. Their values are checked one by one when they are applied
    (a value a control cannot take is skipped there), so an odd one never keeps a session from opening."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError(f"invalid window settings in {where}")
    out = {}
    for key in VIEW_SECTIONS:
        section = raw.get(key)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise SessionError(f"invalid {key} settings in {where}")
        out[key] = dict(section)
    return out


def check_uid(value: Any, seen: set[str]) -> str | None:
    """A volume identity from a document: a short text not seen before, else None (a new one is made)."""
    if isinstance(value, str) and 0 < len(value) <= 64 and value.isprintable() and value not in seen:
        seen.add(value)
        return value
    return None


def check_uids(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(u, str) and len(u) <= 64 for u in value):
        raise SessionError(f"invalid result identities in {where}")
    return list(value)
