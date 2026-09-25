"""The settings of the windows a session keeps: the 3-D view, the post-processing window, the texture analysis.

:func:`collect` reads them from the live windows when a session is saved (a window never opened keeps what
the last session brought, held in ``AppState.ui_state``); :func:`restore` puts them back after a session
opens, into the windows that exist, and leaves the rest in ``AppState.ui_state`` for the windows opened
later (:func:`restore_post` and :func:`restore_texture` are called by the main window when it creates
them). Restoring is lenient: a value a control cannot take is skipped, so a settings entry never keeps a
session from opening. What is *not* here on purpose: the drawing tool of the moment, the export choices,
the application preferences (theme, language, window geometry), which belong to the application.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

POST = "post"
TEXTURE = "texture"
VIEW3D = "view3d"
MAIN = "main"
TEXTURE_DATA = "texture_data"  # the texture analysis itself (result, sweep, region) next to its settings


# ---------------------------------------------------------------------------- helpers
@contextlib.contextmanager
def _blocked(*widgets):
    """Set controls without their signals (the owner is refreshed once afterwards)."""
    previous = [w.blockSignals(True) for w in widgets]
    try:
        yield
    finally:
        for w, was in zip(widgets, previous):
            w.blockSignals(was)


def _select(combo, key) -> bool:
    if key is None:
        return False
    i = combo.findData(key)
    if i >= 0:
        combo.setCurrentIndex(i)
    return i >= 0


def _check(box, value) -> None:
    if isinstance(value, bool):
        box.setChecked(value)


def _number(spin, value, kind=float) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        spin.setValue(kind(min(max(value, spin.minimum()), spin.maximum())))


def _triple(value, kind=int):
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
        return tuple(kind(v) for v in value)
    return None


# ---------------------------------------------------------------------------- 3-D view
def view3d_state(panel) -> dict:
    return {
        "mode": panel.mode_key(),
        "slices_shown": {axis: box.isChecked() for axis, box in panel.slice_visible.items()},
        "iso": float(panel.iso.value()),
        "iso_levels": int(panel.iso_levels.value()),
        "cutaway": panel.iso_cutaway.isChecked(),
        "warp_scale": float(panel.warp_scale.value()),
        "edges": panel.edges.isChecked(),
        "arrows": panel.arrows.isChecked(),
        "stride": int(panel.stride.value()),
        "arrow_scale": float(panel.arrow_scale.value()),
        "volume_slices": panel.volume_slices.isChecked(),
        "outline": panel.outline.isChecked(),
        "background": panel.background_key(),
        "background_chosen": bool(panel._background_default.chosen),
        "camera": panel._camera,
        "turn": int(panel.azimuth.value()),
        "tilt": int(panel.elevation.value()),
        "zoom": float(panel.zoom.value()),
        "animation": {
            "kind": panel.anim_kind_key(),
            "axis": panel.anim_axis.currentData(),
            "direction": panel.anim_direction.currentData(),
            "speed": float(panel.anim_speed.value()),
            "smooth": panel.anim_smooth.isChecked(),
        },
    }


def restore_view3d(panel, d: dict | None) -> None:
    from .view3d_scene import CAMERAS

    if not isinstance(d, dict) or not d:
        return
    panel.stop_animation()
    controls = (
        panel.mode,
        panel.iso,
        panel.iso_levels,
        panel.iso_cutaway,
        panel.warp_scale,
        panel.edges,
        panel.arrows,
        panel.stride,
        panel.arrow_scale,
        panel.volume_slices,
        panel.outline,
        panel.background,
        panel.camera,
        panel.azimuth,
        panel.elevation,
        panel.zoom,
        panel.anim_kind,
        *panel.slice_visible.values(),
    )
    anim = d.get("animation") if isinstance(d.get("animation"), dict) else {}
    with _blocked(*controls):
        _select(panel.mode, d.get("mode"))
        shown = d.get("slices_shown") if isinstance(d.get("slices_shown"), dict) else {}
        for axis, box in panel.slice_visible.items():
            _check(box, shown.get(axis))
        _number(panel.iso, d.get("iso"))
        _number(panel.iso_levels, d.get("iso_levels"), int)
        _check(panel.iso_cutaway, d.get("cutaway"))
        _number(panel.warp_scale, d.get("warp_scale"))
        _check(panel.edges, d.get("edges"))
        _check(panel.arrows, d.get("arrows"))
        _number(panel.stride, d.get("stride"), int)
        _number(panel.arrow_scale, d.get("arrow_scale"))
        _check(panel.volume_slices, d.get("volume_slices"))
        _check(panel.outline, d.get("outline"))
        chosen = d.get("background_chosen") is True and _select(panel.background, d.get("background"))
        if d.get("camera") in CAMERAS:
            _select(panel.camera, d["camera"])
            panel._camera = d["camera"]
        _number(panel.azimuth, d.get("turn"), int)
        _number(panel.elevation, d.get("tilt"), int)
        _number(panel.zoom, d.get("zoom"))
        _select(panel.anim_kind, anim.get("kind"))
    panel._background_default.chosen = bool(chosen)
    if not chosen:
        panel._background_default.apply()  # the theme's background, as for a new window
    panel._on_anim_kind()  # the speed range and unit of the kind (and its default speed) ...
    with _blocked(panel.anim_axis, panel.anim_direction, panel.anim_speed, panel.anim_smooth):
        _select(panel.anim_axis, anim.get("axis"))
        _select(panel.anim_direction, anim.get("direction"))
        _number(panel.anim_speed, anim.get("speed"))  # ... then the saved speed
        _check(panel.anim_smooth, anim.get("smooth"))
    panel._camera_reset_pending = True  # the preset with the saved turn, tilt and zoom
    panel._live_state = None
    panel._on_background()
    panel._update_enabled()
    panel._sync_animation_choices()
    panel.invalidate()


# ---------------------------------------------------------------------------- post-processing window
def post_state(win) -> dict:
    return {
        "open": win.isVisible(),
        "tab": int(win.tabs.currentIndex()),
        "frame": int(win.frame_slider.value()),
        "controls": {
            "method": win.method.currentData(),
            "measure": win.measure.currentData(),
            "fit_window": [int(w.value()) for w in win.fit_window_axes],
            "cube": win.fit_window_lock.isChecked(),
            "disp_smoothing": float(win.disp_smoothing.value()),
            "strain_smoothing": float(win.strain_smoothing.value()),
            "edge_trim": win.edge_trim.isChecked(),
        },
        "display": {
            "field": win.field.currentData(),
            "colormap": win.colormap.currentText(),
            "auto": win.auto_range.isChecked(),
            "min": float(win.vmin.value()),
            "max": float(win.vmax.value()),
            "layout": win.layout_combo.currentData(),
            "show_volume": win.show_volume.isChecked(),
            "equal_scale": win.equal_scale.isChecked(),
        },
        "analysis": {
            "frame": int(win.analysis.frame_slider.value()),
            "results_tab": int(win.analysis.results_tabs.currentIndex()),
        },
    }


def restore_post(win, d: dict | None) -> None:
    """The strain controls (the result carries the strain and the settings it was computed with; controls edited
    since are put back and flagged as such), the display, the frame and the tab."""
    if not isinstance(d, dict) or not d:
        return
    c = d.get("controls") if isinstance(d.get("controls"), dict) else {}
    win._updating = True
    try:
        _select(win.method, c.get("method"))
        _select(win.measure, c.get("measure"))
        widths = _triple(c.get("fit_window"))
        if widths is not None:
            _check(win.fit_window_lock, c.get("cube") if isinstance(c.get("cube"), bool) else len(set(widths)) == 1)
            for w, width in zip(win.fit_window_axes, widths):
                _number(w, width | 1, int)  # odd widths only
        _number(win.disp_smoothing, c.get("disp_smoothing"))
        _number(win.strain_smoothing, c.get("strain_smoothing"))
        _check(win.edge_trim, c.get("edge_trim"))
    finally:
        win._updating = False
    res = win._state.results
    win._stale = bool(res is not None and res.result_strain and win._controls_differ(res.dvc_para))
    win._progress.setValue(win._progress.maximum() if res is not None and res.result_strain else 0)
    disp = d.get("display") if isinstance(d.get("display"), dict) else {}
    win._updating = True
    try:
        with _blocked(win.show_volume, win.equal_scale):
            _check(win.show_volume, disp.get("show_volume"))
            _check(win.equal_scale, disp.get("equal_scale"))
        _select(win.field, disp.get("field"))
        if isinstance(disp.get("colormap"), str) and win.colormap.findText(disp["colormap"]) >= 0:
            win.colormap.setCurrentText(disp["colormap"])
        _select(win.layout_combo, disp.get("layout"))
        _number(win.vmin, disp.get("min"))
        _number(win.vmax, disp.get("max"))
        with _blocked(win.auto_range):
            _check(win.auto_range, disp.get("auto"))
        win.vmin.setEnabled(not win.auto_range.isChecked())
        win.vmax.setEnabled(not win.auto_range.isChecked())
    finally:
        win._updating = False
    if isinstance(d.get("tab"), int) and 0 <= d["tab"] < win.tabs.count():
        win.tabs.setCurrentIndex(d["tab"])
    _number(win.frame_slider, d.get("frame"), int)
    a = d.get("analysis") if isinstance(d.get("analysis"), dict) else {}
    _number(win.analysis.frame_slider, a.get("frame"), int)
    tab = a.get("results_tab")
    if isinstance(tab, int) and 0 <= tab < win.analysis.results_tabs.count():
        win.analysis.results_tabs.setCurrentIndex(tab)
    win._load_data()  # the background volume as the restored box says, then the view and the status
    win._update_status()


# ---------------------------------------------------------------------------- texture analysis
def _region_spec(win) -> tuple[dict, np.ndarray | None]:
    from al_dvc.texture import whole_box

    shape, box = win._shape, win.range_box()
    if shape is None or box is None:
        return {"kind": "whole"}, None
    if win.region.fill_fraction() >= 1.0:
        if tuple(box) == whole_box(shape):
            return {"kind": "whole"}, None
        return {"kind": "box", "box": [list(pair) for pair in box]}, None
    return {"kind": "mask"}, win.region.snapshot()


def _calibration(source: dict | None) -> dict:
    source = source or {}
    return {
        "units": source.get("units", "voxel"),
        "spacing": list(source.get("spacing", (1.0, 1.0, 1.0))),
        "centre": list(source["centre"]) if source.get("centre") else None,
    }


def texture_state(win) -> tuple[dict, dict]:
    """``(settings, analysis)``: the window's settings for the document, the result, sweep and region for the
    archives."""
    region, mask = _region_spec(win)
    settings = {
        "open": win.isVisible(),
        "step": int(win.tabs.currentIndex()),
        "region": region,
        "centre": list(win.region.centre) if win.region.centre is not None else None,
        "cube_size": int(win.cube_size.value()),
        "factor": float(win.factor.value()),
        "sweep": win.sweep_settings(),
        "size_from_rve": win._size_from_rve,
        "result_current": win.result is not None and not win.is_stale,
        "sweep_current": win.sweep is not None and not win.is_sweep_stale,
        "result_source": _calibration(win._result_source),
        "sweep_source": _calibration(win._sweep_source),
        "plot": {
            "background": win.plot_background.currentData(),
            "background_chosen": bool(win._plot_default.chosen),
            "scale": win.plot_scale.currentData(),
            "curves": {axis: box.isChecked() for axis, box in win.curve_checks.items()},
            "band": win.show_band.isChecked(),
        },
    }
    return settings, {"result": win.result, "sweep": win.sweep, "region": mask}


def _stale_source(saved: Any) -> dict:
    """A source that describes a previous input: it keeps the calibration and centre the analysis was made with (the
    table, the plot title) and never equals the input of now."""
    s = saved if isinstance(saved, dict) else {}
    spacing = _triple(s.get("spacing"), float) or (1.0, 1.0, 1.0)
    centre = _triple(s.get("centre"))
    return {"uid": None, "restored": True, "units": str(s.get("units", "voxel")), "spacing": spacing, "centre": centre}


def _restore_region(win, spec: Any, mask) -> None:
    if win._shape is None or not isinstance(spec, dict):
        return  # no reference volume: the region cannot be drawn (the analyses are still shown)
    kind = spec.get("kind")
    if kind == "mask" and mask is not None and tuple(mask.shape) == tuple(win._shape):
        win.region.set_region(mask)
    elif kind == "box" and isinstance(spec.get("box"), list):
        try:
            win.set_range(tuple(tuple(int(v) for v in pair) for pair in spec["box"]))
        except (TypeError, ValueError):
            pass


def restore_texture(win, d: dict | None, data: dict | None) -> None:
    """The texture window as the session left it: region, centre, cube, sweep settings, the two analyses (still
    current, or marked as from a previous input as they were) and the plot settings. ``None`` clears it."""
    from al_dvc.texture import SizeSweep, TextureResult

    d = d if isinstance(d, dict) else {}
    data = data if isinstance(data, dict) else {}
    result = data.get("result") if isinstance(data.get("result"), TextureResult) else None
    sweep = data.get("sweep") if isinstance(data.get("sweep"), SizeSweep) else None
    _restore_region(win, d.get("region"), data.get("region"))
    centre = _triple(d.get("centre"))
    if centre is not None and win._shape is not None:
        win.region.set_centre(centre)
    win._updating = True
    try:
        _number(win.cube_size, d.get("cube_size"), int)
        _number(win.factor, d.get("factor"))
        sw = d.get("sweep") if isinstance(d.get("sweep"), dict) else {}
        _number(win.sweep_start, sw.get("start"), int)
        _number(win.sweep_step, sw.get("step"), int)
        _number(win.sweep_count, sw.get("count"), int)
        _select(win.sweep_criterion, sw.get("criterion"))
        plot = d.get("plot") if isinstance(d.get("plot"), dict) else {}
        with _blocked(win.plot_background, win.plot_scale, win.show_band, *win.curve_checks.values()):
            chosen = plot.get("background_chosen") is True and _select(win.plot_background, plot.get("background"))
            win._plot_default.chosen = bool(chosen)
            if not chosen:
                win._plot_default.apply()
            _select(win.plot_scale, plot.get("scale"))
            curves = plot.get("curves") if isinstance(plot.get("curves"), dict) else {}
            for axis, box in win.curve_checks.items():
                _check(box, curves.get(axis))
            _check(win.show_band, plot.get("band"))
    finally:
        win._updating = False
    rve = d.get("size_from_rve")
    win._size_from_rve = int(rve) if isinstance(rve, int) and not isinstance(rve, bool) else None
    win.result, win.sweep = result, sweep
    win._previous_note = ""
    win._result_source = None if result is None else (win.current_source() if d.get("result_current") is True else None)
    if result is not None and win._result_source is None:
        win._result_source = _stale_source(d.get("result_source"))
    win._sweep_source = None if sweep is None else (win._sweep_input() if d.get("sweep_current") is True else None)
    if sweep is not None and win._sweep_source is None:
        win._sweep_source = _stale_source(d.get("sweep_source"))
    win.recommendation = win._recommend(result)
    win._progress.setValue(1000 if result is not None else 0)
    win._sweep_progress.setValue(1000 if sweep is not None else 0)
    win._fill_table()
    win._redraw()
    win._update_cube_info()
    win._update_size_source()
    step = d.get("step")
    win.go_to_step(step if isinstance(step, int) else win.tabs.currentIndex())
    win._update_overlay()
    win._refresh_validity()


# ---------------------------------------------------------------------------- the whole window
def collect(window) -> dict:
    """The settings of every window of ``window`` (a MainWindow) for a session, and the texture analysis under
    :data:`TEXTURE_DATA`."""
    state = window.state
    out: dict[str, Any] = {k: v for k, v in state.ui_state.items() if k != TEXTURE_DATA}
    if TEXTURE_DATA in state.ui_state:
        out[TEXTURE_DATA] = state.ui_state[TEXTURE_DATA]
    out[MAIN] = {"tab": int(window.center_tabs.currentIndex())}
    out[VIEW3D] = view3d_state(window.view3d)
    post = getattr(window, "strain_window", None)
    if post is not None:
        out[POST] = post_state(post)
    texture = getattr(window, "texture_window", None)
    if texture is not None:
        out[TEXTURE], out[TEXTURE_DATA] = texture_state(texture)
    return out


def restore(window) -> None:
    """Put the settings a session brought (``state.ui_state``) into the windows that exist; the post-processing and
    texture windows the session had open are opened (and restored when they are created)."""
    state = window.state
    ui = state.ui_state
    main = ui.get(MAIN) if isinstance(ui.get(MAIN), dict) else {}
    tab = main.get("tab")
    if isinstance(tab, int) and 0 <= tab < window.center_tabs.count():
        window.center_tabs.setCurrentIndex(tab)
    restore_view3d(window.view3d, ui.get(VIEW3D))
    post = getattr(window, "strain_window", None)
    if post is not None:
        restore_post(post, ui.pop(POST, None))
    elif isinstance(ui.get(POST), dict) and ui[POST].get("open") is True:
        window.open_strain_window()  # restored as it is created
    texture = getattr(window, "texture_window", None)
    if texture is not None:
        restore_texture(texture, ui.pop(TEXTURE, None), ui.pop(TEXTURE_DATA, None))
    elif isinstance(ui.get(TEXTURE), dict) and ui[TEXTURE].get("open") is True:
        window.open_texture_window()


def take_post(state) -> dict | None:
    """The post-processing settings waiting for their window (removed: they are applied once)."""
    value = state.ui_state.pop(POST, None)
    return value if isinstance(value, dict) else None


def take_texture(state) -> tuple[dict | None, dict | None]:
    settings = state.ui_state.pop(TEXTURE, None)
    data = state.ui_state.pop(TEXTURE_DATA, None)
    return (settings if isinstance(settings, dict) else None), (data if isinstance(data, dict) else None)
