"""Strain post-processing window (pyALDIC's ``StrainWindow`` for volumes).

An independent ``QMainWindow`` that takes the displacement results of the main window, computes
strain on demand with its own parameters (method, measure, smoothing, plane-fit window) on a
worker thread, shows displacement and strain fields on a private three-plane canvas, and writes
the strain back into ``AppState.results`` through ``dataclasses.replace`` so the main viewer and
the exports see it too. It never touches the main window's display settings.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dvc.core.data_structures import PipelineResult
from al_dvc.export.export_utils import DISP_FIELDS, STD_FIELDS, STRAIN_FIELDS
from al_dvc.export.slice_plots import LAYOUTS
from al_dvc.strain.compute_strain import compute_strain

from .app_state import AppState
from .field_canvas import FieldSliceCanvas
from .names import field_name, fill_combo, label, retranslate_combo, select_key
from .widgets import CollapsibleSection, combo, dspin, form_label, guard_wheel, headless, make_form

logger = logging.getLogger(__name__)

STRAIN_METHODS = ("plane_fit", "fem", "fd", "direct")
STRAIN_TYPES = ("infinitesimal", "green_lagrange", "euler_almansi", "hencky")
COLORMAPS = ("turbo", "viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r", "jet", "gray")
SIDEBAR_WIDTH = 330

__all__ = ["StrainWindow", "STRAIN_METHODS", "STRAIN_TYPES"]


class StrainCancelled(Exception):
    pass


def _odd_spin() -> QSpinBox:
    """One axis of the fit window: an odd width in nodes (3, 5, 7, 9)."""
    w = QSpinBox()
    w.setRange(3, 9)
    w.setSingleStep(2)
    w.setFixedWidth(52)
    return w


class _StrainWorker(QThread):
    """Compute the strain of every frame off the UI thread; cancellable between frames."""

    progress = Signal(float, str)
    finished_strain = Signal(object)  # list[StrainResult]
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, result: PipelineResult, para, parent=None) -> None:
        super().__init__(parent)
        self._result = result
        self._para = para
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        res = self._result
        mesh = res.dvc_mesh
        strains = []
        try:
            n = len(res.result_disp)
            ops = None
            for i, fr in enumerate(res.result_disp):
                if self._stop:
                    raise StrainCancelled()
                self.progress.emit(i / n, f"frame {i + 1}/{n}")
                U_acc = fr.U_accum if fr.U_accum is not None else fr.U
                F_direct = fr.F if fr.ref_frame == 0 else None
                sr = compute_strain(mesh, self._para, U_acc, F_direct=F_direct, valid=mesh.node_valid, ops=ops)
                strains.append(sr)
            if self._stop:  # a cancel that arrived during the last frame is honoured, not published
                raise StrainCancelled()
            self.progress.emit(1.0, "done")
        except StrainCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # surface to the UI
            logger.exception("Strain computation failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        self.finished_strain.emit(strains)


class StrainWindow(QMainWindow):
    """Strain parameters, compute / cancel, field display and export of the strain of a result."""

    export_requested = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: _StrainWorker | None = None
        self._job: tuple | None = None  # (result, parameters) the running worker was given
        self._stale = False
        self._updating = False
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(1200, 760)

        # ---- canvas (left)
        self.canvas = FieldSliceCanvas()
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        left = QVBoxLayout()
        left.addWidget(self.canvas, 1)
        nav = QHBoxLayout()
        self._btn_prev = QPushButton()
        self._btn_prev.setFixedWidth(36)
        self._btn_next = QPushButton()
        self._btn_next.setFixedWidth(36)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self._frame_label = QLabel()
        self._frame_label.setObjectName("sectionTitle")
        nav.addWidget(self._btn_prev)
        nav.addWidget(self.frame_slider, 1)
        nav.addWidget(self._btn_next)
        nav.addWidget(self._frame_label)
        left.addLayout(nav)
        root.addLayout(left, 1)

        # ---- sidebar (right)
        side = QWidget()
        side.setFixedWidth(SIDEBAR_WIDTH)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(4)
        self.sections: dict[str, CollapsibleSection] = {}
        self.labels: dict[str, QLabel] = {}

        params = CollapsibleSection()
        form = make_form()
        self.method = combo([])
        fill_combo(self.method, "strain_method")
        self.measure = combo([])
        fill_combo(self.measure, "strain_type")
        # fit window: full width in nodes per axis (x, y, z); the solver takes the half-width (width - 1) / 2
        self.fit_window_axes = [_odd_spin() for _ in range(3)]
        self.fit_window = self.fit_window_axes[0]  # x; drives the three while the lock is on
        self.fit_window_lock = QCheckBox()
        self.fit_window_lock.setChecked(True)
        fit_row = QWidget()
        fit_layout = QHBoxLayout(fit_row)
        fit_layout.setContentsMargins(0, 0, 0, 0)
        fit_layout.setSpacing(4)
        for k, w in enumerate(self.fit_window_axes):
            if k:
                fit_layout.addWidget(QLabel("x"))
            fit_layout.addWidget(w)
        fit_layout.addSpacing(6)
        fit_layout.addWidget(self.fit_window_lock)
        fit_layout.addStretch(1)
        self.disp_smoothing = dspin(0.0, 10.0, 1)
        self.strain_smoothing = dspin(0.0, 10.0, 1)
        for w in (self.disp_smoothing, self.strain_smoothing):
            w.setSingleStep(0.5)  # a Gaussian sigma below half a node spacing changes nothing
        self.edge_trim = QCheckBox()
        for key, w in [
            ("method", self.method),
            ("measure", self.measure),
            ("halfwidth", fit_row),
            ("disp_smoothing", self.disp_smoothing),
            ("strain_smoothing", self.strain_smoothing),
        ]:
            label = form_label()
            self.labels[key] = label
            form.addRow(label, w)
        params.add_layout(form)
        params.add_widget(self.edge_trim)
        self._btn_compute = QPushButton()
        self._btn_compute.setProperty("class", "btn-primary")
        self._btn_compute.setMinimumHeight(32)
        self._btn_cancel = QPushButton()
        self._btn_cancel.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self._btn_compute, 2)
        row.addWidget(self._btn_cancel, 1)
        params.add_layout(row)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        params.add_widget(self._progress)
        self._status = QLabel()
        self._status.setObjectName("hint")
        self._status.setWordWrap(True)
        params.add_widget(self._status)
        self.sections["params"] = params
        side_layout.addWidget(params)

        display = CollapsibleSection()
        dform = make_form()
        self.field = combo([])
        self.colormap = combo(list(COLORMAPS))
        self.auto_range = QCheckBox()
        self.auto_range.setChecked(True)
        self.vmin = dspin(-1e9, 1e9, 4)
        self.vmax = dspin(-1e9, 1e9, 4)
        self.vmin.setEnabled(False)
        self.vmax.setEnabled(False)
        self.layout_combo = combo([])
        for key in LAYOUTS:
            self.layout_combo.addItem(key, key)
        self.layout_combo.setCurrentIndex(LAYOUTS.index("grid"))
        self.show_volume = QCheckBox()
        self.show_volume.setChecked(True)
        self.equal_scale = QCheckBox()
        for key, w in [
            ("field", self.field),
            ("colormap", self.colormap),
            ("auto", self.auto_range),
            ("vmin", self.vmin),
            ("vmax", self.vmax),
            ("layout", self.layout_combo),
        ]:
            label = form_label()
            self.labels[key] = label
            dform.addRow(label, w)
        display.add_layout(dform)
        display.add_widget(self.show_volume)
        display.add_widget(self.equal_scale)
        self.sections["display"] = display
        side_layout.addWidget(display)

        actions = CollapsibleSection()
        self._btn_export = QPushButton()
        self._btn_export.setEnabled(False)
        self._btn_png = QPushButton()
        actions.add_widget(self._btn_export)
        actions.add_widget(self._btn_png)
        self.sections["export"] = actions
        side_layout.addWidget(actions)
        side_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setFixedWidth(SIDEBAR_WIDTH + 18)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(scroll, 0)
        self.setCentralWidget(central)
        guard_wheel(side)

        # ---- wiring
        self._btn_compute.clicked.connect(self.compute)
        self._btn_cancel.clicked.connect(self.cancel)
        self._btn_prev.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() - 1))
        self._btn_next.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() + 1))
        self.frame_slider.valueChanged.connect(self._on_frame)
        self.field.currentIndexChanged.connect(lambda _i: self._on_display_changed())
        self.colormap.currentTextChanged.connect(lambda _v: self._on_display_changed())
        self.auto_range.toggled.connect(self._on_auto)
        self.vmin.valueChanged.connect(lambda _v: self._on_display_changed())
        self.vmax.valueChanged.connect(lambda _v: self._on_display_changed())
        self.layout_combo.currentIndexChanged.connect(lambda _i: self._on_display_changed())
        self.show_volume.toggled.connect(lambda _v: self._load_data())
        self.equal_scale.toggled.connect(lambda _v: self._on_display_changed())
        for w in (self.method, self.measure):
            w.currentIndexChanged.connect(lambda _i: self._mark_stale())
        for k, w in enumerate(self.fit_window_axes):
            w.valueChanged.connect(lambda v, axis=k: self._on_fit_window(axis, v))
        self.fit_window_lock.toggled.connect(self._on_fit_window_lock)
        for w in (self.disp_smoothing, self.strain_smoothing):
            w.valueChanged.connect(lambda _v: self._mark_stale())
        self.edge_trim.toggled.connect(lambda _v: self._mark_stale())
        self._btn_export.clicked.connect(self.export_requested.emit)
        self._btn_png.clicked.connect(self._on_save_png)
        self._state.results_changed.connect(self._on_results_changed)
        self.retranslate_ui()
        self._load_params()
        self._load_data()

    # ------------------------------------------------------------------ parameters
    def _source_para(self):
        """The parameters the strain settings start from: the result's own (the run that produced the
        displacement), else the next run's."""
        res = self._state.results
        return res.dvc_para if res is not None else self._state.para

    def _load_params(self) -> None:
        p = self._source_para()
        self._updating = True
        try:
            select_key(self.method, p.strain_method)
            select_key(self.measure, p.strain_type)
            widths = [2 * int(h) + 1 for h in p.strain_plane_fit_halfwidth]
            if len(set(widths)) > 1:
                self.fit_window_lock.setChecked(False)  # a non-cubic window unlocks the axes
            for w, width in zip(self.fit_window_axes, widths):
                w.setValue(width)
            self.disp_smoothing.setValue(float(p.disp_smoothing))
            self.strain_smoothing.setValue(float(p.strain_smoothing))
            self.edge_trim.setChecked(bool(p.strain_edge_trim))
        finally:
            self._updating = False

    def strain_para(self):
        """The run's parameters with this window's strain settings."""
        hw = tuple((int(w.value()) - 1) // 2 for w in self.fit_window_axes)
        return replace(
            self._source_para(),
            strain_method=str(self.method.currentData()),
            strain_type=str(self.measure.currentData()),
            strain_plane_fit_halfwidth=hw,
            disp_smoothing=float(self.disp_smoothing.value()),
            strain_smoothing=float(self.strain_smoothing.value()),
            strain_edge_trim=bool(self.edge_trim.isChecked()),
        )

    STRAIN_FIELDS = (
        "strain_method",
        "strain_type",
        "strain_plane_fit_halfwidth",
        "disp_smoothing",
        "strain_smoothing",
        "strain_edge_trim",
    )

    def _controls_differ(self, para) -> bool:
        """True when the controls no longer describe ``para`` (the settings a strain was computed with)."""
        try:
            now = self.strain_para()
        except (ValueError, TypeError):
            return True
        return any(getattr(now, f) != getattr(para, f) for f in self.STRAIN_FIELDS)

    def _set_controls_enabled(self, enabled: bool) -> None:
        controls = (self.method, self.measure, *self.fit_window_axes, self.fit_window_lock)
        for w in (*controls, self.disp_smoothing, self.strain_smoothing, self.edge_trim):
            w.setEnabled(enabled)

    def _on_fit_window(self, axis: int, value: int) -> None:
        """Odd widths only; with the lock on, one box drives the three."""
        v = int(value)
        if v % 2 == 0:
            self.fit_window_axes[axis].setValue(v + 1)  # re-enters with the odd value
            return
        if self._updating:
            return
        if self.fit_window_lock.isChecked():
            for j, w in enumerate(self.fit_window_axes):
                if j != axis and w.value() != v:
                    w.blockSignals(True)
                    w.setValue(v)
                    w.blockSignals(False)
        self._mark_stale()

    def _on_fit_window_lock(self, locked: bool) -> None:
        if locked and not self._updating:
            self._on_fit_window(0, self.fit_window_axes[0].value())

    def _mark_stale(self) -> None:
        if self._updating:
            return
        self._stale = self._state.results is not None and bool(self._state.results.result_strain)
        self._update_status()

    @property
    def is_stale(self) -> bool:
        """True when strain exists but the parameters changed since it was computed."""
        return self._stale

    # ------------------------------------------------------------------ compute
    def compute(self) -> None:
        res = self._state.results
        if res is None or not res.result_disp or (self._worker is not None and self._worker.isRunning()):
            return
        try:
            para = self.strain_para()
        except (ValueError, TypeError) as exc:
            self._state.log(self.tr("Strain parameters: {error}").format(error=exc), "error")
            return
        self._worker = _StrainWorker(res, para, parent=self)
        self._job = (res, para)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_strain.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._btn_compute.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._set_controls_enabled(False)  # the settings of a running computation are fixed
        self._progress.setValue(0)
        self._status.setText(self.tr("Computing strain..."))
        self._state.log(
            self.tr("Strain: {method}, {measure}, window {w}").format(
                method=label("strain_method", para.strain_method),
                measure=label("strain_type", para.strain_type),
                w=" x ".join(str(int(w.value())) for w in self.fit_window_axes),
            )
        )
        self._worker.start()

    def cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()

    def wait(self, timeout_ms: int = 300_000) -> bool:
        """Block until the worker finishes (tests)."""
        return self._worker.wait(timeout_ms) if self._worker is not None else True

    def shutdown(self, timeout_ms: int = 30_000) -> bool:
        """Cancel a running computation and wait for its thread (application exit); True when settled."""
        if not self._is_running():
            return True
        self._worker.cancel()
        return self._worker.wait(timeout_ms)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Closing the window while computing cancels the computation (asked first, unless headless)."""
        if self._is_running():
            if not headless():
                answer = QMessageBox.question(
                    self, self.tr("Strain computation running"), self.tr("Cancel the strain computation and close the window?")
                )
                if answer != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            self.cancel()
        super().closeEvent(event)

    def _settle(self, progress: int | None = None) -> None:
        """Terminal UI state after success, failure, cancellation or a discarded completion."""
        self._job = None
        self._btn_cancel.setEnabled(False)
        self._set_controls_enabled(True)
        res = self._state.results
        self._btn_compute.setEnabled(res is not None and bool(res.result_disp))
        if progress is not None:
            self._progress.setValue(progress)

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(round(1000 * fraction)))
        self._status.setText(self.tr("Computing strain: {msg}").format(msg=message))

    def _on_finished(self, strains) -> None:
        job = self._job
        res, para = job if job is not None else (None, None)
        if res is None or self._state.results is not res:
            # the displacement result was replaced (new run, session, import) while this strain was computed:
            # it describes another result and is not spliced into the current one
            self._settle(0)
            self._status.setText(self.tr("Discarded: the displacement result changed while the strain was computed."))
            self._state.log(
                self.tr("Strain discarded: the displacement result changed while it was computed; compute again."),
                "warning",
            )
            return
        new = replace(
            res,
            result_strain=list(strains),
            dvc_para=replace(
                res.dvc_para,
                **{
                    "strain_method": para.strain_method,
                    "strain_type": para.strain_type,
                    "strain_plane_fit_halfwidth": para.strain_plane_fit_halfwidth,
                    "disp_smoothing": para.disp_smoothing,
                    "strain_smoothing": para.strain_smoothing,
                    "strain_edge_trim": para.strain_edge_trim,
                },
            ),
        )
        self._stale = self._controls_differ(para)  # edited during the computation: the strain is from ``para``
        self._settle(1000)
        self._state.set_results(new)  # results_changed -> every view (including this one) refreshes
        want = self.field.findData("exx")
        if want >= 0:
            self.field.setCurrentIndex(want)
        self._state.log(self.tr("Strain computed for {n} frame(s)").format(n=len(strains)), "success")

    def _on_failed(self, message: str, detail: str) -> None:
        self._settle()
        self._status.setText(message)
        self._state.log(self.tr("Strain failed: {msg}").format(msg=message), "error")
        self._state.log(detail, "debug")

    def _on_cancelled(self) -> None:
        self._settle(0)
        self._status.setText(self.tr("Cancelled."))

    # ------------------------------------------------------------------ display
    def _available_fields(self) -> list[str]:
        res = self._state.results
        if res is None or not res.result_disp:
            return []
        fields = list(DISP_FIELDS)
        if res.result_disp[0].U_std is not None:
            fields += list(STD_FIELDS)
        if res.result_strain:
            fields += list(STRAIN_FIELDS) + ["det_F", "rotation_deg"]
        return fields

    def _on_results_changed(self) -> None:
        if not self._is_running():
            self._stale = False
            self._load_params()  # the settings of the result on screen (its run, or the strain it carries)
        self._load_data()

    def _load_data(self) -> None:
        res = self._state.results
        background = None
        if res is not None and self.show_volume.isChecked() and self._state.volumes:
            try:
                vol = self._state.volume_array(0)
                if tuple(vol.shape) == tuple(res.volume_shape):
                    background = vol
            except Exception as exc:
                self._state.log(f"strain window: cannot load the reference volume: {exc}", "warning")
        fields = self._available_fields()
        current = self.field.currentData()
        self._updating = True
        try:
            self.field.clear()
            for name in fields:
                self.field.addItem(field_name(name), name)
            idx = self.field.findData(current) if current else -1
            self.field.setCurrentIndex(idx if idx >= 0 else 0)
            n = len(res.result_disp) if res is not None else 0
            self.frame_slider.setRange(0, max(0, n - 1))
        finally:
            self._updating = False
        self.canvas.set_data(res, background)
        self._btn_export.setEnabled(bool(res is not None and res.result_strain))
        self._btn_compute.setEnabled(res is not None and bool(res.result_disp) and not self._is_running())
        self._on_display_changed()
        self._update_status()

    def _is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _on_frame(self, value: int) -> None:
        self._on_display_changed()

    def _on_auto(self, auto: bool) -> None:
        self.vmin.setEnabled(not auto)
        self.vmax.setEnabled(not auto)
        if not auto and self.canvas.last_clim is not None:
            lo, hi = self.canvas.last_clim
            self._updating = True
            try:
                self.vmin.setValue(lo)
                self.vmax.setValue(hi)
            finally:
                self._updating = False
        self._on_display_changed()

    def _on_display_changed(self) -> None:
        if self._updating:
            return
        field = self.field.currentData() or "disp_magnitude"
        clim = None if self.auto_range.isChecked() else (float(self.vmin.value()), float(self.vmax.value()))
        layout = str(self.layout_combo.currentData() or "grid")
        self.canvas.set_view(
            frame=int(self.frame_slider.value()),
            field=field,
            cmap=self.colormap.currentText(),
            clim=clim,
            layout=layout,
            equal_scale=self.equal_scale.isChecked(),
        )
        n = self.frame_slider.maximum() + 1
        self._frame_label.setText(self.tr("Frame {k}/{n}").format(k=self.frame_slider.value() + 1, n=n))
        self._btn_prev.setEnabled(self.frame_slider.value() > 0)
        self._btn_next.setEnabled(self.frame_slider.value() < self.frame_slider.maximum())

    def _update_status(self) -> None:
        if self._is_running():
            return  # the progress messages own the status line
        res = self._state.results
        self._btn_export.setToolTip(
            self.tr("The strain shown was computed with previous settings; compute again before exporting") if self._stale else ""
        )
        if res is None or not res.result_disp:
            self._status.setText(self.tr("No displacement results yet: run an analysis first."))
        elif not res.result_strain:
            self._status.setText(self.tr("Displacement fields are available; press Compute strain."))
        elif self._stale:
            self._status.setText(self.tr("Parameters changed: the shown strain is from the previous settings."))
        else:
            sr = res.result_strain[0]
            valid = int(np.asarray(sr.strain_valid).sum()) if getattr(sr, "strain_valid", None) is not None else 0
            self._status.setText(
                self.tr("Strain: {method}, {measure}; {n} valid nodes in frame 1").format(
                    method=label("strain_method", sr.method), measure=label("strain_type", sr.strain_type), n=valid
                )
            )

    # ------------------------------------------------------------------ export
    def _on_save_png(self) -> None:
        default = str(self._state.output_dir / f"strain_{self.canvas.field}_frame_{self.canvas.frame + 1}.png")
        if headless():
            path = default
        else:
            path, _ = QFileDialog.getSaveFileName(self, self.tr("Save image"), default, "PNG (*.png)")
        if path:
            try:
                out = self.canvas.save_png(path)
            except Exception as exc:  # unwritable folder, bad name
                self._status.setText(self.tr("Cannot save the image: {error}").format(error=exc))
                self._state.log(self.tr("Cannot save the image: {error}").format(error=exc), "error")
                return
            self._state.log(self.tr("Image saved: {path}").format(path=out))

    # ------------------------------------------------------------------ misc
    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("Strain post-processing"))
        self.sections["params"].set_title(self.tr("Strain parameters"))
        self.sections["display"].set_title(self.tr("Display"))
        self.sections["export"].set_title(self.tr("Export"))
        texts = {
            "method": self.tr("Method"),
            "measure": self.tr("Strain measure"),
            "halfwidth": self.tr("Fit window [nodes, x y z]"),
            "disp_smoothing": self.tr("Smooth displacement [sigma, nodes]"),
            "strain_smoothing": self.tr("Smooth strain [sigma, nodes]"),
            "field": self.tr("Field"),
            "colormap": self.tr("Colormap"),
            "auto": self.tr("Auto range"),
            "vmin": self.tr("Min"),
            "vmax": self.tr("Max"),
            "layout": self.tr("Layout"),
        }
        for key, widget_label in self.labels.items():
            widget_label.setText(texts[key])
        retranslate_combo(self.method, "strain_method")
        retranslate_combo(self.measure, "strain_type")
        for i in range(self.field.count()):
            self.field.setItemText(i, field_name(self.field.itemData(i)))
        self.labels["method"].setToolTip(
            self.tr(
                "Plane fitting: least-squares gradient over a window of nodes (robust, smooth).\n"
                "Finite elements: shape-function gradient of the hexahedral mesh. Finite differences: central differences.\n"
                "Solver gradient: the gradient the AL-DVC solver produced (reference frame 0 only)."
            )
        )
        self.labels["halfwidth"].setToolTip(
            self.tr(
                "The gradient at a node is a least-squares fit over the nodes of this window around it "
                "(3 x 3 x 3: the direct neighbours). Larger windows smooth more and lose resolution (plane fitting only)."
            )
        )
        self.fit_window_lock.setText(self.tr("Cube"))
        self.fit_window_lock.setToolTip(self.tr("Keep the fit window cubic: one width for x, y and z"))
        self.labels["disp_smoothing"].setToolTip(
            self.tr(
                "Gaussian smoothing of the displacement field before it is differentiated; sigma in node spacings. "
                "0 = off; 1 blends each node with its neighbours; below 0.5 nothing changes."
            )
        )
        self.labels["strain_smoothing"].setToolTip(
            self.tr(
                "Gaussian smoothing of the strain field after differentiation; sigma in node spacings. "
                "0 = off; below 0.5 nothing changes."
            )
        )
        self.edge_trim.setText(self.tr("Discard nodes whose fit window is incomplete"))
        self.edge_trim.setToolTip(
            self.tr(
                "At the edge of the region and next to masked-out nodes the fit window lacks neighbours. Ticked: those "
                "nodes get no strain (blank). Unticked: strain from the neighbours that exist, less reliable."
            )
        )
        self._btn_compute.setText(self.tr("Compute strain"))
        self._btn_cancel.setText(self.tr("Cancel"))
        self._btn_prev.setText("<")
        self._btn_next.setText(">")
        self._btn_export.setText(self.tr("Export results..."))
        self._btn_png.setText(self.tr("Save image as PNG..."))
        for i, text in enumerate((self.tr("Row"), self.tr("Column"), self.tr("2 x 2"))):
            self.layout_combo.setItemText(i, text)
        self.show_volume.setText(self.tr("Show the reference volume under the field"))
        self.equal_scale.setText(self.tr("Same scale on the three planes"))
        self.canvas.set_empty_text(self.tr("No field to show."))
        self._update_status()
