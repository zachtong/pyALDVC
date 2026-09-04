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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from al_dvc.core.data_structures import PipelineResult
from al_dvc.export.export_utils import DISP_FIELDS, STD_FIELDS, STRAIN_FIELDS
from al_dvc.export.slice_plots import FIELD_LABELS, LAYOUTS
from al_dvc.strain.compute_strain import compute_strain

from .app_state import AppState
from .field_canvas import FieldSliceCanvas
from .widgets import CollapsibleSection, combo, dspin, form_label, guard_wheel, headless, make_form, spin

logger = logging.getLogger(__name__)

STRAIN_METHODS = ("plane_fit", "fem", "fd", "direct")
STRAIN_TYPES = ("infinitesimal", "green_lagrange", "euler_almansi", "hencky")
COLORMAPS = ("turbo", "viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r", "jet", "gray")
SIDEBAR_WIDTH = 330

__all__ = ["StrainWindow", "STRAIN_METHODS", "STRAIN_TYPES"]


class StrainCancelled(Exception):
    pass


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
        self.method = combo(list(STRAIN_METHODS))
        self.measure = combo(list(STRAIN_TYPES))
        self.halfwidth = spin(1, 4, 1)
        self.disp_smoothing = dspin(0.0, 10.0, 2)
        self.strain_smoothing = dspin(0.0, 10.0, 2)
        self.edge_trim = QCheckBox()
        for key, w in [
            ("method", self.method),
            ("measure", self.measure),
            ("halfwidth", self.halfwidth),
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
        self.show_volume = QCheckBox()
        self.show_volume.setChecked(True)
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
        for w in (self.method, self.measure):
            w.currentIndexChanged.connect(lambda _i: self._mark_stale())
        for w in (self.halfwidth, self.disp_smoothing, self.strain_smoothing):
            w.valueChanged.connect(lambda _v: self._mark_stale())
        self.edge_trim.toggled.connect(lambda _v: self._mark_stale())
        self._btn_export.clicked.connect(self.export_requested.emit)
        self._btn_png.clicked.connect(self._on_save_png)
        self._state.results_changed.connect(self._on_results_changed)
        self.retranslate_ui()
        self._load_params()
        self._load_data()

    # ------------------------------------------------------------------ parameters
    def _load_params(self) -> None:
        p = self._state.para
        self._updating = True
        try:
            self.method.setCurrentText(p.strain_method)
            self.measure.setCurrentText(p.strain_type)
            self.halfwidth.setValue(int(p.strain_plane_fit_halfwidth[0]))
            self.disp_smoothing.setValue(float(p.disp_smoothing))
            self.strain_smoothing.setValue(float(p.strain_smoothing))
            self.edge_trim.setChecked(bool(p.strain_edge_trim))
        finally:
            self._updating = False

    def strain_para(self):
        """The run's parameters with this window's strain settings."""
        hw = int(self.halfwidth.value())
        return replace(
            self._state.para,
            strain_method=self.method.currentText(),
            strain_type=self.measure.currentText(),
            strain_plane_fit_halfwidth=(hw, hw, hw),
            disp_smoothing=float(self.disp_smoothing.value()),
            strain_smoothing=float(self.strain_smoothing.value()),
            strain_edge_trim=bool(self.edge_trim.isChecked()),
        )

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
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_strain.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._btn_compute.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)
        self._status.setText(self.tr("Computing strain..."))
        self._state.log(
            self.tr("Strain: {method}, {measure}, window {w}").format(
                method=para.strain_method, measure=para.strain_type, w=2 * int(self.halfwidth.value()) + 1
            )
        )
        self._worker.start()

    def cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()

    def wait(self, timeout_ms: int = 300_000) -> bool:
        """Block until the worker finishes (tests)."""
        return self._worker.wait(timeout_ms) if self._worker is not None else True

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(round(1000 * fraction)))
        self._status.setText(self.tr("Computing strain: {msg}").format(msg=message))

    def _on_finished(self, strains) -> None:
        res = self._state.results
        if res is None:
            return
        para = self.strain_para()
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
        self._stale = False
        self._btn_compute.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setValue(1000)
        self._state.set_results(new)  # results_changed -> every view (including this one) refreshes
        want = self.field.findData("exx")
        if want >= 0:
            self.field.setCurrentIndex(want)
        self._state.log(self.tr("Strain computed for {n} frame(s)").format(n=len(strains)), "success")

    def _on_failed(self, message: str, detail: str) -> None:
        self._btn_compute.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._status.setText(message)
        self._state.log(self.tr("Strain failed: {msg}").format(msg=message), "error")
        self._state.log(detail, "debug")

    def _on_cancelled(self) -> None:
        self._btn_compute.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setValue(0)
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
                self.field.addItem(FIELD_LABELS.get(name, name), name)
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
        layout = str(self.layout_combo.currentData() or "row")
        self.canvas.set_view(
            frame=int(self.frame_slider.value()), field=field, cmap=self.colormap.currentText(), clim=clim, layout=layout
        )
        n = self.frame_slider.maximum() + 1
        self._frame_label.setText(self.tr("Frame {k}/{n}").format(k=self.frame_slider.value() + 1, n=n))
        self._btn_prev.setEnabled(self.frame_slider.value() > 0)
        self._btn_next.setEnabled(self.frame_slider.value() < self.frame_slider.maximum())

    def _update_status(self) -> None:
        res = self._state.results
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
                    method=sr.method, measure=sr.strain_type, n=valid
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
            out = self.canvas.save_png(path)
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
            "halfwidth": self.tr("Plane-fit half-width [nodes]"),
            "disp_smoothing": self.tr("Displacement smoothing [nodes]"),
            "strain_smoothing": self.tr("Strain smoothing [nodes]"),
            "field": self.tr("Field"),
            "colormap": self.tr("Colormap"),
            "auto": self.tr("Auto range"),
            "vmin": self.tr("Min"),
            "vmax": self.tr("Max"),
            "layout": self.tr("Layout"),
        }
        for key, label in self.labels.items():
            label.setText(texts[key])
        self.labels["method"].setToolTip(
            self.tr(
                "plane_fit: least-squares gradient over a window of nodes (robust, smooth);\n"
                "fem: shape-function gradient of the hex mesh; fd: central differences;\n"
                "direct: the gradient the ADMM solver produced (reference frame 0 only)."
            )
        )
        self.edge_trim.setText(self.tr("Trim nodes with an incomplete fitting window"))
        self._btn_compute.setText(self.tr("Compute strain"))
        self._btn_cancel.setText(self.tr("Cancel"))
        self._btn_prev.setText("<")
        self._btn_next.setText(">")
        self._btn_export.setText(self.tr("Export results..."))
        self._btn_png.setText(self.tr("Save image as PNG..."))
        for i, text in enumerate((self.tr("Row"), self.tr("Column"), self.tr("2 x 2"))):
            self.layout_combo.setItemText(i, text)
        self.show_volume.setText(self.tr("Show the reference volume under the field"))
        self.canvas.set_empty_text(self.tr("No field to show."))
        self._update_status()
