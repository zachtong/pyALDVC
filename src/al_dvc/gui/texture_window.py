"""Texture analysis window: correlation lengths of the reference volume and a subset suggestion.

An independent ``QMainWindow`` like the strain window. It analyses the reference volume (inside
the region of interest when one exists) on a worker thread, shows the directional and radial
correlation profiles, the lengths at the three thresholds, the size sweep and the parameter
suggestion, and can write the suggestion into the run parameters or export the numbers.
"""

from __future__ import annotations

import csv
import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from al_dvc.texture import (
    THRESHOLD_LABELS,
    THRESHOLDS,
    SizeSweep,
    TextureResult,
    analyse_texture,
    recommend_parameters,
    size_schedule,
    sweep_sizes,
)

from .app_state import AppState
from .names import fill_combo, retranslate_combo
from .theme import COLORS
from .widgets import CollapsibleSection, combo, dspin, form_label, guard_wheel, headless, make_form, spin

logger = logging.getLogger(__name__)

SIDEBAR_WIDTH = 340
PROFILE_STYLES = {"x": ("C0", "-"), "y": ("C1", "--"), "z": ("C2", ":"), "radial": ("C3", "-.")}
AXES_ROWS = ("x", "y", "z", "radial")

__all__ = ["TextureWindow"]


class _Cancelled(Exception):
    pass


class _TextureWorker(QThread):
    """Analyse the volume (and optionally sweep the sizes) off the UI thread."""

    progress = Signal(float, str)
    finished_analysis = Signal(object, object)  # TextureResult, SizeSweep | None
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, vol, mask, spacing, settings: dict, sweep: dict | None, parent=None) -> None:
        super().__init__(parent)
        self._vol = vol
        self._mask = mask
        self._spacing = spacing
        self._settings = settings
        self._sweep = sweep
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            self.progress.emit(0.0, "autocorrelation")
            result = analyse_texture(self._vol, self._spacing, self._mask, **self._settings)
            sweep = None
            if self._sweep is not None:
                if self._stop:
                    raise _Cancelled()
                sweep = sweep_sizes(
                    self._vol,
                    self._mask,
                    spacing=self._spacing,
                    estimator=self._settings["estimator"],
                    min_overlap=self._settings["min_overlap"],
                    progress=lambda f, m: self.progress.emit(0.1 + 0.9 * f, m),
                    stop=lambda: self._stop,
                    **self._sweep,
                )
                if self._stop:
                    raise _Cancelled()
            if self._stop:  # a cancel during the analysis itself is honoured, not published
                raise _Cancelled()
            self.progress.emit(1.0, "done")
        except _Cancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # surface to the UI
            logger.exception("Texture analysis failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        self.finished_analysis.emit(result, sweep)


class TextureWindow(QMainWindow):
    """Parameters, analyse / cancel, profiles, lengths, size sweep, suggestion and export."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: _TextureWorker | None = None
        self.result: TextureResult | None = None
        self.sweep: SizeSweep | None = None
        self.recommendation = None
        self._job_source: dict | None = None  # the input the running analysis was given
        self._result_source: dict | None = None  # the input ``result`` describes
        self._previous_note = ""  # why the previous result is still on screen (failed / cancelled rerun)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(1240, 780)

        # ---- figures (left)
        self.tabs = QTabWidget()
        self.fig_profiles = Figure(figsize=(7, 5), facecolor=COLORS.BG_CANVAS)
        self.canvas_profiles = FigureCanvas(self.fig_profiles)
        self.fig_sweep = Figure(figsize=(7, 5), facecolor=COLORS.BG_CANVAS)
        self.canvas_sweep = FigureCanvas(self.fig_sweep)
        self.tabs.addTab(self.canvas_profiles, "")
        self.tabs.addTab(self.canvas_sweep, "")
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(self.tabs, 1)

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
        self.estimator = combo([])
        fill_combo(self.estimator, "estimator")
        self.max_lag = spin(0, 512, 4)  # 0 = half the window
        self.min_overlap = dspin(0.05, 1.0, 2)
        self.min_overlap.setValue(0.5)
        self.window_edge = spin(32, 1024, 32)
        self.window_edge.setValue(256)
        self.use_roi = QCheckBox()
        self.use_roi.setChecked(True)
        for key, w in [
            ("estimator", self.estimator),
            ("max_lag", self.max_lag),
            ("min_overlap", self.min_overlap),
            ("window_edge", self.window_edge),
        ]:
            lab = form_label()
            self.labels[key] = lab
            form.addRow(lab, w)
        params.add_layout(form)
        params.add_widget(self.use_roi)
        self._btn_analyse = QPushButton()
        self._btn_analyse.setProperty("class", "btn-primary")
        self._btn_analyse.setMinimumHeight(32)
        self._btn_cancel = QPushButton()
        self._btn_cancel.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self._btn_analyse, 2)
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

        sweep = CollapsibleSection(expanded=False)
        sform = make_form()
        self.sweep_start = spin(8, 512, 8)
        self.sweep_start.setValue(16)
        self.sweep_step = spin(4, 256, 4)
        self.sweep_step.setValue(16)
        self.sweep_count = spin(2, 20, 1)
        self.sweep_count.setValue(8)
        self.sweep_samples = spin(1, 16, 1)
        self.sweep_samples.setValue(4)
        for key, w in [
            ("sweep_start", self.sweep_start),
            ("sweep_step", self.sweep_step),
            ("sweep_count", self.sweep_count),
            ("sweep_samples", self.sweep_samples),
        ]:
            lab = form_label()
            self.labels[key] = lab
            sform.addRow(lab, w)
        sweep.add_layout(sform)
        self.run_sweep = QCheckBox()
        sweep.add_widget(self.run_sweep)
        self.sections["sweep"] = sweep
        side_layout.addWidget(sweep)

        lengths = CollapsibleSection()
        self.table = QTableWidget(len(AXES_ROWS), len(THRESHOLDS))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setFixedHeight(22 * (len(AXES_ROWS) + 1) + 6)
        lengths.add_widget(self.table)
        self._table_hint = QLabel()
        self._table_hint.setObjectName("hint")
        self._table_hint.setWordWrap(True)
        lengths.add_widget(self._table_hint)
        self.sections["lengths"] = lengths
        side_layout.addWidget(lengths)

        suggestion = CollapsibleSection()
        self._suggestion = QLabel()
        self._suggestion.setWordWrap(True)
        self._suggestion.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        suggestion.add_widget(self._suggestion)
        self._btn_apply = QPushButton()
        self._btn_apply.setEnabled(False)
        suggestion.add_widget(self._btn_apply)
        self.sections["suggestion"] = suggestion
        side_layout.addWidget(suggestion)

        export = CollapsibleSection()
        self._btn_csv = QPushButton()
        self._btn_json = QPushButton()
        self._btn_png = QPushButton()
        for b in (self._btn_csv, self._btn_json, self._btn_png):
            b.setEnabled(False)
            export.add_widget(b)
        self.sections["export"] = export
        side_layout.addWidget(export)
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
        self._btn_analyse.clicked.connect(self.analyse)
        self._btn_cancel.clicked.connect(self.cancel)
        self._btn_apply.clicked.connect(self.apply_recommendation)
        self._btn_csv.clicked.connect(self._on_save_csv)
        self._btn_json.clicked.connect(self._on_save_json)
        self._btn_png.clicked.connect(self._on_save_png)
        self._state.volumes_changed.connect(self._on_volumes_changed)
        self._state.mask_changed.connect(self._refresh_validity)
        self._state.params_changed.connect(self._refresh_validity)
        self.use_roi.toggled.connect(lambda _v: self._refresh_validity())
        self.estimator.currentIndexChanged.connect(lambda _i: self._refresh_validity())
        for w in (self.max_lag, self.min_overlap, self.window_edge):
            w.valueChanged.connect(lambda _v: self._refresh_validity())
        self.retranslate_ui()
        self._draw_profiles()
        self._draw_sweep()
        self._on_volumes_changed()

    # ------------------------------------------------------------------ data
    def _reference(self):
        """``(volume, mask)`` of the reference frame, the mask only when "use region" is on.

        Raises ``ValueError`` when the region is unusable (empty, or another shape than the volume): the
        analysis stops with a message instead of silently taking the whole volume."""
        if not self._state.volumes:
            return None, None
        vol = np.asarray(self._state.volume_array(0))
        mask = self._state.reference_mask() if self.use_roi.isChecked() else None
        if mask is not None and mask.shape != vol.shape:
            raise ValueError(
                self.tr("The region of interest ({m}) does not match the reference volume ({v}).").format(
                    m=" x ".join(str(n) for n in mask.shape[::-1]), v=" x ".join(str(n) for n in vol.shape[::-1])
                )
            )
        if mask is not None and not mask.any():
            raise ValueError(self.tr("The region of interest is empty: draw a region or untick the restriction."))
        return vol, mask

    def current_source(self) -> dict | None:
        """What an analysis started now would describe: reference identity, mask revision, calibration, settings."""
        st = self._state
        if not st.volumes:
            return None
        use_roi = bool(self.use_roi.isChecked())
        return {
            "uid": st.volumes[0].uid,
            "mask": st.mask_revision if use_roi else None,
            "use_roi": use_roi,
            "spacing": tuple(float(v) for v in st.para.voxel_size),
            "units": str(getattr(st.para, "units", "voxel") or "voxel"),
            "settings": self.settings(),
        }

    @property
    def is_stale(self) -> bool:
        """True when a result is shown but the reference, the region, the calibration or the settings changed."""
        return self.result is not None and self._result_source != self.current_source()

    def settings(self) -> dict:
        max_lag = int(self.max_lag.value())
        edge = int(self.window_edge.value())
        return {
            "max_lag": None if max_lag <= 0 else max_lag,
            "estimator": str(self.estimator.currentData() or "overlap"),
            "min_overlap": float(self.min_overlap.value()),
            "max_voxels": edge**3,
        }

    def sweep_settings(self, box: tuple[int, int, int]) -> dict:
        sizes = size_schedule(box, int(self.sweep_start.value()), int(self.sweep_step.value()), int(self.sweep_count.value()))
        return {"sizes": sizes, "samples_per_size": int(self.sweep_samples.value())}

    # ------------------------------------------------------------------ analyse
    def analyse(self) -> None:
        if self._is_running():
            return
        try:
            vol, mask = self._reference()
        except ValueError as exc:
            self._status.setText(str(exc))
            self._state.log(self.tr("Texture analysis: {error}").format(error=exc), "error")
            return
        if vol is None:
            self._status.setText(self.tr("Load a reference volume first."))
            return
        spacing = tuple(float(v) for v in self._state.para.voxel_size)
        sweep = None
        if self.run_sweep.isChecked():
            from al_dvc.texture import analysis_window

            win = analysis_window(vol.shape, mask, max_voxels=vol.size)
            box = tuple(s.stop - s.start for s in win)[::-1]  # (nx, ny, nz)
            try:
                sweep = self.sweep_settings(box)
            except ValueError as exc:
                self._status.setText(str(exc))
                return
        self._worker = _TextureWorker(vol, mask, spacing, self.settings(), sweep, parent=self)
        self._job_source = self.current_source()
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_analysis.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._btn_analyse.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        for b in (self._btn_apply, self._btn_csv, self._btn_json, self._btn_png):
            b.setEnabled(False)  # the previous result stays on screen but is not applied or exported meanwhile
        self._progress.setValue(0)
        self._status.setText(self.tr("Analysing the texture..."))
        self._state.log(
            self.tr("Texture analysis started ({estimator}, region: {roi})").format(
                estimator=self.estimator.currentText(), roi=self.tr("yes") if mask is not None else self.tr("no")
            )
        )
        self._worker.start()

    def cancel(self) -> None:
        if self._is_running():
            self._worker.cancel()

    def wait(self, timeout_ms: int = 600_000) -> bool:
        """Block until the worker finishes (tests)."""
        return self._worker.wait(timeout_ms) if self._worker is not None else True

    def shutdown(self, timeout_ms: int = 30_000) -> bool:
        """Cancel a running analysis and wait for its thread (application exit); True when settled."""
        if not self._is_running():
            return True
        self._worker.cancel()
        return self._worker.wait(timeout_ms)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Closing the window while analysing cancels the analysis (asked first, unless headless)."""
        if self._is_running():
            if not headless():
                answer = QMessageBox.question(
                    self, self.tr("Texture analysis running"), self.tr("Cancel the texture analysis and close the window?")
                )
                if answer != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            self.cancel()
        super().closeEvent(event)

    def _is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _settle(self) -> None:
        """Terminal UI state after success, failure or cancellation."""
        self._job_source = None
        self._btn_analyse.setEnabled(bool(self._state.volumes))
        self._btn_cancel.setEnabled(False)
        has = self.result is not None
        for b in (self._btn_csv, self._btn_json, self._btn_png):
            b.setEnabled(has)
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        """Apply only a recommendation that describes the current input; say so when it does not."""
        if self._is_running():
            return
        stale = self.is_stale
        self._btn_apply.setEnabled(self.recommendation is not None and not stale)
        note = self.tr("From a previous input: analyse again before using it") if stale else ""
        for b in (self._btn_apply, self._btn_csv, self._btn_json, self._btn_png):
            b.setToolTip(note)
        self._fill_suggestion()
        self._update_status()

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(round(1000 * fraction)))
        self._status.setText(self.tr("Analysing: {msg}").format(msg=message))

    def _on_finished(self, result, sweep) -> None:
        self.result = result
        self.sweep = sweep
        self._result_source = self._job_source
        self._previous_note = ""
        self.recommendation = recommend_parameters(result) if result.status == "ok" else None
        self._progress.setValue(1000)
        self._settle()
        self._fill_table()
        self._draw_profiles()
        self._draw_sweep()
        if sweep is not None:
            self.tabs.setCurrentIndex(1)
        one = result.length("radial")
        self._state.log(
            self.tr("Texture analysed: 1/e length {L} voxel (radial), {n} voxels").format(
                L=f"{one:.2f}" if one is not None else "-", n=f"{result.acf.n_voxels:,}"
            ),
            "success",
        )

    def _on_failed(self, message: str, detail: str) -> None:
        self._previous_note = self.tr("the new analysis failed")
        self._settle()
        self._status.setText(message)
        self._state.log(self.tr("Texture analysis failed: {msg}").format(msg=message), "error")
        self._state.log(detail, "debug")

    def _on_cancelled(self) -> None:
        self._previous_note = self.tr("the new analysis was cancelled")
        self._settle()
        self._progress.setValue(0)
        self._status.setText(self.tr("Cancelled."))

    def _on_volumes_changed(self) -> None:
        self._btn_analyse.setEnabled(bool(self._state.volumes) and not self._is_running())
        self._refresh_validity()

    # ------------------------------------------------------------------ results
    def apply_recommendation(self) -> None:
        rec = self.recommendation
        if rec is None:
            return
        if self.is_stale:
            self._status.setText(self.tr("The suggestion is from a previous input: analyse again first."))
            return
        self._state.set_params(winsize=tuple(rec.subset), winstepsize=tuple(rec.step))
        self._state.log(
            self.tr("Subset {ws} and step {st} applied from the texture analysis").format(
                ws=" x ".join(str(e + 1) for e in rec.subset), st=" x ".join(str(s) for s in rec.step)
            ),
            "success",
        )
        if self._state.busy:
            self._state.log(self.tr("The active run keeps its parameters; the new subset applies to the next run."), "warning")

    def _fill_table(self) -> None:
        res = self.result
        src = self._result_source or {}
        units = src.get("units", "voxel")  # the calibration the analysis was done with, not today's
        spacing = tuple(src.get("spacing", (1.0, 1.0, 1.0)))
        physical = spacing != (1.0, 1.0, 1.0)
        for r, axis in enumerate(AXES_ROWS):
            for c, t in enumerate(THRESHOLDS):
                text = "-"
                if res is not None and res.status == "ok":
                    cr = res.lengths[axis][float(t)]
                    if cr.found:
                        text = f"{cr.value:.2f}"
                        if physical:
                            text += f" ({res.physical_lengths[axis][float(t)].value:.3g} {units})"
                        if cr.status == "plateau":
                            text += " ~"
                    else:
                        text = {"not_crossed": ">", "invalid": "?"}.get(cr.status, "-")
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    def _fill_suggestion(self) -> None:
        rec = self.recommendation
        if rec is None:
            self._suggestion.setText(self.tr("Analyse the texture to get a subset suggestion."))
            return
        lines = [
            self.tr("Subset {ws} voxel, step {st} ({factor} x the 1/e length per axis)").format(
                ws=" x ".join(str(e + 1) for e in rec.subset), st=" x ".join(str(s) for s in rec.step), factor=f"{rec.factor:g}"
            )
        ]
        lines += [f"  {n}" for n in rec.notes]
        if self.is_stale:
            lines.append(self.tr("(from a previous input: the reference, region, calibration or settings changed)"))
        self._suggestion.setText("\n".join(lines))

    def _update_status(self) -> None:
        if self._is_running():
            return
        if not self._state.volumes:
            self._status.setText(self.tr("No volume loaded: add the reference volume first."))
            return
        res = self.result
        if res is None:
            mask = self._state.reference_mask()
            where = (
                self.tr("inside the region of interest")
                if (mask is not None and self.use_roi.isChecked())
                else self.tr("whole volume")
            )
            self._status.setText(self.tr("Ready: reference volume, {where}.").format(where=where))
        elif res.status != "ok":
            self._status.setText(self.tr("No texture: the analysed region has no grey-value variation."))
        else:
            parts = [self.tr("{n} voxels analysed").format(n=f"{res.acf.n_voxels:,}")]
            if np.isfinite(res.noise_floor):
                parts.append(self.tr("noise floor {v}").format(v=f"{res.noise_floor:.3f}"))
            if res.periodicity is not None:
                axis, period, height = res.periodicity
                parts.append(self.tr("periodic along {axis} ({p} voxel)").format(axis=axis, p=f"{period:.1f}"))
            if self._previous_note:
                parts.append(self.tr("showing the previous result ({why})").format(why=self._previous_note))
            if self.is_stale:
                parts.append(self.tr("the input changed since: analyse again"))
            self._status.setText(", ".join(parts))

    # ------------------------------------------------------------------ figures
    def _style(self, ax) -> None:
        ax.set_facecolor(COLORS.BG_CANVAS)
        ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=8)
        ax.xaxis.label.set_color(COLORS.TEXT_SECONDARY)
        ax.yaxis.label.set_color(COLORS.TEXT_SECONDARY)
        ax.title.set_color(COLORS.TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(COLORS.BORDER)
        ax.grid(alpha=0.2)

    def _draw_profiles(self) -> None:
        fig = self.fig_profiles
        fig.clear()
        ax_lin = fig.add_subplot(2, 1, 1)
        ax_log = fig.add_subplot(2, 1, 2)
        res = self.result
        for ax in (ax_lin, ax_log):
            self._style(ax)
            for t in THRESHOLDS:
                ax.axhline(t, color=COLORS.TEXT_SECONDARY, lw=0.6, alpha=0.5)
        if res is None or res.status != "ok":
            ax_lin.text(
                0.5,
                0.5,
                self.tr("No analysis yet."),
                ha="center",
                va="center",
                color=COLORS.TEXT_SECONDARY,
                transform=ax_lin.transAxes,
            )
        else:
            for axis in AXES_ROWS:
                p = res.profiles[axis]
                color, ls = PROFILE_STYLES[axis]
                name = axis if axis != "radial" else self.tr("radial")
                ax_lin.plot(p.lag, p.mean, color=color, ls=ls, lw=1.4, label=name)
                positive = np.where(p.mean > 0, p.mean, np.nan)  # log axis: non-positive values are masked, not clamped
                ax_log.plot(p.lag, positive, color=color, ls=ls, lw=1.4)
                if axis == "radial" and np.isfinite(p.std).any():
                    ax_lin.fill_between(p.lag, p.mean - p.std, p.mean + p.std, color=color, alpha=0.12)
                for t in THRESHOLDS:
                    cr = res.lengths[axis][float(t)]
                    if cr.found:
                        ax_lin.plot([cr.value], [t], "o", color=color, ms=4)
            ax_log.set_yscale("log")
            ax_log.set_ylim(0.005, 1.2)
            ax_lin.legend(fontsize=8, loc="upper right", frameon=False, labelcolor=COLORS.TEXT_SECONDARY)
        ax_lin.set_ylabel(self.tr("autocorrelation"))
        ax_log.set_ylabel(self.tr("autocorrelation (log)"))
        ax_log.set_xlabel(self.tr("lag [voxel]"))
        fig.tight_layout()
        self.canvas_profiles.draw_idle()

    def _draw_sweep(self) -> None:
        fig = self.fig_sweep
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        self._style(ax)
        sweep = self.sweep
        if sweep is None or not sweep.levels:
            ax.text(
                0.5,
                0.5,
                self.tr("No size sweep yet: tick 'Run the size sweep' and analyse."),
                ha="center",
                va="center",
                color=COLORS.TEXT_SECONDARY,
                transform=ax.transAxes,
            )
        else:
            for t in THRESHOLDS:
                d = sweep.decisions[float(t)]
                label = THRESHOLD_LABELS.get(float(t), f"{t:.2f}")
                line = ax.errorbar(
                    sweep.sizes, sweep.means(t), yerr=sweep.stds(t), marker="o", ms=4, capsize=3, lw=1.2, label=label
                )
                if d.converged:
                    ax.axvline(sweep.sizes[d.start_index], color=line[0].get_color(), ls="--", lw=1)
                    ax.axhspan(d.reference - d.tolerance, d.reference + d.tolerance, color=line[0].get_color(), alpha=0.08)
            ax.legend(fontsize=8, frameon=False, labelcolor=COLORS.TEXT_SECONDARY)
            verdicts = []
            for t in THRESHOLDS:
                d = sweep.decisions[float(t)]
                name = THRESHOLD_LABELS.get(float(t), f"{t:.2f}")
                if d.converged:
                    verdicts.append(
                        self.tr("{name}: stable from {size} voxel").format(name=name, size=sweep.sizes[d.start_index])
                    )
                else:
                    verdicts.append(self.tr("{name}: not stable ({why})").format(name=name, why=d.reason))
            ax.text(
                0.02,
                0.98,
                "\n".join(verdicts),
                ha="left",
                va="top",
                fontsize=7,
                color=COLORS.TEXT_SECONDARY,
                transform=ax.transAxes,
            )
        ax.set_xlabel(self.tr("sub-volume edge [voxel]"))
        ax.set_ylabel(self.tr("correlation length [voxel]"))
        fig.tight_layout()
        self.canvas_sweep.draw_idle()

    # ------------------------------------------------------------------ export
    def _ask_path(self, default: str, filter_text: str) -> str:
        if headless():
            return default
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Save"), default, filter_text)
        return path

    def _write(self, what: str, path, writer) -> None:
        """Run ``writer(path)`` and report success or the file error in the window and the log."""
        try:
            writer(path)
        except Exception as exc:  # unwritable folder, bad name, disk full
            self._status.setText(self.tr("Cannot save {what}: {error}").format(what=what, error=exc))
            self._state.log(self.tr("Cannot save {what}: {error}").format(what=what, error=exc), "error")
            return
        self._state.log(self.tr("{what} saved: {path}").format(what=what, path=path))

    def _on_save_csv(self) -> None:
        if self.result is None:
            return
        path = self._ask_path(str(self._state.output_dir / "texture_profiles.csv"), "CSV (*.csv)")
        if path:
            self._write(self.tr("Profiles"), path, self.save_csv)

    def _on_save_json(self) -> None:
        if self.result is None:
            return
        path = self._ask_path(str(self._state.output_dir / "texture_summary.json"), "JSON (*.json)")
        if path:
            self._write(self.tr("Summary"), path, self.save_json)

    def _on_save_png(self) -> None:
        path = self._ask_path(str(self._state.output_dir / "texture_profiles.png"), "PNG (*.png)")
        if path:
            fig = self.fig_sweep if self.tabs.currentIndex() == 1 else self.fig_profiles

            def save(target, fig=fig):
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(target, dpi=150, facecolor=fig.get_facecolor())

            self._write(self.tr("Image"), path, save)

    def save_csv(self, path) -> None:
        write_profiles_csv(self.result, path)

    def save_json(self, path) -> None:
        write_summary_json(self.result, self.sweep, self.recommendation, path)

    # ------------------------------------------------------------------ misc
    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("Texture analysis"))
        self.tabs.setTabText(0, self.tr("Correlation"))
        self.tabs.setTabText(1, self.tr("Size sweep"))
        self.sections["params"].set_title(self.tr("Texture parameters"))
        self.sections["sweep"].set_title(self.tr("Size sweep"))
        self.sections["lengths"].set_title(self.tr("Correlation lengths [voxel]"))
        self.sections["suggestion"].set_title(self.tr("Subset suggestion"))
        self.sections["export"].set_title(self.tr("Export"))
        texts = {
            "estimator": self.tr("Estimator"),
            "max_lag": self.tr("Maximum lag [voxel]"),
            "min_overlap": self.tr("Minimum overlap"),
            "window_edge": self.tr("Analysis window edge [voxel]"),
            "sweep_start": self.tr("First size [voxel]"),
            "sweep_step": self.tr("Size step [voxel]"),
            "sweep_count": self.tr("Number of sizes"),
            "sweep_samples": self.tr("Positions per size"),
        }
        for key, lab in self.labels.items():
            lab.setText(texts[key])
        retranslate_combo(self.estimator, "estimator")
        self.labels["estimator"].setToolTip(
            self.tr(
                "Overlap-corrected: every lag is divided by the number of voxel pairs behind it (no size bias).\n"
                "Finite window: the plain FFT estimate, as in the DVC Challenge scripts, for comparison."
            )
        )
        self.labels["max_lag"].setToolTip(self.tr("Largest lag analysed per axis; 0 takes half the analysis window."))
        self.labels["min_overlap"].setToolTip(
            self.tr("Lags backed by fewer voxel pairs than this fraction of the region are not reported.")
        )
        self.labels["window_edge"].setToolTip(
            self.tr("The region is cropped about its centre to this edge (cubed) before the FFT, to bound memory and time.")
        )
        self.labels["sweep_samples"].setToolTip(self.tr("Sub-volumes analysed at different positions for every size."))
        self.use_roi.setText(self.tr("Restrict to the region of interest"))
        self.run_sweep.setText(self.tr("Run the size sweep"))
        self._btn_analyse.setText(self.tr("Analyse texture"))
        self._btn_cancel.setText(self.tr("Cancel"))
        self._btn_apply.setText(self.tr("Apply to parameters"))
        self._btn_csv.setText(self.tr("Save profiles as CSV..."))
        self._btn_json.setText(self.tr("Save summary as JSON..."))
        self._btn_png.setText(self.tr("Save image as PNG..."))
        self.table.setHorizontalHeaderLabels([THRESHOLD_LABELS.get(float(t), f"{t:.2f}") for t in THRESHOLDS])
        self.table.setVerticalHeaderLabels(["x", "y", "z", self.tr("radial")])
        self._table_hint.setText(
            self.tr("> above the threshold in the reliable range, ? no valid profile, ~ a plateau at the threshold")
        )
        self._fill_suggestion()
        self._draw_profiles()
        self._draw_sweep()
        self._update_status()


# ---------------------------------------------------------------------- files (shared with the CLI)
def write_profiles_csv(result: TextureResult, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["axis", "lag_voxel", "distance", "mean", "std", "count", "coverage"])
        for axis, p in result.profiles.items():
            for i in range(len(p)):
                w.writerow(
                    [
                        axis,
                        f"{p.lag[i]:.4f}",
                        f"{p.distance[i]:.4f}",
                        f"{p.mean[i]:.6f}",
                        f"{p.std[i]:.6f}",
                        int(p.count[i]),
                        f"{p.coverage[i]:.4f}",
                    ]
                )


def summary_dict(result: TextureResult, sweep: SizeSweep | None, recommendation) -> dict:
    out = {
        "status": result.status,
        "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in result.settings.items()},
        "lengths_voxel": {
            axis: {THRESHOLD_LABELS.get(t, f"{t:.3g}"): {"value": c.value, "status": c.status} for t, c in table.items()}
            for axis, table in result.lengths.items()
        },
        "lengths_physical": {
            axis: {THRESHOLD_LABELS.get(t, f"{t:.3g}"): c.value for t, c in table.items()}
            for axis, table in result.physical_lengths.items()
        },
        "noise_floor": None if not np.isfinite(result.noise_floor) else float(result.noise_floor),
        "periodicity": None
        if result.periodicity is None
        else {"axis": result.periodicity[0], "distance": result.periodicity[1], "height": result.periodicity[2]},
        "spacing": list(result.acf.spacing),
    }
    if recommendation is not None:
        out["recommendation"] = asdict(recommendation)
    if sweep is not None:
        out["sweep"] = {
            "axis": sweep.axis,
            "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in sweep.settings.items()},
            "levels": [
                {
                    "size": list(lvl.size),
                    "n_samples": len(lvl.samples),
                    "mean": {
                        THRESHOLD_LABELS.get(t, f"{t:.3g}"): (None if not np.isfinite(v) else v) for t, v in lvl.mean.items()
                    },
                    "std": {THRESHOLD_LABELS.get(t, f"{t:.3g}"): (None if not np.isfinite(v) else v) for t, v in lvl.std.items()},
                }
                for lvl in sweep.levels
            ],
            "decisions": {
                THRESHOLD_LABELS.get(t, f"{t:.3g}"): {
                    "converged": d.converged,
                    "start_size": None if d.start_index is None else list(sweep.levels[d.start_index].size),
                    "reference": None if not np.isfinite(d.reference) else d.reference,
                    "tolerance": None if not np.isfinite(d.tolerance) else d.tolerance,
                    "reason": d.reason,
                }
                for t, d in sweep.decisions.items()
            },
        }
    return out


def write_summary_json(result: TextureResult, sweep: SizeSweep | None, recommendation, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_dict(result, sweep, recommendation), f, indent=2)
