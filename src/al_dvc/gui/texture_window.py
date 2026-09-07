"""Texture analysis window: how far the grey values of the reference volume stay correlated, and what
subset size that suggests.

Three steps, one tab and one parameter page each; the strip at the top shows what every step produced:

1. **Region** -- the part of the volume whose texture is measured, drawn on its own slice viewer
   (rectangle, ellipse, polygon, brush) or copied from the DVC region of interest. It is *not* the
   DVC region of interest. The analysis uses the bounding box of the region: a window slides inside
   it, so the box fixes the largest shift that can be analysed, ``(box - window) / 2`` per axis.
2. **Representative volume element (RVE)** -- windows of growing size, all centred in the region, are
   analysed with the same shifts; the size from which the correlation lengths stop changing is the
   window of step 3.
3. **Autocorrelation** -- one window: the correlation curves along x, y, z and over spherical shells,
   the lengths at 1/e, 0.1 and 0.01, the noise floor, the periodicity, and the subset suggestion.

The autocorrelation lengths and the subset suggestion stay on screen whatever the step. Both analyses
run on worker threads; results are tagged with the input they describe, so a suggestion from a
previous volume, region, calibration or window cannot be applied by mistake.
"""

from __future__ import annotations

import csv
import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from al_dvc.texture import (
    MAX_RANGE_VOXELS,
    THRESHOLD_LABELS,
    THRESHOLDS,
    SizeSweep,
    TextureResult,
    analyse_range,
    box_size,
    centred_window,
    lag_reach,
    normalise_box,
    recommend_parameters,
    sweep_concentric,
    sweep_sizes_concentric,
)

from .app_state import AppState
from .region_viewer import REGION_COLOR, RegionViewer
from .theme import COLORS
from .widgets import CollapsibleSection, combo, form_label, guard_wheel, headless, make_form, spin

logger = logging.getLogger(__name__)

SIDEBAR_WIDTH = 400
AXES_ROWS = ("x", "y", "z", "radial")
CURVE_STYLES = {"x": ("#60a5fa", "-"), "y": ("#f472b6", "--"), "z": ("#34d399", ":"), "radial": ("#f97316", "-")}
PLOT_THEMES = {  # figure and axes face, text, grid, threshold lines
    "dark": {"face": COLORS.BG_CANVAS, "text": COLORS.TEXT_PRIMARY, "grid": "#4b5563", "threshold": "#fbbf24"},
    "white": {"face": "#ffffff", "text": "#111827", "grid": "#d1d5db", "threshold": "#b45309"},
    "grey": {"face": "#e5e7eb", "text": "#111827", "grid": "#9ca3af", "threshold": "#b45309"},
}
FONT = {"label": 11, "tick": 10, "legend": 10, "note": 9}
WINDOW_STEP = 8  # the window edge box moves in steps of this many voxels
STEPS = ("region", "sweep", "acf")
TAB_REGION, TAB_SWEEP, TAB_ACF = 0, 1, 2
MIN_FILL = 0.5  # below this share of its bounding box, a region gets a warning
SWEEP_CMAP = "viridis"  # one colour per window size in the RVE curves plot

__all__ = ["TextureWindow"]


class _Cancelled(Exception):
    pass


class _TextureWorker(QThread):
    """One analysis off the UI thread: ``kind`` is ``acf`` (one window) or ``sweep`` (window sizes)."""

    progress = Signal(float, str)
    finished_analysis = Signal(object)  # TextureResult
    finished_sweep = Signal(object)  # SizeSweep
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, kind: str, vol, box, spacing, window: int, sweep: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self._vol = vol
        self._box = box
        self._spacing = spacing
        self._window = window
        self._sweep = sweep
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            if self.kind == "acf":
                self.progress.emit(0.0, "autocorrelation")
                out = analyse_range(self._vol, self._box, self._window, self._spacing)
            else:
                out = sweep_concentric(
                    self._vol,
                    self._box,
                    spacing=self._spacing,
                    progress=self.progress.emit,
                    stop=lambda: self._stop,
                    **self._sweep,
                )
            if self._stop:  # a cancel during the computation is honoured, not published
                raise _Cancelled()
            self.progress.emit(1.0, "done")
        except _Cancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # surface to the UI
            logger.exception("Texture analysis failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        if self.kind == "acf":
            self.finished_analysis.emit(out)
        else:
            self.finished_sweep.emit(out)


class _StepStrip(QWidget):
    """Three step buttons with what each step produced; clicking one goes to that step."""

    clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.buttons: list[QPushButton] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for i in range(len(STEPS)):
            b = QPushButton()
            b.setCheckable(True)
            b.setMinimumHeight(34)
            b.clicked.connect(lambda _c, k=i: self.clicked.emit(k))
            group.addButton(b)
            layout.addWidget(b, 1)
            self.buttons.append(b)
            if i < len(STEPS) - 1:
                arrow = QLabel("→")
                arrow.setObjectName("hint")
                layout.addWidget(arrow)
        self.setStyleSheet(
            f"QPushButton {{ text-align: left; padding: 4px 10px; border: 1px solid {COLORS.BORDER}; border-radius: 6px;"
            f" background: {COLORS.BG_PANEL}; color: {COLORS.TEXT_SECONDARY}; }}"
            f"QPushButton:checked {{ border-color: {COLORS.ACCENT}; background: {COLORS.BG_HOVER};"
            f" color: {COLORS.TEXT_PRIMARY}; font-weight: bold; }}"
        )

    def set_current(self, i: int) -> None:
        self.buttons[i].setChecked(True)

    def set_text(self, i: int, text: str, tooltip: str = "") -> None:
        self.buttons[i].setText(text)
        self.buttons[i].setToolTip(tooltip)


def _heading(parent_layout, size: int = 13) -> QLabel:
    lab = QLabel()
    lab.setWordWrap(True)
    lab.setStyleSheet(f"font-size: {size}px; font-weight: bold; color: {COLORS.TEXT_PRIMARY};")
    parent_layout.addWidget(lab)
    return lab


def _hint(parent_layout) -> QLabel:
    lab = QLabel()
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    parent_layout.addWidget(lab)
    return lab


def _notice(parent_layout, size: int = 12) -> QLabel:
    """A statement that must not be missed: orange bar on the left, tinted background."""
    lab = QLabel()
    lab.setWordWrap(True)
    lab.setStyleSheet(
        f"font-size: {size}px; color: {COLORS.TEXT_PRIMARY}; background: rgba(249, 115, 22, 0.14);"
        f" border-left: 4px solid {REGION_COLOR}; border-radius: 4px; padding: 6px 8px;"
    )
    parent_layout.addWidget(lab)
    return lab


class TextureWindow(QMainWindow):
    """Region, representative volume element, autocorrelation; lengths and subset suggestion always in view."""

    TAB_REGION, TAB_SWEEP, TAB_ACF = TAB_REGION, TAB_SWEEP, TAB_ACF

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: _TextureWorker | None = None
        self.result: TextureResult | None = None
        self.sweep: SizeSweep | None = None
        self.recommendation = None
        self._job_source: dict | None = None  # the input the running analysis was given
        self._result_source: dict | None = None  # the input ``result`` describes
        self._sweep_source: dict | None = None  # the input ``sweep`` describes
        self._previous_note = ""  # why the previous result is still on screen (failed / cancelled rerun)
        self._shape: tuple[int, int, int] | None = None  # (nz, ny, nx) the region viewer holds
        self._volume_uid = None
        self._window_from_rve: int | None = None  # the window the RVE step wrote, None when set by hand
        self._updating = False
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(1360, 880)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.steps = _StepStrip()
        root.addWidget(self.steps)
        body = QHBoxLayout()
        body.setSpacing(8)
        root.addLayout(body, 1)

        # ---- plots and region (left) --------------------------------------
        self.plot_background = combo([])
        for key in PLOT_THEMES:
            self.plot_background.addItem(key, key)
        self.plot_scale = combo([])
        for key in ("linear", "log"):
            self.plot_scale.addItem(key, key)
        self.curve_checks: dict[str, QCheckBox] = {}
        for axis in AXES_ROWS:
            cb = QCheckBox()
            cb.setChecked(True)
            self.curve_checks[axis] = cb
        self.show_band = QCheckBox()
        self.show_band.setChecked(True)
        self._btn_reset_view = QPushButton()
        self._plot_labels = {k: QLabel() for k in ("background", "scale", "curves")}
        self._plot_tools = QWidget()
        tools = QHBoxLayout(self._plot_tools)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(6)
        tools.addWidget(self._plot_labels["background"])
        tools.addWidget(self.plot_background)
        tools.addSpacing(8)
        tools.addWidget(self._plot_labels["scale"])
        tools.addWidget(self.plot_scale)
        tools.addSpacing(8)
        tools.addWidget(self._plot_labels["curves"])
        for cb in self.curve_checks.values():
            tools.addWidget(cb)
        tools.addWidget(self.show_band)
        tools.addStretch(1)
        tools.addWidget(self._btn_reset_view)

        self.tabs = QTabWidget()
        self.region = RegionViewer()
        region_page = QWidget()
        rlay = QVBoxLayout(region_page)
        rlay.setContentsMargins(0, 6, 0, 0)
        rlay.setSpacing(6)
        self._region_banner = _notice(rlay, size=13)
        rlay.addWidget(self.region, 1)
        self.tabs.addTab(region_page, "")
        self.fig_sweep = Figure(figsize=(7, 6))
        self.canvas_sweep = FigureCanvas(self.fig_sweep)
        self.fig_profiles = Figure(figsize=(7, 5))
        self.canvas_profiles = FigureCanvas(self.fig_profiles)
        self.toolbar_sweep = NavigationToolbar2QT(self.canvas_sweep, self)
        self.toolbar_profiles = NavigationToolbar2QT(self.canvas_profiles, self)
        for tb in (self.toolbar_profiles, self.toolbar_sweep):
            tb.setIconSize(tb.iconSize() * 0.8)
        self.tabs.addTab(self._plot_page(self.canvas_sweep, self.toolbar_sweep), "")
        self.tabs.addTab(self._plot_page(self.canvas_profiles, self.toolbar_profiles), "")
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(self._plot_tools)
        left.addWidget(self.tabs, 1)
        body.addLayout(left, 1)

        # ---- sidebar (right): one parameter page per step, then the standing results ----
        side = QWidget()
        side.setFixedWidth(SIDEBAR_WIDTH)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)
        self.pages = QStackedWidget()
        self.pages.setSizePolicy(self.pages.sizePolicy().horizontalPolicy(), self.pages.sizePolicy().verticalPolicy())
        self.labels: dict[str, QLabel] = {}
        self._headings: dict[str, QLabel] = {}
        self._hints: dict[str, QLabel] = {}
        self._next: dict[str, QPushButton] = {}

        # step 1: region
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._headings["region"] = _heading(lay)
        self._hints["region"] = _hint(lay)
        lay.addWidget(self.region.tools)
        rbuttons = QHBoxLayout()
        self._btn_region_roi = QPushButton()
        self._btn_region_all = QPushButton()
        rbuttons.addWidget(self._btn_region_roi)
        rbuttons.addWidget(self._btn_region_all)
        lay.addLayout(rbuttons)
        self.labels["box"] = form_label()
        lay.addWidget(self.labels["box"])
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        self.range_lo: dict[str, object] = {}
        self.range_hi: dict[str, object] = {}
        self._range_axis_labels: dict[str, QLabel] = {}
        for row, axis in enumerate(("x", "y", "z")):
            lab = QLabel(axis)
            lab.setFixedWidth(14)
            lo = spin(0, 1, 1, width=72)
            hi = spin(1, 2, 1, width=72)
            dash = QLabel("–")
            grid.addWidget(lab, row, 0)
            grid.addWidget(lo, row, 1)
            grid.addWidget(dash, row, 2)
            grid.addWidget(hi, row, 3)
            self.range_lo[axis], self.range_hi[axis] = lo, hi
            self._range_axis_labels[axis] = lab
        grid.setColumnStretch(4, 1)
        lay.addLayout(grid)
        self._range_info = _hint(lay)
        self._next["region"] = QPushButton()
        lay.addWidget(self._next["region"])
        lay.addStretch(1)
        self.pages.addWidget(page)

        # step 2: representative volume element
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._headings["sweep"] = _heading(lay)
        self._hints["sweep"] = _hint(lay)
        self._sweep_region = _hint(lay)
        sform = make_form()
        self.sweep_start = spin(8, 512, WINDOW_STEP)
        self.sweep_start.setValue(16)
        self.sweep_step = spin(4, 256, 4)
        self.sweep_step.setValue(16)
        for key, w in [("sweep_start", self.sweep_start), ("sweep_step", self.sweep_step)]:
            lab = form_label()
            self.labels[key] = lab
            sform.addRow(lab, w)
        lay.addLayout(sform)
        self._btn_sweep = QPushButton()
        self._btn_sweep.setProperty("class", "btn-primary")
        self._btn_sweep.setMinimumHeight(32)
        self._btn_sweep_cancel = QPushButton()
        self._btn_sweep_cancel.setEnabled(False)
        srow = QHBoxLayout()
        srow.addWidget(self._btn_sweep, 2)
        srow.addWidget(self._btn_sweep_cancel, 1)
        lay.addLayout(srow)
        self._sweep_progress = QProgressBar()
        self._sweep_progress.setRange(0, 1000)
        self._sweep_progress.setTextVisible(False)
        lay.addWidget(self._sweep_progress)
        self._sweep_box = QGroupBox()
        self._sweep_box.setObjectName("analysisBox")
        vbox = QVBoxLayout(self._sweep_box)
        vbox.setSpacing(4)
        self._sweep_status = QLabel()
        self._sweep_status.setWordWrap(True)
        vbox.addWidget(self._sweep_status)
        self._btn_use_size = QPushButton()
        self._btn_use_size.setProperty("class", "btn-primary")
        self._btn_use_size.setEnabled(False)
        vbox.addWidget(self._btn_use_size)
        lay.addWidget(self._sweep_box)
        self._next["sweep"] = QPushButton()
        lay.addWidget(self._next["sweep"])
        lay.addStretch(1)
        self.pages.addWidget(page)

        # step 3: autocorrelation
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._headings["acf"] = _heading(lay)
        self._hints["acf"] = _hint(lay)
        self._acf_region = _hint(lay)
        form = make_form()
        self.window_size = spin(WINDOW_STEP, 1024, WINDOW_STEP)
        self.window_size.setValue(64)
        lab = form_label()
        self.labels["window_size"] = lab
        form.addRow(lab, self.window_size)
        lay.addLayout(form)
        self._window_source = _hint(lay)
        self._btn_analyse = QPushButton()
        self._btn_analyse.setProperty("class", "btn-primary")
        self._btn_analyse.setMinimumHeight(32)
        self._btn_cancel = QPushButton()
        self._btn_cancel.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self._btn_analyse, 2)
        row.addWidget(self._btn_cancel, 1)
        lay.addLayout(row)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        lay.addWidget(self._progress)
        self._status = _hint(lay)
        lay.addStretch(1)
        self.pages.addWidget(page)
        side_layout.addWidget(self.pages)

        # standing results: autocorrelation lengths and the subset suggestion
        self._lengths_box = QGroupBox()
        self._lengths_box.setObjectName("analysisBox")
        lbox = QVBoxLayout(self._lengths_box)
        lbox.setSpacing(4)
        self.table = QTableWidget(len(AXES_ROWS), len(THRESHOLDS))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setFixedHeight(30 * (len(AXES_ROWS) + 1) + 8)
        font = self.table.font()
        font.setPointSizeF(font.pointSizeF() + 1)
        self.table.setFont(font)
        lbox.addWidget(self.table)
        self._table_hint = _hint(lbox)
        side_layout.addWidget(self._lengths_box)

        self._suggestion_box = QGroupBox()
        self._suggestion_box.setObjectName("analysisBox")
        sbox = QVBoxLayout(self._suggestion_box)
        sbox.setSpacing(4)
        self._suggestion = QLabel()
        self._suggestion.setWordWrap(True)
        self._suggestion.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._suggestion.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {COLORS.TEXT_PRIMARY};")
        sbox.addWidget(self._suggestion)
        self._suggestion_notes = _hint(sbox)
        self._btn_apply = QPushButton()
        self._btn_apply.setProperty("class", "btn-primary")
        self._btn_apply.setMinimumHeight(30)
        self._btn_apply.setEnabled(False)
        sbox.addWidget(self._btn_apply)
        side_layout.addWidget(self._suggestion_box)

        self.export_section = CollapsibleSection(expanded=False)
        self._btn_csv = QPushButton()
        self._btn_json = QPushButton()
        self._btn_png = QPushButton()
        for b in (self._btn_csv, self._btn_json, self._btn_png):
            b.setEnabled(False)
            self.export_section.add_widget(b)
        side_layout.addWidget(self.export_section)
        side_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setFixedWidth(SIDEBAR_WIDTH + 18)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body.addWidget(scroll, 0)
        self.setCentralWidget(central)
        guard_wheel(side)

        # ---- wiring ----------------------------------------------------------
        self.steps.clicked.connect(self.go_to_step)
        self.tabs.currentChanged.connect(self.go_to_step)
        self._next["region"].clicked.connect(lambda: self.go_to_step(TAB_SWEEP))
        self._next["sweep"].clicked.connect(lambda: self.go_to_step(TAB_ACF))
        self._btn_analyse.clicked.connect(self.analyse)
        self._btn_cancel.clicked.connect(self.cancel)
        self._btn_sweep.clicked.connect(self.run_sweep_analysis)
        self._btn_sweep_cancel.clicked.connect(self.cancel)
        self._btn_use_size.clicked.connect(self.use_sweep_size)
        self._btn_apply.clicked.connect(self.apply_recommendation)
        self._btn_csv.clicked.connect(self._on_save_csv)
        self._btn_json.clicked.connect(self._on_save_json)
        self._btn_png.clicked.connect(self._on_save_png)
        self._btn_reset_view.clicked.connect(self.reset_view)
        self._btn_region_all.clicked.connect(self.set_range_whole)
        self._btn_region_roi.clicked.connect(self.use_dvc_roi)
        self.region.region_changed.connect(self._on_region_changed)
        for w in (*self.range_lo.values(), *self.range_hi.values()):
            w.valueChanged.connect(lambda _v: self._on_box_spins())
        self.window_size.valueChanged.connect(lambda _v: self._on_window_changed())
        for w in (self.sweep_start, self.sweep_step):
            w.valueChanged.connect(lambda _v: self._refresh_validity())
        self.plot_background.currentIndexChanged.connect(lambda _i: self._redraw())
        self.plot_scale.currentIndexChanged.connect(lambda _i: self._draw_profiles())
        for cb in (*self.curve_checks.values(), self.show_band):
            cb.toggled.connect(lambda _v: self._draw_profiles())
        self._state.volumes_changed.connect(self._on_volumes_changed)
        self._state.mask_changed.connect(self._refresh_validity)
        self._state.params_changed.connect(self._refresh_validity)
        self.retranslate_ui()
        self._redraw()
        self._on_volumes_changed()
        self.go_to_step(TAB_REGION)

    @staticmethod
    def _plot_page(canvas, toolbar) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas, 1)
        layout.addWidget(toolbar)
        return page

    # ------------------------------------------------------------------ steps
    def go_to_step(self, i: int) -> None:
        """Show step ``i`` everywhere: the strip, the tab and the parameter page."""
        i = int(max(0, min(len(STEPS) - 1, i)))
        if self.tabs.currentIndex() != i:
            self.tabs.setCurrentIndex(i)
            return  # currentChanged brings us back here
        self.pages.setCurrentIndex(i)
        self.steps.set_current(i)
        self._update_plot_tools()

    def _update_steps(self) -> None:
        """What every step produced, in the strip at the top."""
        box = self.range_box()
        size = " x ".join(str(v) for v in box_size(box)) if box is not None else "—"
        self.steps.set_text(0, self.tr("1  Region") + f"   ·   {size}", self.tr("The texture analysis region (bounding box)"))
        if self.sweep is None:
            rve = "—"
        else:
            n = self.sweep_size()
            rve = self.tr("window {n}").format(n=n) if n is not None else self.tr("not stable")
            rve += "  ⚠" if self.is_sweep_stale else "  ✓"
        self.steps.set_text(
            1, self.tr("2  Representative volume element (RVE)") + f"   ·   {rve}", self.tr("The window size to use")
        )
        if self.result is None:
            acf = "—"
        elif self.result.status != "ok":
            acf = self.tr("no texture")
        else:
            one = self.result.length("radial")
            acf = self.tr("1/e length {L}").format(L=f"{one:.1f}" if one is not None else "-")
            acf += "  ⚠" if self.is_stale else "  ✓"
        self.steps.set_text(
            2, self.tr("3  Autocorrelation") + f"   ·   {acf}", self.tr("Correlation lengths and subset suggestion")
        )

    # ------------------------------------------------------------------ region (step 1)
    def _volume_shape(self) -> tuple[int, int, int] | None:
        return self._state.volume_shape() if self._state.volumes else None

    def _reference(self):
        """The reference volume, ``None`` without one."""
        if not self._state.volumes:
            return None
        return np.asarray(self._state.volume_array(0))

    def _setup_region(self) -> None:
        """Give the region viewer the reference volume when it changes (the region then is the whole volume)."""
        shape = self._volume_shape()
        uid = self._state.volumes[0].uid if self._state.volumes else None
        if shape == self._shape and uid == self._volume_uid:
            return
        self._shape, self._volume_uid = shape, uid
        self._updating = True
        try:
            nz, ny, nx = shape if shape is not None else (1, 1, 1)
            for axis, n in (("x", nx), ("y", ny), ("z", nz)):
                self.range_lo[axis].setRange(0, max(0, n - 2))
                self.range_hi[axis].setRange(2, max(2, n))
            for w in (*self.range_lo.values(), *self.range_hi.values(), self._btn_region_all):
                w.setEnabled(shape is not None)
            if shape is not None:  # a window that leaves room to shift: at most half the smallest edge
                fit = max(WINDOW_STEP, (min(shape) // 2) // WINDOW_STEP * WINDOW_STEP)
                self.window_size.setValue(min(int(self.window_size.value()), fit))
        finally:
            self._updating = False
        self.region.set_volume(self._reference())  # emits region_changed

    def range_box(self):
        """``((x0, x1), (y0, y1), (z0, z1))``: the bounding box of the region, ``None`` without one."""
        return self.region.box()

    def set_range(self, box) -> None:
        """Make the box ``((x0, x1), (y0, y1), (z0, z1))`` the region (clipped to the volume)."""
        if self._shape is None:
            return
        self.region.set_box(normalise_box(box, self._shape))

    def set_range_whole(self) -> None:
        self.region.set_region(None)

    def use_dvc_roi(self) -> None:
        """Copy the DVC region of interest of the reference volume into the region."""
        mask = self._state.reference_mask()
        if mask is None or self._shape is None or mask.shape != tuple(self._shape):
            self._range_info.setText(self.tr("No region of interest on the reference volume."))
            return
        if not mask.any():
            self._range_info.setText(self.tr("The region of interest is empty."))
            return
        self.region.set_region(mask)
        self._state.log(self.tr("Texture region copied from the DVC region of interest"))

    def _on_region_changed(self) -> None:
        self._sync_box_spins()
        self._update_range_info()
        self._refresh_validity()

    def _sync_box_spins(self) -> None:
        box = self.range_box()
        if box is None:
            return
        self._updating = True
        try:
            for axis, (lo, hi) in zip(("x", "y", "z"), box):
                self.range_lo[axis].setValue(lo)
                self.range_hi[axis].setValue(hi)
        finally:
            self._updating = False

    def _on_box_spins(self) -> None:
        """Typing the box replaces the region by that box."""
        if self._updating or self._shape is None:
            return
        try:
            box = normalise_box(
                tuple((int(self.range_lo[a].value()), int(self.range_hi[a].value())) for a in ("x", "y", "z")), self._shape
            )
        except ValueError:
            return
        if box != self.range_box() or self.region.fill_fraction() < 1.0:
            self.region.set_box(box)

    def _region_text(self) -> str:
        box = self.range_box()
        if box is None:
            return self.tr("No region")
        return self.tr("Region {size} voxel").format(size=" x ".join(str(v) for v in box_size(box)))

    def _update_range_info(self) -> None:
        box = self.range_box()
        if self._shape is None:
            self._range_info.setText(self.tr("Load a reference volume first."))
            return
        if box is None:
            self._range_info.setText(self.tr("The region is empty: draw a shape or press Whole volume."))
            return
        size = box_size(box)
        fill = self.region.fill_fraction()
        parts = [
            self.tr("Bounding box {size} voxel ({mv} M voxels), {pct} % of it inside the drawn shape.").format(
                size=" x ".join(str(v) for v in size), mv=f"{np.prod(size) / 1e6:.1f}", pct=f"{100 * fill:.0f}"
            )
        ]
        if np.prod(size) > MAX_RANGE_VOXELS:
            parts.append(
                self.tr("Too large: reduce the region to {edge} voxel cubed at most.").format(
                    edge=int(round(MAX_RANGE_VOXELS ** (1 / 3)))
                )
            )
        elif fill < MIN_FILL:
            parts.append(
                self.tr(
                    "The analysis uses the whole bounding box: voxels inside the box but outside the shape take part too. "
                    "A box-like region avoids that."
                )
            )
        self._range_info.setText(" ".join(parts))
        self._sweep_region.setText(self._region_text())
        self._update_acf_region()

    def _update_acf_region(self) -> None:
        box = self.range_box()
        if box is None:
            self._acf_region.setText(self._region_text())
            return
        window = centred_window(box, int(self.window_size.value()))
        reach = lag_reach(box, window)
        text = (
            self._region_text()
            + "; "
            + self.tr("window {w}: shifts up to {reach} voxel").format(
                w=" x ".join(str(b - a) for a, b in window), reach=" x ".join(str(v) for v in reach)
            )
        )
        if min(reach) < 1:
            text += " " + self.tr("The window fills the region on one axis: enlarge the region or reduce the window.")
        self._acf_region.setText(text)

    def _on_window_changed(self) -> None:
        if self._updating:
            return
        self._update_acf_region()
        self._update_window_source()
        self._refresh_validity()

    def _update_window_source(self) -> None:
        w = int(self.window_size.value())
        rve = self._window_from_rve
        if rve is None:
            text = self.tr("Set by hand. Step 2 suggests the window from the texture itself.")
        elif rve == w:
            text = self.tr("From the RVE analysis ({n} voxel).").format(n=rve)
        else:
            text = self.tr("Set by hand; the RVE analysis suggested {n} voxel.").format(n=rve)
        self._window_source.setText(text)
        self._window_source.setStyleSheet(f"color: {REGION_COLOR};" if rve is not None and rve != w else "")

    # ------------------------------------------------------------------ inputs
    def current_source(self) -> dict | None:
        """What an analysis started now would describe: reference identity, region, window, calibration."""
        st = self._state
        box = self.range_box()
        if not st.volumes or box is None:
            return None
        return {
            "uid": st.volumes[0].uid,
            "range": box,
            "window": int(self.window_size.value()),
            "spacing": tuple(float(v) for v in st.para.voxel_size),
            "units": str(getattr(st.para, "units", "voxel") or "voxel"),
        }

    def _sweep_input(self) -> dict | None:
        src = self.current_source()
        if src is None:
            return None
        return {k: v for k, v in src.items() if k != "window"} | {"sweep": self.sweep_settings()}

    @property
    def is_stale(self) -> bool:
        """True when a result is shown but the reference, the region, the window or the calibration changed."""
        return self.result is not None and self._result_source != self.current_source()

    @property
    def is_sweep_stale(self) -> bool:
        return self.sweep is not None and self._sweep_source != self._sweep_input()

    def sweep_settings(self) -> dict:
        start = int(self.sweep_start.value())
        return {"start": start, "step": int(self.sweep_step.value()), "min_lag": max(8, start)}

    # ------------------------------------------------------------------ analyses
    def _start(self, kind: str) -> None:
        if self._is_running():
            return
        vol = self._reference()
        box = self.range_box()
        status = self._status if kind == "acf" else self._sweep_status
        if vol is None:
            status.setText(self.tr("Load a reference volume first."))
            return
        if box is None:
            status.setText(self.tr("The region is empty: draw a shape or press Whole volume."))
            return
        if np.prod(box_size(box)) > MAX_RANGE_VOXELS:
            status.setText(self.tr("The region is too large for one analysis: reduce it."))
            return
        spacing = tuple(float(v) for v in self._state.para.voxel_size)
        window = int(self.window_size.value())
        sweep = None
        if kind == "acf":
            if min(lag_reach(box, centred_window(box, window))) < 1:
                status.setText(self.tr("The window fills the region: enlarge the region or reduce the window."))
                return
        else:
            sweep = self.sweep_settings()
            try:
                sizes = sweep_sizes_concentric(box, **sweep)
            except ValueError as exc:
                status.setText(str(exc))
                return
            if len(sizes) < 2:
                status.setText(self.tr("The region allows only one window size: enlarge it or reduce the first size."))
                return
        self._worker = _TextureWorker(kind, vol, box, spacing, window, sweep, parent=self)
        self._job_source = self.current_source() if kind == "acf" else self._sweep_input()
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_analysis.connect(self._on_finished)
        self._worker.finished_sweep.connect(self._on_sweep_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._btn_analyse.setEnabled(False)
        self._btn_sweep.setEnabled(False)
        (self._btn_cancel if kind == "acf" else self._btn_sweep_cancel).setEnabled(True)
        for b in (self._btn_apply, self._btn_csv, self._btn_json, self._btn_png, self._btn_use_size):
            b.setEnabled(False)  # the previous results stay on screen but are not applied or exported meanwhile
        if kind == "acf":
            self._progress.setValue(0)
            self._status.setText(self.tr("Analysing the texture..."))
        else:
            self._sweep_progress.setValue(0)
            self._sweep_status.setText(self.tr("Sweeping the window sizes..."))
        self._state.log(
            self.tr("{what} started: region {size} voxel").format(
                what=self.tr("Autocorrelation analysis") if kind == "acf" else self.tr("RVE analysis"),
                size=" x ".join(str(v) for v in box_size(box)),
            )
        )
        self._worker.start()

    def analyse(self) -> None:
        """The autocorrelation analysis: profiles, correlation lengths, subset suggestion."""
        self._start("acf")

    def run_sweep_analysis(self) -> None:
        """The RVE analysis: correlation length against the size of the window."""
        self._start("sweep")

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
        has_volume = bool(self._state.volumes)
        self._btn_analyse.setEnabled(has_volume)
        self._btn_sweep.setEnabled(has_volume)
        self._btn_cancel.setEnabled(False)
        self._btn_sweep_cancel.setEnabled(False)
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
        self._btn_use_size.setEnabled(self.sweep_size() is not None and not self.is_sweep_stale)
        self._btn_region_roi.setEnabled(self._shape is not None and self._state.reference_mask() is not None)
        self._fill_suggestion()
        self._update_status()
        self._update_sweep_status()
        self._update_steps()

    def _on_progress(self, fraction: float, message: str) -> None:
        if self._worker is not None and self._worker.kind == "sweep":
            self._sweep_progress.setValue(int(round(1000 * fraction)))
            self._sweep_status.setText(self.tr("Sweeping: {msg}").format(msg=message))
        else:
            self._progress.setValue(int(round(1000 * fraction)))
            self._status.setText(self.tr("Analysing: {msg}").format(msg=message))

    def _on_finished(self, result) -> None:
        self.result = result
        self._result_source = self._job_source
        self._previous_note = ""
        self.recommendation = recommend_parameters(result) if result.status == "ok" else None
        self._progress.setValue(1000)
        self._settle()
        self._fill_table()
        self._draw_profiles()
        self.go_to_step(TAB_ACF)
        one = result.length("radial")
        self._state.log(
            self.tr("Texture analysed: 1/e length {L} voxel (radial), window {w} voxel").format(
                L=f"{one:.2f}" if one is not None else "-",
                w=" x ".join(str(b - a) for a, b in result.settings["window_xyz"]),
            ),
            "success",
        )

    def _on_sweep_finished(self, sweep) -> None:
        self.sweep = sweep
        self._sweep_source = self._job_source
        self._sweep_progress.setValue(1000)
        self._settle()
        self._draw_sweep()
        self.go_to_step(TAB_SWEEP)
        size = self.sweep_size()
        self._state.log(
            self.tr("RVE analysis done: {verdict}").format(
                verdict=self.tr("stable from {size} voxel").format(size=size)
                if size is not None
                else self.tr("no stable size in the region")
            ),
            "success",
        )

    def _on_failed(self, message: str, detail: str) -> None:
        kind = self._worker.kind if self._worker is not None else "acf"
        if kind == "acf":
            self._previous_note = self.tr("the new analysis failed")
        self._settle()
        (self._status if kind == "acf" else self._sweep_status).setText(message)
        self._state.log(self.tr("Texture analysis failed: {msg}").format(msg=message), "error")
        self._state.log(detail, "debug")

    def _on_cancelled(self) -> None:
        kind = self._worker.kind if self._worker is not None else "acf"
        if kind == "acf":
            self._previous_note = self.tr("the new analysis was cancelled")
            self._progress.setValue(0)
        else:
            self._sweep_progress.setValue(0)
        self._settle()
        (self._status if kind == "acf" else self._sweep_status).setText(self.tr("Cancelled."))

    def _on_volumes_changed(self) -> None:
        self._setup_region()
        has = bool(self._state.volumes) and not self._is_running()
        self._btn_analyse.setEnabled(has)
        self._btn_sweep.setEnabled(has)
        self._refresh_validity()

    # ------------------------------------------------------------------ results
    def sweep_size(self) -> int | None:
        """The window edge the RVE analysis recommends: where the 1/e length became stable (else the largest
        stable threshold), ``None`` when nothing stabilised."""
        sweep = self.sweep
        if sweep is None or not sweep.levels:
            return None
        for t in THRESHOLDS:
            d = sweep.decisions.get(float(t))
            if d is not None and d.converged and d.start_index is not None:
                return int(max(sweep.levels[d.start_index].size))
        return None

    def use_sweep_size(self) -> None:
        """Write the stable size into the window of step 3 (rounded up to its step) and go there."""
        size = self.sweep_size()
        if size is None or self.is_sweep_stale:
            return
        edge = int(np.ceil(size / WINDOW_STEP) * WINDOW_STEP)
        edge = max(self.window_size.minimum(), min(self.window_size.maximum(), edge))
        self._window_from_rve = edge
        self.window_size.setValue(edge)
        self._update_window_source()
        self._state.log(self.tr("Window set to {edge} voxel from the RVE analysis").format(edge=edge))
        self.go_to_step(TAB_ACF)

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

    def _crossing_text(self, cr, physical, axis, t, units) -> str:
        if cr.found:
            text = f"{cr.value:.1f}"
            if physical and self.result is not None:
                text += f"  ({self.result.physical_lengths[axis][float(t)].value:.3g} {units})"
            if cr.status == "plateau":
                text += "  " + self.tr("plateau")
            return text
        return {"not_crossed": self.tr("not reached"), "invalid": self.tr("no profile")}.get(cr.status, "-")

    def _fill_table(self) -> None:
        res = self.result
        src = self._result_source or {}
        units = src.get("units", "voxel")  # the calibration the analysis was done with, not today's
        spacing = tuple(src.get("spacing", (1.0, 1.0, 1.0)))
        physical = spacing != (1.0, 1.0, 1.0)
        bold = self.table.font()
        bold.setBold(True)
        for r, axis in enumerate(AXES_ROWS):
            for c, t in enumerate(THRESHOLDS):
                text = "-"
                if res is not None and res.status == "ok":
                    text = self._crossing_text(res.lengths[axis][float(t)], physical, axis, t, units)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 0 and text != "-":
                    item.setFont(bold)  # the 1/e column is the one the suggestion uses
                self.table.setItem(r, c, item)

    def _fill_suggestion(self) -> None:
        rec = self.recommendation
        if rec is None:
            self._suggestion.setText(self.tr("Run step 3 to get a subset suggestion."))
            self._suggestion_notes.setText("")
            return
        self._suggestion.setText(
            self.tr("Subset {ws} voxel, step {st}").format(
                ws=" x ".join(str(e + 1) for e in rec.subset), st=" x ".join(str(s) for s in rec.step)
            )
        )
        notes = [self.tr("{factor} x the 1/e correlation length per axis").format(factor=f"{rec.factor:g}")]
        notes += list(rec.notes)
        if self.is_stale:
            notes.append(self.tr("From a previous input: the reference, region, window or calibration changed."))
        self._suggestion_notes.setText("\n".join(notes))

    def _update_status(self) -> None:
        if self._is_running():
            return
        if not self._state.volumes:
            self._status.setText(self.tr("No volume loaded: add the reference volume first."))
            return
        res = self.result
        if res is None:
            self._status.setText(self.tr("Ready."))
        elif res.status != "ok":
            self._status.setText(self.tr("No texture: the window has no grey-value variation."))
        else:
            reach = " x ".join(str(v) for v in res.acf.max_lag)
            parts = [
                self.tr("window {w} voxel, shifts up to {reach}").format(
                    w=" x ".join(str(b - a) for a, b in res.settings["window_xyz"]), reach=reach
                )
            ]
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

    def _update_sweep_status(self) -> None:
        if self._is_running():
            return
        sweep = self.sweep
        if sweep is None or not sweep.levels:
            self._sweep_status.setText(self.tr("No RVE analysis yet."))
            self._btn_use_size.setText(self.tr("Use the stable size as the window"))
            return
        size = self.sweep_size()
        text = (
            self.tr("The correlation length is stable from {size} voxel: use that as the window.").format(size=size)
            if size is not None
            else self.tr("No stable size in this region: enlarge the region or reduce the first size.")
        )
        values = []
        for t in THRESHOLDS:
            d = sweep.decisions.get(float(t))
            if d is not None and d.converged:
                values.append(f"{THRESHOLD_LABELS.get(float(t), f'{t:.2f}')}: {d.reference:.2f} ± {d.tolerance:.2f}")
        if values:
            text += " " + self.tr("Stable lengths [voxel]: {values}.").format(values="; ".join(values))
        if self.is_sweep_stale:
            text += " " + self.tr("(from a previous input: run again)")
        self._sweep_status.setText(text)
        self._btn_use_size.setText(
            self.tr("Use {size} voxel as the window → step 3").format(size=size)
            if size is not None
            else self.tr("Use the stable size as the window")
        )

    # ------------------------------------------------------------------ figures
    def _theme(self) -> dict:
        return PLOT_THEMES.get(str(self.plot_background.currentData() or "dark"), PLOT_THEMES["dark"])

    def _style(self, ax, th: dict) -> None:
        ax.set_facecolor(th["face"])
        ax.tick_params(colors=th["text"], labelsize=FONT["tick"])
        ax.xaxis.label.set_color(th["text"])
        ax.yaxis.label.set_color(th["text"])
        ax.xaxis.label.set_size(FONT["label"])
        ax.yaxis.label.set_size(FONT["label"])
        ax.title.set_color(th["text"])
        for spine in ax.spines.values():
            spine.set_color(th["grid"])
        ax.grid(color=th["grid"], alpha=0.5, lw=0.6)

    def _empty(self, ax, th: dict, text: str) -> None:
        ax.text(0.5, 0.5, text, ha="center", va="center", color=th["text"], fontsize=FONT["label"], transform=ax.transAxes)

    def _threshold_lines(self, ax, th: dict) -> None:
        for t in THRESHOLDS:
            ax.axhline(t, color=th["threshold"], ls="--", lw=1.1, alpha=0.9)
            ax.annotate(
                THRESHOLD_LABELS.get(float(t), f"{t:.2f}"),
                xy=(1.0, t),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                fontsize=FONT["note"],
                color=th["threshold"],
            )

    def _redraw(self) -> None:
        self._draw_profiles()
        self._draw_sweep()

    def reset_view(self) -> None:
        """Back to the full plot after zooming or panning."""
        (self.toolbar_sweep if self.tabs.currentIndex() == TAB_SWEEP else self.toolbar_profiles).home()

    def _update_plot_tools(self) -> None:
        tab = self.tabs.currentIndex()
        self._plot_tools.setVisible(tab != TAB_REGION)
        on_profiles = tab == TAB_ACF
        controls = (self._plot_labels["scale"], self.plot_scale, self._plot_labels["curves"], self.show_band)
        for w in (*controls, *self.curve_checks.values()):
            w.setVisible(on_profiles)

    def _draw_profiles(self) -> None:
        fig = self.fig_profiles
        th = self._theme()
        fig.clear()
        fig.set_facecolor(th["face"])
        ax = fig.add_subplot(1, 1, 1)
        self._style(ax, th)
        res = self.result
        log = str(self.plot_scale.currentData() or "linear") == "log"
        if res is None or res.status != "ok":
            self._empty(ax, th, self.tr("No analysis yet."))
        else:
            handles = []
            for axis in AXES_ROWS:
                if not self.curve_checks[axis].isChecked():
                    continue
                p = res.profiles[axis]
                color, ls = CURVE_STYLES[axis]
                name = axis if axis != "radial" else self.tr("radial")
                y = np.where(p.mean > 0, p.mean, np.nan) if log else p.mean  # log axis: non-positive values are masked
                handles.append(ax.plot(p.lag, y, color=color, ls=ls, lw=1.8, label=name)[0])
                if axis == "radial" and self.show_band.isChecked() and np.isfinite(p.std).any():
                    lo, hi = p.mean - p.std, p.mean + p.std
                    if log:
                        lo = np.where(lo > 0, lo, np.nan)
                    handles.append(ax.fill_between(p.lag, lo, hi, color=color, alpha=0.18, label=self.tr("radial: mean ± 1 std")))
                for t in THRESHOLDS:
                    cr = res.lengths[axis][float(t)]
                    if cr.found:
                        ax.plot([cr.value], [t], "o", color=color, ms=5)
            self._threshold_lines(ax, th)
            if log:
                ax.set_yscale("log")
                positive = np.concatenate(
                    [res.profiles[a].mean[res.profiles[a].mean > 0] for a in AXES_ROWS if self.curve_checks[a].isChecked()]
                    or [np.array([1.0])]
                )
                ax.set_ylim(max(1e-4, float(positive.min()) * 0.7), 1.3)
            if handles:
                ax.legend(handles=handles, fontsize=FONT["legend"], loc="upper right", frameon=False, labelcolor=th["text"])
            src = self._result_source or {}
            box = src.get("range")
            title = self.tr("window {w} voxel").format(w=" x ".join(str(b - a) for a, b in res.settings["window_xyz"]))
            if box is not None:
                title = self.tr("Region {size} voxel").format(size=" x ".join(str(v) for v in box_size(box))) + ", " + title
            ax.set_title(title, fontsize=FONT["note"], color=th["text"])
        ax.set_ylabel(self.tr("autocorrelation") + (self.tr(" (log)") if log else ""))
        ax.set_xlabel(self.tr("shift [voxel]"))
        fig.tight_layout()
        self.canvas_profiles.draw_idle()

    def _draw_sweep(self) -> None:
        """Top: the radial curve of every window size. Bottom: the correlation lengths against the size."""
        fig = self.fig_sweep
        th = self._theme()
        fig.clear()
        fig.set_facecolor(th["face"])
        ax_curves, ax_len = fig.subplots(2, 1, gridspec_kw={"hspace": 0.32})
        self._style(ax_curves, th)
        self._style(ax_len, th)
        sweep = self.sweep
        if sweep is None or not sweep.levels:
            self._empty(ax_curves, th, self.tr("No RVE analysis yet: press Run RVE analysis."))
        else:
            sizes = [max(lvl.size) for lvl in sweep.levels]
            stable = self.sweep_size()
            cmap = colormaps[SWEEP_CMAP]
            n = len(sweep.levels)
            for i, lvl in enumerate(sweep.levels):
                p = getattr(lvl, "radial", None)
                if p is None:
                    continue
                is_stable = stable is not None and sizes[i] == stable
                ax_curves.plot(
                    p.lag,
                    p.mean,
                    color=cmap(0.1 + 0.85 * i / max(1, n - 1)),
                    lw=3.0 if is_stable else 1.3,
                    alpha=1.0 if is_stable else 0.85,
                    label=self.tr("{n} (stable)").format(n=sizes[i]) if is_stable else f"{sizes[i]}",
                )
            self._threshold_lines(ax_curves, th)
            ax_curves.legend(
                fontsize=FONT["note"],
                frameon=False,
                labelcolor=th["text"],
                loc="upper right",
                ncol=2 if n > 8 else 1,
                title=self.tr("window [voxel]"),
                title_fontsize=FONT["note"],
            )
            ax_curves.get_legend().get_title().set_color(th["text"])
            src = self._sweep_source or {}
            box = src.get("range")
            if box is not None:
                ax_curves.set_title(
                    self.tr("Region {size} voxel").format(size=" x ".join(str(v) for v in box_size(box))),
                    fontsize=FONT["note"],
                    color=th["text"],
                )
            verdicts = []
            for t in THRESHOLDS:
                d = sweep.decisions[float(t)]
                label = THRESHOLD_LABELS.get(float(t), f"{t:.2f}")
                line = ax_len.plot(sizes, sweep.means(t), marker="o", ms=5, lw=1.6, label=label)[0]
                if d.converged:
                    ax_len.axvline(sizes[d.start_index], color=line.get_color(), ls="--", lw=1.1)
                    ax_len.axhspan(d.reference - d.tolerance, d.reference + d.tolerance, color=line.get_color(), alpha=0.08)
                    ax_len.axhline(d.reference, color=line.get_color(), ls=":", lw=1.0, alpha=0.9)
                    verdicts.append(
                        self.tr("{name}: stable from {size} voxel, length {value} ± {tol} voxel").format(
                            name=label, size=sizes[d.start_index], value=f"{d.reference:.2f}", tol=f"{d.tolerance:.2f}"
                        )
                    )
                else:
                    verdicts.append(self.tr("{name}: not stable ({why})").format(name=label, why=d.reason))
            ax_len.legend(fontsize=FONT["legend"], frameon=False, labelcolor=th["text"], loc="upper right")
            ax_len.text(
                0.02,
                0.02,
                "\n".join(verdicts),
                ha="left",
                va="bottom",
                fontsize=FONT["note"],
                color=th["text"],
                transform=ax_len.transAxes,
            )
        ax_curves.set_xlabel(self.tr("shift [voxel]"))
        ax_curves.set_ylabel(self.tr("radial autocorrelation"))
        ax_len.set_xlabel(self.tr("window edge [voxel]"))
        ax_len.set_ylabel(self.tr("correlation length [voxel]"))
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
        sweep_tab = self.tabs.currentIndex() == TAB_SWEEP
        name = "texture_rve.png" if sweep_tab else "texture_profiles.png"
        path = self._ask_path(str(self._state.output_dir / name), "PNG (*.png)")
        if path:
            fig = self.fig_sweep if sweep_tab else self.fig_profiles

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
        self.tabs.setTabText(TAB_REGION, self.tr("1. Region"))
        self.tabs.setTabText(TAB_SWEEP, self.tr("2. RVE"))
        self.tabs.setTabText(TAB_ACF, self.tr("3. Autocorrelation"))
        self.tabs.setTabToolTip(TAB_SWEEP, self.tr("Representative volume element: the window size to use"))
        self.export_section.set_title(self.tr("Export"))
        self._lengths_box.setTitle(self.tr("Autocorrelation lengths [voxel]"))
        self._suggestion_box.setTitle(self.tr("Subset suggestion"))
        self._sweep_box.setTitle(self.tr("Result"))
        self._region_banner.setText(
            self.tr(
                "Texture analysis region: the part of the volume whose texture is measured. "
                "This is not the DVC region of interest; the two never affect each other."
            )
        )
        self._headings["region"].setText(self.tr("1. Texture analysis region"))
        self._hints["region"].setText(
            self.tr(
                "Draw the region on the slices (rectangle, ellipse, polygon, brush) or copy the DVC region of interest. "
                "The analysis uses its bounding box, which is drawn dashed."
            )
        )
        self._headings["sweep"].setText(self.tr("2. Representative volume element (RVE) analysis"))
        self._hints["sweep"].setText(
            self.tr(
                "Windows of growing size, all centred in the region, are analysed with the same shifts. "
                "The size from which the correlation length stops changing is the RVE: the window for step 3."
            )
        )
        self._headings["acf"].setText(self.tr("3. Autocorrelation analysis"))
        self._hints["acf"].setText(
            self.tr(
                "One window is compared with its shifted copies inside the region: the curves along x, y, z and over "
                "spherical shells give the correlation lengths and the subset suggestion."
            )
        )
        texts = {
            "window_size": self.tr("Window [voxel]"),
            "sweep_start": self.tr("First size [voxel]"),
            "sweep_step": self.tr("Size step [voxel]"),
            "box": self.tr("Bounding box [voxel]"),
        }
        for key, lab in self.labels.items():
            lab.setText(texts[key])
        self.labels["window_size"].setToolTip(
            self.tr(
                "Edge of the cubic window compared with its shifted copies. The shifts reach (region - window) / 2 "
                "on every axis, so a larger window inside the same region sees shorter shifts."
            )
        )
        self.labels["sweep_start"].setToolTip(self.tr("Edge of the smallest window analysed"))
        self.labels["sweep_step"].setToolTip(self.tr("Growth of the window edge from one size to the next"))
        self.labels["box"].setToolTip(self.tr("Typing a box replaces the drawn region by that box"))
        self._btn_region_all.setText(self.tr("Whole volume"))
        self._btn_region_roi.setText(self.tr("Same as DVC ROI"))
        self._btn_region_roi.setToolTip(
            self.tr("Copy the DVC region of interest of the reference volume into the texture region")
        )
        for axis, lab in self._range_axis_labels.items():
            lab.setToolTip(self.tr("First and last voxel (exclusive) of the box along {axis}").format(axis=axis))
        self._next["region"].setText(self.tr("Next: RVE analysis →"))
        self._next["sweep"].setText(self.tr("Next: autocorrelation →"))
        self._btn_analyse.setText(self.tr("Run autocorrelation analysis"))
        self._btn_cancel.setText(self.tr("Cancel"))
        self._btn_sweep.setText(self.tr("Run RVE analysis"))
        self._btn_sweep_cancel.setText(self.tr("Cancel"))
        self._btn_apply.setText(self.tr("Apply to parameters"))
        self._btn_csv.setText(self.tr("Save profiles as CSV..."))
        self._btn_json.setText(self.tr("Save summary as JSON..."))
        self._btn_png.setText(self.tr("Save image as PNG..."))
        self._btn_reset_view.setText(self.tr("Reset view"))
        self._btn_reset_view.setToolTip(
            self.tr("Undo zooming and panning (drag with the toolbar's magnifier or hand to zoom or pan)")
        )
        self._plot_labels["background"].setText(self.tr("Background"))
        self._plot_labels["scale"].setText(self.tr("Scale"))
        self._plot_labels["curves"].setText(self.tr("Curves"))
        for i, text in enumerate((self.tr("Dark"), self.tr("White"), self.tr("Grey"))):
            self.plot_background.setItemText(i, text)
        for i, text in enumerate((self.tr("Linear"), self.tr("Log"))):
            self.plot_scale.setItemText(i, text)
        for axis, cb in self.curve_checks.items():
            cb.setText(axis if axis != "radial" else self.tr("radial"))
        self.show_band.setText(self.tr("± 1 std band"))
        self.show_band.setToolTip(self.tr("Spread of the radial curve inside each shell (direction dependence plus noise)"))
        self.table.setHorizontalHeaderLabels([THRESHOLD_LABELS.get(float(t), f"{t:.2f}") for t in THRESHOLDS])
        self.table.setVerticalHeaderLabels(["x", "y", "z", self.tr("radial")])
        self._table_hint.setText(
            self.tr(
                'Distance at which the correlation drops to 1/e, 0.1 and 0.01. "not reached": still above the threshold '
                'within the shifts the region allows; "no profile": no valid curve; '
                '"plateau": the curve flattens at the threshold.'
            )
        )
        self.region.retranslate_ui()
        self._fill_table()
        self._fill_suggestion()
        self._redraw()
        self._update_range_info()
        self._update_window_source()
        self._update_status()
        self._update_sweep_status()
        self._update_steps()


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
