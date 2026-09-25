"""The Analysis tab of the post-processing window: statistics of a result, frame by frame and region by region.

Left: the three slices of the field (as measured or with the motion removed, or the nodes used) with the regions
outlined; regions and lines are drawn there. Right: what to take statistics of, over which region, which nodes
count, which motion to remove (fitted over which region, and whether the main window and the exports show it),
the regions, exports. Below: the summary table with the confidence interval of each mean, histograms, the series
over frames, the comparison of the regions, the profile along an axis, the line and the virtual extensometer,
the noise floor and the homogeneous deformation.

Every computation runs on a worker thread and is tagged with the controls it was made with: the current frame
recomputes on its own after any change (a newer request waits for the running one); the long jobs -- series,
noise floor, profiles over frames, the line over frames with the extensometer -- run on request and are marked
out of date when the controls move on. Nothing here changes the result: corrections are views (``al_dvc.analysis``).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, fields

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from al_dvc.analysis import FIELD_GROUPS, FILTER_REASONS, MOTION_KINDS, Correction, NodeFilter, available_groups, field_unit

from .analysis_charts import draw_histograms, fmt
from .analysis_export import TAB_LINE, TAB_REGIONS, AnalysisExportMixin
from .analysis_jobs import (
    NODES_USED,
    AnalysisCancelled,  # noqa: F401 - re-exported for scripts
    CurrentFrame,
    Worker,
    compute_current,
    histogram,
    line_job,
    noise_job,
    profiles_job,
    same_source,
    series_job,
)
from .analysis_pages import PROFILE_AXES, REGION_COLUMNS, TABLE_COLUMNS, AnalysisPagesMixin
from .analysis_regions import RegionDrawer, RegionsPanel, draw_overlays, grid_bounds
from .app_state import AppState
from .field_canvas import FieldSliceCanvas
from .names import field_name
from .theme import SIDE_COLUMN
from .theme_manager import connect_theme
from .widgets import CollapsibleSection, combo, dspin, form_label, guard_wheel, make_form, spin

logger = logging.getLogger(__name__)

__all__ = ["NODES_USED", "AnalysisTab", "CurrentFrame", "compute_current", "same_source"]

_histogram = histogram  # the name the tests and scripts know
_fmt = fmt

SIDEBAR_WIDTH = 330
DEBOUNCE_MS = 150
LONG_KINDS = ("series", "noise", "profiles", "line")
ALL_NODES = -1  # the region combos' "every node" entry


class AnalysisTab(AnalysisExportMixin, AnalysisPagesMixin, QWidget):
    """Statistics of the result in ``AppState``; see the module docstring."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._updating = False
        self._pushing = False  # this tab is setting the display correction (its echo is not an outside change)
        self._current_worker: Worker | None = None
        self._current_pending = False
        self._long_worker: Worker | None = None
        self.current: CurrentFrame | None = None
        self.long: dict[str, tuple] = {}  # kind -> (key, payload) of the last finished long job
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.request_current)
        self.sections: dict[str, CollapsibleSection] = {}
        self.labels: dict[str, QLabel] = {}

        # ---- canvas and frame (left)
        self.canvas = FieldSliceCanvas()
        self.drawer = RegionDrawer(self.canvas, self)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.canvas, 1)
        self.canvas_hint = QLabel()
        self.canvas_hint.setObjectName("hint")
        self.canvas_hint.setWordWrap(True)
        left_layout.addWidget(self.canvas_hint)
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("<")
        self._btn_prev.setFixedWidth(36)
        self._btn_next = QPushButton(">")
        self._btn_next.setFixedWidth(36)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self._frame_label = QLabel()
        self._frame_label.setObjectName("sectionTitle")
        nav.addWidget(self._btn_prev)
        nav.addWidget(self.frame_slider, 1)
        nav.addWidget(self._btn_next)
        nav.addWidget(self._frame_label)
        left_layout.addLayout(nav)

        # ---- sidebar (right)
        side = QWidget()
        side.setFixedWidth(SIDEBAR_WIDTH)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(4)

        what = CollapsibleSection()
        form = make_form()
        self.group = combo([])
        self.shown = combo([])
        self.stats_region_combo = combo([])
        for key, w in (("group", self.group), ("shown", self.shown), ("stats_region", self.stats_region_combo)):
            lab = form_label()
            self.labels[key] = lab
            form.addRow(lab, w)
        what.add_layout(form)
        self.sections["what"] = what
        side_layout.addWidget(what)

        nodes = CollapsibleSection()
        self.converged_only = QCheckBox()
        self.converged_only.setChecked(True)
        self.drop_outliers = QCheckBox()
        self.drop_outliers.setChecked(True)
        self.drop_cut = QCheckBox()
        nodes.add_widget(self.converged_only)
        nodes.add_widget(self.drop_outliers)
        nform = make_form()
        self.min_zncc = dspin(0.0, 1.0, 2)
        self.min_zncc.setSingleStep(0.05)
        self.edge_layers = spin(0, 10, 1)
        for key, w in (("min_zncc", self.min_zncc), ("edge_layers", self.edge_layers)):
            lab = form_label()
            self.labels[key] = lab
            nform.addRow(lab, w)
        nodes.add_layout(nform)
        nodes.add_widget(self.drop_cut)
        self.nodes_info = QLabel()
        self.nodes_info.setObjectName("hint")
        self.nodes_info.setWordWrap(True)
        nodes.add_widget(self.nodes_info)
        self.sections["nodes"] = nodes
        side_layout.addWidget(nodes)

        motion = CollapsibleSection()
        mform = make_form()
        self.motion = combo([])
        for kind in MOTION_KINDS:
            self.motion.addItem(kind, kind)
        self.fit_region_combo = combo([])
        for key, w in (("motion", self.motion), ("fit_region", self.fit_region_combo)):
            lab = form_label()
            self.labels[key] = lab
            mform.addRow(lab, w)
        motion.add_layout(mform)
        self.apply_main = QCheckBox()
        motion.add_widget(self.apply_main)
        self.motion_info = QLabel()
        self.motion_info.setObjectName("hint")
        self.motion_info.setWordWrap(True)
        self.motion_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        motion.add_widget(self.motion_info)
        self.sections["motion"] = motion
        side_layout.addWidget(motion)

        regions = CollapsibleSection()
        self.regions_panel = RegionsPanel(state)
        regions.add_widget(self.regions_panel)
        self.sections["regions"] = regions
        side_layout.addWidget(regions)

        export = CollapsibleSection()
        self._btn_csv = QPushButton()
        self._btn_json = QPushButton()
        self._btn_png = QPushButton()
        self._btn_copy = QPushButton()
        for b in (self._btn_copy, self._btn_csv, self._btn_json, self._btn_png):
            export.add_widget(b)
        self.status = QLabel()
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        export.add_widget(self.status)
        self.sections["export"] = export
        side_layout.addWidget(export)
        side_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setObjectName(SIDE_COLUMN)
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setFixedWidth(SIDEBAR_WIDTH + 18)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        guard_wheel(side)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(left, 1)
        top_layout.addWidget(scroll, 0)

        # ---- results (bottom)
        self.results_tabs = QTabWidget()
        self._run: dict[str, QPushButton] = {}
        self._cancel: dict[str, QPushButton] = {}
        self._bars: dict[str, QProgressBar] = {}
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_tabs.addTab(self.table, "")
        self.hist_figure, self.hist_canvas = self._figure()
        self.results_tabs.addTab(self.hist_canvas, "")
        self.results_tabs.addTab(self._build_series_page(), "")
        self.results_tabs.addTab(self._build_regions_page(), "")
        self.results_tabs.addTab(self._build_profile_page(), "")
        self.results_tabs.addTab(self._build_line_page(), "")
        self.results_tabs.addTab(self._build_noise_page(), "")
        self.hom_table = QTableWidget(0, 3)
        self.hom_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_tabs.addTab(self.hom_table, "")
        self._btn_series, self._btn_cancel, self._progress = self._run["series"], self._cancel["series"], self._bars["series"]
        self._btn_noise = self._run["noise"]

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.results_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(splitter)

        # ---- wiring
        self._btn_prev.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() - 1))
        self._btn_next.clicked.connect(lambda: self.frame_slider.setValue(self.frame_slider.value() + 1))
        self.frame_slider.valueChanged.connect(lambda _v: self._changed())
        self.group.currentIndexChanged.connect(self._on_group)
        for w in (self.shown, self.stats_region_combo, self.fit_region_combo, self.motion, self.profile_axis):
            w.currentIndexChanged.connect(lambda _i: self._changed())
        for w in (self.converged_only, self.drop_outliers, self.drop_cut):
            w.toggled.connect(lambda _v: self._changed())
        self.min_zncc.valueChanged.connect(lambda _v: self._changed())
        self.edge_layers.valueChanged.connect(lambda _v: self._changed())
        for w in self.line_spins:
            w.valueChanged.connect(lambda _v: self._on_line_edited())
        self.apply_main.toggled.connect(self._on_apply)
        self.reduction.currentIndexChanged.connect(lambda _i: self._draw_series())
        self.compare_regions.toggled.connect(lambda _v: self._mark_long_stale())
        for w in self.nominal:
            w.valueChanged.connect(lambda _v: self._mark_long_stale())
        self._run["series"].clicked.connect(self.compute_series)
        self._run["noise"].clicked.connect(self.compute_noise_floor)
        self._run["profiles"].clicked.connect(self.compute_profiles)
        self._run["line"].clicked.connect(self.compute_line)
        self.min_share.valueChanged.connect(lambda _v: self._mark_long_stale())
        for b in self._cancel.values():
            b.clicked.connect(self.cancel)
        self._btn_pick_line.clicked.connect(lambda: self._start_drawing("line"))
        self._btn_copy.clicked.connect(self.copy_table)
        self._btn_csv.clicked.connect(lambda: self.export_csv())
        self._btn_json.clicked.connect(lambda: self.export_json())
        self._btn_png.clicked.connect(lambda: self.export_png())
        self.results_tabs.currentChanged.connect(self._on_results_tab)
        self.regions_panel.draw_requested.connect(self._start_drawing)
        self.regions_panel.selection_changed.connect(lambda _rid: self.canvas.redraw())
        self.drawer.finished.connect(self._on_drawn)
        self.drawer.cancelled.connect(lambda: self.canvas_hint.setText(self.tr("Drawing cancelled.")))
        self.canvas.set_decorator(self._decorate)
        self._state.results_changed.connect(self.load)
        self._state.regions_changed.connect(self._on_regions_changed)
        self._state.correction_changed.connect(self._sync_apply)
        self._state.analysis_restored.connect(self.restore_settings)
        connect_theme(self.apply_theme)
        saved = dict(self._state.analysis_settings or {})  # a session loaded before this tab existed
        self.retranslate_ui()
        self.load()  # saves the default controls into the state ...
        if saved:
            self.restore_settings(saved)  # ... so the session's settings are the ones read before

    # ------------------------------------------------------------------ pages
    # ------------------------------------------------------------------ controls
    def node_filter(self) -> NodeFilter:
        return NodeFilter(
            converged_only=self.converged_only.isChecked(),
            drop_outliers=self.drop_outliers.isChecked(),
            min_zncc=float(self.min_zncc.value()),
            edge_layers=int(self.edge_layers.value()),
            drop_cut=self.drop_cut.isChecked(),
        )

    def fields(self) -> tuple[str, ...]:
        return FIELD_GROUPS.get(self.group.currentData() or "displacement", ())

    def frame(self) -> int:
        return int(self.frame_slider.value())

    def motion_kind(self) -> str:
        return str(self.motion.currentData() or "none")

    def _region_of(self, box) -> object:
        rid = box.currentData()
        if rid is None or rid == ALL_NODES:
            return None
        return self.regions_panel.region(int(rid))

    def stats_region(self):
        """The region the statistics are taken over, ``None`` for every node."""
        return self._region_of(self.stats_region_combo)

    def fit_region(self):
        """The region the removed motion is fitted over, ``None`` for every node."""
        return self._region_of(self.fit_region_combo)

    def compare_field(self) -> str:
        shown = self.shown.currentData()
        return shown if shown and shown != NODES_USED else (self.fields() or ("disp_u",))[0]

    def line_points(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        v = [float(w.value()) for w in self.line_spins]
        return (v[0], v[1], v[2]), (v[3], v[4], v[5])

    def set_line(self, p0, p1) -> None:
        self._updating = True
        try:
            for w, v in zip(self.line_spins, [*p0, *p1]):
                w.setValue(float(v))
        finally:
            self._updating = False
        self._on_line_edited()

    def _line_ok(self) -> bool:
        p0, p1 = self.line_points()
        return p0 != p1

    def current_key(self) -> tuple:
        return (
            self._state.results,
            self.frame(),
            self.group.currentData(),
            self.node_filter(),
            self.motion_kind(),
            self.shown.currentData(),
            self.stats_region(),
            self.fit_region(),
            tuple(self._state.regions),
            self.profile_axis.currentText(),
            self.line_points() if self._line_ok() else None,
        )

    def series_key(self) -> tuple:
        """What the series over frames depends on (not the frame on screen, not the applied displacement)."""
        compare = self.compare_regions.isChecked()
        return (
            self._state.results,
            self.group.currentData(),
            self.node_filter(),
            self.motion_kind(),
            None if compare else self.stats_region(),
            self.fit_region(),
            compare,
            tuple(self._state.regions) if compare else (),
            int(self.min_share.value()),
        )

    def noise_key(self) -> tuple:
        """What the noise floor depends on: the frame, the nodes, the region and the applied displacement (not the
        motion shown, which the noise floor does not use)."""
        return (
            self._state.results,
            self.frame(),
            self.node_filter(),
            tuple(float(w.value()) for w in self.nominal),
            self.stats_region(),
        )

    def profiles_key(self) -> tuple:
        return (
            self._state.results,
            self.compare_field(),
            self.node_filter(),
            self.motion_kind(),
            self.fit_region(),
            self.stats_region(),
            self.profile_axis.currentText(),
        )

    def line_key(self) -> tuple:
        return (
            self._state.results,
            self.node_filter(),
            self.line_points(),
            self.compare_field(),
            self.motion_kind(),
            self.fit_region(),
        )

    def long_key(self, kind: str) -> tuple:
        return {
            "series": self.series_key,
            "noise": self.noise_key,
            "profiles": self.profiles_key,
            "line": self.line_key,
        }[kind]()

    def long_is_current(self, kind: str) -> bool:
        done = self.long.get(kind)
        return done is not None and same_source(done[0], self.long_key(kind))

    @property
    def series(self):
        return self.long.get("series")

    @property
    def noise(self):
        return self.long.get("noise")

    def series_is_current(self) -> bool:
        return self.long_is_current("series")

    def noise_is_current(self) -> bool:
        return self.long_is_current("noise")

    # ------------------------------------------------------------------ loading and the region lists
    def load(self) -> None:
        """A new result (or none): rebuild the lists and start over."""
        res = self._state.results
        self._updating = True
        try:
            current = self.group.currentData() or self._state.analysis_settings.get("group")
            self.group.clear()
            for g in available_groups(res):
                self.group.addItem(self._group_label(g), g)
            idx = self.group.findData(current) if current else -1
            self.group.setCurrentIndex(idx if idx >= 0 else 0)
            n = len(res.result_disp) if res is not None else 0
            self.frame_slider.setRange(0, max(0, n - 1))
            self._fill_shown()
            self._fill_region_combos()
        finally:
            self._updating = False
        background = None
        if res is not None and self._state.volumes and not self._state.volumes[0].missing:
            try:
                vol = self._state.volume_array(0)
                if tuple(vol.shape) == tuple(res.volume_shape):
                    background = vol
            except Exception as exc:
                self._state.log(f"statistics: cannot load the reference volume: {exc}", "warning")
        self.drawer.cancel()
        self.regions_panel.set_result(res)
        self.canvas.set_data(res, background)
        if res is not None and not self._line_ok():
            self._default_line(res)
        self.long = {}
        self._changed()

    def _default_line(self, res) -> None:
        """Along x through the middle of the node grid."""
        lo, hi = grid_bounds(res)
        c = 0.5 * (lo + hi)
        self.set_line((lo[0], c[1], c[2]), (hi[0], c[1], c[2]))

    def _fill_shown(self) -> None:
        current = self.shown.currentData() or self._state.analysis_settings.get("shown")
        self.shown.clear()
        for f in self.fields():
            self.shown.addItem(field_name(f), f)
        self.shown.addItem(self.tr("Nodes used"), NODES_USED)
        idx = self.shown.findData(current) if current else -1
        self.shown.setCurrentIndex(idx if idx >= 0 else 0)

    def _fill_region_combos(self) -> None:
        for box in (self.stats_region_combo, self.fit_region_combo):
            current = box.currentData()
            box.blockSignals(True)
            box.clear()
            box.addItem(self.tr("All nodes"), ALL_NODES)
            for r in self._state.regions:
                box.addItem(r.name, r.id)
            idx = box.findData(current) if current is not None else -1
            box.setCurrentIndex(idx if idx >= 0 else 0)
            box.blockSignals(False)

    def _on_regions_changed(self) -> None:
        self._fill_region_combos()
        self.canvas.redraw()
        self._changed()

    def _on_group(self, _index: int) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._fill_shown()
        finally:
            self._updating = False
        self._changed()

    def _on_line_edited(self) -> None:
        if self._updating:
            return
        if self.results_tabs.currentIndex() == TAB_LINE:
            self.canvas.redraw()
        self._changed()

    def _on_results_tab(self, _index: int) -> None:
        self.canvas.redraw()  # the line is outlined on the slices while its tab is open
        self._update_buttons()

    def _changed(self) -> None:
        if self._updating:
            return
        n = self.frame_slider.maximum() + 1
        self._frame_label.setText(self.tr("Frame {k}/{n}").format(k=self.frame() + 1, n=n))
        self._btn_prev.setEnabled(self.frame() > 0)
        self._btn_next.setEnabled(self.frame() < self.frame_slider.maximum())
        if self.apply_main.isChecked():
            self._push_correction()
        self._save_settings()
        self._debounce.start()
        self._mark_long_stale()

    # ------------------------------------------------------------------ the correction in the main window
    def correction(self) -> Correction:
        return Correction(motion=self.motion_kind(), node_filter=self.node_filter(), fit_region=self.fit_region())

    def _push_correction(self) -> None:
        self._pushing = True
        try:
            self._state.set_display_correction(self.correction())
        finally:
            self._pushing = False

    def _on_apply(self, checked: bool) -> None:
        if self._updating:
            return
        self._pushing = True
        try:
            self._state.set_display_correction(self.correction() if checked else None)
        finally:
            self._pushing = False
        self._save_settings()

    def _sync_apply(self) -> None:
        """The correction was changed outside this tab ("As measured" in the main window, a session)."""
        if self._pushing:
            return
        self.apply_main.blockSignals(True)
        self.apply_main.setChecked(self._state.display_correction is not None)
        self.apply_main.blockSignals(False)
        self._save_settings()

    # ------------------------------------------------------------------ settings kept with the session
    def settings(self) -> dict:
        stats_r, fit_r = self.stats_region(), self.fit_region()
        return {
            "group": self.group.currentData(),
            "shown": self.shown.currentData(),
            "stats_region": None if stats_r is None else stats_r.id,
            "fit_region": None if fit_r is None else fit_r.id,
            "motion": self.motion_kind(),
            "node_filter": asdict(self.node_filter()),
            "apply": self.apply_main.isChecked(),
            "profile_axis": self.profile_axis.currentText(),
            "line": [list(p) for p in self.line_points()],
            "compare_regions": self.compare_regions.isChecked(),
            "min_share": int(self.min_share.value()),
            "reduction": self.reduction.currentData(),
            "nominal": [float(w.value()) for w in self.nominal],
        }

    def _save_settings(self) -> None:
        # without a result the field lists are empty: saving now would lose a session's choice of field before the
        # result it applies to is loaded
        if not self._updating and self._state.results is not None:
            self._state.analysis_settings = self.settings()

    def restore_settings(self, settings: dict | None = None) -> None:
        """Put the controls back as ``settings`` (default ``AppState.analysis_settings``, a loaded session) says;
        anything invalid or unknown keeps the control as it is."""
        s = dict((self._state.analysis_settings if settings is None else settings) or {})
        self._updating = True
        try:
            self._fill_region_combos()

            def pick(box, value) -> None:
                idx = box.findData(value) if value is not None else -1
                if idx >= 0:
                    box.setCurrentIndex(idx)

            pick(self.group, s.get("group"))
            self._fill_shown()
            pick(self.shown, s.get("shown"))
            pick(self.stats_region_combo, s.get("stats_region", ALL_NODES) or ALL_NODES)
            pick(self.fit_region_combo, s.get("fit_region", ALL_NODES) or ALL_NODES)
            pick(self.motion, s.get("motion"))
            pick(self.reduction, s.get("reduction"))
            if s.get("profile_axis") in PROFILE_AXES:
                self.profile_axis.setCurrentText(s["profile_axis"])
            nf = s.get("node_filter")
            if isinstance(nf, dict):
                known = {f.name for f in fields(NodeFilter)}
                try:
                    f = NodeFilter(**{k: v for k, v in nf.items() if k in known})
                    self.converged_only.setChecked(bool(f.converged_only))
                    self.drop_outliers.setChecked(bool(f.drop_outliers))
                    self.min_zncc.setValue(float(f.min_zncc))
                    self.edge_layers.setValue(int(f.edge_layers))
                    self.drop_cut.setChecked(bool(f.drop_cut))
                except (TypeError, ValueError):
                    pass
            line = s.get("line")
            try:
                p = np.asarray(line, dtype=np.float64).reshape(2, 3)
                if np.all(np.isfinite(p)):
                    for w, v in zip(self.line_spins, p.ravel()):
                        w.setValue(float(v))
            except (TypeError, ValueError):
                pass
            nominal = s.get("nominal")
            if isinstance(nominal, list) and len(nominal) == 3:
                try:
                    for w, v in zip(self.nominal, nominal):
                        w.setValue(float(v))
                except (TypeError, ValueError):
                    pass
            self.compare_regions.setChecked(bool(s.get("compare_regions", False)))
            if isinstance(s.get("min_share"), (int, float)):
                self.min_share.setValue(int(s["min_share"]))
            self.apply_main.setChecked(self._state.display_correction is not None)
        finally:
            self._updating = False
        self.canvas.redraw()
        self._changed()

    # ------------------------------------------------------------------ drawing regions and the line
    def _start_drawing(self, kind: str) -> None:
        if self._state.results is None:
            return
        self.drawer.start(kind)
        hints = {
            "rect": self.tr("Drag a rectangle on one of the slices. Esc cancels."),
            "ellipse": self.tr("Drag an ellipse on one of the slices. Esc cancels."),
            "polygon": self.tr("Click the corners on one slice; double-click, right-click or Enter closes it. Esc cancels."),
            "line": self.tr("Click the two ends of the line on the slices (each at the slice shown). Esc cancels."),
        }
        if kind == "line":
            self.results_tabs.setCurrentIndex(TAB_LINE)
        self.canvas_hint.setText(hints[kind])

    def _on_drawn(self, drawn: dict) -> None:
        self.canvas_hint.setText("")
        if drawn.get("kind") == "line":
            self.set_line(drawn["p0"], drawn["p1"])
            return
        region = self.regions_panel.add_drawn(drawn["plane"], drawn["kind"], drawn["points"])
        if region is not None:
            self.canvas_hint.setText(
                self.tr("{name} added: it goes through the whole node grid; set its depth in Regions.").format(name=region.name)
            )

    def _decorate(self, axes, indices, shape) -> None:
        show_line = self.results_tabs.currentIndex() == TAB_LINE and self._line_ok()
        draw_overlays(
            axes,
            indices,
            shape,
            self._state.regions,
            selected=self.regions_panel.selected_id(),
            line=self.line_points() if show_line else None,
        )

    # ------------------------------------------------------------------ the current frame
    def request_current(self) -> None:
        """Compute the current frame's statistics on the worker (after the running job, if any)."""
        res = self._state.results
        if res is None or not res.result_disp:
            self.current = None
            self.canvas.set_override(None)
            self._show_current()
            return
        if self._current_worker is not None and self._current_worker.isRunning():
            self._current_pending = True
            return
        key = self.current_key()
        flds = self.fields()
        nf, motion, shown = self.node_filter(), self.motion_kind(), self.shown.currentData() or flds[0]
        frame = self.frame()
        stats_r, fit_r, regions = self.stats_region(), self.fit_region(), tuple(self._state.regions)
        axis = self.profile_axis.currentText()
        line = key[-1]

        def job(_stop, _progress):
            return compute_current(res, frame, flds, nf, motion, shown, key, stats_r, fit_r, regions, axis, line)

        worker = Worker(key, job, self)
        worker.done.connect(self._on_current_done)
        worker.failed.connect(self._on_current_failed)
        self._current_worker = worker
        self.status.setText(self.tr("Computing..."))
        worker.start()

    def _after_current(self) -> None:
        if self._current_pending:
            self._current_pending = False
            self.request_current()

    def _on_current_done(self, key, payload: CurrentFrame) -> None:
        if not same_source(key, self.current_key()):
            self._after_current()  # the controls moved on; the pending request is the one that counts
            return
        self.current = payload
        self._show_current()
        self._after_current()

    def _on_current_failed(self, _key, message: str) -> None:
        self.status.setText(self.tr("Statistics failed: {error}").format(error=message))
        self._state.log(self.tr("Statistics failed: {error}").format(error=message), "error")
        self._after_current()

    def wait(self, timeout_ms: int = 60_000) -> bool:
        """Process events until no statistics job is running or waiting (tests, scripts)."""
        import time

        from PySide6.QtWidgets import QApplication

        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            QApplication.processEvents()
            busy = self._debounce.isActive() or self._current_pending
            busy |= any(w is not None and w.isRunning() for w in (self._current_worker, self._long_worker))
            if not busy:
                QApplication.processEvents()
                return True
            QThread.msleep(10)
        return False

    def _clear_current(self) -> None:
        self.table.setRowCount(0)
        self.motion_info.setText("")
        self.hist_figure.clear()
        self.hist_canvas.draw_idle()
        self.hom_table.setRowCount(0)
        self.regions_table.setRowCount(0)
        for fig, canvas in (
            (self.regions_figure, self.regions_canvas),
            (self.profile_figure, self.profile_canvas),
            (self.line_figure, self.line_canvas),
        ):
            fig.clear()
            canvas.draw_idle()

    def _show_current(self) -> None:
        cur = self.current
        res = self._state.results
        self.status.setText("")
        if cur is None or res is None:
            self._clear_current()
            self.nodes_info.setText(self.tr("No results yet: run an analysis first."))
            self._update_buttons()
            return
        if cur.error:  # nothing from the previous fit may stay on screen
            self._clear_current()
            self.nodes_info.setText(self.tr("Too few nodes for this correction: {error}").format(error=cur.error))
            self.canvas.set_override(None)
            self._update_buttons()
            return
        self._show_nodes(cur.selection)
        self._show_motion(cur.fit)
        self._show_table(cur.stats)
        draw_histograms(self.hist_figure, cur.stats, cur.histograms)
        self.hist_canvas.draw_idle()
        self._show_homogeneous(cur.homogeneous)
        self._show_regions(cur)
        self._draw_profile()
        self._draw_line()
        shown = self.shown.currentData()
        label = self.tr("Nodes used (1) and left out (0)") if shown == NODES_USED else self._label_with_unit(shown)
        if self.motion_kind() != "none" and shown != NODES_USED:
            label += " - " + self.tr("motion removed")
        self.canvas.set_view(frame=self.frame(), field=shown if shown != NODES_USED else "disp_magnitude")
        self.canvas.set_override(cur.canvas, label)
        self._update_buttons()

    def _unit(self, name: str | None) -> str:
        return field_unit(name, self._state.results.dvc_para) if name and self._state.results is not None else ""

    def _label_with_unit(self, name: str | None) -> str:
        if not name:
            return ""
        unit = self._unit(name)
        text = field_name(name)
        return f"{text} [{unit}]" if unit and unit != "deg" else text

    def _length_unit(self) -> str:
        return self._unit("disp_u")

    def _show_nodes(self, selection) -> None:
        removed = [
            self.tr("{reason} {n}").format(reason=self._reason_label(r), n=f"{selection.removed.get(r, 0):,}")
            for r in FILTER_REASONS
            if selection.removed.get(r, 0)
        ]
        text = self.tr("{n} of {c} nodes used").format(n=f"{selection.n:,}", c=f"{selection.candidates:,}")
        region = self.stats_region()
        if region is not None:
            text += " " + self.tr("in {region}").format(region=region.name)
        if removed:
            text += " (" + ", ".join(removed) + ")"
        self.nodes_info.setText(text)

    def _show_motion(self, fit) -> None:
        kind = self.motion_kind()
        unit = self._length_unit()
        if fit is None:
            self.motion_info.setText(self.tr("Nothing removed: the displacements as measured."))
            return
        t = ", ".join(fmt(v) for v in fit.translation)
        lines = [self.tr("Translation: {t} {unit}").format(t=t, unit=unit)]
        if kind in ("rigid", "affine"):
            axis = fit.rotation_vector_deg
            norm = float(np.linalg.norm(axis))
            direction = ", ".join(fmt(v, 3) for v in (axis / norm if norm > 0 else axis))
            lines.append(self.tr("Rotation: {a} deg about ({d})").format(a=fmt(fit.rotation_deg), d=direction))
            lines.append(self.tr("Euler x, y, z: {e} deg").format(e=", ".join(fmt(v, 3) for v in fit.euler_xyz_deg)))
        where = self.fit_region()
        lines.append(
            self.tr("Residual RMS: {r} {unit} over {n} nodes").format(r=fmt(fit.residual_rms), unit=unit, n=f"{fit.n:,}")
            + ("" if where is None else " " + self.tr("of {region}").format(region=where.name))
        )
        notes = {
            "translation": self.tr("Strain unchanged: a translation does not deform."),
            "rigid": self.tr("Strain recomputed without the rotation; Green-Lagrange and principal stretches do not change."),
            "affine": self.tr("Displacement: the non-affine part. Strain unchanged; the fitted strain is in Homogeneous."),
        }
        lines.append(notes.get(kind, ""))
        self.motion_info.setText("\n".join(line for line in lines if line))

    # ------------------------------------------------------------------ long jobs
    def _start_long(self, kind: str, fn, info: QLabel) -> None:
        if self._long_worker is not None and self._long_worker.isRunning():
            return
        worker = Worker((kind, self.long_key(kind)), fn, self)
        worker.done.connect(self._on_long_done)
        worker.failed.connect(self._on_long_failed)
        bar = self._bars.get(kind)
        if bar is not None:
            worker.progress.connect(lambda f, b=bar: b.setValue(int(1000 * f)))
            bar.setValue(0)
        self._long_worker = worker
        info.setText(self.tr("Computing every frame...") if kind != "noise" else self.tr("Computing..."))
        self._update_buttons()
        worker.start()

    def _results_or_none(self):
        res = self._state.results
        return res if res is not None and res.result_disp else None

    def compute_series(self) -> None:
        res = self._results_or_none()
        if res is None:
            return
        flds, nf, motion = self.fields(), self.node_filter(), self.motion_kind()
        compare = self.compare_regions.isChecked()
        stats_r, fit_r, regions = self.stats_region(), self.fit_region(), tuple(self._state.regions)
        share = float(self.min_share.value()) / 100.0

        def job(stop, progress):
            return series_job(res, flds, nf, motion, stats_r, fit_r, compare, regions, share, stop, progress)

        self._start_long("series", job, self.series_info)

    def compute_noise_floor(self) -> None:
        res = self._results_or_none()
        if res is None:
            return
        frame, nf, region = self.frame(), self.node_filter(), self.stats_region()
        nominal = [float(w.value()) for w in self.nominal]

        def job(stop, progress):
            return noise_job(res, frame, nf, nominal, region, stop, progress)

        self._start_long("noise", job, self.noise_info)

    def compute_profiles(self) -> None:
        res = self._results_or_none()
        if res is None:
            return
        field, nf, motion = self.compare_field(), self.node_filter(), self.motion_kind()
        fit_r, stats_r, axis = self.fit_region(), self.stats_region(), self.profile_axis.currentText()

        def job(stop, progress):
            return profiles_job(res, field, nf, motion, fit_r, stats_r, axis, stop, progress)

        self._start_long("profiles", job, self.profile_info)

    def compute_line(self) -> None:
        """The line over every frame, the field at its ends and the extensometer between them."""
        res = self._results_or_none()
        if res is None or not self._line_ok():
            return
        p0, p1 = self.line_points()
        field, nf, motion, fit_r = self.compare_field(), self.node_filter(), self.motion_kind(), self.fit_region()

        def job(stop, progress):
            return line_job(res, p0, p1, field, nf, motion, fit_r, stop, progress)

        self._start_long("line", job, self.line_info)

    def cancel(self) -> None:
        """Stop a long job at the next frame and drop a waiting current-frame request (the one running finishes:
        it takes a fraction of a second)."""
        self._debounce.stop()
        self._current_pending = False
        if self._long_worker is not None and self._long_worker.isRunning():
            self._long_worker.cancel()

    def is_running(self) -> bool:
        return any(w is not None and w.isRunning() for w in (self._current_worker, self._long_worker))

    def shutdown(self, timeout_ms: int = 30_000) -> bool:
        self.cancel()
        self.drawer.cancel()
        self._debounce.stop()
        self._current_pending = False
        ok = True
        for w in (self._current_worker, self._long_worker):
            if w is not None and w.isRunning():
                ok &= w.wait(timeout_ms)
        return ok

    def _info_of(self, kind: str) -> QLabel:
        return {"series": self.series_info, "noise": self.noise_info, "profiles": self.profile_info}.get(kind, self.line_info)

    def _on_long_done(self, key, payload) -> None:
        kind, source = key
        if source[0] is not self._state.results:  # a different result arrived while this ran
            self._update_buttons()
            return
        self.long[kind] = (source, payload)
        {
            "series": self._draw_series,
            "noise": self._show_noise,
            "profiles": self._draw_profile,
            "line": self._draw_line,
        }[kind]()
        if kind in self._bars:
            self._bars[kind].setValue(1000)
        if kind == "profiles":
            self.profile_info.setText(self.tr("{n} frames").format(n=len(payload)))
        self._update_buttons()

    def _on_long_failed(self, key, message: str) -> None:
        kind, _source = key
        target = self._info_of(kind)
        target.setText(self.tr("Cancelled.") if not message else self.tr("Statistics failed: {error}").format(error=message))
        if message:
            self._state.log(self.tr("Statistics failed: {error}").format(error=message), "error")
        if kind in self._bars:
            self._bars[kind].setValue(0)
        self._update_buttons()

    def _mark_long_stale(self) -> None:
        stale = self.tr("Out of date: the controls changed. Compute again.")
        redraw = {
            "series": self._draw_series,
            "noise": self._show_noise,
            "profiles": self._draw_profile,
            "line": self._draw_line,
        }
        for kind in LONG_KINDS:
            if kind not in self.long:
                continue
            info = self._info_of(kind)
            if not self.long_is_current(kind):
                if info.text() != stale:
                    if kind in ("profiles", "line"):
                        redraw[kind]()  # the chart drops what no longer matches the controls
                    info.setText(stale)
            elif info.text() == stale:  # the controls came back to what it was computed with
                redraw[kind]()
        self._update_buttons()

    # ------------------------------------------------------------------ exports
    def _update_buttons(self) -> None:
        running_long = self._long_worker is not None and self._long_worker.isRunning()
        has = self._results_or_none() is not None
        for kind, b in self._run.items():
            b.setEnabled(has and not running_long and (kind != "line" or self._line_ok()))
        running_kind = self._long_worker.key[0] if running_long else None
        for kind, b in self._cancel.items():
            b.setEnabled(running_kind == kind)
        self._btn_pick_line.setEnabled(has)
        self._btn_csv.setEnabled(self._csv_ready(self._csv_kind()))
        self._btn_json.setEnabled(self.current is not None and self.current.selection is not None)
        shown_table = self.regions_table if self.results_tabs.currentIndex() == TAB_REGIONS else self.table
        self._btn_copy.setEnabled(shown_table.rowCount() > 0)
        chart = self._chart()
        self._btn_png.setEnabled(chart is not None and bool(chart[0].axes))

    # ------------------------------------------------------------------ texts
    def _group_label(self, key: str) -> str:
        return {
            "displacement": self.tr("Displacement"),
            "uncertainty": self.tr("Uncertainty"),
            "strain": self.tr("Strain tensor"),
            "principal": self.tr("Principal strains"),
            "equivalent": self.tr("Equivalent strains"),
            "rotation": self.tr("Local rotation"),
            "det_F": self.tr("Volume ratio det F"),
            "zncc": self.tr("Correlation (ZNCC)"),
        }.get(key, key)

    def _reason_label(self, key: str) -> str:
        return {
            "invalid": self.tr("outside the region"),
            "not_converged": self.tr("not converged"),
            "outlier": self.tr("rejected by the median test"),
            "low_zncc": self.tr("low ZNCC"),
            "edge": self.tr("edge"),
            "cut": self.tr("cut subset"),
        }.get(key, key)

    def tab_titles(self) -> list[str]:
        return [
            self.tr("Summary"),
            self.tr("Histogram"),
            self.tr("Over frames"),
            self.tr("Regions"),
            self.tr("Profile"),
            self.tr("Line"),
            self.tr("Noise floor"),
            self.tr("Homogeneous"),
        ]

    def _stat_headers(self) -> dict[str, str]:
        return {
            "n": self.tr("Nodes"),
            "mean": self.tr("Mean"),
            "std": self.tr("Std"),
            "median": self.tr("Median"),
            "robust_std": self.tr("Robust std"),
            "p05": self.tr("P5"),
            "p95": self.tr("P95"),
            "min": self.tr("Min"),
            "max": self.tr("Max"),
            "rms": self.tr("RMS"),
            "n_eff": self.tr("Eff. nodes"),
            "ci95": self.tr("95 % CI ±"),
        }

    def _stat_tips(self) -> dict[str, str]:
        return {
            "std": self.tr("Population standard deviation (divided by N), as the DVC Challenge 2.0 paper"),
            "robust_std": self.tr("1.4826 x median absolute deviation: not moved by a few wild nodes"),
            "n_eff": self.tr(
                "Effective number of independent nodes: neighbouring nodes share voxels and are smoothed together, "
                "so N nodes carry the information of fewer"
            ),
            "ci95": self.tr(
                "Half width of the 95 % confidence interval of the mean, from the effective number of nodes "
                "(not from N, which would make it far too narrow)"
            ),
        }

    def retranslate_ui(self) -> None:
        self.sections["what"].set_title(self.tr("Quantity"))
        self.sections["nodes"].set_title(self.tr("Nodes used"))
        self.sections["motion"].set_title(self.tr("Rigid-body motion"))
        self.sections["regions"].set_title(self.tr("Regions"))
        self.sections["export"].set_title(self.tr("Export"))
        texts = {
            "group": self.tr("Statistics of"),
            "shown": self.tr("On the slices"),
            "stats_region": self.tr("Statistics over"),
            "min_zncc": self.tr("Minimum ZNCC"),
            "edge_layers": self.tr("Drop edge layers"),
            "motion": self.tr("Remove"),
            "fit_region": self.tr("Fitted over"),
        }
        for key, lab in self.labels.items():
            lab.setText(texts[key])
        self.labels["motion"].setToolTip(
            self.tr(
                "Fitted over the nodes used and removed from the displacement; the stored result does not change. "
                "Translation: the mean displacement (the DVC Challenge 2.0 noise floor removes only this). "
                "Rigid: translation and rotation (exact, not linearised). Affine: the homogeneous deformation too, "
                "leaving the non-affine part."
            )
        )
        self.labels["stats_region"].setToolTip(
            self.tr("The region whose nodes the table, the histograms, the series and the profile describe")
        )
        self.labels["fit_region"].setToolTip(
            self.tr(
                "The nodes the removed motion is fitted to: every node, or a region that moves rigidly (a grip, "
                "an undeformed part). The fitted motion is removed from the whole field."
            )
        )
        self.apply_main.setText(self.tr("Also in the main window and exports"))
        self.apply_main.setToolTip(
            self.tr(
                "The slices, the 3-D view and the CSV, ParaView, image and report exports show the field with this "
                "motion removed (a badge says so); the npz and mat archives keep the result as measured."
            )
        )
        motion_names = {
            "none": self.tr("Nothing"),
            "translation": self.tr("Translation"),
            "rigid": self.tr("Rigid"),
            "affine": self.tr("Affine"),
        }
        for i in range(self.motion.count()):
            self.motion.setItemText(i, motion_names.get(self.motion.itemData(i), ""))
        red_names = {
            "mean_std": self.tr("Mean and std"),
            "mean_ci": self.tr("Mean and 95 % CI"),
            "median_robust": self.tr("Median and robust std"),
            "rms": self.tr("RMS"),
            "max": self.tr("Maximum"),
            "min": self.tr("Minimum"),
            "n": self.tr("Nodes used"),
        }
        for i in range(self.reduction.count()):
            self.reduction.setItemText(i, red_names.get(self.reduction.itemData(i), ""))
        for i in range(self.group.count()):
            self.group.setItemText(i, self._group_label(self.group.itemData(i)))
        for box in (self.stats_region_combo, self.fit_region_combo):
            if box.count():
                box.setItemText(0, self.tr("All nodes"))
        self.converged_only.setText(self.tr("Converged nodes only"))
        self.converged_only.setToolTip(
            self.tr(
                "A node whose local solve did not converge carries the global step's value or an inpainted one, "
                "not a measurement."
            )
        )
        self.drop_outliers.setText(self.tr("Leave out nodes rejected by the median test"))
        self.drop_cut.setText(self.tr("Leave out cut subsets"))
        self.labels["edge_layers"].setToolTip(
            self.tr(
                "Leave out this many node layers next to the edge of the grid or of the region (the strain fit is weaker there)."
            )
        )
        headers, tips = self._stat_headers(), self._stat_tips()
        self.table.setHorizontalHeaderLabels([headers[k] for k in TABLE_COLUMNS])
        self.regions_table.setHorizontalHeaderLabels([headers[k] for k in REGION_COLUMNS])
        for table, columns in ((self.table, TABLE_COLUMNS), (self.regions_table, REGION_COLUMNS)):
            for c, key in enumerate(columns):
                if key in tips:
                    table.horizontalHeaderItem(c).setToolTip(tips[key])
        for i, title in enumerate(self.tab_titles()):
            self.results_tabs.setTabText(i, title)
        self._run["series"].setText(self.tr("Compute over all frames"))
        self._run["profiles"].setText(self.tr("All frames"))
        self._run["profiles"].setToolTip(self.tr("The profile of every frame, coloured by frame"))
        self._run["line"].setText(self.tr("All frames and extensometer"))
        self._run["line"].setToolTip(
            self.tr(
                "The line in every frame, the field at its two ends, and the virtual extensometer: the distance "
                "between the two points as the material moves, strain = L / L0 - 1 (a length does not change with a "
                "rigid motion, so nothing needs removing)."
            )
        )
        self._min_share_label.setText(self.tr("Gap below"))
        self.min_share.setToolTip(
            self.tr(
                "A frame where the nodes used are fewer than this share of the region's nodes is left as a gap "
                "(0: never). A mean of a handful of nodes is not the mean of the region."
            )
        )
        for b in self._cancel.values():
            b.setText(self.tr("Cancel"))
        self.compare_regions.setText(self.tr("Compare regions"))
        self.compare_regions.setToolTip(
            self.tr("One curve per region (and all nodes) for the field on the slices, instead of every field of one region")
        )
        self._profile_axis_label.setText(self.tr("Along"))
        self._line_labels[0].setText(self.tr("From x, y, z"))
        self._line_labels[1].setText(self.tr("to"))
        self._btn_pick_line.setText(self.tr("Pick on the slices"))
        self._btn_pick_line.setToolTip(self.tr("Click the two ends on the slices; each lies on the slice shown"))
        self._run["noise"].setText(self.tr("Compute noise floor"))
        self._nominal_label.setText(self.tr("Applied displacement x, y, z"))
        self._run["noise"].setToolTip(
            self.tr(
                "For a static or known-translation pair: bias and noise floor per component, strain precision, "
                "virtual strain gauge."
            )
        )
        self._btn_copy.setText(self.tr("Copy table"))
        self._btn_csv.setText(self.tr("Save as CSV..."))
        self._btn_csv.setToolTip(
            self.tr("The data of the tab below: the series over frames, the regions, the profile or the line")
        )
        self._btn_json.setText(self.tr("Save summary as JSON..."))
        self._btn_png.setText(self.tr("Save chart as PNG..."))
        self.canvas.set_empty_text(self.tr("No field to show."))
        self.regions_panel.retranslate_ui()
        if self.shown.count():
            self.shown.setItemText(self.shown.count() - 1, self.tr("Nodes used"))
            for i in range(self.shown.count() - 1):
                self.shown.setItemText(i, field_name(self.shown.itemData(i)))
        if self.current is not None:
            self._show_current()
