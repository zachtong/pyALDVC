"""Result display controls, the post-processing entry points, a compact summary and exports.

Order, top to bottom: the two post-processing windows (texture analysis needs only a volume,
strain needs a result), the display controls, the export, and a two-line summary whose
per-frame details are folded away.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dvc.core.data_structures import STATUS_NAMES
from al_dvc.export.export_utils import DISP_FIELDS, STD_FIELDS, STRAIN_FIELDS

from ..app_state import AppState
from ..names import field_name, status_name
from ..widgets import CollapsibleSection, guard_wheel, headless

COLORMAPS = ["turbo", "viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r", "jet", "gray"]


class ResultsPanel(QWidget):
    """Field / frame / colour controls, a text summary, the strain window and export buttons."""

    strain_requested = Signal()
    texture_requested = Signal()
    export_requested = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # the two post-processing windows first: they are what the user looks for after a run
        self._analysis_group = QGroupBox()
        self._analysis_group.setObjectName("analysisBox")
        agrid = QVBoxLayout(self._analysis_group)
        agrid.setSpacing(6)
        self._btn_texture = QPushButton()
        self._btn_texture.setProperty("class", "btn-primary")
        self._btn_texture.setMinimumHeight(32)
        self._btn_texture.setEnabled(False)
        self._btn_texture.clicked.connect(self.texture_requested.emit)
        self._btn_strain = QPushButton()
        self._btn_strain.setProperty("class", "btn-primary")
        self._btn_strain.setMinimumHeight(32)
        self._btn_strain.setEnabled(False)
        self._btn_strain.clicked.connect(self.strain_requested.emit)
        agrid.addWidget(self._btn_texture)
        agrid.addWidget(self._btn_strain)
        self._analysis_hint = QLabel()
        self._analysis_hint.setObjectName("hint")
        self._analysis_hint.setWordWrap(True)
        agrid.addWidget(self._analysis_hint)
        layout.addWidget(self._analysis_group)

        self._display_group = QGroupBox()
        form = QFormLayout(self._display_group)
        self.frame = QSpinBox()
        self.frame.setRange(1, 1)
        self.frame.setFixedWidth(64)
        self._btn_prev = QPushButton("<")
        self._btn_next = QPushButton(">")
        for b in (self._btn_prev, self._btn_next):
            b.setFixedSize(32, 26)
            b.setStyleSheet("padding: 2px 2px;")
        self.frame.setFixedHeight(26)
        frame_widget = QWidget()
        frame_widget.setStyleSheet("background: transparent;")
        frame_widget.setMinimumHeight(28)  # the row is as tall as its buttons: nothing is clipped by the group's title
        frame_row = QHBoxLayout(frame_widget)
        frame_row.setContentsMargins(0, 0, 0, 0)
        frame_row.setSpacing(4)
        frame_row.addWidget(self._btn_prev)
        frame_row.addWidget(self.frame)
        frame_row.addWidget(self._btn_next)
        frame_row.addStretch(1)
        self.field = QComboBox()
        self.colormap = QComboBox()
        self.colormap.addItems(COLORMAPS)
        self.auto_range = QCheckBox()
        self.auto_range.setChecked(True)
        self.vmin = QDoubleSpinBox()
        self.vmax = QDoubleSpinBox()
        for s in (self.vmin, self.vmax):
            s.setRange(-1e9, 1e9)
            s.setDecimals(4)
            s.setEnabled(False)
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(int(100 * state.overlay_alpha))
        self.show_overlay = QCheckBox()
        self.show_overlay.setChecked(True)
        self._no_result = QLabel()  # shown when the selected volume has no computed field (reference, uncomputed frame)
        self._no_result.setObjectName("hint")
        self._no_result.setWordWrap(True)
        self._no_result.hide()
        form.addRow(self._no_result)
        self._labels: dict[str, QLabel] = {}
        for key, widget in [
            ("frame", frame_widget),
            ("field", self.field),
            ("colormap", self.colormap),
            ("auto", self.auto_range),
            ("vmin", self.vmin),
            ("vmax", self.vmax),
            ("alpha", self.alpha),
            ("show", self.show_overlay),
        ]:
            label = QLabel()
            self._labels[key] = label
            form.addRow(label, widget)
        layout.addWidget(self._display_group)

        self._export_group = QGroupBox()
        grid = QGridLayout(self._export_group)
        self._btn_export = QPushButton()
        self._btn_export.setMinimumHeight(30)
        self._btn_export.clicked.connect(self.export_requested.emit)
        grid.addWidget(self._btn_export, 0, 0)
        self._export_status = QLabel()
        self._export_status.setWordWrap(True)
        self._export_status.setObjectName("hint")
        grid.addWidget(self._export_status, 1, 0)
        layout.addWidget(self._export_group)

        # a two-line summary; the per-frame lines live in a folded section
        self._summary_group = QGroupBox()
        sl = QVBoxLayout(self._summary_group)
        sl.setSpacing(4)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sl.addWidget(self._summary)
        self._details_section = CollapsibleSection(expanded=False)
        self._details = QLabel()
        self._details.setWordWrap(True)
        self._details.setObjectName("hint")
        self._details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._details_section.add_widget(self._details)
        sl.addWidget(self._details_section)
        layout.addWidget(self._summary_group)
        guard_wheel(self)
        layout.addStretch(1)

        self.frame.valueChanged.connect(self._on_frame)
        self._btn_prev.clicked.connect(lambda: self.frame.setValue(self.frame.value() - 1))
        self._btn_next.clicked.connect(lambda: self.frame.setValue(self.frame.value() + 1))
        self.field.currentIndexChanged.connect(self._on_field_index)
        self.colormap.currentTextChanged.connect(lambda v: self._set(colormap=v))
        self.auto_range.toggled.connect(self._on_auto)
        self.vmin.valueChanged.connect(lambda v: self._on_limit("min", float(v)))
        self.vmax.valueChanged.connect(lambda v: self._on_limit("max", float(v)))
        self.alpha.valueChanged.connect(lambda v: self._set(overlay_alpha=v / 100.0))
        self.show_overlay.toggled.connect(lambda v: self._set(show_overlay=bool(v)))
        self._state.results_changed.connect(self.refresh)
        self._state.current_frame_changed.connect(lambda _i: self.refresh())
        self._state.volumes_changed.connect(self.refresh)
        self._state.display_changed.connect(self._sync_display)  # a loaded session, a change made elsewhere
        self.retranslate_ui()
        self.refresh()

    def minimumSizeHint(self):  # noqa: N802
        """The scroll area must scroll rather than squeeze the rows into each other."""
        return self.sizeHint()

    # ------------------------------------------------------------------ binding
    def _set(self, **values) -> None:
        if not self._updating:
            self._state.set_display(**values)

    def _on_frame(self, value: int) -> None:
        """The frame box selects the volume of that result: frame k is deformed volume k."""
        if not self._updating:
            self._state.set_current_frame(int(value))

    def _on_field_index(self, index: int) -> None:
        name = self.field.itemData(index) if index >= 0 else None
        if name:
            self._set(display_field=str(name))

    def select_field(self, name: str) -> bool:
        """Select a field by its internal name (``disp_u``, ``exx``, ...); False when absent."""
        i = self.field.findData(name)
        if i < 0:
            return False
        self.field.setCurrentIndex(i)
        return True

    def field_names(self) -> list[str]:
        return [self.field.itemData(i) for i in range(self.field.count())]

    def _on_auto(self, auto: bool) -> None:
        self.vmin.setEnabled(not auto)
        self.vmax.setEnabled(not auto)
        self._set(color_auto=bool(auto))

    @staticmethod
    def _limit_gap(value: float) -> float:
        """Smallest distance kept between the two colour limits."""
        return max(abs(value) * 1e-3, 1e-6)

    @staticmethod
    def _set_spin(spin, value: float) -> None:
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    def _on_limit(self, which: str, value: float) -> None:
        """Keep min < max: editing one bound pushes the other, so a reversed pair never reaches the display."""
        if self._updating:
            return
        lo, hi = float(self.vmin.value()), float(self.vmax.value())
        if which == "min" and lo >= hi:
            hi = lo + self._limit_gap(lo)
            self._set_spin(self.vmax, hi)
        elif which == "max" and hi <= lo:
            lo = hi - self._limit_gap(hi)
            self._set_spin(self.vmin, lo)
        self._set(color_min=lo, color_max=hi)

    def _sync_display(self) -> None:
        """Every display control shows the state's value: what is rendered is what the widgets say."""
        st = self._state
        lo, hi = float(st.color_min), float(st.color_max)
        if hi <= lo:  # a session edited by hand: repair the pair before it is shown
            hi = lo + self._limit_gap(lo)
            st.color_max = hi
        was = self._updating
        self._updating = True
        try:
            self.colormap.setCurrentText(st.colormap)
            self.auto_range.setChecked(bool(st.color_auto))
            self.vmin.setEnabled(not st.color_auto)
            self.vmax.setEnabled(not st.color_auto)
            self._set_spin(self.vmin, lo)
            self._set_spin(self.vmax, hi)
            self.alpha.setValue(int(round(100 * float(st.overlay_alpha))))
            self.show_overlay.setChecked(bool(st.show_overlay))
        finally:
            self._updating = was

    # ------------------------------------------------------------------ view
    def refresh(self) -> None:
        res = self._state.results
        has = res is not None and bool(res.result_disp)
        self._display_group.setEnabled(has)
        self._export_group.setEnabled(has)
        self._btn_strain.setEnabled(has)
        self._btn_texture.setEnabled(bool(self._state.volumes))
        self._updating = True
        try:
            self.field.clear()
            if has:
                fields = list(DISP_FIELDS)
                if res.result_disp[0].U_std is not None:
                    fields += list(STD_FIELDS)
                if res.result_strain:
                    fields += list(STRAIN_FIELDS)
                for name in fields:
                    self.field.addItem(field_name(name), name)
                if self._state.display_field in fields:
                    self.field.setCurrentIndex(self.field.findData(self._state.display_field))
                else:
                    self._state.display_field = fields[0]
                self.frame.setRange(1, len(res.result_disp))
                self.frame.setValue(max(1, min(self._state.current_frame, len(res.result_disp))))
            self._sync_display()
        finally:
            self._updating = False
        # the selected volume may have no field of its own: the reference row, a row added after the run, a
        # frame the run never reached. Nothing is substituted; the hint says so.
        self._no_result.setVisible(has and self._state.result_frame() is None)
        if has:
            short, details = self._summary_text()
            self._summary.setText(short)
            self._details.setText(details)
        else:
            self._summary.setText(self.tr("No results yet."))
            self._details.setText("")
        self._details_section.setVisible(has)

    def _summary_text(self) -> tuple[str, str]:
        """``(two-line summary, per-frame details)``."""
        res = self._state.results
        mesh = res.dvc_mesh
        nz, ny, nx = mesh.grid_shape
        sx, sy, sz = mesh.spacing
        head = self.tr("{n} nodes, {g} grid, step {s}").format(
            n=f"{mesh.n_nodes:,}", g=f"{nx} x {ny} x {nz}", s=" x ".join(f"{v:g}" for v in (sx, sy, sz))
        )
        conv = [float(np.mean(fr.status == 0)) if fr.status is not None else float("nan") for fr in res.result_disp]
        z = [
            float(np.nanmedian(fr.zncc)) if fr.zncc is not None and np.isfinite(fr.zncc).any() else float("nan")
            for fr in res.result_disp
        ]
        t = res.timings
        second = self.tr("{k} frame(s): converged {c}, median ZNCC {z}, {t:.1f} s").format(
            k=len(res.result_disp),
            c=f"{100 * float(np.nanmin(conv)):.0f} %" if conv and np.isfinite(conv).any() else "-",
            z=f"{float(np.nanmin(z)):.3f}" if z and np.isfinite(z).any() else "-",
            t=t.get("total", 0.0),
        )
        short = head + "\n" + second
        if res.stopped_early:
            short += "\n" + self.tr("Stopped early at frame {k}: {why}").format(k=res.stopped_at_frame, why=res.stop_reason)
        lines = []
        for k, fr in enumerate(res.result_disp):
            codes, counts = np.unique(fr.status, return_counts=True) if fr.status is not None else ([], [])
            status = ", ".join(f"{status_name(STATUS_NAMES.get(int(c), c))} {int(n)}" for c, n in zip(codes, counts))
            z = np.nanmedian(fr.zncc) if fr.zncc is not None and np.isfinite(fr.zncc).any() else float("nan")
            beta = f", beta {fr.admm.beta:.3g}, ADMM {fr.admm.n_steps}" if fr.admm is not None else ""
            std = ""
            if fr.U_std is not None and np.isfinite(fr.U_std).any():
                med = np.nanmedian(fr.U_std, axis=0)
                std = self.tr(", std {u:.3f}/{v:.3f}/{w:.3f}").format(u=med[0], v=med[1], w=med[2])
            lines.append(
                self.tr("Frame {k} (ref {r}): ZNCC {z:.3f}{beta}{std}; {status}").format(
                    k=k + 1, r=fr.ref_frame, z=z, beta=beta, std=std, status=status
                )
            )
        lines.append(
            self.tr("Time: total {t:.1f} s (local {l:.1f} s, ADMM local {s1:.1f} s)").format(
                t=t.get("total", 0.0), l=t.get("local_icgn", 0.0), s1=t.get("subpb1", 0.0)
            )
        )
        return short, "\n".join(lines)

    # ------------------------------------------------------------------ export
    _EXPORT_FILES = {
        "npz": ("aldvc.npz", "NumPy (*.npz)"),
        "mat": ("aldvc.mat", "MATLAB (*.mat)"),
        "report": ("aldvc_report.pdf", "PDF (*.pdf)"),
    }

    def _ask_target(self, kind: str) -> Path | None:
        """File (npz / mat / report) or folder (csv / vtk) chosen by the user; the last folder is remembered.

        Headless (tests, self-test) the target goes straight into ``state.output_dir``.
        """
        out = Path(self._state.output_dir)
        if headless():
            out.mkdir(parents=True, exist_ok=True)
            return out / self._EXPORT_FILES[kind][0] if kind in self._EXPORT_FILES else out / kind
        if kind in self._EXPORT_FILES:
            name, flt = self._EXPORT_FILES[kind]
            path, _ = QFileDialog.getSaveFileName(self, self.tr("Export results"), str(out / name), flt)
            if not path:
                return None
            target = Path(path)
            self._state.set_output_dir(target.parent)
            return target
        folder = QFileDialog.getExistingDirectory(self, self.tr("Export folder"), str(out))
        if not folder:
            return None
        self._state.set_output_dir(folder)
        return Path(folder) / kind

    def export(self, kind: str) -> Path | None:
        res = self._state.results
        if res is None:
            return None
        target = self._ask_target(kind)
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            from al_dvc.export import export_csv, export_mat, export_npz, export_report, export_vtk

            if kind == "npz":
                path = export_npz(res, target)
            elif kind == "mat":
                path = export_mat(res, target)
            elif kind == "csv":
                path = export_csv(res, target)[0].parent
            elif kind == "vtk":
                path = export_vtk(res, target)[0].parent
            elif kind == "report":
                path = export_report(res, target)
            else:
                raise ValueError(kind)
        except Exception as exc:
            self._export_status.setText(self.tr("Export failed: {error}").format(error=exc))
            self._state.log(f"export {kind}: {exc}", "error")
            return None
        finally:
            QApplication.restoreOverrideCursor()
        self._export_status.setText(self.tr("Written: {path}").format(path=path))
        self._state.log(self.tr("Exported {kind} to {path}").format(kind=kind, path=path))
        return Path(path)

    def retranslate_ui(self) -> None:
        self._display_group.setTitle(self.tr("Display"))
        for i in range(self.field.count()):
            self.field.setItemText(i, field_name(self.field.itemData(i)))
        self._summary_group.setTitle(self.tr("Summary"))
        self._details_section.set_title(self.tr("Details"))
        self._analysis_group.setTitle(self.tr("Post-processing"))
        self._btn_texture.setText(self.tr("Texture analysis..."))
        self._analysis_hint.setText(
            self.tr("Texture: subset size from the reference volume. Strain: from the displacement result.")
        )
        self._export_group.setTitle(self.tr("Export"))
        self._btn_strain.setText(self.tr("Strain post-processing..."))
        texts = {
            "frame": self.tr("Frame"),
            "field": self.tr("Field"),
            "colormap": self.tr("Colormap"),
            "auto": self.tr("Auto range"),
            "vmin": self.tr("Min"),
            "vmax": self.tr("Max"),
            "alpha": self.tr("Overlay opacity"),
            "show": self.tr("Show overlay"),
        }
        for key, label in self._labels.items():
            label.setText(texts[key])
        self._no_result.setText(self.tr("No result for this volume"))
        self._btn_export.setText(self.tr("Export results..."))
        self._export_status.setText(self.tr("npz, mat, CSV, ParaView, PDF report and slice images"))
        if self._state.results is None:
            self._summary.setText(self.tr("No results yet."))
