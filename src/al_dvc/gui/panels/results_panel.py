"""Result display controls, summary and exports."""

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
from al_dvc.export.slice_plots import FIELD_LABELS

from ..app_state import AppState
from ..widgets import guard_wheel, headless

COLORMAPS = ["turbo", "viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r", "jet", "gray"]


class ResultsPanel(QWidget):
    """Field / frame / colour controls, a text summary, the strain window and export buttons."""

    strain_requested = Signal()
    export_requested = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._display_group = QGroupBox()
        form = QFormLayout(self._display_group)
        self.frame = QSpinBox()
        self.frame.setRange(1, 1)
        self.frame.setFixedWidth(64)
        self._btn_prev = QPushButton("<")
        self._btn_prev.setFixedWidth(28)
        self._btn_next = QPushButton(">")
        self._btn_next.setFixedWidth(28)
        frame_widget = QWidget()
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

        self._summary_group = QGroupBox()
        sl = QVBoxLayout(self._summary_group)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sl.addWidget(self._summary)
        layout.addWidget(self._summary_group)

        self._btn_strain = QPushButton()
        self._btn_strain.setProperty("class", "btn-primary")
        self._btn_strain.setMinimumHeight(30)
        self._btn_strain.setEnabled(False)
        self._btn_strain.clicked.connect(self.strain_requested.emit)
        layout.addWidget(self._btn_strain)
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
        guard_wheel(self)
        layout.addStretch(1)

        self.frame.valueChanged.connect(lambda v: self._set(display_frame=int(v) - 1))
        self._btn_prev.clicked.connect(lambda: self.frame.setValue(self.frame.value() - 1))
        self._btn_next.clicked.connect(lambda: self.frame.setValue(self.frame.value() + 1))
        self.field.currentIndexChanged.connect(self._on_field_index)
        self.colormap.currentTextChanged.connect(lambda v: self._set(colormap=v))
        self.auto_range.toggled.connect(self._on_auto)
        self.vmin.valueChanged.connect(lambda v: self._set(color_min=float(v)))
        self.vmax.valueChanged.connect(lambda v: self._set(color_max=float(v)))
        self.alpha.valueChanged.connect(lambda v: self._set(overlay_alpha=v / 100.0))
        self.show_overlay.toggled.connect(lambda v: self._set(show_overlay=bool(v)))
        self._state.results_changed.connect(self.refresh)
        self.retranslate_ui()
        self.refresh()

    # ------------------------------------------------------------------ binding
    def _set(self, **values) -> None:
        if not self._updating:
            self._state.set_display(**values)

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

    # ------------------------------------------------------------------ view
    def refresh(self) -> None:
        res = self._state.results
        has = res is not None and bool(res.result_disp)
        self._display_group.setEnabled(has)
        self._export_group.setEnabled(has)
        self._btn_strain.setEnabled(has)
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
                    self.field.addItem(FIELD_LABELS.get(name, name), name)
                if self._state.display_field in fields:
                    self.field.setCurrentIndex(self.field.findData(self._state.display_field))
                else:
                    self._state.display_field = fields[0]
                self.frame.setRange(1, len(res.result_disp))
                self.frame.setValue(self._state.display_frame + 1)
            self.colormap.setCurrentText(self._state.colormap)
        finally:
            self._updating = False
        self._summary.setText(self._summary_text() if has else self.tr("No results yet."))

    def _summary_text(self) -> str:
        res = self._state.results
        mesh = res.dvc_mesh
        lines = [self.tr("Nodes: {n} on a {g} grid, spacing {s}").format(n=mesh.n_nodes, g=mesh.grid_shape, s=mesh.spacing)]
        for k, fr in enumerate(res.result_disp):
            codes, counts = np.unique(fr.status, return_counts=True) if fr.status is not None else ([], [])
            status = ", ".join(f"{STATUS_NAMES.get(int(c), c)} {int(n)}" for c, n in zip(codes, counts))
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
        t = res.timings
        lines.append(
            self.tr("Time: total {t:.1f} s (local {l:.1f} s, ADMM local {s1:.1f} s)").format(
                t=t.get("total", 0.0), l=t.get("local_icgn", 0.0), s1=t.get("subpb1", 0.0)
            )
        )
        if res.stopped_early:
            lines.append(self.tr("Stopped early at frame {k}: {why}").format(k=res.stopped_at_frame, why=res.stop_reason))
        return "\n".join(lines)

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
        self._summary_group.setTitle(self.tr("Summary"))
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
        self._btn_export.setText(self.tr("Export results..."))
        self._export_status.setText(self.tr("npz, mat, CSV, ParaView, PDF report and slice images"))
        if self._state.results is None:
            self._summary.setText(self.tr("No results yet."))
