"""Export dialog: destination, data formats, fields, frames, images and the PDF report in one place.

Follows pyALDIC's ``ExportDialog``: everything is chosen up front, the work runs on a worker
thread with progress, and the folder can be opened when done. Headless (tests) the dialog
works without any file chooser.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from al_dvc.core.data_structures import PipelineResult
from al_dvc.export.export_utils import DISP_FIELDS, STD_FIELDS, STRAIN_FIELDS
from al_dvc.export.slice_plots import LAYOUTS

from ..app_state import AppState
from ..names import field_name
from ..widgets import combo, guard_wheel, headless, spin

logger = logging.getLogger(__name__)

__all__ = ["ExportConfig", "ExportDialog", "run_export"]


@dataclass
class ExportConfig:
    """What to write. ``fields`` apply to CSV / VTK / images / report; npz and mat always hold everything."""

    out_dir: Path
    basename: str = "aldvc"
    npz: bool = True
    mat: bool = False
    csv: bool = False
    vtk: bool = False
    report: bool = False
    images: bool = False
    fields: list[str] = field(default_factory=lambda: ["disp_u", "disp_v", "disp_w"])
    frames: list[int] | None = None  # None = all
    image_layout: str = "grid"
    image_cmap: str = "turbo"
    image_dpi: int = 150
    image_light: bool = True
    image_background: bool = True
    image_equal_scale: bool = False

    def formats(self) -> list[str]:
        return [k for k in ("npz", "mat", "csv", "vtk", "report", "images") if getattr(self, k)]


def run_export(result: PipelineResult, cfg: ExportConfig, background=None, progress_fn=None, log_fn=None) -> list[Path]:
    """Write every selected format and return the paths (files or folders) produced."""
    from al_dvc.export import export_csv, export_mat, export_npz, export_report, export_vtk
    from al_dvc.export.slice_plots import export_field_images

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = cfg.formats()
    if not steps:
        raise ValueError("nothing selected to export")
    written: list[Path] = []
    fields = [f for f in cfg.fields if f in DISP_FIELDS or f in STD_FIELDS or (f in STRAIN_FIELDS and result.result_strain)]
    if not fields:
        fields = ["disp_u", "disp_v", "disp_w"]
    for i, step in enumerate(steps):
        if progress_fn is not None:
            progress_fn(i / len(steps), step)
        if step == "npz":
            written.append(export_npz(result, out / f"{cfg.basename}.npz"))
        elif step == "mat":
            written.append(export_mat(result, out / f"{cfg.basename}.mat"))
        elif step == "csv":
            written.append(export_csv(result, out / "csv", cfg.basename, fields=fields, frames=cfg.frames)[0].parent)
        elif step == "vtk":
            written.append(export_vtk(result, out / "vtk", cfg.basename, fields=fields, frames=cfg.frames)[0].parent)
        elif step == "report":
            written.append(export_report(result, out / f"{cfg.basename}_report.pdf", fields=fields))
        elif step == "images":
            files = export_field_images(
                result,
                out / "images",
                fields,
                frames=cfg.frames,
                layout=cfg.image_layout,
                cmap=cfg.image_cmap,
                background=background if cfg.image_background else None,
                dpi=cfg.image_dpi,
                light=cfg.image_light,
                equal_scale=cfg.image_equal_scale,
                progress_fn=(lambda f, m: progress_fn((i + f) / len(steps), f"images: {m}")) if progress_fn else None,
            )
            if files:
                written.append(files[0].parent)
        if log_fn is not None:
            log_fn(f"exported {step}: {written[-1]}")
    if progress_fn is not None:
        progress_fn(1.0, "done")
    return written


class _ExportWorker(QThread):
    progress = Signal(float, str)
    finished_paths = Signal(object)
    failed = Signal(str, str)
    log = Signal(str)

    def __init__(self, result, cfg: ExportConfig, background, parent=None) -> None:
        super().__init__(parent)
        self._result = result
        self._cfg = cfg
        self._background = background

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            paths = run_export(
                self._result,
                self._cfg,
                self._background,
                progress_fn=lambda f, m: self.progress.emit(float(f), str(m)),
                log_fn=self.log.emit,
            )
        except Exception as exc:
            logger.exception("Export failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        self.finished_paths.emit(paths)


def open_folder(path: Path) -> None:
    """Show a folder in the desktop file manager (no-op headless)."""
    if headless():
        return
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606 - opening a folder for the user
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])  # noqa: S603, S607
        else:
            subprocess.Popen(["xdg-open", str(path)])  # noqa: S603, S607
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.warning("cannot open folder %s: %s", path, exc)


class ExportDialog(QDialog):
    """Choose the destination, formats, fields and frames; run the export with progress."""

    def __init__(self, state: AppState, parent: QWidget | None = None, preselect_strain: bool = False) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: _ExportWorker | None = None
        self.setModal(False)
        self.resize(560, 640)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ---- destination
        self._dest_group = QGroupBox()
        dest = QVBoxLayout(self._dest_group)
        row = QHBoxLayout()
        self.folder = QLineEdit(str(state.output_dir))
        self._btn_browse = QPushButton()
        self._btn_open = QPushButton()
        row.addWidget(self.folder, 1)
        row.addWidget(self._btn_browse)
        row.addWidget(self._btn_open)
        dest.addLayout(row)
        name_row = QHBoxLayout()
        self._name_label = QLabel()
        self.basename = QLineEdit("aldvc")
        self.basename.setFixedWidth(180)
        name_row.addWidget(self._name_label)
        name_row.addWidget(self.basename)
        name_row.addStretch(1)
        dest.addLayout(name_row)
        layout.addWidget(self._dest_group)

        # ---- formats
        self._format_group = QGroupBox()
        fmt = QVBoxLayout(self._format_group)
        self.checks: dict[str, QCheckBox] = {k: QCheckBox() for k in ("npz", "mat", "csv", "vtk", "report", "images")}
        self.checks["npz"].setChecked(True)
        grid_row1 = QHBoxLayout()
        grid_row2 = QHBoxLayout()
        for key in ("npz", "mat", "csv"):
            grid_row1.addWidget(self.checks[key])
        for key in ("vtk", "report", "images"):
            grid_row2.addWidget(self.checks[key])
        fmt.addLayout(grid_row1)
        fmt.addLayout(grid_row2)
        layout.addWidget(self._format_group)

        # ---- fields and frames
        self._fields_group = QGroupBox()
        fl = QVBoxLayout(self._fields_group)
        self.fields = QListWidget()
        self.fields.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.fields.setMaximumHeight(150)
        fl.addWidget(self.fields)
        sel = QHBoxLayout()
        self._btn_all = QPushButton()
        self._btn_none = QPushButton()
        self._btn_disp = QPushButton()
        self._btn_strain = QPushButton()
        for b in (self._btn_all, self._btn_none, self._btn_disp, self._btn_strain):
            b.setFlat(True)
            sel.addWidget(b)
        sel.addStretch(1)
        fl.addLayout(sel)
        frames = QHBoxLayout()
        self._frames_label = QLabel()
        self.all_frames = QCheckBox()
        self.all_frames.setChecked(True)
        self.frame_from = spin(1, 1, 1, width=70)
        self.frame_to = spin(1, 1, 1, width=70)
        self._to_label = QLabel("-")
        frames.addWidget(self._frames_label)
        frames.addWidget(self.all_frames)
        frames.addWidget(self.frame_from)
        frames.addWidget(self._to_label)
        frames.addWidget(self.frame_to)
        frames.addStretch(1)
        fl.addLayout(frames)
        layout.addWidget(self._fields_group)

        # ---- images
        self._image_group = QGroupBox()
        il = QHBoxLayout(self._image_group)
        self._layout_label = QLabel()
        self.image_layout = combo([], width=100)
        for key in LAYOUTS:
            self.image_layout.addItem(key, key)
        self.image_layout.setCurrentIndex(LAYOUTS.index("grid"))
        self._cmap_label = QLabel()
        self.image_cmap = combo(["turbo", "viridis", "plasma", "inferno", "coolwarm", "RdBu_r", "jet", "gray"], width=100)
        self._dpi_label = QLabel()
        self.image_dpi = QSpinBox()
        self.image_dpi.setRange(50, 600)
        self.image_dpi.setValue(150)
        self.image_dpi.setFixedWidth(70)
        self.image_light = QCheckBox()
        self.image_light.setChecked(True)
        self.image_bg = QCheckBox()
        self.image_bg.setChecked(True)
        self.image_equal = QCheckBox()
        for w in (self._layout_label, self.image_layout, self._cmap_label, self.image_cmap, self._dpi_label, self.image_dpi):
            il.addWidget(w)
        il.addWidget(self.image_light)
        il.addWidget(self.image_bg)
        il.addWidget(self.image_equal)
        il.addStretch(1)
        layout.addWidget(self._image_group)

        # ---- run
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)
        self._status = QLabel()
        self._status.setObjectName("hint")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._btn_export = QPushButton()
        self._btn_export.setProperty("class", "btn-primary")
        self._btn_export.setMinimumHeight(30)
        self._btn_close = QPushButton()
        buttons.addWidget(self._btn_export)
        buttons.addWidget(self._btn_close)
        layout.addLayout(buttons)
        guard_wheel(self)

        # ---- wiring
        self._btn_browse.clicked.connect(self._on_browse)
        self._btn_open.clicked.connect(lambda: open_folder(Path(self.folder.text() or ".")))
        self._btn_export.clicked.connect(self.start)
        self._btn_close.clicked.connect(self.close)
        self._btn_all.clicked.connect(lambda: self._check_fields(lambda _n: True))
        self._btn_none.clicked.connect(lambda: self._check_fields(lambda _n: False))
        self._btn_disp.clicked.connect(lambda: self._check_fields(lambda n: n in DISP_FIELDS))
        self._btn_strain.clicked.connect(lambda: self._check_fields(lambda n: n in STRAIN_FIELDS))
        self.all_frames.toggled.connect(self._on_all_frames)
        self.checks["images"].toggled.connect(lambda v: self._image_group.setEnabled(v))
        self._image_group.setEnabled(False)
        self._state.results_changed.connect(self.refresh)
        self.retranslate_ui()
        self.refresh()
        if preselect_strain:
            self._check_fields(lambda n: n in STRAIN_FIELDS or n in DISP_FIELDS)
            self.checks["csv"].setChecked(True)

    # ------------------------------------------------------------------ fields
    def refresh(self) -> None:
        res = self._state.results
        has = res is not None and bool(res.result_disp)
        checked = {
            self.fields.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.fields.count())
            if self.fields.item(i).checkState() == Qt.CheckState.Checked
        }
        self.fields.clear()
        names: list[str] = []
        if has:
            names = list(DISP_FIELDS)
            if res.result_disp[0].U_std is not None:
                names += list(STD_FIELDS)
            if res.result_strain:
                names += list(STRAIN_FIELDS)
        for name in names:
            item = QListWidgetItem(field_name(name))
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            default = name in ("disp_u", "disp_v", "disp_w") if not checked else name in checked
            item.setCheckState(Qt.CheckState.Checked if default else Qt.CheckState.Unchecked)
            self.fields.addItem(item)
        n = len(res.result_disp) if has else 1
        for s in (self.frame_from, self.frame_to):
            s.setRange(1, max(1, n))
        self.frame_to.setValue(max(1, n))
        self._btn_export.setEnabled(has and not self._running())
        self._btn_strain.setEnabled(has and bool(res.result_strain))
        if not has:
            self._status.setText(self.tr("No results to export yet."))
        elif not res.result_strain:
            self._status.setText(self.tr("Displacement only: compute strain in the strain window to export strain fields."))
        else:
            self._status.setText("")

    def _check_fields(self, predicate) -> None:
        for i in range(self.fields.count()):
            item = self.fields.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Checked if predicate(name) else Qt.CheckState.Unchecked)

    def selected_fields(self) -> list[str]:
        return [
            self.fields.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.fields.count())
            if self.fields.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _on_all_frames(self, all_frames: bool) -> None:
        self.frame_from.setEnabled(not all_frames)
        self.frame_to.setEnabled(not all_frames)

    def config(self) -> ExportConfig:
        frames = None
        if not self.all_frames.isChecked():
            lo, hi = sorted((int(self.frame_from.value()), int(self.frame_to.value())))
            frames = list(range(lo - 1, hi))
        return ExportConfig(
            out_dir=Path(self.folder.text() or "aldvc_results"),
            basename=self.basename.text().strip() or "aldvc",
            npz=self.checks["npz"].isChecked(),
            mat=self.checks["mat"].isChecked(),
            csv=self.checks["csv"].isChecked(),
            vtk=self.checks["vtk"].isChecked(),
            report=self.checks["report"].isChecked(),
            images=self.checks["images"].isChecked(),
            fields=self.selected_fields(),
            frames=frames,
            image_layout=str(self.image_layout.currentData() or "grid"),
            image_cmap=self.image_cmap.currentText(),
            image_dpi=int(self.image_dpi.value()),
            image_light=self.image_light.isChecked(),
            image_background=self.image_bg.isChecked(),
            image_equal_scale=self.image_equal.isChecked(),
        )

    # ------------------------------------------------------------------ run
    def _running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _on_browse(self) -> None:
        if headless():
            return
        folder = QFileDialog.getExistingDirectory(self, self.tr("Export folder"), self.folder.text())
        if folder:
            self.folder.setText(folder)

    def start(self) -> None:
        res = self._state.results
        if res is None or self._running():
            return
        cfg = self.config()
        if not cfg.formats():
            self._status.setText(self.tr("Select at least one format."))
            return
        background = None
        if cfg.images and cfg.image_background and self._state.volumes:
            try:
                vol = self._state.volume_array(0)
                if tuple(vol.shape) == tuple(res.volume_shape):
                    background = vol
            except Exception as exc:
                self._state.log(f"export: cannot load the reference volume for the images: {exc}", "warning")
        self._state.set_output_dir(cfg.out_dir)
        self._worker = _ExportWorker(res, cfg, background, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_paths.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.log.connect(lambda m: self._state.log(m))
        self._btn_export.setEnabled(False)
        self._progress.setValue(0)
        self._status.setText(self.tr("Exporting..."))
        self._worker.start()

    def wait(self, timeout_ms: int = 300_000) -> bool:
        return self._worker.wait(timeout_ms) if self._worker is not None else True

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(round(1000 * fraction)))
        self._status.setText(self.tr("Exporting: {msg}").format(msg=message))

    def _on_finished(self, paths) -> None:
        self._btn_export.setEnabled(True)
        self._progress.setValue(1000)
        self._status.setText(self.tr("Done: {n} item(s) written to {folder}").format(n=len(paths), folder=self.folder.text()))
        self._state.log(self.tr("Export finished: {folder}").format(folder=self.folder.text()), "success")

    def _on_failed(self, message: str, detail: str) -> None:
        self._btn_export.setEnabled(True)
        self._status.setText(self.tr("Export failed: {error}").format(error=message))
        self._state.log(self.tr("Export failed: {error}").format(error=message), "error")
        self._state.log(detail, "debug")

    # ------------------------------------------------------------------ misc
    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("Export results"))
        self._dest_group.setTitle(self.tr("Destination"))
        self._btn_browse.setText(self.tr("Browse..."))
        self._btn_open.setText(self.tr("Open folder"))
        self._name_label.setText(self.tr("Base name"))
        self._format_group.setTitle(self.tr("Formats"))
        texts = {
            "npz": self.tr("NumPy archive (.npz)"),
            "mat": self.tr("MATLAB (.mat)"),
            "csv": self.tr("CSV per frame"),
            "vtk": self.tr("ParaView (.vti + .pvd)"),
            "report": self.tr("PDF report"),
            "images": self.tr("Slice images (PNG)"),
        }
        for key, box in self.checks.items():
            box.setText(texts[key])
        self.checks["npz"].setToolTip(self.tr("Everything: parameters, mesh, every displacement and strain array"))
        self.checks["csv"].setToolTip(self.tr("Node coordinates and the selected fields, one file per frame"))
        self._fields_group.setTitle(self.tr("Fields (CSV, ParaView, images, report)"))
        self._btn_all.setText(self.tr("All"))
        self._btn_none.setText(self.tr("None"))
        self._btn_disp.setText(self.tr("Displacement"))
        self._btn_strain.setText(self.tr("Strain"))
        self._frames_label.setText(self.tr("Frames"))
        self.all_frames.setText(self.tr("all"))
        self._image_group.setTitle(self.tr("Images"))
        self._layout_label.setText(self.tr("Layout"))
        self._cmap_label.setText(self.tr("Colormap"))
        self._dpi_label.setText(self.tr("DPI"))
        self.image_light.setText(self.tr("White background"))
        self.image_bg.setText(self.tr("Volume under the field"))
        self.image_equal.setText(self.tr("Same scale"))
        for i, text in enumerate((self.tr("Row"), self.tr("Column"), self.tr("2 x 2"))):
            self.image_layout.setItemText(i, text)
        self._btn_export.setText(self.tr("Export"))
        self._btn_close.setText(self.tr("Close"))
