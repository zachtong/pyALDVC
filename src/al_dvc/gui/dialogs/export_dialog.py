"""Export dialog: destination, data formats, fields, frames, images and the PDF report in one place.

Follows pyALDIC's ``ExportDialog``: everything is chosen up front, the work runs on a worker
thread with progress, and the folder can be opened when done. The settings of a running job are
frozen (the controls are disabled) and every message about the job reads the captured
configuration, never the widgets. Every format is attempted even when one fails, and the outcome
lists what was written and what was not. Headless (tests) the dialog works without any file
chooser or question box.
"""

from __future__ import annotations

import logging
import os
import re
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
    QMessageBox,
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

__all__ = [
    "ExportConfig",
    "ExportDialog",
    "ExportError",
    "ExportOutcome",
    "existing_outputs",
    "export_formats",
    "run_export",
    "validate_basename",
]

FIELD_FORMATS = ("csv", "vtk", "report", "images")  # the formats that take the field selection
FRAME_FORMATS = ("csv", "vtk", "images")  # the formats that take the frame range; npz / mat / report hold every frame
_BAD_NAME_CHARS = re.compile(r'[<>:"|?*\\/\x00-\x1f]')


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

    def needs_fields(self) -> bool:
        return any(getattr(self, k) for k in FIELD_FORMATS)


@dataclass
class ExportOutcome:
    """What one export job produced, format by format (a failed format does not stop the others)."""

    written: dict[str, list[Path]] = field(default_factory=dict)  # format -> files or folders
    errors: dict[str, str] = field(default_factory=dict)  # format -> "ExcType: message"
    tracebacks: dict[str, str] = field(default_factory=dict)

    @property
    def paths(self) -> list[Path]:
        return [p for paths in self.written.values() for p in paths]

    @property
    def ok(self) -> bool:
        return not self.errors


class ExportError(RuntimeError):
    """A format failed; ``outcome`` still lists what the other formats wrote."""

    def __init__(self, outcome: ExportOutcome) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in outcome.errors.items()))
        self.outcome = outcome


def validate_basename(name: str) -> str | None:
    """Why ``name`` cannot be the base of the output files (``None`` when it can).

    Only a leaf file name is accepted: no folder separators (a pasted path would escape the
    chosen folder), no ``.`` / ``..``, none of the characters Windows refuses, not empty.
    Returns one of ``"empty"``, ``"dots"``, ``"characters"``.
    """
    text = name.strip()
    if not text:
        return "empty"
    if text in (".", ".."):
        return "dots"
    if _BAD_NAME_CHARS.search(text):
        return "characters"
    return None


def existing_outputs(cfg: ExportConfig) -> list[Path]:
    """Files an export with ``cfg`` would overwrite: the archive / report names, the per-frame files by pattern."""
    out = Path(cfg.out_dir)
    found: list[Path] = []
    for key, name in (("npz", f"{cfg.basename}.npz"), ("mat", f"{cfg.basename}.mat"), ("report", f"{cfg.basename}_report.pdf")):
        if getattr(cfg, key) and (out / name).exists():
            found.append(out / name)
    if cfg.csv:
        found += sorted((out / "csv").glob(f"{cfg.basename}_*.csv"))
    if cfg.vtk:
        found += sorted((out / "vtk").glob(f"{cfg.basename}*.pvd")) + sorted((out / "vtk").glob(f"{cfg.basename}_*.vti"))
    if cfg.images:
        for name in cfg.fields:
            found += sorted((out / "images").glob(f"{name}_frame_*.png"))
    return found


def _write_format(step: str, result, cfg: ExportConfig, out: Path, fields: list[str], background, progress_fn, i: int, n: int):
    from al_dvc.export import export_csv, export_mat, export_npz, export_report, export_vtk
    from al_dvc.export.slice_plots import export_field_images

    if step == "npz":
        return [Path(export_npz(result, out / f"{cfg.basename}.npz"))]
    if step == "mat":
        return [Path(export_mat(result, out / f"{cfg.basename}.mat"))]
    if step == "csv":
        return [Path(export_csv(result, out / "csv", cfg.basename, fields=fields, frames=cfg.frames)[0]).parent]
    if step == "vtk":
        return [Path(export_vtk(result, out / "vtk", cfg.basename, fields=fields, frames=cfg.frames)[0]).parent]
    if step == "report":
        return [Path(export_report(result, out / f"{cfg.basename}_report.pdf", fields=fields))]
    if step == "images":
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
            progress_fn=(lambda f, m: progress_fn((i + f) / n, f"images: {m}")) if progress_fn else None,
        )
        return [Path(files[0]).parent] if files else []
    raise ValueError(f"unknown export format {step!r}")


def export_formats(result: PipelineResult, cfg: ExportConfig, background=None, progress_fn=None, log_fn=None) -> ExportOutcome:
    """Write every selected format, going on after a failure; the outcome says what was written and what failed.

    Raises ``ValueError`` before writing anything when nothing is selected or when a format that
    takes fields has none.
    """
    steps = cfg.formats()
    if not steps:
        raise ValueError("nothing selected to export")
    fields = [f for f in cfg.fields if f in DISP_FIELDS or f in STD_FIELDS or (f in STRAIN_FIELDS and result.result_strain)]
    if not fields and cfg.needs_fields():
        raise ValueError("no field selected: CSV, ParaView, images and the report need at least one")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outcome = ExportOutcome()
    for i, step in enumerate(steps):
        if progress_fn is not None:
            progress_fn(i / len(steps), step)
        try:
            paths = _write_format(step, result, cfg, out, fields, background, progress_fn, i, len(steps))
        except Exception as exc:
            logger.exception("export %s failed", step)
            outcome.errors[step] = f"{type(exc).__name__}: {exc}"
            outcome.tracebacks[step] = traceback.format_exc()
            if log_fn is not None:
                log_fn(f"export {step} failed: {outcome.errors[step]}")
            continue
        outcome.written[step] = paths
        if log_fn is not None and paths:
            log_fn(f"exported {step}: {paths[-1]}")
    if progress_fn is not None:
        progress_fn(1.0, "done")
    return outcome


def run_export(result: PipelineResult, cfg: ExportConfig, background=None, progress_fn=None, log_fn=None) -> list[Path]:
    """Write every selected format and return the paths (files or folders) produced.

    Every format is attempted; a failure is raised afterwards as :class:`ExportError`, whose
    ``outcome`` still lists the files that were written.
    """
    outcome = export_formats(result, cfg, background, progress_fn, log_fn)
    if outcome.errors:
        raise ExportError(outcome)
    return outcome.paths


class _ExportWorker(QThread):
    progress = Signal(float, str)
    finished_outcome = Signal(object)  # ExportOutcome, complete or partial
    failed = Signal(str, str)  # nothing could be written: message, traceback
    log = Signal(str)

    def __init__(self, result, cfg: ExportConfig, background, parent=None) -> None:
        super().__init__(parent)
        self._result = result
        self._cfg = cfg
        self._background = background

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            outcome = export_formats(
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
        self.finished_outcome.emit(outcome)


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
        self._job_cfg: ExportConfig | None = None  # the configuration of the running (or last) job
        self._n_frames = 1
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
        # the settings of a job are frozen while it runs: these controls are disabled together
        self._job_controls: list[QWidget] = [
            self.folder,
            self._btn_browse,
            self.basename,
            *self.checks.values(),
            self.fields,
            self._btn_all,
            self._btn_none,
            self._btn_disp,
            self._btn_strain,
            self.all_frames,
            self.frame_from,
            self.frame_to,
            self._image_group,
        ]

        # ---- wiring
        self._btn_browse.clicked.connect(self._on_browse)
        self._btn_open.clicked.connect(self._open_folder)
        self._btn_export.clicked.connect(self.start)
        self._btn_close.clicked.connect(self.close)
        self._btn_all.clicked.connect(lambda: self._check_fields(lambda _n: True))
        self._btn_none.clicked.connect(lambda: self._check_fields(lambda _n: False))
        self._btn_disp.clicked.connect(lambda: self._check_fields(lambda n: n in DISP_FIELDS))
        self._btn_strain.clicked.connect(lambda: self._check_fields(lambda n: n in STRAIN_FIELDS))
        self.all_frames.toggled.connect(self._on_all_frames)
        self.checks["images"].toggled.connect(lambda v: self._image_group.setEnabled(v and not self._running()))
        self._image_group.setEnabled(False)
        self._state.results_changed.connect(self.refresh)
        self.retranslate_ui()
        self.refresh()
        if preselect_strain:
            self.preselect_strain()

    # ------------------------------------------------------------------ fields
    def preselect_strain(self) -> None:
        """Tick the displacement and strain fields and the CSV format (the Strain window's export entry)."""
        self._check_fields(lambda n: n in STRAIN_FIELDS or n in DISP_FIELDS)
        self.checks["csv"].setChecked(True)

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
        n = max(1, len(res.result_disp) if has else 1)
        previous, self._n_frames = self._n_frames, n
        for s in (self.frame_from, self.frame_to):
            s.setRange(1, n)
        # a range the user narrowed survives a refresh; "up to the last frame" follows the frame count
        if self.frame_to.value() > n or self.frame_to.value() == previous:
            self.frame_to.setValue(n)
        if self.frame_from.value() > n:
            self.frame_from.setValue(1)
        self._btn_export.setEnabled(has and not self._running())
        self._btn_strain.setEnabled(has and bool(res.result_strain) and not self._running())
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
        enabled = not all_frames and not self._running()
        self.frame_from.setEnabled(enabled)
        self.frame_to.setEnabled(enabled)

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

    @property
    def job_config(self) -> ExportConfig | None:
        """The configuration captured when the running (or last) job started."""
        return self._job_cfg

    # ------------------------------------------------------------------ run
    def _running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _on_browse(self) -> None:
        if headless():
            return
        folder = QFileDialog.getExistingDirectory(self, self.tr("Export folder"), self.folder.text())
        if folder:
            self.folder.setText(folder)

    def _open_folder(self) -> None:
        """While a job runs the button opens the job's folder; idle, the folder in the box."""
        if self._running() and self._job_cfg is not None:
            open_folder(Path(self._job_cfg.out_dir))
        else:
            open_folder(Path(self.folder.text() or "."))

    def _set_job_controls_enabled(self, enabled: bool) -> None:
        for w in self._job_controls:
            w.setEnabled(enabled)
        if enabled:
            res = self._state.results
            self._btn_strain.setEnabled(res is not None and bool(res.result_strain))
            self._image_group.setEnabled(self.checks["images"].isChecked())
            self._on_all_frames(self.all_frames.isChecked())

    def _basename_message(self, problem: str) -> str:
        return {
            "empty": self.tr("Base name: type a file name."),
            "dots": self.tr("Base name: '.' and '..' are not file names."),
            "characters": self.tr('Base name: a file name only, without folder separators or the characters < > : " | ? *'),
        }[problem]

    def _confirm_overwrite(self, cfg: ExportConfig) -> bool:
        """Ask before files of an earlier export are replaced (headless: go ahead)."""
        clashes = existing_outputs(cfg)
        if not clashes or headless():
            return True
        answer = QMessageBox.question(
            self,
            self.tr("Overwrite files?"),
            self.tr("{n} file(s) in {folder} would be overwritten, e.g. {name}. Continue?").format(
                n=len(clashes), folder=cfg.out_dir, name=clashes[0].name
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def start(self) -> None:
        res = self._state.results
        if res is None or self._running():
            return
        problem = validate_basename(self.basename.text())
        if problem is not None:
            self._status.setText(self._basename_message(problem))
            return
        cfg = self.config()
        if not cfg.formats():
            self._status.setText(self.tr("Select at least one format."))
            return
        if cfg.needs_fields() and not cfg.fields:
            self._status.setText(self.tr("Select at least one field: CSV, ParaView, images and the report need one."))
            return
        if not self._confirm_overwrite(cfg):
            self._status.setText(self.tr("Export cancelled: choose another base name or folder."))
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
        self._job_cfg = cfg
        self._worker = _ExportWorker(res, cfg, background, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_outcome.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.log.connect(lambda m: self._state.log(m))
        self._set_job_controls_enabled(False)
        self._btn_export.setEnabled(False)
        self._progress.setValue(0)
        self._status.setText(self.tr("Exporting to {folder}...").format(folder=cfg.out_dir))
        self._worker.start()

    def wait(self, timeout_ms: int = 300_000) -> bool:
        return self._worker.wait(timeout_ms) if self._worker is not None else True

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(round(1000 * fraction)))
        self._status.setText(self.tr("Exporting: {msg}").format(msg=message))

    def _job_folder(self) -> Path:
        return Path(self._job_cfg.out_dir) if self._job_cfg is not None else Path(self.folder.text() or ".")

    def _on_finished(self, outcome: ExportOutcome) -> None:
        folder = self._job_folder()
        self._set_job_controls_enabled(True)
        self._btn_export.setEnabled(self._state.results is not None)
        self._progress.setValue(1000)
        names = ", ".join(p.name for p in outcome.paths) or "-"
        if outcome.ok:
            self._status.setText(self.tr("Done: {names} written to {folder}").format(names=names, folder=folder))
            self._state.log(self.tr("Export finished: {names} in {folder}").format(names=names, folder=folder), "success")
        else:
            failed = "; ".join(f"{k}: {v}" for k, v in outcome.errors.items())
            text = self.tr("Export incomplete. Written: {names}. Failed: {failed}").format(names=names, failed=failed)
            self._status.setText(text)
            self._state.log(text, "error")
            for tb in outcome.tracebacks.values():
                self._state.log(tb, "debug")
        npz = outcome.written.get("npz")
        if npz and hasattr(self._state, "results_path"):
            self._state.results_path = str(npz[0])

    def _on_failed(self, message: str, detail: str) -> None:
        self._set_job_controls_enabled(True)
        self._btn_export.setEnabled(self._state.results is not None)
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
        self.basename.setToolTip(self.tr("File name without folder or extension; existing files are replaced after a question"))
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
        self._frames_label.setText(self.tr("Frames (CSV, ParaView, images)"))
        frames_tip = self.tr(
            "The frame range applies to CSV, ParaView and the slice images; the NumPy archive, the MATLAB file "
            "and the PDF report always hold every frame."
        )
        for w in (self._frames_label, self.all_frames, self.frame_from, self.frame_to):
            w.setToolTip(frames_tip)
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
