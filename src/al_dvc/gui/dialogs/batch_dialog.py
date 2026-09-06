"""Batch dialog: queue session files, run them in a worker thread, show progress and outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..app_state import AppState
from ..batch import DEFAULT_EXPORTS, EXPORT_KINDS, BatchJob, BatchRunner
from ..session import SESSION_SUFFIX

COLUMNS = ("session", "status", "nodes", "frames", "converged", "time", "message")
SESSION_FILTER = f"pyALDVC sessions (*{SESSION_SUFFIX});;All files (*)"
MAX_LOG_LINES = 2000


class BatchWorker(QThread):
    """Runs a :class:`BatchRunner` off the UI thread."""

    job_changed = Signal(int, object)  # index, BatchJob
    progress = Signal(int, int, float, str)  # index, n, fraction, message
    finished_all = Signal(object)  # list[BatchJob]

    def __init__(self, sessions, exports, checkpoints: bool, compute_strain: bool, parent=None) -> None:
        super().__init__(parent)
        self._stop = False
        self._runner = BatchRunner(
            sessions,
            exports=exports,
            checkpoints=checkpoints,
            compute_strain=compute_strain,
            progress_fn=lambda i, n, f, m: self.progress.emit(i, n, f, m),
            job_fn=lambda i, j: self.job_changed.emit(i, j),
            stop_fn=lambda: self._stop,
        )

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        jobs = self._runner.run()
        self.finished_all.emit(jobs)


class BatchDialog(QDialog):
    """Queue of sessions with per-job status; non-modal so the main window stays usable."""

    def __init__(
        self, state: AppState, parent: QWidget | None = None, open_session: Callable[[str], object] | None = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._open_session = open_session
        self._worker: BatchWorker | None = None
        self.jobs: list[BatchJob] = []
        self.setMinimumSize(860, 520)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)

        self._btn = {k: QPushButton() for k in ("add", "add_current", "remove", "clear", "start", "stop", "open", "close")}
        self._btn["start"].setProperty("class", "btn-primary")
        self._btn["start"].setMinimumHeight(30)
        self.exports = {k: QCheckBox() for k in EXPORT_KINDS}
        for k in DEFAULT_EXPORTS:
            self.exports[k].setChecked(True)
        self.checkpoints = QCheckBox()
        self.checkpoints.setChecked(True)
        self.strain = QCheckBox()
        self.strain.setChecked(True)
        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        self.current = QProgressBar()
        self.current.setRange(0, 1000)
        self._overall_label = QLabel()
        self._current_label = QLabel()
        self._exports_label = QLabel()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(MAX_LOG_LINES)
        self._summary = QLabel()
        self._summary.setObjectName("hint")
        self._summary.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        self._queue_group = QGroupBox()
        queue = QVBoxLayout(self._queue_group)
        row = QHBoxLayout()
        for k in ("add", "add_current", "remove", "clear"):
            row.addWidget(self._btn[k])
        row.addStretch(1)
        queue.addLayout(row)
        queue.addWidget(self.table, stretch=1)
        layout.addWidget(self._queue_group, stretch=2)
        self._options_group = QGroupBox()
        opts = QHBoxLayout(self._options_group)
        opts.addWidget(self._exports_label)
        for k in EXPORT_KINDS:
            opts.addWidget(self.exports[k])
        opts.addSpacing(16)
        opts.addWidget(self.checkpoints)
        opts.addWidget(self.strain)
        opts.addStretch(1)
        layout.addWidget(self._options_group)
        self._progress_group = QGroupBox()
        prog = QVBoxLayout(self._progress_group)
        prog.addWidget(self._overall_label)
        prog.addWidget(self.overall)
        prog.addWidget(self._current_label)
        prog.addWidget(self.current)
        self.log.setObjectName("console")
        prog.addWidget(self.log, stretch=1)
        prog.addWidget(self._summary)
        layout.addWidget(self._progress_group, stretch=1)
        row2 = QHBoxLayout()
        row2.addWidget(self._btn["start"])
        row2.addWidget(self._btn["stop"])
        row2.addWidget(self._btn["open"])
        row2.addStretch(1)
        row2.addWidget(self._btn["close"])
        layout.addLayout(row2)

        self._btn["add"].clicked.connect(self._on_add)
        self._btn["add_current"].clicked.connect(self._on_add_current)
        self._btn["remove"].clicked.connect(self.remove_selected)
        self._btn["clear"].clicked.connect(self.clear)
        self._btn["start"].clicked.connect(self.start)
        self._btn["stop"].clicked.connect(self.stop)
        self._btn["open"].clicked.connect(self._on_open_selected)
        self._btn["close"].clicked.connect(self.close)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.retranslate_ui()
        self._update_buttons()

    # ------------------------------------------------------------------ queue
    def sessions(self) -> list[str]:
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(self.table.rowCount())]

    def add_sessions(self, paths) -> int:
        """Append session files (duplicates and non-existing files are skipped); returns how many were added."""
        existing = set(self.sessions())
        added = 0
        for p in paths:
            path = str(Path(p))
            if path in existing or not Path(path).is_file():
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            item = QTableWidgetItem(Path(path).name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.table.setItem(r, 0, item)
            for c in range(1, len(COLUMNS)):
                self.table.setItem(r, c, QTableWidgetItem(""))
            self.table.item(r, 1).setText(self.tr("pending"))
            existing.add(path)
            added += 1
        self._update_buttons()
        return added

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._update_buttons()

    def clear(self) -> None:
        self.table.setRowCount(0)
        self.jobs = []
        self._update_buttons()

    def _on_add(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("Add sessions"), "", SESSION_FILTER)
        if files:
            self.add_sessions(files)

    def _on_add_current(self) -> None:
        """Queue the session file on disk: unsaved edits are not part of the batch, and an unsaved session cannot be."""
        path = self._state.session_path
        if path is None:
            self.append_log(self.tr("Save the session first: the batch runs the file on disk."))
            return
        if self.add_sessions([str(path)]):
            self.append_log(self.tr("Queued the saved file {path}; edits not yet saved are not included.").format(path=path))

    def _on_open_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if rows and self._open_session is not None:
            self._open_session(self.table.item(rows[0], 0).data(Qt.ItemDataRole.UserRole))

    # ------------------------------------------------------------------ running
    def selected_exports(self) -> tuple[str, ...]:
        return tuple(k for k in EXPORT_KINDS if self.exports[k].isChecked())

    def start(self) -> bool:
        if self._worker is not None and self._worker.isRunning():
            return False
        sessions = self.sessions()
        if not sessions:
            return False
        exports = self.selected_exports()
        self.jobs = []
        for r in range(self.table.rowCount()):
            for c in range(1, len(COLUMNS)):
                self.table.item(r, c).setText("")
            self.table.item(r, 1).setText(self.tr("pending"))
        self.overall.setValue(0)
        self.current.setValue(0)
        self._summary.setText("")
        self.append_log(
            self.tr("Batch of {n} session(s); exports: {exports}").format(n=len(sessions), exports=", ".join(exports) or "-")
        )
        self._worker = BatchWorker(sessions, exports, self.checkpoints.isChecked(), self.strain.isChecked(), self)
        self._worker.job_changed.connect(self._on_job_changed)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_all.connect(self._on_finished)
        # the buttons settle on the thread's own termination: ``finished_all`` is emitted inside ``run`` while
        # ``isRunning()`` can still be true
        self._worker.finished.connect(self._update_buttons)
        self._worker.start()
        self._update_buttons()
        return True

    def stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self.append_log(self.tr("Stop requested: the current session finishes its step, the rest is skipped."))
            self._update_buttons()

    def wait(self, timeout_ms: int = 600_000) -> bool:
        return True if self._worker is None else self._worker.wait(timeout_ms)

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _on_job_changed(self, index: int, job: BatchJob) -> None:
        while len(self.jobs) <= index:
            self.jobs.append(None)  # type: ignore[arg-type]
        self.jobs[index] = job
        if index >= self.table.rowCount():
            return
        cells = {
            1: self.tr(job.status),
            2: str(job.n_nodes) if job.n_nodes else "",
            3: str(job.n_frames) if job.n_frames else "",
            4: "" if job.converged != job.converged else f"{100 * job.converged:.1f} %",  # NaN check
            5: f"{job.elapsed:.1f} s" if job.elapsed else "",
            6: job.message,
        }
        for c, text in cells.items():
            self.table.item(index, c).setText(text)
        if job.status != "running":
            self.append_log(f"[{job.status}] {job.session.name}: {job.message} ({job.elapsed:.1f} s)")
            if job.traceback:
                self.append_log(job.traceback.rstrip())
        else:
            self.append_log(self.tr("Running {name} ...").format(name=job.session.name))

    def _on_progress(self, index: int, n: int, fraction: float, message: str) -> None:
        self.current.setValue(int(1000 * max(0.0, min(1.0, fraction))))
        self.overall.setValue(int(1000 * (index + max(0.0, min(1.0, fraction))) / max(n, 1)))
        self._current_label.setText(f"{index + 1}/{n}: {message}")

    def _on_finished(self, jobs) -> None:
        self.jobs = list(jobs)
        n_ok = sum(1 for j in self.jobs if j.status == "done")
        n_fail = sum(1 for j in self.jobs if j.status == "failed")
        n_other = len(self.jobs) - n_ok - n_fail
        self.overall.setValue(1000)
        self._summary.setText(
            self.tr("Finished: {ok} done, {failed} failed, {other} stopped or skipped.").format(
                ok=n_ok, failed=n_fail, other=n_other
            )
        )
        self.append_log(BatchRunner.summary_table(self.jobs))
        self._update_buttons()

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    # ------------------------------------------------------------------ misc
    def _update_buttons(self) -> None:
        running = self.running
        has_rows = self.table.rowCount() > 0
        selected = bool(self.table.selectedItems())
        self._btn["start"].setEnabled(has_rows and not running)
        self._btn["stop"].setEnabled(running)
        self._btn["remove"].setEnabled(selected and not running)
        self._btn["clear"].setEnabled(has_rows and not running)
        self._btn["add"].setEnabled(not running)
        self._btn["add_current"].setEnabled(not running and self._state.session_path is not None)
        self._btn["open"].setEnabled(selected and self._open_session is not None and not running)
        # the options were captured when the batch started: editing them now would describe the next batch only
        for w in (*self.exports.values(), self.checkpoints, self.strain):
            w.setEnabled(not running)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.running:
            self.stop()
            self.wait(60_000)
        super().closeEvent(event)

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("Batch run"))
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Session"),
                self.tr("Status"),
                self.tr("Nodes"),
                self.tr("Frames"),
                self.tr("Converged"),
                self.tr("Time"),
                self.tr("Message"),
            ]
        )
        texts = {
            "add": self.tr("Add sessions..."),
            "add_current": self.tr("Add current session"),
            "remove": self.tr("Remove"),
            "clear": self.tr("Clear"),
            "start": self.tr("Start"),
            "stop": self.tr("Stop"),
            "open": self.tr("Open in window"),
            "close": self.tr("Close"),
        }
        for k, t in texts.items():
            self._btn[k].setText(t)
        self._btn["add_current"].setToolTip(
            self.tr("Queues the session file saved on disk; edits not yet saved are not part of the batch")
        )
        self._exports_label.setText(self.tr("Exports:"))
        self._queue_group.setTitle(self.tr("Queue"))
        self._options_group.setTitle(self.tr("Options"))
        self._progress_group.setTitle(self.tr("Progress and log"))
        for k in EXPORT_KINDS:
            self.exports[k].setText(k)
        self.checkpoints.setText(self.tr("Write checkpoints"))
        self.strain.setText(self.tr("Compute strain"))
        self._overall_label.setText(self.tr("Overall progress"))
        self._current_label.setText(self.tr("Current session"))
