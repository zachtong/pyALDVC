"""Run / stop the pipeline, show progress and the log."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget

from ..app_state import AppState, RunState
from ..pipeline_worker import PipelineWorker

MAX_LOG_LINES = 2000


class RunPanel(QWidget):
    """Buttons, progress bar, message and log console; owns the worker thread."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: PipelineWorker | None = None
        self._started = 0.0
        self._btn_run = QPushButton()
        self._btn_run.setObjectName("primary")
        self._btn_stop = QPushButton()
        self._btn_stop.setEnabled(False)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._message = QLabel()
        self._message.setWordWrap(True)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(MAX_LOG_LINES)
        self._log.setMinimumHeight(120)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(self._btn_run)
        row.addWidget(self._btn_stop)
        layout.addLayout(row)
        layout.addWidget(self._progress)
        layout.addWidget(self._message)
        layout.addWidget(self._log)

        self._btn_run.clicked.connect(self.start)
        self._btn_stop.clicked.connect(self.stop)
        self._state.progress_updated.connect(self._on_progress)
        self._state.log_message.connect(self.append_log)
        self._state.run_state_changed.connect(self._on_run_state)
        self._state.volumes_changed.connect(self._update_buttons)
        self.retranslate_ui()
        self._update_buttons()

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        entries = list(self._state.volumes)
        if len(entries) < 2:
            self._state.log(self.tr("At least two volumes are needed."), "warning")
            return
        try:
            volumes = [e.load() for e in entries]
            masks_list = [e.load_mask() for e in entries]
        except Exception as exc:
            self._state.log(self.tr("Cannot load the volumes: {error}").format(error=exc), "error")
            return
        from al_dvc.core.config import validate_dvcpara
        from al_dvc.utils.validation import validate_para_against_volume

        try:
            validate_dvcpara(self._state.para)
            validate_para_against_volume(self._state.para, tuple(int(s) for s in volumes[0].shape))
        except ValueError as exc:
            self._state.log(self.tr("Parameters do not fit the volume: {error}").format(error=exc), "error")
            return
        masks = masks_list if any(m is not None for m in masks_list) else None
        if masks is not None:
            masks = [m if m is not None else np.ones(v.shape, dtype=bool) for m, v in zip(masks_list, volumes)]
        out = Path(self._state.output_dir)
        checkpoint = out / "checkpoints" if self._state.write_checkpoints else None
        self._worker = PipelineWorker(self._state.para, volumes, masks, checkpoint_dir=checkpoint, parent=self)
        self._worker.progress.connect(self._state.set_progress)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._started = time.perf_counter()
        self._state.set_results(None)
        self._state.set_run_state(RunState.RUNNING)
        self._state.set_progress(0.0, self.tr("Starting..."))
        self._state.log(
            self.tr("Run started: {n} frames, subset {ws}, step {st}").format(
                n=len(volumes), ws=self._state.para.winsize, st=self._state.para.winstepsize
            )
        )
        self._timer.start()
        self._worker.start()

    def stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._state.set_run_state(RunState.STOPPING)
            self._state.log(self.tr("Stop requested; finishing the current frame..."), "warning")

    def wait(self, timeout_ms: int = 600_000) -> bool:
        """Block until the worker finishes (tests)."""
        return self._worker.wait(timeout_ms) if self._worker is not None else True

    # ------------------------------------------------------------------ slots
    def _on_finished(self, result) -> None:
        self._timer.stop()
        elapsed = time.perf_counter() - self._started
        self._state.set_results(result)
        self._state.set_run_state(RunState.DONE)
        self._state.set_progress(1.0, self.tr("Done in {s:.1f} s").format(s=elapsed))
        if result.stopped_early:
            self._state.log(self.tr("Stopped early: {n} frame(s) kept").format(n=result.n_frames), "warning")
        else:
            self._state.log(self.tr("Finished {n} frame(s) in {s:.1f} s").format(n=result.n_frames, s=elapsed))

    def _on_failed(self, message: str, detail: str) -> None:
        self._timer.stop()
        self._state.set_run_state(RunState.FAILED)
        self._state.set_progress(0.0, message)
        self._state.log(message, "error")
        self._state.log(detail, "debug")

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(round(1000 * fraction)))
        self._message.setText(message)

    def _on_run_state(self, state: RunState) -> None:
        self._update_buttons()

    def _tick(self) -> None:
        elapsed = time.perf_counter() - self._started
        self._btn_stop.setText(self.tr("Stop  ({s:.0f} s)").format(s=elapsed))

    def _update_buttons(self) -> None:
        running = self._state.run_state in (RunState.RUNNING, RunState.STOPPING)
        self._btn_run.setEnabled(not running and len(self._state.volumes) >= 2)
        self._btn_stop.setEnabled(self._state.run_state == RunState.RUNNING)
        if not running:
            self._btn_stop.setText(self.tr("Stop"))

    def append_log(self, message: str, level: str = "info") -> None:
        prefix = {"warning": "[warning] ", "error": "[error] ", "debug": "    "}.get(level, "")
        self._log.appendPlainText(prefix + message)

    def retranslate_ui(self) -> None:
        self._btn_run.setText(self.tr("Run AL-DVC"))
        self._btn_stop.setText(self.tr("Stop"))
