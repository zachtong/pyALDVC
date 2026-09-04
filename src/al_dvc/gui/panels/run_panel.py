"""Run / stop the pipeline and show its progress (top of the right sidebar).

The results stay in memory and are exported afterwards from the results panel; a checkpoint
directory is written only when the advanced option is on, and a directory left by a different
run is replaced instead of raising.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from ..app_state import AppState, RunState
from ..pipeline_worker import PipelineWorker

CHECKPOINT_SUBDIR = "checkpoints"


class RunPanel(QWidget):
    """Run and stop buttons, progress bar and status line; owns the worker thread."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker: PipelineWorker | None = None
        self._started = 0.0
        self._btn_run = QPushButton()
        self._btn_run.setProperty("class", "btn-primary")
        self._btn_run.setMinimumHeight(32)
        self._btn_stop = QPushButton()
        self._btn_stop.setEnabled(False)
        self._btn_stop.setMinimumHeight(32)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        self._message = QLabel()
        self._message.setWordWrap(True)
        self._message.setObjectName("hint")
        self._elapsed = QLabel()
        self._elapsed.setObjectName("hint")
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self._btn_run, 2)
        row.addWidget(self._btn_stop, 1)
        layout.addLayout(row)
        layout.addWidget(self._progress)
        status = QHBoxLayout()
        status.addWidget(self._message, 1)
        status.addWidget(self._elapsed, 0)
        layout.addLayout(status)

        self._btn_run.clicked.connect(self.start)
        self._btn_stop.clicked.connect(self.stop)
        self._state.progress_updated.connect(self._on_progress)
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
        from dataclasses import replace

        from al_dvc.core.config import validate_dvcpara
        from al_dvc.utils.validation import validate_para_against_volume

        shape = tuple(int(s) for s in volumes[0].shape)
        para = self._state.para
        voi = self._state.effective_voi()
        if voi is not None and (para.voi is None or para.voi.is_whole):
            para = replace(para, voi=voi)
            box = voi.clamp(shape)
            self._state.log(
                self.tr("Analysed box from the region of interest: x {x0}-{x1}, y {y0}-{y1}, z {z0}-{z1}").format(
                    x0=box.x[0], x1=box.x[1], y0=box.y[0], y1=box.y[1], z0=box.z[0], z1=box.z[1]
                )
            )
        try:
            validate_dvcpara(para)
            validate_para_against_volume(para, shape)
        except ValueError as exc:
            self._state.log(self.tr("Parameters do not fit the volume: {error}").format(error=exc), "error")
            return
        masks = masks_list if any(m is not None for m in masks_list) else None
        if masks is not None:
            masks = [m if m is not None else np.ones(v.shape, dtype=bool) for m, v in zip(masks_list, volumes)]
        checkpoint = Path(self._state.output_dir) / CHECKPOINT_SUBDIR if self._state.write_checkpoints else None
        self._worker = PipelineWorker(para, volumes, masks, checkpoint_dir=checkpoint, resume="auto", parent=self)
        self._worker.progress.connect(self._state.set_progress)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._started = time.perf_counter()
        self._state.set_results(None)
        self._state.set_run_state(RunState.RUNNING)
        self._state.set_progress(0.0, self.tr("Starting..."))
        self._state.log(
            self.tr("Run started: {n} frames, subset {ws}, step {st}").format(
                n=len(volumes), ws=para.winsize[0] + 1, st=para.winstepsize[0]
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
            self._state.log(self.tr("Finished {n} frame(s) in {s:.1f} s").format(n=result.n_frames, s=elapsed), "success")

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
        self._elapsed.setText(self.tr("{s:.0f} s").format(s=elapsed))

    def _update_buttons(self) -> None:
        running = self._state.run_state in (RunState.RUNNING, RunState.STOPPING)
        self._btn_run.setEnabled(not running and len(self._state.volumes) >= 2)
        self._btn_stop.setEnabled(self._state.run_state == RunState.RUNNING)

    def retranslate_ui(self) -> None:
        self._btn_run.setText(self.tr("Run AL-DVC"))
        self._btn_stop.setText(self.tr("Stop"))
