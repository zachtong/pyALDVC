"""Run the AL-DVC pipeline on a background thread with progress and stop support."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from al_dvc.core.config import DVCPara

logger = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """Execute :func:`al_dvc.run_aldvc` off the UI thread.

    Signals: ``progress(fraction, message)``, ``log(message, level)``,
    ``finished_result(PipelineResult)`` and ``failed(title, detail)``. A
    stop request ends the run after the current frame; the frames finished so
    far come back as a partial result (``stopped_early`` set).
    """

    progress = Signal(float, str)
    log = Signal(str, str)
    finished_result = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        para: DVCPara,
        volumes: list,
        masks: list | None,
        checkpoint_dir: str | Path | None = None,
        compute_strain: bool = True,
        resume: bool | str = "auto",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._resume = resume
        self._para = para
        self._volumes = volumes
        self._masks = masks
        self._checkpoint_dir = checkpoint_dir
        self._compute_strain = compute_strain
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop

    def run(self) -> None:  # noqa: D401 - QThread entry point
        from al_dvc.core.pipeline import run_aldvc

        try:
            result = run_aldvc(
                self._para,
                self._volumes,
                masks=self._masks,
                progress_fn=lambda frac, msg: self.progress.emit(float(frac), str(msg)),
                stop_fn=self._should_stop,
                compute_strain=self._compute_strain,
                checkpoint_dir=self._checkpoint_dir,
                resume=self._resume,
            )
        except Exception as exc:  # surface to the UI, never crash the thread
            logger.exception("Pipeline failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        self.finished_result.emit(result)
