"""Saving and opening sessions off the UI thread, with progress in the status bar (the main window's File menu).

A session with results can hold hundreds of megabytes, so the writing and the reading run on a worker
thread (:class:`SessionWorker`) while the window stays responsive: the snapshot a save writes is taken on
the UI thread first (``session.prepare_save``), and a session that was read is applied on the UI thread
afterwards (``session.apply_session`` and ``session_views.restore``). Errors come back as a message, never
as a crash. Callers that need the outcome at once (the unsaved-changes prompt, scripts and tests) pass
``wait=True``: the call then returns when the job is done, processing events meanwhile.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressBar

from al_dvc import __version__

from . import session_views
from .session import SessionError, apply_session, load_session, prepare_save, write_session

logger = logging.getLogger(__name__)

PROGRESS_STEPS = 1000


class SessionWorker(QThread):
    """``fn(progress)`` on a worker thread; the outcome is left in :attr:`result` or :attr:`error`."""

    progress = Signal(float, str)

    def __init__(self, kind: str, fn: Callable[[Callable[[float, str], None]], Any], path: Path, parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.path = Path(path)
        self._fn = fn
        self.result: Any = None
        self.error: str | None = None
        self.handled = False

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            self.result = self._fn(lambda f, m="": self.progress.emit(float(f), str(m)))
        except SessionError as exc:
            self.error = str(exc)
        except Exception as exc:  # never crash the thread: the window reports it
            logger.exception("session %s failed", self.kind)
            self.error = f"{type(exc).__name__}: {exc}"


class SessionOps(QObject):
    """The main window's session jobs: one at a time, each with a progress bar in the status bar."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        self._worker: SessionWorker | None = None
        self._was_dirty = False
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, PROGRESS_STEPS)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.setVisible(False)
        window.statusBar().addPermanentWidget(self.progress_bar)

    @property
    def state(self):
        return self._window.state

    @property
    def busy(self) -> bool:
        """A session is being saved or opened."""
        return self._worker is not None

    def tr(self, text: str) -> str:  # noqa: D102 - the window's translation context
        return self._window.tr(text)

    # ------------------------------------------------------------------ jobs
    def _start(self, worker: SessionWorker, wait: bool):
        self._worker = worker
        worker.progress.connect(lambda f, _m, w=worker: self._on_progress(w, f))
        worker.finished.connect(lambda w=worker: self._on_finished(w))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._on_progress(worker, 0.0)
        worker.start()
        if not wait:
            return None
        while not worker.isFinished():
            QApplication.processEvents()
            QThread.msleep(5)
        worker.wait()
        return self._on_finished(worker)

    def wait(self, timeout_ms: int = 600_000) -> bool:
        """Until the running job is done (tests, closing); False on a timeout."""
        worker = self._worker
        if worker is None:
            return True
        ok = worker.wait(timeout_ms)
        if ok:
            self._on_finished(worker)
        return ok

    def _on_progress(self, worker: SessionWorker, fraction: float) -> None:
        if worker is not self._worker:
            return
        self.progress_bar.setValue(int(round(PROGRESS_STEPS * max(0.0, min(1.0, fraction)))))
        what = self.tr("Saving the session: {pct} %") if worker.kind == "save" else self.tr("Opening the session: {pct} %")
        self._window.statusBar().showMessage(what.format(pct=int(round(100 * fraction))))

    def _on_finished(self, worker: SessionWorker):
        if worker.handled:
            return worker.result if worker.error is None else None
        worker.handled = True
        if worker is self._worker:
            self._worker = None
        self.progress_bar.setVisible(False)
        self._window.statusBar().clearMessage()
        worker.deleteLater()
        if worker.kind == "save":
            return self._saved(worker)
        return self._opened(worker)

    # ------------------------------------------------------------------ saving
    def save(self, path: str | Path, wait: bool = False) -> Path | None:
        """Save the session to ``path``; with ``wait`` the written path (None on failure), else None at once."""
        window, state = self._window, self.state
        results_path = state.results_path  # the archive the export dialog or a batch actually wrote
        if results_path is None and state.results is not None:
            candidate = Path(state.output_dir) / "aldvc.npz"
            if candidate.exists():
                results_path = str(candidate)
        try:
            plan = prepare_save(state, path, results_path, session_views.collect(window))
        except SessionError as exc:
            window._message("critical", self.tr("Cannot save session"), str(exc))
            return None
        except Exception as exc:  # a bug must not take the window down: say it, keep the session as it is
            logger.exception("preparing the session save failed")
            window._message("critical", self.tr("Cannot save session"), f"{type(exc).__name__}: {exc}")
            return None
        # edits made while the file is written make the session dirty again; a failure restores the flag
        self._was_dirty = state.dirty
        state.dirty = False
        worker = SessionWorker("save", lambda progress: write_session(plan, progress), plan.path, self)
        return self._start(worker, wait)

    def _saved(self, worker: SessionWorker) -> Path | None:
        window, state = self._window, self.state
        if worker.error is not None:
            state.dirty = state.dirty or self._was_dirty
            window._message("critical", self.tr("Cannot save session"), worker.error)
            return None
        p = Path(worker.result)
        state.session_path = p
        window.setWindowTitle(f"pyALDVC {__version__} - {p.name}")
        window._on_run_state_changed(state.run_state)
        state.log(self.tr("Session saved: {path}").format(path=p))
        window.remember_session(p)
        return p

    # ------------------------------------------------------------------ opening
    def open(self, path: str | Path, wait: bool = False) -> list[str] | None:
        """Read the session at ``path`` on a worker, then apply it; with ``wait`` the saved paths of the volumes still
        missing (``[]`` also on failure), else None at once."""
        worker = SessionWorker("open", lambda progress: load_session(path, progress=progress), Path(path), self)
        return self._start(worker, wait)

    def _opened(self, worker: SessionWorker) -> list[str]:
        window, state = self._window, self.state
        if worker.error is not None:
            window._message("critical", self.tr("Cannot open session"), worker.error)
            return []
        if getattr(window, "closing", False):  # the window is closing: a session read meanwhile is not applied
            return []
        if state.busy:  # a run was started while the file was read: it must keep its session
            window._message(
                "warning", self.tr("Open session"), self.tr("A run is in progress. Stop it before changing the session.")
            )
            return []
        data, path = worker.result, str(worker.path)
        try:
            missing = apply_session(data, state, path, locate_folder_cb=window.locate_volumes_folder)
        except SessionError as exc:
            window._message("critical", self.tr("Cannot open session"), str(exc))
            return []
        try:
            window.viewer.sync_from_state()
            session_views.restore(window)  # window settings: a failure here leaves the session itself open
        except Exception as exc:
            logger.exception("restoring the window settings of %s failed", path)
            state.log(f"the window settings of the session could not all be restored: {type(exc).__name__}: {exc}", "warning")
        window.setWindowTitle(f"pyALDVC {__version__} - {Path(path).name}")
        window._on_run_state_changed(state.run_state)
        if missing:
            window._message(
                "warning",
                self.tr("Missing volumes"),
                self.tr("{n} volume file(s) of the session were not found:\n{files}").format(
                    n=len(missing), files="\n".join(missing[:8])
                )
                + "\n\n"
                + self.tr(
                    "Results, masks and settings were restored. Move the files back, or open the session again and "
                    "choose the folder that holds them."
                ),
            )
        state.log(self.tr("Session loaded: {path}").format(path=path))
        window.remember_session(path)
        QApplication.processEvents()  # the views react to the restored state before it is declared clean
        state.mark_clean()
        return missing

    def ask_folder(self, missing: str) -> str | None:
        """Ask once for the folder of the first missing volume (never headless: a dialog would block for ever)."""
        window = self._window
        if window.headless:
            self.state.log(f"volume not found (headless: no folder asked for): {missing}", "warning")
            return None
        QMessageBox.information(
            window,
            self.tr("Locate the session's volumes"),
            self.tr(
                "Volume files of the session were not found, for example:\n{path}\n\nThe results were restored. To show "
                "the grey values, choose the folder that now holds this file; the other missing files are searched for "
                "there too."
            ).format(path=missing),
        )
        folder = QFileDialog.getExistingDirectory(
            window, self.tr("Folder holding {name}").format(name=Path(missing).name), str(Path(missing).parent)
        )
        return folder or None
