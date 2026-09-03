"""Compile the Numba kernels in the background, before the user needs them.

The solver kernels are JIT-compiled on first use and cached on disk, so the
cost is paid once per installation -- but it lands on the first click of
*Run* and freezes the interface for tens of seconds with nothing to explain
it. Compilation therefore starts shortly after the window appears, on a
daemon thread (compilation cannot be interrupted, and a daemon thread is
reclaimed at interpreter shutdown without the abort-on-close problem of a
``QThread``). The one Qt object is the signal carrier on the main thread.
"""

from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

START_DELAY_MS = 2000  # after the first paint, while the user reaches for the mouse
REPORT_THRESHOLD_S = 1.0  # below this the cache was already warm


class KernelWarmup(QObject):
    """Runs one small pipeline on a daemon thread to populate the JIT cache."""

    compiled = Signal(float)  # elapsed seconds; not emitted when the cache was warm or warm-up failed

    def start(self) -> None:
        threading.Thread(target=self._run, name="pyALDVC-kernel-warmup", daemon=True).start()

    def _run(self) -> None:
        started = time.perf_counter()
        try:
            from al_dvc.solver.warmup import warmup

            warmup()
        except Exception:
            logger.exception("Kernel warm-up failed; the first run will compile the kernels itself")
            return
        elapsed = time.perf_counter() - started
        logger.info("Kernel warm-up finished in %.1fs", elapsed)
        if elapsed >= REPORT_THRESHOLD_S:
            self.compiled.emit(elapsed)
