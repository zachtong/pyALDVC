"""Numba import shim.

Every module that needs JIT compilation imports ``njit``, ``prange`` and the
``HAS_NUMBA`` / ``JIT_CACHE`` flags from here, never from ``numba`` directly.

Behaviour:
    - If Numba is installed, ``njit`` / ``prange`` are the real decorators.
    - If Numba is missing, ``njit`` becomes a no-op decorator and ``prange``
      becomes ``range`` so the pure-Python fallbacks still run (slowly).
    - ``JIT_CACHE`` is probed: on-disk caching is requested only when the
      package directory is writable (it is not inside frozen bundles or
      read-only site-packages).
"""

from __future__ import annotations

import os
import warnings

try:  # pragma: no cover - exercised implicitly by every kernel test
    import numba as _numba
    from numba import njit as _njit, prange as _prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _numba = None
    HAS_NUMBA = False

    def _njit(*args, **kwargs):  # type: ignore[misc]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

    _prange = range  # type: ignore[assignment]


def _probe_cache_dir() -> bool:
    """Return True when the package directory can hold Numba cache files."""
    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(here, "solver", "__pycache__")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        probe = os.path.join(cache_dir, ".write_probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


JIT_CACHE: bool = HAS_NUMBA and _probe_cache_dir()

if HAS_NUMBA and os.environ.get("ALDVC_DISABLE_JIT_CACHE"):
    JIT_CACHE = False

njit = _njit
prange = _prange


def get_num_threads() -> int:
    """Number of threads Numba will use for ``prange`` loops."""
    if HAS_NUMBA:
        return int(_numba.get_num_threads())
    return 1


def set_num_threads(n: int) -> None:
    """Set the number of Numba threads (clamped to the machine's CPU count)."""
    if not HAS_NUMBA:
        return
    n = max(1, min(int(n), int(_numba.config.NUMBA_NUM_THREADS)))
    _numba.set_num_threads(n)


if not HAS_NUMBA:  # pragma: no cover
    warnings.warn(
        "Numba is not installed: al_dvc will run pure-Python kernels, "
        "which are 100-1000x slower. Install numba for production use.",
        RuntimeWarning,
        stacklevel=1,
    )
