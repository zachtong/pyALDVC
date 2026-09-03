"""PyInstaller runtime hook: environment defaults for the frozen build.

Runs before ``al_dvc``, matplotlib or numba is imported. Numba's cache lands
in the user-wide cache directory under ``sys.frozen`` (a ``NUMBA_CACHE_DIR``
would be ignored because the source files do not exist in the bundle);
``al_dvc._numba_compat`` copes with an unwritable one.
"""

import os
import sys
import tempfile

if getattr(sys, "frozen", False):
    _base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~") or tempfile.gettempdir()
    # A writable matplotlib config directory avoids a full font scan at every start.
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_base, "pyALDVC", "mpl"))
    # Ignore ambient matplotlib configuration of the launching directory.
    os.environ.pop("MATPLOTLIBRC", None)

    # The windowed executable (console=False) starts with sys.stdout/stderr set
    # to None; anything that writes to them directly (C extensions, third-party
    # warnings) would raise. Send both to a log file next to the GUI log instead.
    if sys.stdout is None or sys.stderr is None:
        try:
            _log_dir = os.path.join(_base, "pyALDVC")
            os.makedirs(_log_dir, exist_ok=True)
            _stream = open(os.path.join(_log_dir, "stderr.log"), "a", encoding="utf-8", buffering=1)
        except OSError:
            _stream = open(os.devnull, "w", encoding="utf-8")
        if sys.stdout is None:
            sys.stdout = _stream
        if sys.stderr is None:
            sys.stderr = _stream
