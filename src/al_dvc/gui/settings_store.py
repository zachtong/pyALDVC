"""Where the GUI keeps its settings, and how tests and scripts keep away from the user's.

Two stores, both under the organisation "pyALDVC": ``QSettings()`` (application "pyALDVC": language,
theme) and :func:`gui_settings` (application "gui": window layout, recent sessions, toggles). On
Windows both live in the registry under ``HKCU\\Software\\pyALDVC``.

Both are opened in the *default* format. ``QSettings("pyALDVC", "gui")`` would always open the
native store and ignore a redirect of the default format, so a test or a screenshot script
overwrote the user's window layout and cleared their recent sessions. :func:`use_private_settings`
sends both stores to ini files in a folder of its own; ``create_application`` calls it for every
offscreen run (tests, report and screenshot scripts), which is never the user's session.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QSettings

SETTINGS_ORG = "pyALDVC"
SETTINGS_APP = "gui"


def gui_settings() -> QSettings:
    """The GUI's own store (window layout, recent sessions, toggles), in the default format."""
    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, SETTINGS_ORG, SETTINGS_APP)


def settings_are_private() -> bool:
    """True once the default format no longer points at the user's native store."""
    return QSettings.defaultFormat() != QSettings.Format.NativeFormat


def use_private_settings(folder: str | Path | None = None) -> Path:
    """Send every settings store of this process to ini files in ``folder``; call before any window.

    Without ``folder`` a temporary one is made and removed when the process ends. Returns the folder.
    """
    if folder is None:
        path = Path(tempfile.mkdtemp(prefix="pyaldvc-settings-"))
        atexit.register(shutil.rmtree, path, ignore_errors=True)
    else:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(path))
    return path
