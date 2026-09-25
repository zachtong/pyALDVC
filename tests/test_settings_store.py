"""The GUI's settings stores: the user's stay where 1.0.1 kept them; tests and offscreen scripts never touch them."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from al_dvc.gui.settings_store import (  # noqa: E402
    SETTINGS_APP,
    SETTINGS_ORG,
    gui_settings,
    settings_are_private,
    use_private_settings,
)


def test_the_gui_store_follows_the_test_redirect():
    """Window layout, recent sessions and toggles go to the test folder, not to the registry."""
    settings = gui_settings()
    assert settings_are_private() and settings.format() == QSettings.Format.IniFormat
    assert settings.organizationName() == SETTINGS_ORG and settings.applicationName() == SETTINGS_APP
    settings.setValue("probe/value", 7)
    settings.sync()
    path = Path(settings.fileName())
    assert path.is_file() and path.suffix == ".ini"
    app_store = QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, SETTINGS_ORG, "pyALDVC")
    assert Path(app_store.fileName()).parent == path.parent  # language and theme share the private folder
    settings.remove("probe")


def test_the_user_store_is_where_it_was():
    """In the application (native default format) the store is the one ``QSettings(org, app)`` opened in 1.0.1."""
    native = QSettings(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, SETTINGS_ORG, SETTINGS_APP)
    assert native.fileName() == QSettings(SETTINGS_ORG, SETTINGS_APP).fileName()


def test_use_private_settings_makes_the_folder(tmp_path):
    folder = tmp_path / "nested" / "settings"
    session_file = gui_settings().fileName()  # <session folder>/pyALDVC/gui.ini
    try:
        assert use_private_settings(folder) == folder and folder.is_dir()
        settings = gui_settings()
        settings.setValue("probe", 1)
        settings.sync()
        assert (folder / SETTINGS_ORG / f"{SETTINGS_APP}.ini").is_file()
    finally:  # back to the session's folder for the other tests
        use_private_settings(Path(session_file).parents[1])
    assert settings_are_private() and gui_settings().fileName() == session_file


def test_an_offscreen_application_keeps_away_from_the_user_store():
    """A report or screenshot script (offscreen, no redirect of its own) gets a temporary store, removed at exit."""
    code = textwrap.dedent(
        """
        import json, os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtCore import QSettings
        from al_dvc.gui.app import create_application
        from al_dvc.gui.settings_store import gui_settings
        create_application(["probe"])
        gui_settings().setValue("probe", 1)
        gui_settings().sync()
        print(json.dumps({
            "ini": QSettings.defaultFormat() == QSettings.Format.IniFormat,
            "gui": gui_settings().fileName(),
            "app": QSettings().fileName(),
            "written": os.path.isfile(gui_settings().fileName()),
        }))
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "PYALDVC_THEME"}
    env["QT_QPA_PLATFORM"] = "offscreen"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    info = json.loads(out.stdout.strip().splitlines()[-1])
    gui_file, app_file = Path(info["gui"]), Path(info["app"])
    assert info["ini"] and info["written"]
    assert gui_file.name == f"{SETTINGS_APP}.ini" and app_file.name == "pyALDVC.ini"
    folder = gui_file.parents[1]
    assert folder.name.startswith("pyaldvc-settings-") and app_file.parents[1] == folder
    assert not folder.exists()  # removed when the script ended
