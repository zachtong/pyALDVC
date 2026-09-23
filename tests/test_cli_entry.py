"""``al-dvc`` alone opens the application, as pyALDIC's ``al-dic`` does; the commands (``al-dvc run``, ...) are unchanged."""

import json
import sys

import numpy as np
import pytest

from al_dvc import cli
from al_dvc.io.volume_io import save_volume


@pytest.fixture
def window(monkeypatch):
    """What would have opened the window, recorded instead: one list of arguments per launch."""
    import al_dvc.gui.app as app

    launches: list[list[str]] = []
    monkeypatch.setattr(app, "main", lambda argv=None: launches.append(list(argv)[1:]) or 0)
    monkeypatch.setattr(cli, "_display_available", lambda: True)
    return launches


def test_al_dvc_alone_opens_the_application(window):
    assert cli.main([]) == 0
    assert window == [[]]


def test_a_session_file_opens_in_the_application(window):
    assert cli.main(["study/scan.aldvc"]) == 0
    assert cli.main(["SCAN.ALDVC"]) == 0
    assert window == [["study/scan.aldvc"], ["SCAN.ALDVC"]]


def test_the_self_test_runs_even_without_a_display(window, monkeypatch):
    """It runs offscreen, so a server without a display can check its install too."""
    monkeypatch.setattr(cli, "_display_available", lambda: False)
    assert cli.main(["--self-test", "report.txt"]) == 0
    assert cli.main(["--self-test"]) == 0
    assert window == [["--self-test", "report.txt"], ["--self-test"]]


def test_the_commands_are_unchanged(window, tmp_path, capsys):
    save_volume(tmp_path / "v.npy", np.zeros((4, 5, 6), np.float32))
    assert cli.main(["info", str(tmp_path / "v.npy")]) == 0
    assert json.loads(capsys.readouterr().out)["shape_zyx"] == [4, 5, 6]
    assert window == []
    assert cli.main(["gui"]) == 0 and cli.main(["gui", "a.aldvc"]) == 0  # the old spelling still opens it
    assert window == [[], ["a.aldvc"]]
    with pytest.raises(SystemExit):
        cli.main(["no-such-command"])


def test_without_a_display_al_dvc_lists_the_commands_instead(window, monkeypatch, capsys):
    """An SSH session on a cluster: Qt would abort, so the user gets the commands and why."""
    monkeypatch.setattr(cli, "_display_available", lambda: False)
    assert cli.main([]) == 2
    out, err = capsys.readouterr()
    assert "run" in out and "batch" in out and "display" in err
    assert window == []


def test_a_display_is_missing_only_on_a_linux_machine_without_one(monkeypatch):
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert cli._display_available() is False
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert cli._display_available() is True
    monkeypatch.delenv("WAYLAND_DISPLAY")
    monkeypatch.setenv("DISPLAY", ":0")
    assert cli._display_available() is True
    monkeypatch.delenv("DISPLAY")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert cli._display_available() is True
    monkeypatch.delenv("QT_QPA_PLATFORM")
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(sys, "platform", platform)
        assert cli._display_available() is True


def test_the_help_says_what_al_dvc_alone_does(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    assert "Without a command, al-dvc opens the application" in " ".join(capsys.readouterr().out.split())


def test_both_names_are_installed():
    """``al-dvc`` for everything; ``al-dvc-gui`` keeps working for shortcuts and notes made with 0.9."""
    tomllib = pytest.importorskip("tomllib")
    from pathlib import Path

    scripts = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["al-dvc"] == "al_dvc.cli:main"
    assert scripts["al-dvc-gui"] == "al_dvc.gui.app:main"
