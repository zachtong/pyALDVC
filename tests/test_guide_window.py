"""The built-in texture analysis guide and the independent sub-windows."""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.gui.app import MainWindow, create_application  # noqa: E402
from al_dvc.gui.guide_window import ASSETS, MEDIA  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return create_application(["pytest"])


def _pump(n=10):
    for _ in range(n):
        QApplication.processEvents()


def test_guide_opens_from_the_texture_window_and_plays_its_demonstrations(qapp):
    assert all((ASSETS / name).is_file() for name in MEDIA.values())  # shipped with the package
    window = MainWindow()
    window.show()
    guide = window.open_guide_window()
    _pump()
    assert guide.isVisible() and guide.parent() is None and window.open_guide_window() is guide
    assert len(guide.movies) == 2 and all(m.isValid() for m in guide.movies)
    assert len(guide.pixmaps) == 1 and not guide.pixmaps[0].isNull()
    tw = window.open_texture_window()
    assert tw.parent() is None  # independent windows: not pinned above the main window
    guide.hide()
    tw._btn_guide.click()  # the texture window opens the same guide
    _pump()
    assert guide.isVisible()
    mgr = qapp._pyaldvc_lang_mgr
    mgr.load("zh_CN")
    _pump()
    assert guide.windowTitle() != "Texture analysis guide"
    mgr.load("en")
    _pump()
    assert guide.windowTitle() == "Texture analysis guide"
    window.close()  # closes the independent windows with it
    _pump()
    assert not guide.isVisible() and not tw.isVisible()
