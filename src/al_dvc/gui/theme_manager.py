"""Switching the colour theme of the running application (*Settings > Theme*).

One :class:`ThemeManager` per ``QApplication`` (:func:`theme_manager`) applies a theme: it makes the
palette current (:func:`al_dvc.gui.theme.set_current_theme`), re-applies the application stylesheet,
renders again what carries colours of its own -- widgets styled with :func:`themed`, buttons whose
icon was set with :func:`al_dvc.gui.icons.set_icon`, matplotlib toolbars, Windows title bars --,
stores the choice in ``QSettings`` and emits :attr:`ThemeManager.theme_changed`. The canvases redraw
on that signal (:func:`connect_theme`), reading their colours at draw time, so nothing waits for a
restart. :func:`create_application <al_dvc.gui.app.create_application>` applies :func:`startup_theme`
(the saved choice) before the first window is built.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TypeVar

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QComboBox, QWidget

from .icons import ICON_PROPERTY, icon
from .theme import DEFAULT_THEME, THEMES, build_stylesheet, current_colors, current_theme, set_current_theme, style_text
from .window_chrome import set_title_bar_theme

logger = logging.getLogger(__name__)

# in the application's settings, QSettings(), like the language (see settings_store for the two stores)
SETTINGS_KEY = "ui/theme"
ENV_THEME = "PYALDVC_THEME"  # environment override of the saved choice, like PYALDVC_LANGUAGE
STYLE_PROPERTY = "pyaldvcStyle"  # the style sheet template of a widget styled with themed()

W = TypeVar("W", bound=QWidget)

__all__ = [
    "ENV_THEME",
    "SETTINGS_KEY",
    "ThemeDefault",
    "ThemeManager",
    "apply_title_bar",
    "connect_theme",
    "refresh_toolbar_icons",
    "restyle",
    "startup_theme",
    "theme_manager",
    "themed",
]


def startup_theme() -> str:
    """The theme to start with: ``$PYALDVC_THEME`` when it names one (scripts that capture one theme whatever was
    chosen), else the choice saved last time (``QSettings`` ``ui/theme``), else the dark default."""
    forced = os.environ.get(ENV_THEME, "")
    if forced in THEMES:
        return forced
    name = QSettings().value(SETTINGS_KEY, DEFAULT_THEME, type=str)
    return name if name in THEMES else DEFAULT_THEME


class ThemeManager(QObject):
    """Applies a theme to the whole application; :attr:`theme_changed` tells the canvases to redraw."""

    theme_changed = Signal(str)

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._app = app

    @property
    def name(self) -> str:
        return current_theme()

    def set_theme(self, name: str, persist: bool = True) -> None:
        """Switch to ``name`` (``"dark"`` or ``"light"``) now; ``persist`` remembers it for the next start."""
        if name not in THEMES:
            raise ValueError(f"unknown theme {name!r}; available: {sorted(THEMES)}")
        changed = name != current_theme()
        set_current_theme(name)
        css = build_stylesheet()
        if self._app.styleSheet() != css:  # setting a sheet re-polishes every widget: not for the same sheet
            self._app.setStyleSheet(css)
        if persist:
            QSettings().setValue(SETTINGS_KEY, name)
        if not changed:
            return
        for window in self._app.topLevelWidgets():
            restyle(window)
        logger.info("theme: %s", name)
        self.theme_changed.emit(name)


def theme_manager(app: QApplication | None = None) -> ThemeManager:
    """The application's theme manager, created on first use."""
    app = app if app is not None else QApplication.instance()
    if app is None:
        raise RuntimeError("the theme manager needs a QApplication")
    manager = getattr(app, "_pyaldvc_theme_mgr", None)
    if manager is None:
        manager = ThemeManager(app)
        app._pyaldvc_theme_mgr = manager  # type: ignore[attr-defined]
    return manager


def connect_theme(slot) -> None:
    """Call ``slot(name)`` after every theme change. ``slot`` should be a method of a ``QObject``, so the
    connection ends with the object."""
    theme_manager().theme_changed.connect(slot)


def themed(widget: W, template: str) -> W:
    """Style ``widget`` with ``template``, a style sheet whose ``{FIELD}`` placeholders name colours of
    :class:`~al_dvc.gui.theme.Colors` (literal braces doubled); it is filled again after every theme change."""
    widget.setProperty(STYLE_PROPERTY, template)
    widget.setStyleSheet(style_text(template))
    return widget


def apply_title_bar(window: QWidget) -> None:
    """The Windows title bar of ``window`` in the theme's variant (dark or light)."""
    set_title_bar_theme(window, current_colors().IS_DARK)


def restyle(root: QWidget) -> None:
    """Render ``root`` and its children again in the current theme: themed style sheets, remembered icons,
    matplotlib toolbar icons and, for a window that already has a native window, the title bar."""
    for widget in (root, *root.findChildren(QWidget)):
        template = widget.property(STYLE_PROPERTY)
        if template:
            widget.setStyleSheet(style_text(str(template)))
        name = widget.property(ICON_PROPERTY)
        if name and hasattr(widget, "setIcon"):
            widget.setIcon(icon(str(name)))
    from matplotlib.backends.backend_qt import NavigationToolbar2QT

    for toolbar in root.findChildren(NavigationToolbar2QT):
        refresh_toolbar_icons(toolbar)
    if root.isWindow() and root.internalWinId():
        apply_title_bar(root)


def refresh_toolbar_icons(toolbar) -> None:
    """Draw a matplotlib toolbar's icons in the theme's text colour.

    matplotlib picks their colour itself from the toolbar's background -- 3.10 when the toolbar is built (the text
    colour on a dark background, black otherwise), 3.11 at every paint (white or black) -- and the light theme
    leaves that background transparent, which reads as black: matplotlib 3.11 drew white icons on the white
    window. The icons are made here from matplotlib's own images instead, so they follow the theme alone.
    """
    colour = QColor(current_colors().TEXT_PRIMARY)
    try:
        for text, _tip, image, callback in toolbar.toolitems:
            if text is not None and callback in toolbar._actions:
                toolbar._actions[callback].setIcon(_tinted_icon(image, colour, toolbar.devicePixelRatioF()))
    except (AttributeError, TypeError, ValueError, OSError) as exc:  # another matplotlib: the icons stay as they are
        logger.debug("matplotlib toolbar icons not refreshed: %s", exc)


def _tinted_icon(image: str, colour: QColor, pixel_ratio: float) -> QIcon:
    """matplotlib's toolbar image ``image`` (its large PNG when there is one) with its black drawn in ``colour``."""
    import matplotlib

    folder = Path(matplotlib.get_data_path()) / "images"
    path = folder / f"{image}_large.png"
    if not path.is_file():
        path = folder / f"{image}.png"
    if not path.is_file():
        raise OSError(f"no toolbar image {image!r} in {folder}")
    pixmap = QPixmap(str(path))
    pixmap.setDevicePixelRatio(pixel_ratio or 1.0)
    mask = pixmap.createMaskFromColor(QColor("black"), Qt.MaskMode.MaskOutColor)
    pixmap.fill(colour)
    pixmap.setMask(mask)
    return QIcon(pixmap)


class ThemeDefault(QObject):
    """Keeps ``combo`` on the entry the theme suggests -- the item whose data is the palette field ``field``
    (e.g. ``VIEW3D_BACKGROUND``) -- until the user picks an entry; from then on that choice stays."""

    def __init__(self, combo: QComboBox, field: str) -> None:
        super().__init__(combo)
        self._combo = combo
        self._field = field
        self.chosen = False
        combo.activated.connect(self._on_activated)
        connect_theme(self._on_theme_changed)
        self.apply()

    def _on_activated(self, _index: int) -> None:
        self.chosen = True  # activated is emitted for the user's picks only, never for setCurrentIndex

    def _on_theme_changed(self, _name: str) -> None:
        if not self.chosen:
            self.apply()

    def apply(self) -> None:
        """Select the theme's entry (a missing one leaves the combo as it is)."""
        index = self._combo.findData(getattr(current_colors(), self._field))
        if index >= 0:
            self._combo.setCurrentIndex(index)
