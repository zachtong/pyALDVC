"""Vector icons for the tool buttons, rendered from inline SVG in the theme colours.

The icons are simple 20 x 20 line drawings (stroke ``currentColor`` replaced at render time), so
they follow the dark theme and stay crisp at any DPI without shipping bitmap assets.
"""

# ruff: noqa: E501  (inline SVG paths are long by nature)
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QToolButton

from .theme import COLORS

ICON_SIZE = 18
BUTTON_SIZE = 30

_STROKE = 'fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"'

SVG: dict[str, str] = {
    "rectangle": f'<rect x="3" y="4.5" width="14" height="11" rx="1" {_STROKE}/>',
    "ellipse": f'<ellipse cx="10" cy="10" rx="7" ry="5" {_STROKE}/>',
    "polygon": f'<polygon points="4,7 11,3 17,8 14,16 6,15" {_STROKE}/>',
    "brush": f'<path d="M13.5 3.5 L16.5 6.5 L8 15 L5 12 Z" {_STROKE}/><path d="M5 12 C3 13 4 16 2.5 17.5 C6 17.5 8 16 8 15" {_STROKE}/>',
    "box": f'<path d="M4 7 L11 4 L17 7 L17 14 L10 17 L4 14 Z M4 7 L10 10 L17 7 M10 10 L10 17" {_STROKE}/>',
    "replace": '<rect x="3" y="3" width="14" height="14" rx="1.5" fill="{c}" opacity="0.9"/>',
    "add": f'<circle cx="10" cy="10" r="7" {_STROKE}/><path d="M10 6.5 V13.5 M6.5 10 H13.5" {_STROKE}/>',
    "cut": f'<circle cx="10" cy="10" r="7" {_STROKE}/><path d="M6.5 10 H13.5" {_STROKE}/>',
    "undo": f'<path d="M7.5 5.5 L3.5 9 L7.5 12.5" {_STROKE}/><path d="M3.5 9 H12 A4 4 0 0 1 12 17 H8" {_STROKE}/>',
    "redo": f'<path d="M12.5 5.5 L16.5 9 L12.5 12.5" {_STROKE}/><path d="M16.5 9 H8 A4 4 0 0 0 8 17 H12" {_STROKE}/>',
    "invert": f'<circle cx="10" cy="10" r="7" {_STROKE}/><path d="M10 3 A7 7 0 0 1 10 17 Z" fill="{{c}}"/>',
    "fill": f'<rect x="3" y="3" width="14" height="14" rx="1.5" {_STROKE}/><rect x="6" y="6" width="8" height="8" fill="{{c}}"/>',
    "clear": f'<rect x="3" y="3" width="14" height="14" rx="1.5" {_STROKE}/><path d="M6.5 6.5 L13.5 13.5 M13.5 6.5 L6.5 13.5" {_STROKE}/>',
    "remove": f'<path d="M4 6 H16 M8 6 V4 H12 V6 M6 6 L6.8 16 H13.2 L14 6" {_STROKE}/>',
    "save": f'<path d="M4 3 H13.5 L16.5 6 V17 H4 Z" {_STROKE}/><path d="M6.5 3 V8 H12.5 V3 M6.5 17 V12 H13.5 V17" {_STROKE}/>',
    "auto": f'<path d="M3.5 16.5 L12 8" {_STROKE}/><path d="M13.5 3 L14.3 5.2 L16.5 6 L14.3 6.8 L13.5 9 L12.7 6.8 L10.5 6 L12.7 5.2 Z" fill="{{c}}"/><path d="M5 4 L5.5 5.5 L7 6 L5.5 6.5 L5 8 L4.5 6.5 L3 6 L4.5 5.5 Z" fill="{{c}}"/>',
    "refresh": f'<path d="M16 10 A6 6 0 1 1 14.2 5.8" {_STROKE}/><path d="M14.5 2.5 V6.5 H10.5" {_STROKE}/>',
    "camera": f'<path d="M3 7 H7 L8.5 5 H11.5 L13 7 H17 V16 H3 Z" {_STROKE}/><circle cx="10" cy="11.5" r="2.8" {_STROKE}/>',
    "play": f'<polygon points="6,4 16,10 6,16" {_STROKE}/>',
    "stop": f'<rect x="5" y="5" width="10" height="10" rx="1" {_STROKE}/>',
    "pause": f'<rect x="5" y="4" width="3.5" height="12" rx="1" {_STROKE}/><rect x="11.5" y="4" width="3.5" height="12" rx="1" {_STROKE}/>',
    "record": f'<circle cx="10" cy="10" r="6" {_STROKE}/><circle cx="10" cy="10" r="2.5" fill="{{c}}" stroke="none"/>',
    "copy": f'<rect x="7" y="7" width="9.5" height="9.5" rx="1.5" {_STROKE}/><path d="M4 13 V4 H13" {_STROKE}/>',
    "eye": f'<path d="M2.5 10 C5 6 8 4.5 10 4.5 C12 4.5 15 6 17.5 10 C15 14 12 15.5 10 15.5 C8 15.5 5 14 2.5 10 Z" {_STROKE}/><circle cx="10" cy="10" r="2.5" {_STROKE}/>',
}


def svg_source(name: str, color: str) -> str:
    body = SVG[name].replace("{c}", color)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" width="20" height="20">{body}</svg>'


@lru_cache(maxsize=256)
def pixmap(name: str, color: str, size: int = ICON_SIZE) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(svg_source(name, color).encode("utf-8")))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    return pm


def icon(name: str, size: int = ICON_SIZE) -> QIcon:
    """Theme-coloured icon: secondary text colour normally, primary when active or checked, muted when disabled."""
    ic = QIcon()
    ic.addPixmap(pixmap(name, COLORS.TEXT_SECONDARY, size), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(pixmap(name, COLORS.TEXT_PRIMARY, size), QIcon.Mode.Active, QIcon.State.Off)
    ic.addPixmap(pixmap(name, "#ffffff", size), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(pixmap(name, "#ffffff", size), QIcon.Mode.Active, QIcon.State.On)
    ic.addPixmap(pixmap(name, COLORS.TEXT_MUTED, size), QIcon.Mode.Disabled, QIcon.State.Off)
    ic.addPixmap(pixmap(name, COLORS.TEXT_MUTED, size), QIcon.Mode.Disabled, QIcon.State.On)
    return ic


def tool_button(name: str, tooltip: str = "", checkable: bool = False) -> QToolButton:
    """A square icon button of the toolbar style (``QToolButton#tool`` in the theme)."""
    b = QToolButton()
    b.setObjectName("tool")
    b.setIcon(icon(name))
    b.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
    b.setFixedSize(BUTTON_SIZE, BUTTON_SIZE)
    b.setCheckable(checkable)
    b.setToolTip(tooltip)
    b.setAutoRaise(True)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return b


__all__ = ["BUTTON_SIZE", "ICON_SIZE", "SVG", "icon", "pixmap", "svg_source", "tool_button"]
