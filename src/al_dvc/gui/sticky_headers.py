"""Sticky section titles for a scrolling sidebar (pyALDIC's sticky-headers overlay).

While the left column scrolls, the title of every section whose header has passed the top of the
viewport is pinned in a stack at the top, so the outline stays visible. Clicking a pinned title
scrolls back to that section.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from .theme import COLORS

HEADER_HEIGHT = 24

__all__ = ["StickyHeadersOverlay"]


class _StickyTitle(QLabel):
    """Clone of a section title; a click scrolls the original section into view."""

    def __init__(self, text: str, on_click) -> None:
        super().__init__(text)
        self._on_click = on_click
        self.setObjectName("sectionTitle")
        self.setFixedHeight(HEADER_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background: {COLORS.BG_DARKEST}; padding-left: 8px; border: none;")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._on_click()


class StickyHeadersOverlay(QWidget):
    """Pinned stack of section titles at the top of ``scroll_area``'s viewport.

    ``sections`` are widgets that each expose a ``label`` (their title ``QLabel``); the whole
    section widget is what gets scrolled into view on click.
    """

    def __init__(self, scroll_area: QScrollArea, sections: list[QWidget]) -> None:
        super().__init__(scroll_area.viewport())
        self._scroll = scroll_area
        self._sections = list(sections)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"background: {COLORS.BG_DARKEST}; border-bottom: 1px solid {COLORS.BORDER};")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._proxies: list[_StickyTitle] = []
        for section in self._sections:
            proxy = _StickyTitle(section.label.text(), lambda s=section: self.scroll_to(s))
            proxy.hide()
            self._layout.addWidget(proxy)
            self._proxies.append(proxy)
        self._scroll.verticalScrollBar().valueChanged.connect(lambda _v: self.refresh())
        self._scroll.viewport().installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.refresh()
        return super().eventFilter(obj, event)

    def scroll_to(self, section: QWidget) -> None:
        self._scroll.ensureWidgetVisible(section, 0, 0)
        bar = self._scroll.verticalScrollBar()
        top = section.mapTo(self._scroll.widget(), section.rect().topLeft()).y()
        bar.setValue(min(bar.maximum(), max(0, top)))

    def pinned_titles(self) -> list[str]:
        """Titles currently pinned, top to bottom (tests)."""
        return [p.text() for p in self._proxies if p.isVisible()]

    def refresh(self) -> None:
        """Pin every section whose title has scrolled above the stack built so far (stacked style)."""
        vp = self._scroll.viewport()
        width = vp.width()
        if width <= 0:
            return
        self.setFixedWidth(width)
        stack_bottom = 0
        any_visible = False
        for proxy, section in zip(self._proxies, self._sections):
            proxy.setText(section.label.text())
            top = section.label.mapTo(vp, section.label.rect().topLeft()).y()
            if top < stack_bottom:  # stacked: a header that passed the top stays pinned
                proxy.show()
                proxy.setFixedWidth(width)
                stack_bottom += proxy.height()
                any_visible = True
            else:
                proxy.hide()
        if any_visible:
            self.setFixedHeight(stack_bottom)
            self.move(0, 0)
            self.show()
            self.raise_()
        else:
            self.hide()
