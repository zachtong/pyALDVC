"""Control rows that wrap instead of squeezing: a flow layout, groups that stay together, an eliding label.

A row of controls in a QHBoxLayout cannot become narrower than the sum of its controls; below that
width Qt draws them over each other (clipped check-box texts, labels across combo boxes). The flow
layout starts a new line instead and keeps a single line while there is room. A stretch marks where
a line's spare width goes, so a group can sit at the right edge as in a box layout; a wrapped line
starts at the left. Adapted from Qt's Flow Layout example (BSD licence), with the stretch, vertical
centring on each line, and items that never get less than their size hint (except an
:class:`ElidedLabel`, which takes at most the width of a line and ends in "...").
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLayout, QLayoutItem, QSizePolicy, QSpacerItem, QWidget

H_SPACING = 6  # px between the items of a line
V_SPACING = 4  # px between lines
GROUP_SPACING = 4  # px between the widgets of a group (a label and its control)
ELIDED_MIN_WIDTH = 24  # px: an eliding label keeps room for a few characters and the "..."


class FlowLayout(QLayout):
    """Items left to right, wrapping onto new lines; a line's spare width goes to its stretch, if any."""

    def __init__(self, parent: QWidget | None = None, h_spacing: int = H_SPACING, v_spacing: int = V_SPACING):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    # -- QLayout interface ---------------------------------------------------------------------------
    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 (Qt override)
        self._items.append(item)

    def addStretch(self) -> None:  # noqa: N802 (named like QBoxLayout.addStretch)
        """Spare width of the line goes here (no effect at the start of a wrapped line)."""
        self.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 (Qt override)
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 (Qt override)
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 (Qt override)
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt override)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt override)
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 (Qt override)
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        """Everything on one line."""
        sizes = [size for _item, size in self._visible(None)]
        m = self.contentsMargins()
        width = sum(s.width() for s in sizes) + self._h_spacing * max(0, len(sizes) - 1)
        height = max((s.height() for s in sizes), default=0)
        return QSize(width + m.left() + m.right(), height + m.top() + m.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802 (Qt override)
        """As narrow as the widest item: every other item can go on a line of its own."""
        m = self.contentsMargins()
        width = height = 0
        for item, size in self._visible(None):
            width = max(width, max(item.minimumSize().width(), 0) if _elides(item) else size.width())
            height = max(height, size.height())
        return QSize(width + m.left() + m.right(), height + m.top() + m.bottom())

    # -- layout --------------------------------------------------------------------------------------
    def _visible(self, line_width: int | None) -> list[tuple[QLayoutItem, QSize]]:
        """(item, size) of the items that take space; a stretch comes with an empty size."""
        out = []
        for item in self._items:
            if item.spacerItem() is not None:
                out.append((item, QSize(0, 0)))
                continue
            if item.isEmpty():  # a hidden widget
                continue
            hint = item.sizeHint()
            if hint.width() <= 0:  # a group whose widgets are all hidden
                continue
            width = hint.width()
            if line_width is not None and _elides(item):  # an eliding label: at most one line
                width = max(min(width, line_width), item.minimumSize().width())
            out.append((item, QSize(width, hint.height())))
        if line_width is None:
            out = [(item, size) for item, size in out if item.spacerItem() is None]
        return out

    def _arrange(self, rect: QRect, apply: bool) -> int:
        """Break the items into lines within ``rect``; place them when ``apply``. Returns the height used."""
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        lines: list[list[tuple[QLayoutItem, QSize]]] = [[]]
        used = 0
        for item, size in self._visible(max(area.width(), 1)):
            line = lines[-1]
            if item.spacerItem() is not None:
                if line:  # a stretch that opens a line would push nothing
                    line.append((item, size))
                continue
            has_items = any(it.spacerItem() is None for it, _s in line)
            needed = used + (self._h_spacing if has_items else 0) + size.width()
            if has_items and needed > area.width():
                lines.append([])
                line = lines[-1]
                needed = size.width()
            line.append((item, size))
            used = needed
        if apply:  # a group with nothing to show keeps no area (it would lie over its neighbours and take clicks)
            for item in self._items:
                if item.spacerItem() is None and not item.isEmpty() and item.sizeHint().width() <= 0:
                    item.setGeometry(QRect(area.topLeft(), QSize(0, 0)))
        y = area.y()
        placed = 0
        for line in lines:
            items = [(it, s) for it, s in line if it.spacerItem() is None]
            if not items:
                continue
            if placed:
                y += self._v_spacing
            height = max(s.height() for _it, s in items)
            if apply:
                self._place_line(line, items, area, y, height)
            y += height
            placed += 1
        return y - rect.y() + m.bottom()

    def _place_line(self, line, items, area: QRect, y: int, height: int) -> None:
        content = sum(s.width() for _it, s in items) + self._h_spacing * (len(items) - 1)
        spare = max(0, area.width() - content)
        x = area.x()
        first = True
        for item, size in line:
            if item.spacerItem() is not None:
                x += spare  # the first stretch of the line takes the spare width
                spare = 0
                continue
            if not first:
                x += self._h_spacing
            item.setGeometry(QRect(QPoint(x, y + (height - size.height()) // 2), size))
            x += size.width()
            first = False


def _elides(item: QLayoutItem) -> bool:
    return isinstance(item.widget(), ElidedLabel)


def flow_group(*widgets: QWidget, spacing: int = GROUP_SPACING) -> QWidget:
    """Widgets that stay on one line together (a label and its control) inside a :class:`FlowLayout`."""
    group = QWidget()
    row = QHBoxLayout(group)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for widget in widgets:
        row.addWidget(widget)
    return group


def flow_row(*items: QWidget | None, h_spacing: int = H_SPACING, v_spacing: int = V_SPACING) -> QWidget:
    """A widget holding a :class:`FlowLayout` of ``items``; ``None`` places a stretch."""
    row = QWidget()
    layout = FlowLayout(row, h_spacing, v_spacing)
    for item in items:
        if item is None:
            layout.addStretch()
        else:
            layout.addWidget(item)
    return row


class ElidedLabel(QLabel):
    """A one-line label that ends in "..." when it is narrower than its text; ``text()`` keeps the whole text.

    Its size hint is the whole text, its minimum a few characters, so a layout short of room
    shortens the label instead of squeezing its neighbours. Show the whole text in the tooltip.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), ELIDED_MIN_WIDTH), hint.height())

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        rect = self.contentsRect()
        text = self.text()
        elided = self.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, rect.width())
        if elided == text:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        self.style().drawItemText(
            painter, rect, int(self.alignment()), self.palette(), self.isEnabled(), elided, self.foregroundRole()
        )
        painter.end()
