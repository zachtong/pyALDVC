"""Small widgets shared by the panels: guarded inputs, compact forms, collapsible sections, the console log.

The look follows pyALDIC: labels of a fixed width on the left, inputs of a fixed width next to
them (never stretched across the panel), section headers that fold, and a monospace console at
the bottom of the right sidebar.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QLocale, QObject, Qt, QTime, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .theme import COLORS

LABEL_WIDTH = 150  # px, parameter labels
FIELD_WIDTH = 110  # px, spin boxes
COMBO_WIDTH = 150  # px, drop-down lists
CONSOLE_MIN_HEIGHT = 140
CONSOLE_MAX_HEIGHT = 240
CONSOLE_MAX_LINES = 3000

__all__ = [
    "COMBO_WIDTH",
    "FIELD_WIDTH",
    "LABEL_WIDTH",
    "CollapsibleSection",
    "ConsoleLog",
    "LocaleSafeDoubleSpinBox",
    "WheelGuard",
    "combo",
    "dspin",
    "form_label",
    "guard_wheel",
    "headless",
    "make_form",
    "spin",
]


def headless() -> bool:
    """True under the offscreen platform (tests, self-test): file dialogs would block forever."""
    app = QApplication.instance()
    return app is not None and app.platformName() == "offscreen"


# --------------------------------------------------------------------------- wheel guard
class WheelGuard(QObject):
    """Event filter: the mouse wheel changes a spin box or combo only while it has keyboard focus.

    Scrolling a long parameter panel must never edit the value the pointer happens to rest on.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget) and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


def guard_wheel(root: QWidget) -> WheelGuard:
    """Install a :class:`WheelGuard` on every spin box and combo below ``root`` (and on ``root`` itself)."""
    guard = WheelGuard(root)
    widgets = list(root.findChildren(QAbstractSpinBox)) + list(root.findChildren(QComboBox))
    if isinstance(root, (QAbstractSpinBox, QComboBox)):
        widgets.append(root)
    for w in widgets:
        w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        w.installEventFilter(guard)
    return guard


# --------------------------------------------------------------------------- inputs
def _c_locale_no_grouping() -> QLocale:
    locale = QLocale.c()
    locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator | QLocale.NumberOption.OmitGroupSeparator)
    return locale


class LocaleSafeDoubleSpinBox(QDoubleSpinBox):
    """``QDoubleSpinBox`` that reads a dot decimal on every OS locale and accepts a typed comma as a dot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setLocale(_c_locale_no_grouping())

    def validate(self, text: str, pos: int) -> object:  # noqa: D102 - Qt API
        return super().validate(text.replace(",", "."), pos)

    def valueFromText(self, text: str) -> float:  # noqa: N802, D102 - Qt API
        return super().valueFromText(text.replace(",", "."))

    def fixup(self, text: str) -> str:  # noqa: D102 - Qt API
        return super().fixup(text.replace(",", "."))


def spin(lo: int, hi: int, step: int = 1, width: int = FIELD_WIDTH) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setFixedWidth(width)
    return s


def dspin(lo: float, hi: float, decimals: int, width: int = FIELD_WIDTH) -> QDoubleSpinBox:
    s = LocaleSafeDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(10 ** (-decimals) if decimals else 1.0)
    s.setFixedWidth(width)
    return s


def combo(items: list[str], width: int = COMBO_WIDTH) -> QComboBox:
    c = QComboBox()
    c.addItems(items)
    c.setMinimumWidth(width)
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    return c


def make_form(parent: QWidget | None = None) -> QFormLayout:
    """A form whose fields keep their own width (no stretching) with left-aligned labels of a fixed width."""
    form = QFormLayout(parent) if parent is not None else QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    form.setHorizontalSpacing(10)
    form.setVerticalSpacing(6)
    form.setContentsMargins(4, 2, 4, 6)
    return form


def form_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setFixedWidth(LABEL_WIDTH)
    label.setWordWrap(True)
    return label


# --------------------------------------------------------------------------- collapsible section
class CollapsibleSection(QWidget):
    """A header that folds its content on click (arrow + uppercase title), like pyALDIC's sidebar sections."""

    toggled = Signal(bool)

    def __init__(self, title: str = "", expanded: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._expanded = expanded
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._header = QPushButton()
        self._header.setObjectName("sectionHeader")
        self._header.setFlat(True)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setFixedHeight(28)
        self._header.clicked.connect(self.toggle)
        outer.addWidget(self._header)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 2, 8, 8)
        self._content_layout.setSpacing(6)
        outer.addWidget(self._content)
        self._content.setVisible(expanded)
        self.set_title(title)

    def set_title(self, title: str) -> None:
        self._title = title
        arrow = "▼" if self._expanded else "▶"  # black down / right triangle: present in the CJK fonts too
        self._header.setText(f"{arrow}  {title.upper()}".replace("&", "&&"))  # a bare & would be a mnemonic

    def add_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._content_layout.addLayout(layout)

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._content.setVisible(expanded)
        self.set_title(self._title)
        self.toggled.emit(expanded)

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)


# --------------------------------------------------------------------------- console
class ConsoleLog(QWidget):
    """Timestamped, colour-coded, read-only message log with a clear button (bottom of the right sidebar)."""

    _COLORS = {
        "info": COLORS.TEXT_SECONDARY,
        "debug": COLORS.TEXT_MUTED,
        "warning": COLORS.WARNING,
        "error": COLORS.DANGER,
        "success": COLORS.SUCCESS,
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 0, 0)
        self._title = QLabel()
        self._title.setObjectName("sectionTitle")
        header.addWidget(self._title)
        header.addStretch(1)
        self._btn_clear = QPushButton()
        self._btn_clear.setFlat(True)
        self._btn_clear.setFixedHeight(22)
        self._btn_clear.clicked.connect(self.clear)
        header.addWidget(self._btn_clear)
        layout.addLayout(header)
        self._view = QTextEdit()
        self._view.setObjectName("console")
        self._view.setReadOnly(True)
        self._view.setMinimumHeight(CONSOLE_MIN_HEIGHT)
        self._view.setMaximumHeight(CONSOLE_MAX_HEIGHT)
        self._view.document().setMaximumBlockCount(CONSOLE_MAX_LINES)
        self._view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._view, 1)
        self._count = 0
        self.retranslate_ui()

    def append_log(self, message: str, level: str = "info") -> None:
        color = self._COLORS.get(level, COLORS.TEXT_SECONDARY)
        stamp = QTime.currentTime().toString("HH:mm:ss")
        text = str(message).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        if level == "debug":
            text = "&nbsp;&nbsp;&nbsp;&nbsp;" + text
        self._view.append(f'<span style="color:{color}">{stamp}  {text}</span>')
        self._count += 1
        bar = self._view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear(self) -> None:
        self._view.clear()
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def text(self) -> str:
        return self._view.toPlainText()

    def retranslate_ui(self) -> None:
        self._title.setText(self.tr("Console"))
        self._btn_clear.setText(self.tr("Clear"))
