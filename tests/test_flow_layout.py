"""Control rows that wrap instead of squeezing: the flow layout, and the rows of the viewer, 3-D view and strain sidebar."""

import os
from itertools import combinations

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLabel,
    QWidget,
)

from al_dvc.gui.flow_layout import ELIDED_MIN_WIDTH, ElidedLabel, flow_group, flow_row  # noqa: E402

WINDOW_WIDTHS = (1200, 1440, 1920)  # the smallest window the application allows, its default, a full-HD screen


@pytest.fixture(scope="module")
def qapp():
    from al_dvc.gui.app import create_application

    return create_application([])


def _box(width: int, height: int = 20) -> QWidget:
    widget = QWidget()
    widget.setFixedSize(width, height)
    return widget


def _arrange(row: QWidget, width: int) -> int:
    """Lay the row out at ``width`` (no event loop needed); returns the height it takes."""
    height = row.layout().heightForWidth(width)
    row.layout().setGeometry(QRect(0, 0, width, height))
    return height


def test_one_line_while_there_is_room(qapp):
    a, b, c = _box(50), _box(60), _box(70)
    row = flow_row(a, b, c)
    assert _arrange(row, 300) == 20
    assert [w.geometry().topLeft().toTuple() for w in (a, b, c)] == [(0, 0), (56, 0), (122, 0)]
    hint, minimum = row.layout().sizeHint(), row.layout().minimumSize()
    assert hint.toTuple() == (50 + 60 + 70 + 2 * 6, 20)  # everything on one line
    assert minimum.width() == 70  # as narrow as the widest item


def test_wraps_instead_of_drawing_over_each_other(qapp):
    a, b, c = _box(50), _box(60), _box(70)
    row = flow_row(a, b, c)
    assert _arrange(row, 130) == 20 + 4 + 20
    assert a.geometry().topLeft().toTuple() == (0, 0) and b.geometry().topLeft().toTuple() == (56, 0)
    assert c.geometry().topLeft().toTuple() == (0, 24)  # a new line, from the left
    assert _arrange(row, 70) == 3 * 20 + 2 * 4  # one item per line at the minimum width
    assert [w.geometry().x() for w in (a, b, c)] == [0, 0, 0]


def test_stretch_sends_the_rest_to_the_right_edge(qapp):
    a, b, c = _box(50), _box(60), _box(40)
    row = flow_row(a, None, b, c)
    _arrange(row, 300)
    assert a.geometry().x() == 0 and c.geometry().right() == 299 and b.geometry().x() == 300 - 40 - 6 - 60
    _arrange(row, 110)  # b and c wrap: the new line starts at the left
    assert a.geometry().topLeft().toTuple() == (0, 0) and b.geometry().topLeft().toTuple() == (0, 24)
    assert c.geometry().topLeft().toTuple() == (66, 24)


def test_hidden_items_and_empty_groups_take_no_space(qapp):
    a, b, c = _box(50), _box(60), _box(70)
    label, spin = _box(30), _box(40)
    group = flow_group(label, spin)
    row = flow_row(a, b, group, c)
    row.show()  # size changes of a group reach the row through the event loop of shown widgets
    b.hide()
    label.hide()
    spin.hide()
    QApplication.processEvents()
    _arrange(row, 400)
    assert c.geometry().x() == 56  # neither the hidden box nor the group with nothing to show left a gap
    assert group.geometry().isEmpty()  # the empty group lies over nothing (it would take the clicks)
    spin.show()
    QApplication.processEvents()
    _arrange(row, 400)
    assert group.geometry().width() == 40 and c.geometry().x() == 56 + 40 + 6  # back once it has a widget
    row.close()


def test_items_are_centred_on_their_line(qapp):
    small, tall = _box(40, 20), _box(40, 30)
    row = flow_row(small, tall)
    assert _arrange(row, 200) == 30
    assert small.geometry().y() == 5 and tall.geometry().y() == 0


def test_elided_label_keeps_its_text_and_takes_at_most_a_line(qapp):
    text = "6 x 5 x 4 = 120 nodes (108 in the region), subset 17 x 17 x 17, overlap 53 %"
    label = ElidedLabel(text)
    assert label.text() == text and label.minimumSizeHint().width() <= ELIDED_MIN_WIDTH
    assert label.sizeHint().width() > 200
    row = flow_row(_box(50), label)
    _arrange(row, 150)
    assert label.geometry().width() == 150 and label.geometry().y() > 0  # its own line, shortened
    label.resize(80, label.sizeHint().height())
    assert not label.grab().isNull()  # painted with the ellipsis
    assert row.layout().minimumSize().width() == 50  # the label never forces the row wider


# ---------------------------------------------------------------------------------------------- the real rows
def _smart_min_width(widget: QWidget) -> int:
    """The width below which Qt would squeeze the widget: its own minimum, else its minimum size hint."""
    return widget.minimumWidth() if widget.minimumWidth() > 0 else widget.minimumSizeHint().width()


def _assert_row_fits(row: QWidget) -> None:
    """Nothing in ``row`` is squeezed below its minimum, drawn over a neighbour, or outside the row."""
    layout = row.layout()
    rects = []
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget()
        if widget is None or not widget.isVisible() or widget.width() <= 0:
            continue
        rect = widget.geometry()
        assert rect.left() >= 0 and rect.right() < row.width(), (widget, rect, row.width())
        rects.append((widget, rect))
    assert rects
    for (w1, r1), (w2, r2) in combinations(rects, 2):
        assert not r1.intersects(r2), (w1, r1, w2, r2)
    controls = (QAbstractButton, QAbstractSpinBox, QComboBox, QLabel)
    for widget in row.findChildren(QWidget):
        if isinstance(widget, controls) and not isinstance(widget, ElidedLabel) and widget.isVisible():
            assert widget.width() >= _smart_min_width(widget), (widget, widget.text() if hasattr(widget, "text") else "")


def _settle(window) -> None:
    for _ in range(4):
        QApplication.processEvents()


@pytest.mark.parametrize("width", WINDOW_WIDTHS)
def test_slice_viewer_controls_fit(qapp, width):
    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    window.resize(width, 900)
    window.show()
    _settle(window)
    viewer = window.viewer
    _assert_row_fits(viewer._controls_row)
    if width >= 1440:  # the default window keeps the controls on one line
        assert viewer._controls_row.height() < 2 * viewer.layout_combo.height()
    # the status line: the grid and the configuration share one line and shorten instead of overlapping
    viewer._lattice_label.setText("6 x 5 x 4 = 120 nodes (108 in the region), subset 17 x 17 x 17, overlap 53 %")
    viewer._config_label.setText("(field at reference positions, image deformed)")
    _settle(window)
    lattice, config = viewer._lattice_label.geometry(), viewer._config_label.geometry()
    assert not lattice.intersects(config) and config.right() < viewer.width()
    window.close()


@pytest.mark.parametrize("width", WINDOW_WIDTHS)
@pytest.mark.parametrize("mode", ["surface", "warped"])
def test_view3d_rows_fit(qapp, width, mode):
    from al_dvc.gui.app import MainWindow

    window = MainWindow()
    window.resize(width, 900)
    window.show()
    window.center_tabs.setCurrentWidget(window.view3d)
    view = window.view3d
    view.mode.setCurrentIndex(view.mode.findData(mode))
    _settle(window)
    rows = {view.mode.parentWidget(), view.background.parentWidget(), view.anim_kind.parentWidget()}
    rows.add(view.volume_slices.parentWidget())
    assert len(rows) == 4
    for row in rows:
        _assert_row_fits(row)
    top = view.mode.parentWidget()
    assert view._labels["mode"].y() == top.y()  # the leading label stays level with the first line
    window.close()


def test_strain_fit_window_goes_under_its_label(qapp):
    from al_dvc.gui.app_state import AppState
    from al_dvc.gui.strain_window import StrainWindow

    window = StrainWindow(AppState())
    window.show()
    _settle(window)
    row = window.fit_window_axes[0].parentWidget()
    label = window.labels["halfwidth"]
    assert row.y() > label.y()  # wider than the field column: a line of its own
    assert row.width() >= row.sizeHint().width()
    widgets = [w for w in row.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly) if w.isVisible()]
    assert len(widgets) == 6  # three widths, two 'x', 'Cube'
    for w1, w2 in combinations(widgets, 2):
        assert not w1.geometry().intersects(w2.geometry()), (w1, w2)
    lock = window.fit_window_lock
    assert lock.width() >= lock.sizeHint().width()  # 'Cube' no longer cut off
    window.close()
