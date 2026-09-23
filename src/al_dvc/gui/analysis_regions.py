"""Regions of the Statistics tab: typed or drawn on the slices, listed with their node counts, outlined.

* :func:`draw_overlays` outlines the regions -- their cut through each slice -- and the line of the Line tab on
  the three planes of a :class:`~al_dvc.gui.field_canvas.FieldSliceCanvas`.
* :class:`RegionDrawer` turns mouse gestures on those planes into a rectangle, an ellipse or a polygon (extruded
  through the whole node grid along the plane's normal) or into the two end points of a line. Esc cancels.
* :class:`RegionsPanel` lists the regions of ``AppState.regions`` and edits the selected one.

Regions live in the reference configuration, in voxels ``[x, y, z]`` (``al_dvc.analysis.regions``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from al_dvc.analysis.regions import AXES, DEFAULT_COLORS, PLANES, Region, next_region_id

from .i18n import tr
from .widgets import combo, dspin, guard_wheel

PLANE_OF_AXES = ("xy", "xz", "yz")  # the canvas' axes 0, 1, 2: (horizontal, vertical) = (x, y), (x, z), (y, z)
DEPTH_INDEX = {"xy": 0, "xz": 1, "yz": 2}  # which of the canvas indices (iz, iy, ix) is the plane's depth
DRAW_KINDS = ("rect", "ellipse", "polygon", "line")
PARAMETRIC = ("box", "sphere", "cylinder", "slab")
OUTLINE_SAMPLES = 160  # samples along the longer side of a plane when outlining a region
MIN_DRAG = 1.0  # voxels: a smaller rectangle or ellipse is a click, not a region
COORD_LIMIT = 1e6
EDIT_DELAY_MS = 300


# ------------------------------------------------------------------ geometry
def grid_bounds(result) -> tuple[np.ndarray, np.ndarray]:
    """The first and last node of the node grid, ``[x, y, z]`` (voxels)."""
    m = result.dvc_mesh
    lo = np.array([m.x0[0], m.y0[0], m.z0[0]], dtype=np.float64)
    hi = np.array([m.x0[-1], m.y0[-1], m.z0[-1]], dtype=np.float64)
    return lo, hi


def _r(v) -> float:
    return float(np.round(float(v), 1))


def default_region(shape: str, rid: int, bounds: tuple[np.ndarray, np.ndarray]) -> Region:
    """A new ``shape`` region in the middle of ``bounds`` (the node grid), big enough to hold nodes."""
    lo, hi = (np.asarray(b, dtype=np.float64) for b in bounds)
    c = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1.0)
    if shape == "box":
        params = {"lo": [_r(v) for v in c - span / 4], "hi": [_r(v) for v in c + span / 4]}
    elif shape == "sphere":
        params = {"centre": [_r(v) for v in c], "radius": _r(max(span.min() / 4, 1.0))}
    elif shape == "cylinder":
        params = {"axis": "z", "centre": [_r(v) for v in c], "radius": _r(max(span[:2].min() / 4, 1.0))}
        params.update({"lo": _r(lo[2]), "hi": _r(hi[2])})
    elif shape == "slab":
        params = {"axis": "z", "lo": _r(c[2] - span[2] / 6), "hi": _r(c[2] + span[2] / 6)}
    else:
        raise ValueError(f"no default for a {shape!r} region")
    return Region(id=rid, name=f"R{rid}", shape=shape, params=params, color=DEFAULT_COLORS[(rid - 1) % len(DEFAULT_COLORS)])


def prism_region(plane: str, outline: str, points, rid: int, bounds: tuple[np.ndarray, np.ndarray]) -> Region:
    """A drawn outline on ``plane`` extruded through the whole node grid along the plane's normal."""
    lo, hi = bounds
    n = PLANES[plane][2]
    pts = [[_r(a), _r(b)] for a, b in np.asarray(points, dtype=np.float64).reshape(-1, 2)]
    params = {"plane": plane, "outline": outline, "points": pts, "lo": _r(lo[n]), "hi": _r(hi[n])}
    return Region(id=rid, name=f"R{rid}", shape="prism", params=params, color=DEFAULT_COLORS[(rid - 1) % len(DEFAULT_COLORS)])


def plane_point(plane: str, a: float, b: float, depth: float) -> np.ndarray:
    """The voxel position ``[x, y, z]`` of the in-plane coordinates ``(a, b)`` on the slice at ``depth``."""
    ia, ib, n = PLANES[plane]
    p = np.zeros(3)
    p[ia], p[ib], p[n] = a, b, depth
    return p


def _pane_size(plane: str, shape) -> tuple[int, int]:
    nz, ny, nx = (int(s) for s in shape)
    return {"xy": (nx, ny), "xz": (nx, nz), "yz": (ny, nz)}[plane]


def region_cut(region: Region, plane: str, depth: float, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(a, b, inside)``: the region on the slice ``plane`` at ``depth``, sampled on a grid that has one row and
    column outside the pane on every side, where ``inside`` is always False -- so the outline closes at the
    edge of the pane when the region goes past it."""
    w, h = _pane_size(plane, shape)
    step = max(0.5, max(w, h) / OUTLINE_SAMPLES)
    a = np.arange(-step - 0.5, w - 0.5 + 2 * step, step)
    b = np.arange(-step - 0.5, h - 0.5 + 2 * step, step)
    A, B = np.meshgrid(a, b)
    ia, ib, n = PLANES[plane]
    X = np.zeros((A.size, 3))
    X[:, ia], X[:, ib], X[:, n] = A.ravel(), B.ravel(), depth
    inside = region.contains(X).reshape(A.shape)
    inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = False
    return a, b, inside


def draw_overlays(axes, indices, shape, regions, selected: int | None = None, line=None, line_color: str = "#ffffff") -> None:
    """Outline every visible region on the three planes (the selected one thicker) and the line ``(p0, p1)``."""
    for k, ax in enumerate(axes[:3]):
        plane = PLANE_OF_AXES[k]
        depth = float(indices[DEPTH_INDEX[plane]])
        for region in regions:
            if not region.visible:
                continue
            a, b, inside = region_cut(region, plane, depth, shape)
            if not inside.any():
                continue
            lw = 2.2 if region.id == selected else 1.2
            ax.contour(a, b, inside.astype(np.float32), levels=[0.5], colors=[region.color], linewidths=lw)
            rows, cols = np.nonzero(inside)
            top = rows.max()
            left = cols[rows == top].min()
            ax.text(a[left], b[top], f" {region.name}", color=region.color, fontsize=7, va="bottom", ha="left", clip_on=True)
        if line is not None:
            ia, ib, _n = PLANES[plane]
            p0, p1 = (np.asarray(p, dtype=np.float64) for p in line)
            ax.plot([p0[ia], p1[ia]], [p0[ib], p1[ib]], "--", color=line_color, lw=1.2)
            ax.plot([p0[ia]], [p0[ib]], "o", color=line_color, ms=4)
            ax.plot([p1[ia]], [p1[ib]], "s", color=line_color, ms=4)


# ------------------------------------------------------------------ drawing on the slices
class RegionDrawer(QObject):
    """Mouse gestures on a :class:`FieldSliceCanvas` into a drawn outline or a line.

    ``rect`` / ``ellipse``: press and drag on one plane. ``polygon``: click the vertices on one plane, then
    double-click, right-click or press Enter to close it (three vertices at least). ``line``: click the two end
    points, each on the slice shown (they may be on different planes). Esc cancels. :attr:`finished` carries
    ``{"kind", "plane", "points"}`` for an outline, ``{"kind": "line", "p0", "p1"}`` for a line.
    """

    finished = Signal(object)
    cancelled = Signal()

    def __init__(self, canvas, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self.kind: str | None = None
        self._plane: str | None = None
        self._points: list[tuple[float, float]] = []
        self._line: list[np.ndarray] = []
        self._drag: tuple[float, float] | None = None
        self._artists: list = []
        mpl = canvas.canvas
        mpl.mpl_connect("button_press_event", self._on_press)
        mpl.mpl_connect("motion_notify_event", self._on_motion)
        mpl.mpl_connect("button_release_event", self._on_release)
        mpl.mpl_connect("key_press_event", self._on_key)

    @property
    def active(self) -> bool:
        return self.kind is not None

    def start(self, kind: str) -> None:
        if kind not in DRAW_KINDS:
            raise ValueError(f"unknown drawing {kind!r}; use one of {DRAW_KINDS}")
        self._reset()
        self.kind = kind
        mpl = self._canvas.canvas
        mpl.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        mpl.setFocus()
        mpl.setCursor(Qt.CursorShape.CrossCursor)

    def cancel(self) -> None:
        was = self.active
        self._reset()
        if was:
            self.cancelled.emit()

    def _reset(self) -> None:
        self.kind = None
        self._plane = None
        self._points = []
        self._line = []
        self._drag = None
        self._clear_artists()
        self._canvas.canvas.unsetCursor()

    def _clear_artists(self) -> None:
        for artist in self._artists:
            if artist.axes is not None:  # a redraw of the canvas may have removed it already
                artist.remove()
        self._artists = []
        self._canvas.canvas.draw_idle()

    def _axes_of(self, plane: str):
        return self._canvas.axes[PLANE_OF_AXES.index(plane)]

    def _plane_of(self, event) -> str | None:
        for k, ax in enumerate(self._canvas.axes[:3]):
            if event.inaxes is ax:
                return PLANE_OF_AXES[k]
        return None

    # ---- matplotlib events -> the gesture methods below
    def _on_press(self, event) -> None:
        plane = self._plane_of(event)
        if self.active and plane is not None and event.xdata is not None:
            self.press(plane, float(event.xdata), float(event.ydata), button=int(event.button or 1), double=bool(event.dblclick))

    def _on_motion(self, event) -> None:
        plane = self._plane_of(event)
        if self.active and plane is not None and event.xdata is not None:
            self.move(plane, float(event.xdata), float(event.ydata))

    def _on_release(self, event) -> None:
        plane = self._plane_of(event)
        if self.active and self._drag is not None:
            if plane == self._plane and event.xdata is not None:
                self.release(float(event.xdata), float(event.ydata))
            else:
                self.release(*self._drag)  # released outside the plane: where the pointer last was on it

    def _on_key(self, event) -> None:
        if self.active and event.key:
            self.key(str(event.key))

    # ---- gestures (the tests drive these directly)
    def press(self, plane: str, a: float, b: float, button: int = 1, double: bool = False) -> None:
        if not self.active:
            return
        if button == 3:  # right button: close a polygon, else give up
            if self.kind == "polygon" and len(self._points) >= 3:
                self._finish_polygon()
            else:
                self.cancel()
            return
        if self.kind in ("rect", "ellipse"):
            self._plane = plane
            self._points = [(a, b)]
            self._drag = (a, b)
            self._rubber_band(a, b)
        elif self.kind == "polygon":
            if self._plane is None:
                self._plane = plane
            elif plane != self._plane:
                return  # a polygon lies on one plane
            if double and len(self._points) >= 3:
                self._finish_polygon()
                return
            if not self._points or np.hypot(a - self._points[-1][0], b - self._points[-1][1]) > 1e-9:
                self._points.append((a, b))
            self._draw_polyline()
        elif self.kind == "line":
            depth = float(self._canvas.indices[DEPTH_INDEX[plane]])
            self._line.append(plane_point(plane, a, b, depth))
            ax = self._axes_of(plane)
            self._artists += ax.plot([a], [b], "o", color="#ffffff", ms=5)
            self._canvas.canvas.draw_idle()
            if len(self._line) == 2:
                p0, p1 = self._line
                self._reset()
                if np.linalg.norm(p1 - p0) > 0:
                    self.finished.emit({"kind": "line", "p0": p0.tolist(), "p1": p1.tolist()})
                else:
                    self.cancelled.emit()

    def move(self, plane: str, a: float, b: float) -> None:
        if self.kind in ("rect", "ellipse") and self._drag is not None and plane == self._plane:
            self._drag = (a, b)
            self._rubber_band(a, b)

    def release(self, a: float, b: float) -> None:
        if self.kind not in ("rect", "ellipse") or self._drag is None:
            return
        (a0, b0) = self._points[0]
        self._drag = None
        if abs(a - a0) < MIN_DRAG or abs(b - b0) < MIN_DRAG:  # a click: wait for a real drag
            self._points = []
            self._clear_artists()
            return
        out = {"kind": self.kind, "plane": self._plane, "points": [[a0, b0], [a, b]]}
        self._reset()
        self.finished.emit(out)

    def key(self, key: str) -> None:
        if key == "escape":
            self.cancel()
        elif key in ("enter", "return") and self.kind == "polygon" and len(self._points) >= 3:
            self._finish_polygon()

    def _finish_polygon(self) -> None:
        out = {"kind": "polygon", "plane": self._plane, "points": [list(p) for p in self._points]}
        self._reset()
        self.finished.emit(out)

    # ---- feedback while drawing
    def _rubber_band(self, a: float, b: float) -> None:
        from matplotlib.patches import Ellipse, Rectangle

        self._clear_artists()
        (a0, b0) = self._points[0]
        ax = self._axes_of(self._plane)
        if self.kind == "rect":
            patch = Rectangle((min(a0, a), min(b0, b)), abs(a - a0), abs(b - b0), fill=False, ec="#ffffff", lw=1.2, ls="--")
        else:
            patch = Ellipse((0.5 * (a0 + a), 0.5 * (b0 + b)), abs(a - a0), abs(b - b0), fill=False, ec="#ffffff", lw=1.2, ls="--")
        self._artists.append(ax.add_patch(patch))
        self._canvas.canvas.draw_idle()

    def _draw_polyline(self) -> None:
        self._clear_artists()
        ax = self._axes_of(self._plane)
        pts = np.asarray(self._points)
        self._artists += ax.plot(pts[:, 0], pts[:, 1], "o-", color="#ffffff", lw=1.2, ms=3)
        self._canvas.canvas.draw_idle()


# ------------------------------------------------------------------ the list and the editor
class RegionsPanel(QWidget):
    """The regions (``AppState.regions``): add, draw, rename, show or hide, edit, delete."""

    draw_requested = Signal(str)  # "rect" | "ellipse" | "polygon"
    selection_changed = Signal(object)  # the selected region's id, or None

    def __init__(self, state, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._result = None
        self._updating = False
        self._editor_for: tuple[int, str] | None = None  # (id, shape) the editor was built for
        self._empty_hint = False  # the hint names regions without nodes
        self._edits: dict[str, object] = {}
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(EDIT_DELAY_MS)
        self._apply_timer.timeout.connect(self.apply_editor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.table = QTableWidget(0, 3)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(90)
        self.table.setMaximumHeight(150)
        layout.addWidget(self.table)

        grid = QGridLayout()
        grid.setSpacing(4)
        self.add_buttons: dict[str, QPushButton] = {}
        for i, shape in enumerate(PARAMETRIC):
            b = QPushButton()
            b.clicked.connect(lambda _c=False, s=shape: self.add(s))
            self.add_buttons[shape] = b
            grid.addWidget(b, 0, i)
        self.draw_buttons: dict[str, QPushButton] = {}
        for i, kind in enumerate(("rect", "ellipse", "polygon")):
            b = QPushButton()
            b.clicked.connect(lambda _c=False, k=kind: self.draw_requested.emit(k))
            self.draw_buttons[kind] = b
            grid.addWidget(b, 1, i)
        self._btn_delete = QPushButton()
        self._btn_delete.clicked.connect(self.delete_selected)
        grid.addWidget(self._btn_delete, 1, 3)
        layout.addLayout(grid)

        self._editor = QWidget()
        self._form = QFormLayout(self._editor)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setSpacing(4)
        layout.addWidget(self._editor)
        self.hint = QLabel()
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.itemChanged.connect(self._on_item_changed)
        self._state.regions_changed.connect(self.refresh)
        self.retranslate_ui()
        self.refresh()

    # ------------------------------------------------------------------ data
    def set_result(self, result) -> None:
        self._result = result
        self.refresh()

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._result is not None:
            return grid_bounds(self._result)
        shape = self._state.volume_shape() if self._state.volumes else None
        hi = np.array([shape[2] - 1, shape[1] - 1, shape[0] - 1], dtype=np.float64) if shape else np.full(3, 100.0)
        return np.zeros(3), hi

    def regions(self) -> list[Region]:
        return list(self._state.regions)

    def region(self, rid: int | None) -> Region | None:
        return next((r for r in self._state.regions if r.id == rid), None)

    def selected_id(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return None if item is None else int(item.data(Qt.ItemDataRole.UserRole))

    def select(self, rid: int | None) -> None:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and int(item.data(Qt.ItemDataRole.UserRole)) == rid:
                self.table.selectRow(r)
                return
        self.table.clearSelection()

    def _replace(self, region: Region) -> None:
        self._state.set_regions([region if r.id == region.id else r for r in self._state.regions])

    # ------------------------------------------------------------------ actions
    def add(self, shape: str) -> Region:
        region = default_region(shape, next_region_id(self._state.regions), self._bounds())
        self._state.set_regions([*self._state.regions, region])
        self.select(region.id)
        return region

    def add_drawn(self, plane: str, outline: str, points) -> Region | None:
        """A drawn outline as a new region; ``None`` (and the reason in the hint) when it is not a valid one."""
        try:
            region = prism_region(plane, outline, points, next_region_id(self._state.regions), self._bounds())
        except ValueError as exc:
            self.hint.setText(tr("Not a region: {error}").format(error=exc))
            return None
        self._state.set_regions([*self._state.regions, region])
        self.select(region.id)
        return region

    def delete_selected(self) -> None:
        rid = self.selected_id()
        if rid is None:
            return
        self._state.set_regions([r for r in self._state.regions if r.id != rid])

    def cycle_color(self) -> None:
        region = self.region(self.selected_id())
        if region is None:
            return
        i = DEFAULT_COLORS.index(region.color) if region.color in DEFAULT_COLORS else -1
        self._replace(replace(region, color=DEFAULT_COLORS[(i + 1) % len(DEFAULT_COLORS)]))

    # ------------------------------------------------------------------ table
    def refresh(self) -> None:
        keep = self.selected_id()
        regions = self._state.regions
        counts = None
        if self._result is not None:
            counts = [int(r.node_mask(self._result).sum()) for r in regions]
        self._updating = True
        try:
            self.table.setRowCount(len(regions))
            for row, r in enumerate(regions):
                name = QTableWidgetItem(r.name)
                name.setData(Qt.ItemDataRole.UserRole, r.id)
                name.setFlags(name.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
                name.setCheckState(Qt.CheckState.Checked if r.visible else Qt.CheckState.Unchecked)
                name.setForeground(_brush(r.color))
                name.setToolTip(tr("Tick to outline it on the slices; double-click to rename"))
                self.table.setItem(row, 0, name)
                shape = QTableWidgetItem(self._shape_label(r))
                shape.setFlags(shape.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, 1, shape)
                n = QTableWidgetItem("-" if counts is None else f"{counts[row]:,}")
                n.setFlags(n.flags() & ~Qt.ItemFlag.ItemIsEditable)
                n.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, 2, n)
            self.table.resizeColumnsToContents()
        finally:
            self._updating = False
        if keep is not None and self.region(keep) is not None:
            self.select(keep)
        else:
            self._on_selection()
        empty = [r.name for r, c in zip(regions, counts or []) if c == 0]
        if empty:
            self.hint.setText(tr("No node inside: {names}").format(names=", ".join(empty)))
        elif self._empty_hint:
            self.hint.setText("")
        self._empty_hint = bool(empty)
        self._btn_delete.setEnabled(self.selected_id() is not None)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != 0:
            return
        region = self.region(int(item.data(Qt.ItemDataRole.UserRole)))
        if region is None:
            return
        name = item.text().strip() or region.name
        visible = item.checkState() == Qt.CheckState.Checked
        if name != region.name or visible != region.visible:
            self._replace(replace(region, name=name, visible=visible))

    def _on_selection(self) -> None:
        rid = self.selected_id()
        self._btn_delete.setEnabled(rid is not None)
        self._build_editor(self.region(rid))
        self.selection_changed.emit(rid)

    # ------------------------------------------------------------------ editor
    def _clear_form(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._edits = {}

    def _spin(self, key: str, value: float) -> None:
        w = dspin(-COORD_LIMIT, COORD_LIMIT, 1, width=70)
        w.setValue(float(value))
        w.valueChanged.connect(lambda _v: self._apply_timer.start())
        self._edits[key] = w

    def _vec_row(self, label: str, key: str, values) -> None:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(2)
        for i, v in enumerate(values):
            self._spin(f"{key}{i}", v)
            h.addWidget(self._edits[f"{key}{i}"])
        h.addStretch(1)
        self._form.addRow(label, row)

    def _num_row(self, label: str, key: str, value: float) -> None:
        self._spin(key, value)
        self._form.addRow(label, self._edits[key])

    def _axis_row(self, value: str) -> None:
        w = combo(list(AXES), width=70)
        w.setCurrentText(value)
        w.currentIndexChanged.connect(lambda _i: self._apply_timer.start())
        self._edits["axis"] = w
        self._form.addRow(tr("Axis"), w)

    def _build_editor(self, region: Region | None) -> None:
        key = None if region is None else (region.id, region.shape)
        if key == self._editor_for and region is not None:
            self._sync_editor(region)
            return
        self._apply_timer.stop()
        self._editor_for = key
        self._clear_form()
        if region is None:
            self._editor.setVisible(False)
            return
        self._editor.setVisible(True)
        p = region.params
        name = QLineEdit(region.name)
        name.editingFinished.connect(self.apply_editor)
        self._edits["name"] = name
        color = QPushButton()
        color.setFixedWidth(28)
        color.setStyleSheet(f"background: {region.color};")
        color.setToolTip(tr("Next colour"))
        color.clicked.connect(self.cycle_color)
        self._edits["color"] = color
        head = QWidget()
        hh = QHBoxLayout(head)
        hh.setContentsMargins(0, 0, 0, 0)
        hh.addWidget(name, 1)
        hh.addWidget(color)
        self._form.addRow(tr("Name"), head)
        if region.shape == "box":
            self._vec_row(tr("From x, y, z"), "lo", p["lo"])
            self._vec_row(tr("To x, y, z"), "hi", p["hi"])
        elif region.shape == "sphere":
            self._vec_row(tr("Centre x, y, z"), "centre", p["centre"])
            self._num_row(tr("Radius"), "radius", p["radius"])
        elif region.shape == "cylinder":
            self._axis_row(p["axis"])
            self._vec_row(tr("Centre x, y, z"), "centre", p["centre"])
            self._num_row(tr("Radius"), "radius", p["radius"])
            self._num_row(tr("From (along the axis)"), "lo", p["lo"])
            self._num_row(tr("To (along the axis)"), "hi", p["hi"])
        elif region.shape == "slab":
            self._axis_row(p["axis"])
            self._num_row(tr("From"), "lo", p["lo"])
            self._num_row(tr("To"), "hi", p["hi"])
        else:
            normal = "xyz"[PLANES[p["plane"]][2]]
            self._num_row(tr("From {axis}").format(axis=normal), "lo", p["lo"])
            self._num_row(tr("To {axis}").format(axis=normal), "hi", p["hi"])
        self._form.addRow(QLabel(tr("Voxels, reference volume; bounds included.")))
        guard_wheel(self._editor)

    def _sync_editor(self, region: Region) -> None:
        """The editor shows ``region`` (a change made elsewhere) without being rebuilt under the user's cursor."""
        for w in self._edits.values():
            w.blockSignals(True)
        try:
            p = region.params
            if not self._edits["name"].hasFocus():
                self._edits["name"].setText(region.name)
            self._edits["color"].setStyleSheet(f"background: {region.color};")
            for key, w in self._edits.items():
                if key in ("name", "color"):
                    continue
                if key == "axis":
                    w.setCurrentText(p["axis"])
                elif key[-1].isdigit() and key[:-1] in p:
                    w.setValue(float(p[key[:-1]][int(key[-1])]))
                elif key in p:
                    w.setValue(float(p[key]))
        finally:
            for w in self._edits.values():
                w.blockSignals(False)

    def apply_editor(self) -> bool:
        """The editor's values into the selected region; False (and the reason in the hint) when invalid."""
        self._apply_timer.stop()
        region = self.region(self.selected_id())
        if region is None or not self._edits:
            return False
        params = dict(region.params)
        for key in ("lo", "hi", "centre"):
            if f"{key}0" in self._edits:
                params[key] = [float(self._edits[f"{key}{i}"].value()) for i in range(3)]
        for key in ("radius", "lo", "hi"):
            if key in self._edits:
                params[key] = float(self._edits[key].value())
        if "axis" in self._edits:
            params["axis"] = self._edits["axis"].currentText()
        name = self._edits["name"].text().strip() or region.name
        try:
            new = replace(region, name=name, params=params)
        except ValueError as exc:
            self.hint.setText(tr("Not applied: {error}").format(error=exc))
            return False
        self.hint.setText("")
        if new != region:
            self._replace(new)
        return True

    # ------------------------------------------------------------------ texts
    @staticmethod
    def _shape_label(region: Region) -> str:
        if region.shape == "prism":
            outline = {"rect": tr("Rectangle"), "ellipse": tr("Ellipse"), "polygon": tr("Polygon")}
            return f"{outline.get(region.params.get('outline'), '')} {region.params.get('plane', '').upper()}"
        return {"box": tr("Box"), "sphere": tr("Sphere"), "cylinder": tr("Cylinder"), "slab": tr("Slab")}.get(
            region.shape, region.shape
        )

    def retranslate_ui(self) -> None:
        self.table.setHorizontalHeaderLabels([tr("Region"), tr("Shape"), tr("Nodes")])
        names = {"box": tr("Box"), "sphere": tr("Sphere"), "cylinder": tr("Cylinder"), "slab": tr("Slab")}
        tips = {
            "box": tr("Add a box in the middle of the node grid; edit its corners below"),
            "sphere": tr("Add a sphere in the middle of the node grid"),
            "cylinder": tr("Add a cylinder along z through the node grid"),
            "slab": tr("Add a slab: every node between two planes normal to an axis"),
        }
        for shape, b in self.add_buttons.items():
            b.setText("+ " + names[shape])
            b.setToolTip(tips[shape])
        draw = {"rect": tr("Rectangle"), "ellipse": tr("Ellipse"), "polygon": tr("Polygon")}
        draw_tips = {
            "rect": tr("Drag on a slice; the region goes through the whole node grid along the slice's normal"),
            "ellipse": tr("Drag on a slice; the region goes through the whole node grid along the slice's normal"),
            "polygon": tr("Click the corners on a slice, then double-click or right-click to close; Esc cancels"),
        }
        for kind, b in self.draw_buttons.items():
            b.setText(draw[kind])
            b.setToolTip(draw_tips[kind])
        self._btn_delete.setText(tr("Delete"))
        self.refresh()


def _brush(color: str):
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(color))
