"""Region viewer of the texture window: three slices of the reference volume and the drawing tools that
select the *texture analysis region*.

The region is a boolean volume of its own (a :class:`~al_dvc.gui.mask_editor.MaskEditor` that starts
as the whole volume); it has nothing to do with the DVC region of interest of the main window, and
the two never touch. Voxels outside the region are tinted orange (the DVC mask is red) and the
bounding box of the region, which is what the analysis uses, is drawn dashed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QComboBox, QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget

from al_dvc.texture import box_of_mask

from .icons import tool_button
from .mask_editor import MaskEditor, MaskOp
from .theme import COLORS
from .widgets import guard_wheel

PLANE_OF_AXIS = ("xy", "xz", "yz")  # axes[0], axes[1], axes[2]
NORMAL_OF_PLANE = {"xy": "z", "xz": "y", "yz": "x"}
TOOLS = ("rectangle", "ellipse", "polygon", "brush")
MODES = ("replace", "add", "cut")
DEPTHS = ("all", "current", "range")
EDIT_BUTTONS = ("undo", "redo", "fill", "clear")
REGION_COLOR = "#f97316"  # orange: the texture region, never to be confused with the red DVC mask
REGION_TINT = ListedColormap([[0.976, 0.451, 0.086, 1.0]])
OUTSIDE_ALPHA = 0.42
PREVIEW_COLOR = "#ffd166"
CROSSHAIR_COLOR = COLORS.TEXT_MUTED
BRUSH_MIN_MOVE = 0.5  # voxels between recorded stroke points
LEFT, RIGHT = 1, 3
POINT_DECIMALS = 2
DISPLAY_SAMPLE = 2_000_000  # voxels looked at for the grey-level limits

__all__ = ["RegionSettings", "RegionTools", "RegionViewer"]


@dataclass(frozen=True)
class RegionSettings:
    tool: str
    mode: str
    depth: str
    depth_range: tuple[int, int]
    radius: int


def _voxel_point(event) -> tuple[float, float]:
    return (round(float(event.xdata), POINT_DECIMALS), round(float(event.ydata), POINT_DECIMALS))


class RegionTools(QWidget):
    """Shape, mode, depth and brush size of the next gesture, and undo / redo / fill / clear."""

    settings_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tool = "rectangle"
        self._mode = "replace"
        self.tool_buttons = {k: tool_button(k, checkable=True) for k in TOOLS}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for k, b in self.tool_buttons.items():
            group.addButton(b)
            b.clicked.connect(lambda _c, key=k: self.set_tool(key))
        self.tool_buttons[self._tool].setChecked(True)
        self.mode_buttons = {k: tool_button(k, checkable=True) for k in MODES}
        mgroup = QButtonGroup(self)
        mgroup.setExclusive(True)
        for k, b in self.mode_buttons.items():
            mgroup.addButton(b)
            b.clicked.connect(lambda _c, key=k: self.set_mode(key))
        self.mode_buttons[self._mode].setChecked(True)
        self.depth = QComboBox()
        for key in DEPTHS:
            self.depth.addItem(key, key)
        self.depth.setMinimumWidth(110)
        self.depth_from = QSpinBox()
        self.depth_to = QSpinBox()
        for s in (self.depth_from, self.depth_to):
            s.setRange(0, 100000)
            s.setFixedWidth(56)
        self.radius = QSpinBox()
        self.radius.setRange(1, 200)
        self.radius.setValue(4)
        self.radius.setFixedWidth(56)
        self.edit_buttons = {k: tool_button(k) for k in EDIT_BUTTONS}
        self._labels = {k: QLabel() for k in ("tool", "mode", "depth", "to", "radius")}
        self._labels["to"].setText("-")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row1 = QHBoxLayout()
        row1.setSpacing(2)
        row1.addWidget(self._labels["tool"])
        for k in TOOLS:
            row1.addWidget(self.tool_buttons[k])
        row1.addSpacing(8)
        row1.addWidget(self._labels["radius"])
        row1.addWidget(self.radius)
        row1.addStretch(1)
        row2 = QHBoxLayout()
        row2.setSpacing(2)
        row2.addWidget(self._labels["mode"])
        for k in MODES:
            row2.addWidget(self.mode_buttons[k])
        row2.addSpacing(8)
        for k in EDIT_BUTTONS:
            row2.addWidget(self.edit_buttons[k])
        row2.addStretch(1)
        row3 = QHBoxLayout()
        row3.setSpacing(2)
        row3.addWidget(self._labels["depth"])
        row3.addWidget(self.depth)
        row3.addWidget(self.depth_from)
        row3.addWidget(self._labels["to"])
        row3.addWidget(self.depth_to)
        row3.addStretch(1)
        for row in (row1, row2, row3):
            layout.addLayout(row)
        for key in ("tool", "mode", "depth"):
            self._labels[key].setFixedWidth(58)
        guard_wheel(self)
        self.depth.currentIndexChanged.connect(self._on_depth_changed)
        for s in (self.depth_from, self.depth_to, self.radius):
            s.valueChanged.connect(lambda _v: self.settings_changed.emit())
        self._on_depth_changed()
        self.retranslate_ui()

    def settings(self) -> RegionSettings:
        return RegionSettings(
            tool=self._tool,
            mode=self._mode,
            depth=str(self.depth.currentData()),
            depth_range=(int(self.depth_from.value()), int(self.depth_to.value())),
            radius=int(self.radius.value()),
        )

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(f"tool must be one of {TOOLS}, got {tool!r}")
        self._tool = tool
        self.tool_buttons[tool].setChecked(True)
        self.settings_changed.emit()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self._mode = mode
        self.mode_buttons[mode].setChecked(True)
        self.settings_changed.emit()

    def set_depth(self, kind: str, first: int | None = None, last: int | None = None) -> None:
        if kind not in DEPTHS:
            raise ValueError(f"depth must be one of {DEPTHS}, got {kind!r}")
        if first is not None:
            self.depth_from.setValue(int(first))
        if last is not None:
            self.depth_to.setValue(int(last))
        self.depth.setCurrentIndex(DEPTHS.index(kind))

    def set_depth_limit(self, n: int) -> None:
        """The slices a depth range can address: ``0 .. n - 1``."""
        for s in (self.depth_from, self.depth_to):
            s.setRange(0, max(0, n - 1))
        self.depth_to.setValue(max(0, n - 1))

    def _on_depth_changed(self, *_a) -> None:
        ranged = str(self.depth.currentData()) == "range"
        for w in (self.depth_from, self.depth_to, self._labels["to"]):
            w.setEnabled(ranged)
        self.settings_changed.emit()

    def retranslate_ui(self) -> None:
        self._labels["tool"].setText(self.tr("Tool"))
        self._labels["mode"].setText(self.tr("Mode"))
        self._labels["depth"].setText(self.tr("Depth"))
        self._labels["radius"].setText(self.tr("Brush"))
        tips = {
            "rectangle": self.tr("Rectangle: drag on a slice"),
            "ellipse": self.tr("Ellipse: drag on a slice"),
            "polygon": self.tr("Polygon: click the corners, right-click or Enter to close"),
            "brush": self.tr("Brush: paint on a slice"),
        }
        for k, b in self.tool_buttons.items():
            b.setToolTip(tips[k])
        modes = {
            "replace": self.tr("Replace: the shape becomes the region"),
            "add": self.tr("Add the shape to the region"),
            "cut": self.tr("Cut the shape out of the region"),
        }
        for k, b in self.mode_buttons.items():
            b.setToolTip(modes[k])
        edits = {
            "undo": self.tr("Undo"),
            "redo": self.tr("Redo"),
            "fill": self.tr("Whole volume"),
            "clear": self.tr("Empty the region"),
        }
        for k, b in self.edit_buttons.items():
            b.setToolTip(edits[k])
        for i, text in enumerate((self.tr("All slices"), self.tr("This slice"), self.tr("Slice range"))):
            self.depth.setItemText(i, text)
        self.depth.setToolTip(self.tr("Which slices along the normal of the drawn plane a shape reaches"))
        self.radius.setToolTip(self.tr("Brush radius [voxel]"))


class RegionViewer(QWidget):
    """XY, XZ and YZ slices of one volume with a drawable region (orange outside, dashed bounding box)."""

    region_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._vol: np.ndarray | None = None
        self._editor: MaskEditor | None = None
        self._gesture: dict | None = None
        self._vmin, self._vmax = 0.0, 1.0
        self.tools = RegionTools()  # placed by the owner, next to the other controls of the region step
        self.figure = Figure(figsize=(9, 3.6))
        self.figure.set_facecolor(COLORS.BG_CANVAS)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        gs = self.figure.add_gridspec(1, 3, left=0.045, right=0.99, bottom=0.13, top=0.90, wspace=0.28)
        self.axes = [self.figure.add_subplot(gs[0, i]) for i in range(3)]
        self.sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        self._slider_values: dict[str, QLabel] = {}
        sliders = QHBoxLayout()
        sliders.setSpacing(8)
        for axis in ("z", "y", "x"):
            lab = QLabel()
            lab.setObjectName("hint")
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 0)
            val = QLabel("0")
            val.setObjectName("hint")
            val.setFixedWidth(36)
            sliders.addWidget(lab)
            sliders.addWidget(s, 1)
            sliders.addWidget(val)
            self.sliders[axis] = s
            self._slider_labels[axis] = lab
            self._slider_values[axis] = val
            s.valueChanged.connect(lambda _v, a=axis: self._on_slider(a))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(sliders)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.tools.settings_changed.connect(self._on_settings_changed)
        self.tools.edit_buttons["undo"].clicked.connect(self.undo)
        self.tools.edit_buttons["redo"].clicked.connect(self.redo)
        self.tools.edit_buttons["fill"].clicked.connect(lambda: self.set_region(None))
        self.tools.edit_buttons["clear"].clicked.connect(lambda: self.apply(MaskOp("empty")))
        self.retranslate_ui()
        self.redraw()

    # ------------------------------------------------------------------ volume and region
    def set_volume(self, vol: np.ndarray | None) -> None:
        """Show ``vol`` (``(nz, ny, nx)``) with the whole volume as the region; ``None`` empties the viewer."""
        self._cancel_gesture()
        self._vol = None if vol is None else np.asarray(vol)
        if self._vol is None:
            self._editor = None
            for s in self.sliders.values():
                s.setRange(0, 0)
            self.tools.setEnabled(False)
        else:
            nz, ny, nx = self._vol.shape
            self._editor = MaskEditor(self._vol.shape, base=np.ones(self._vol.shape, dtype=bool))
            step = max(1, int(round((self._vol.size / DISPLAY_SAMPLE) ** (1 / 3))))
            sample = self._vol[::step, ::step, ::step]
            lo, hi = np.percentile(sample, [1, 99])
            self._vmin, self._vmax = float(lo), float(hi if hi > lo else lo + 1.0)
            for axis, n in (("z", nz), ("y", ny), ("x", nx)):
                s = self.sliders[axis]
                s.blockSignals(True)
                s.setRange(0, n - 1)
                s.setValue(n // 2)
                s.blockSignals(False)
            self.tools.set_depth_limit(max(nz, ny, nx))
            self.tools.setEnabled(True)
        self._refresh_edit_buttons()
        self.redraw()
        self.region_changed.emit()

    @property
    def shape(self) -> tuple[int, int, int] | None:
        return None if self._vol is None else tuple(int(s) for s in self._vol.shape)

    @property
    def mask(self) -> np.ndarray | None:
        """The region as a boolean ``(nz, ny, nx)`` volume, ``None`` without a volume."""
        return None if self._editor is None else self._editor.mask

    def box(self):
        """Bounding box ``((x0, x1), (y0, y1), (z0, z1))`` of the region, ``None`` when it is empty."""
        m = self.mask
        if m is None or not m.any():
            return None
        return box_of_mask(m)

    def fill_fraction(self) -> float:
        """Share of the bounding box that the region covers (1.0 for a box)."""
        m = self.mask
        box = self.box()
        if m is None or box is None:
            return 0.0
        return float(m.sum()) / float(np.prod([b - a for a, b in box]))

    def set_region(self, mask: np.ndarray | None) -> None:
        """Replace the region by ``mask`` (``None``: the whole volume); the drawing history is dropped."""
        if self._editor is None:
            return
        base = np.ones(self._editor.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self._cancel_gesture()
        self._editor.reset(base)
        self._after_edit()

    def set_box(self, box) -> None:
        """Replace the region by the box ``((x0, x1), (y0, y1), (z0, z1))``."""
        if self._editor is None:
            return
        (x0, x1), (y0, y1), (z0, z1) = box
        m = np.zeros(self._editor.shape, dtype=bool)
        m[z0:z1, y0:y1, x0:x1] = True
        self.set_region(m)

    def apply(self, op: MaskOp) -> None:
        if self._editor is None:
            return
        self._editor.apply(op)
        self._after_edit()

    def undo(self) -> None:
        if self._editor is not None and self._editor.undo():
            self._after_edit()

    def redo(self) -> None:
        if self._editor is not None and self._editor.redo():
            self._after_edit()

    def _after_edit(self) -> None:
        self._refresh_edit_buttons()
        self.redraw()
        self.region_changed.emit()

    def _refresh_edit_buttons(self) -> None:
        ed = self._editor
        self.tools.edit_buttons["undo"].setEnabled(ed is not None and ed.can_undo)
        self.tools.edit_buttons["redo"].setEnabled(ed is not None and ed.can_redo)

    # ------------------------------------------------------------------ slices
    def slice_indices(self) -> tuple[int, int, int]:
        return (int(self.sliders["z"].value()), int(self.sliders["y"].value()), int(self.sliders["x"].value()))

    def set_slices(self, iz: int, iy: int, ix: int) -> None:
        for axis, v in (("z", iz), ("y", iy), ("x", ix)):
            self.sliders[axis].setValue(int(v))

    def _on_slider(self, axis: str) -> None:
        self._cancel_gesture()
        self.redraw()

    def redraw(self) -> None:
        vol = self._vol
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(COLORS.BG_CANVAS)
            ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(COLORS.BORDER)
        for axis, val in self._slider_values.items():
            val.setText(str(self.sliders[axis].value()) if vol is not None else "-")
        if vol is None:
            self.axes[1].text(
                0.5,
                0.5,
                self.tr("Load a reference volume first."),
                ha="center",
                va="center",
                color=COLORS.TEXT_SECONDARY,
                transform=self.axes[1].transAxes,
            )
            for ax in self.axes:
                ax.set_axis_off()
            self.canvas.draw_idle()
            return
        iz, iy, ix = self.slice_indices()
        nz, ny, nx = vol.shape
        m = self.mask
        panes = [
            (self.axes[0], vol[iz], m[iz], (nx, ny), "x", "y", f"XY  z = {iz}", (ix, iy)),
            (self.axes[1], vol[:, iy, :], m[:, iy, :], (nx, nz), "x", "z", f"XZ  y = {iy}", (ix, iz)),
            (self.axes[2], vol[:, :, ix], m[:, :, ix], (ny, nz), "y", "z", f"YZ  x = {ix}", (iy, iz)),
        ]
        box = self.box()
        spans = None
        if box is not None:
            (x0, x1), (y0, y1), (z0, z1) = box
            spans = {"xy": ((x0, x1), (y0, y1)), "xz": ((x0, x1), (z0, z1)), "yz": ((y0, y1), (z0, z1))}
        for plane, (ax, img, m2d, (w, h), xl, yl, title, (ch, cv)) in zip(PLANE_OF_AXIS, panes):
            ax.set_axis_on()
            extent = [-0.5, w - 0.5, -0.5, h - 0.5]
            ax.imshow(img, cmap="gray", origin="lower", vmin=self._vmin, vmax=self._vmax, extent=extent)
            outside = np.ma.masked_where(m2d, np.ones_like(m2d, dtype=np.float32))
            ax.imshow(outside, cmap=REGION_TINT, origin="lower", alpha=OUTSIDE_ALPHA, extent=extent, interpolation="nearest")
            if m2d.any() and not m2d.all():
                ax.contour(m2d.astype(np.float32), levels=[0.5], colors=[REGION_COLOR], linewidths=0.8)
            if spans is not None:
                (h0, h1), (v0, v1) = spans[plane]
                ax.add_patch(
                    Rectangle((h0 - 0.5, v0 - 0.5), h1 - h0, v1 - v0, fill=False, ec=REGION_COLOR, lw=1.6, ls="--", alpha=0.95)
                )
            ax.axvline(ch, color=CROSSHAIR_COLOR, lw=0.6, alpha=0.6)
            ax.axhline(cv, color=CROSSHAIR_COLOR, lw=0.6, alpha=0.6)
            ax.set_title(title, color=COLORS.TEXT_SECONDARY, fontsize=8)
            ax.set_xlabel(xl, color=COLORS.TEXT_SECONDARY, fontsize=7)
            ax.set_ylabel(yl, color=COLORS.TEXT_SECONDARY, fontsize=7)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ gestures
    def _plane_at(self, event) -> str | None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return None
        for ax, plane in zip(self.axes, PLANE_OF_AXIS):
            if event.inaxes is ax:
                return plane
        return None

    def _depth_for(self, plane: str) -> tuple[int, int] | None:
        s = self.tools.settings()
        if s.depth == "all":
            return None
        if s.depth == "current":
            iz, iy, ix = self.slice_indices()
            i = {"z": iz, "y": iy, "x": ix}[NORMAL_OF_PLANE[plane]]
            return (i, i)
        first, last = sorted(s.depth_range)
        return (first, last)

    def _context(self, plane: str) -> dict:
        s = self.tools.settings()
        return {"mode": s.mode, "radius": float(s.radius), "depth": self._depth_for(plane)}

    def _on_settings_changed(self) -> None:
        if self._gesture is not None:
            self._cancel_gesture()
            self.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if self._editor is None:
            return
        plane = self._plane_at(event)
        if plane is None:
            return
        tool = self.tools.settings().tool
        p = _voxel_point(event)
        if tool == "polygon":
            if event.button == RIGHT or getattr(event, "dblclick", False):
                self._finish_polygon(plane)
                return
            if event.button != LEFT:
                return
            if self._gesture is None or self._gesture["plane"] != plane:
                self._cancel_gesture()
                self._gesture = {
                    "plane": plane,
                    "shape": "polygon",
                    "points": [p],
                    "artists": [],
                    "context": self._context(plane),
                }
            else:
                self._gesture["points"].append(p)
            self._update_preview()
            return
        if event.button != LEFT:
            return
        self._cancel_gesture()
        self._gesture = {"plane": plane, "shape": tool, "points": [p], "artists": [], "context": self._context(plane)}
        self._update_preview()

    def _on_motion(self, event) -> None:
        g = self._gesture
        if g is None or g["shape"] == "polygon":
            return
        p = self._gesture_point(event)
        if p is None:
            return
        if g["shape"] == "brush":
            last = g["points"][-1]
            if abs(p[0] - last[0]) + abs(p[1] - last[1]) >= BRUSH_MIN_MOVE:
                g["points"].append(p)
        else:
            g["points"] = [g["points"][0], p]
        self._update_preview()

    def _on_release(self, event) -> None:
        g = self._gesture
        if g is None or g["shape"] == "polygon" or event.button != LEFT:
            return
        p = self._gesture_point(event)
        if p is not None:
            if g["shape"] == "brush":
                g["points"].append(p)
            else:
                g["points"] = [g["points"][0], p]
        self._commit_gesture()

    def _plane_extent(self, plane: str) -> tuple[int, int]:
        nz, ny, nx = self._vol.shape if self._vol is not None else (1, 1, 1)
        return {"xy": (nx, ny), "xz": (nx, nz), "yz": (ny, nz)}[plane]

    def _gesture_point(self, event) -> tuple[float, float] | None:
        """Pointer position on the gesture's plane, clamped to the slice edge when the drag leaves it."""
        g = self._gesture
        if g is None:
            return None
        ax = self.axes[PLANE_OF_AXIS.index(g["plane"])]
        if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
            h, v = _voxel_point(event)
        else:
            if event.x is None or event.y is None:
                return None
            x, y = ax.transData.inverted().transform((float(event.x), float(event.y)))
            h, v = round(float(x), POINT_DECIMALS), round(float(y), POINT_DECIMALS)
        w, hgt = self._plane_extent(g["plane"])
        return (min(max(h, -0.5), w - 0.5), min(max(v, -0.5), hgt - 0.5))

    def _on_key(self, event) -> None:
        if event.key == "escape":
            self._cancel_gesture()
            self.canvas.draw_idle()
        elif event.key == "enter" and self._gesture is not None and self._gesture["shape"] == "polygon":
            self._finish_polygon(self._gesture["plane"])

    def _finish_polygon(self, plane: str) -> None:
        g = self._gesture
        if g is None or g["shape"] != "polygon" or g["plane"] != plane:
            return
        if len(g["points"]) < 3:
            self._cancel_gesture()
            self.canvas.draw_idle()
            return
        self._commit_gesture()

    def _commit_gesture(self) -> None:
        g = self._gesture
        self._gesture = None
        if g is None:
            return
        self._remove_artists(g)
        pts = g["points"]
        if g["shape"] in ("rectangle", "ellipse") and len(pts) < 2:
            self.canvas.draw_idle()
            return
        ctx = g["context"]
        try:
            op = MaskOp(
                shape=g["shape"], plane=g["plane"], points=tuple(pts), depth=ctx["depth"], mode=ctx["mode"], radius=ctx["radius"]
            )
        except ValueError:
            self.canvas.draw_idle()
            return
        self.apply(op)

    def _cancel_gesture(self) -> None:
        if self._gesture is not None:
            self._remove_artists(self._gesture)
            self._gesture = None

    @staticmethod
    def _remove_artists(g: dict) -> None:
        for a in g.get("artists", []):
            try:
                a.remove()
            except Exception:
                pass
        g["artists"] = []

    def _update_preview(self) -> None:
        g = self._gesture
        if g is None:
            return
        self._remove_artists(g)
        ax = self.axes[PLANE_OF_AXIS.index(g["plane"])]
        pts = np.asarray(g["points"], dtype=float)
        if g["shape"] == "rectangle" and len(pts) == 2:
            (h1, v1), (h2, v2) = pts
            g["artists"].append(
                ax.add_patch(
                    Rectangle((min(h1, h2), min(v1, v2)), abs(h2 - h1), abs(v2 - v1), fill=False, ec=PREVIEW_COLOR, lw=1.0)
                )
            )
        elif g["shape"] == "ellipse" and len(pts) == 2:
            (h1, v1), (h2, v2) = pts
            g["artists"].append(
                ax.add_patch(
                    Ellipse(((h1 + h2) / 2, (v1 + v2) / 2), abs(h2 - h1), abs(v2 - v1), fill=False, ec=PREVIEW_COLOR, lw=1.0)
                )
            )
        elif g["shape"] == "polygon":
            closed = np.vstack([pts, pts[:1]]) if len(pts) > 2 else pts
            g["artists"].extend(ax.plot(closed[:, 0], closed[:, 1], "-o", color=PREVIEW_COLOR, lw=1.0, ms=3))
        elif g["shape"] == "brush":
            lw = max(1.0, 2.0 * g["context"]["radius"] * self._points_per_voxel(ax))
            g["artists"].extend(ax.plot(pts[:, 0], pts[:, 1], "-", color=PREVIEW_COLOR, lw=lw, alpha=0.5, solid_capstyle="round"))
        self.canvas.draw_idle()

    def _points_per_voxel(self, ax) -> float:
        try:
            p0 = ax.transData.transform((0.0, 0.0))
            p1 = ax.transData.transform((1.0, 0.0))
            return float(abs(p1[0] - p0[0])) * 72.0 / float(self.figure.dpi)
        except Exception:
            return 1.0

    def retranslate_ui(self) -> None:
        for axis, lab in self._slider_labels.items():
            lab.setText(axis)
        self.tools.retranslate_ui()
        self.redraw()
