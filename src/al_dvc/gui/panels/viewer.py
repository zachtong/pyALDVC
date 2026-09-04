"""Three-plane slice viewer: the current volume, result overlays, and mask drawing."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from al_dvc.export.export_utils import field_array

from ..app_state import AppState
from ..mask_editor import MaskOp
from ..theme import COLORS
from .mask_tools import MaskToolbar

PLANE_OF_AXIS = ("xy", "xz", "yz")  # axes[0], axes[1], axes[2]
NORMAL_OF_PLANE = {"xy": "z", "xz": "y", "yz": "x"}
MASK_TINT = ListedColormap([[1.0, 0.25, 0.25, 1.0]])
MASK_EDGE = "#ff5555"
PREVIEW_COLOR = "#ffd166"
BRUSH_MIN_MOVE = 0.5  # voxels between recorded stroke points
LEFT, RIGHT = 1, 3
LAYOUTS = ("row", "column", "grid")  # three slices side by side, stacked, or XY / XZ left and YZ top-right


POINT_DECIMALS = 2  # voxel coordinates of a gesture are kept to 0.01 voxel: the display round trip adds pixel noise


def _voxel_point(event) -> tuple[float, float]:
    """Data coordinates of a mouse event on the slice, rounded to :data:`POINT_DECIMALS`.

    The pixel-to-voxel transform returns 19.9999... for a pointer on voxel 20; rounding makes
    gestures (and the mask operations stored in a session) independent of the canvas size.
    """
    return (round(float(event.xdata), POINT_DECIMALS), round(float(event.ydata), POINT_DECIMALS))


class SliceViewer(QWidget):
    """XY, XZ and YZ slices through the current frame, with sliders for the three positions."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._volume: np.ndarray | None = None
        self._volume_index: int | None = None
        self._vmin = 0.0
        self._vmax = 1.0
        self.figure = Figure(figsize=(9, 3.4), facecolor=COLORS.BG_CANVAS)
        self.canvas = FigureCanvas(self.figure)
        self.axes: list = []
        self.cax = None  # one colorbar axes of fixed position: the image axes never shrink on redraw
        self._cbar = None
        self._build_axes(getattr(state, "slice_layout", LAYOUTS[0]))
        self.layout_combo = QComboBox()
        for key in LAYOUTS:
            self.layout_combo.addItem(key, key)
        self.layout_combo.setCurrentIndex(max(0, LAYOUTS.index(getattr(state, "slice_layout", LAYOUTS[0]))))
        self._layout_label = QLabel()
        self.sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        self.mask_tools = MaskToolbar(state)
        # drawing gesture in progress: plane, shape, points (h, v), preview artists
        self._gesture: dict | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mask_tools)
        layout_row = QHBoxLayout()
        layout_row.addStretch(1)
        layout_row.addWidget(self._layout_label)
        layout_row.addWidget(self.layout_combo)
        layout.addLayout(layout_row)
        layout.addWidget(self.canvas, stretch=1)
        rows = QHBoxLayout()
        for axis in ("z", "y", "x"):
            col = QVBoxLayout()
            lab = QLabel()
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 0)
            s.valueChanged.connect(lambda v, a=axis: self._on_slider(a, v))
            col.addWidget(lab)
            col.addWidget(s)
            rows.addLayout(col)
            self.sliders[axis] = s
            self._slider_labels[axis] = lab
        layout.addLayout(rows)
        self._empty = QLabel()
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setObjectName("hint")
        layout.addWidget(self._empty)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.layout_combo.currentIndexChanged.connect(lambda i: self.set_layout(LAYOUTS[i]))

        self._state.volumes_changed.connect(self._on_volumes_changed)
        self._state.current_frame_changed.connect(lambda _i: self._on_volumes_changed())
        self._state.results_changed.connect(self.redraw)
        self._state.display_changed.connect(self.redraw)
        self._state.mask_changed.connect(self.redraw)
        self.retranslate_ui()
        self._on_volumes_changed()

    # ------------------------------------------------------------------ layout
    def _build_axes(self, key: str) -> None:
        """Create the three image axes and the colorbar axes for one of :data:`LAYOUTS`.

        ``self.axes`` always holds (XY, XZ, YZ) in that order whatever the arrangement; the colorbar
        lives in its own axes so drawing it never steals space from the images (the old
        ``figure.colorbar(ax=...)`` shrank them on every redraw).
        """
        if key not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {key!r}")
        fig = self.figure
        fig.clear()
        self._cbar = None
        if key == "row":
            gs = fig.add_gridspec(1, 3, left=0.05, right=0.90, bottom=0.14, top=0.90, wspace=0.32)
            cells = [gs[0, 0], gs[0, 1], gs[0, 2]]
            cax_rect = (0.925, 0.16, 0.014, 0.68)
        elif key == "column":
            gs = fig.add_gridspec(3, 1, left=0.14, right=0.86, bottom=0.05, top=0.96, hspace=0.42)
            cells = [gs[0, 0], gs[1, 0], gs[2, 0]]
            cax_rect = (0.90, 0.22, 0.02, 0.56)
        else:  # grid: XY top-left, XZ bottom-left, YZ top-right, colorbar in the free cell
            gs = fig.add_gridspec(2, 2, left=0.07, right=0.94, bottom=0.08, top=0.94, wspace=0.28, hspace=0.34)
            cells = [gs[0, 0], gs[1, 0], gs[0, 1]]
            cax_rect = (0.60, 0.10, 0.02, 0.32)
        self.axes = [fig.add_subplot(cell) for cell in cells]
        self.cax = fig.add_axes(cax_rect)
        self.cax.set_visible(False)
        self._layout = key

    def set_layout(self, key: str) -> None:
        """Arrange the three slices as a row, a column or a 2 x 2 grid (remembered in the state)."""
        if key == getattr(self, "_layout", None):
            return
        self._cancel_gesture()
        self._build_axes(key)
        if self.layout_combo.currentData() != key:
            self.layout_combo.blockSignals(True)
            self.layout_combo.setCurrentIndex(LAYOUTS.index(key))
            self.layout_combo.blockSignals(False)
        if getattr(self._state, "slice_layout", None) != key:
            self._state.slice_layout = key
        self.redraw()

    @property
    def layout_key(self) -> str:
        return self._layout

    # ------------------------------------------------------------------ data
    def _on_volumes_changed(self) -> None:
        self._cancel_gesture()
        idx = self._state.current_frame
        if not self._state.volumes or idx >= len(self._state.volumes):
            self._volume = None
            self._volume_index = None
            self.redraw()
            return
        try:
            vol = self._state.volume_array(idx)
        except Exception as exc:
            self._state.log(f"cannot load frame {idx}: {exc}", "error")
            self._volume = None
            self.redraw()
            return
        self._volume = np.asarray(vol)
        self._volume_index = idx
        finite = self._volume[np.isfinite(self._volume)] if self._volume.dtype.kind == "f" else self._volume
        sample = finite.ravel()[:: max(1, finite.size // 200000)] if finite.size else np.zeros(1)
        self._vmin, self._vmax = (
            (float(np.percentile(sample, 0.5)), float(np.percentile(sample, 99.5))) if sample.size else (0.0, 1.0)
        )
        nz, ny, nx = self._volume.shape
        for axis, n in (("z", nz), ("y", ny), ("x", nx)):
            s = self.sliders[axis]
            s.blockSignals(True)
            s.setRange(0, n - 1)
            cur = self._state.slice_index.get(axis)
            s.setValue(min(cur, n - 1) if cur is not None else n // 2)
            s.blockSignals(False)
            self._state.slice_index[axis] = s.value()
        self.mask_tools.depth_from.setMaximum(max(nz, ny, nx) - 1)
        self.mask_tools.depth_to.setMaximum(max(nz, ny, nx) - 1)
        self.redraw()

    def _on_slider(self, axis: str, value: int) -> None:
        self._state.set_slice(axis, int(value))  # emits display_changed -> redraw here and in the 3-D view

    def slice_indices(self) -> tuple[int, int, int]:
        """``(iz, iy, ix)`` of the displayed slices (clipped to the volume)."""
        nz, ny, nx = self._volume.shape if self._volume is not None else (1, 1, 1)
        si = self._state.slice_index
        iz = int(np.clip(si.get("z") if si.get("z") is not None else nz // 2, 0, nz - 1))
        iy = int(np.clip(si.get("y") if si.get("y") is not None else ny // 2, 0, ny - 1))
        ix = int(np.clip(si.get("x") if si.get("x") is not None else nx // 2, 0, nx - 1))
        return iz, iy, ix

    # ------------------------------------------------------------------ drawing
    def _field_grid(self):
        """``(grid (nz, ny, nx) over nodes, mesh, label)`` of the displayed field or ``None``."""
        res = self._state.results
        if res is None or not self._state.show_overlay or not res.result_disp:
            return None
        frame = min(self._state.display_frame, len(res.result_disp) - 1)
        try:
            values = field_array(res, frame, self._state.display_field)
        except ValueError:
            return None
        return res.dvc_mesh.to_grid(values), res.dvc_mesh, self._state.display_field

    def redraw(self) -> None:
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(COLORS.BG_CANVAS)
            ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(COLORS.BORDER)
        self.cax.clear()
        self.cax.set_visible(False)
        self._cbar = None
        if self._volume is None:
            self._empty.setVisible(True)
            self.canvas.draw_idle()
            return
        self._empty.setVisible(False)
        vol = self._volume
        nz, ny, nx = vol.shape
        iz, iy, ix = self.slice_indices()
        self._slider_labels["z"].setText(f"z = {iz}")
        self._slider_labels["y"].setText(f"y = {iy}")
        self._slider_labels["x"].setText(f"x = {ix}")
        panes = [
            (self.axes[0], vol[iz], (nx, ny), "x", "y", f"XY  z = {iz}"),
            (self.axes[1], vol[:, iy, :], (nx, nz), "x", "z", f"XZ  y = {iy}"),
            (self.axes[2], vol[:, :, ix], (ny, nz), "y", "z", f"YZ  x = {ix}"),
        ]
        overlay = self._field_grid()
        mappable = None
        for ax, img, (w, h), xl, yl, title in panes:
            ax.imshow(img, cmap="gray", origin="lower", vmin=self._vmin, vmax=self._vmax, extent=[-0.5, w - 0.5, -0.5, h - 0.5])
            ax.set_title(title, color=COLORS.TEXT_SECONDARY, fontsize=8)
            ax.set_xlabel(xl, color=COLORS.TEXT_SECONDARY, fontsize=7)
            ax.set_ylabel(yl, color=COLORS.TEXT_SECONDARY, fontsize=7)
        self._draw_mask(iz, iy, ix)
        if overlay is not None:
            grid, mesh, label = overlay
            x0, y0, z0 = mesh.x0, mesh.y0, mesh.z0
            hx, hy, hz = mesh.spacing
            finite = grid[np.isfinite(grid)]
            if self._state.color_auto and finite.size:
                vmin, vmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
                if vmin == vmax:
                    vmax = vmin + 1e-12
            else:
                vmin, vmax = self._state.color_min, self._state.color_max
            kz = int(np.argmin(np.abs(z0 - iz)))
            ky = int(np.argmin(np.abs(y0 - iy)))
            kx = int(np.argmin(np.abs(x0 - ix)))
            overlays = [
                (self.axes[0], grid[kz], [x0[0] - hx / 2, x0[-1] + hx / 2, y0[0] - hy / 2, y0[-1] + hy / 2]),
                (self.axes[1], grid[:, ky, :], [x0[0] - hx / 2, x0[-1] + hx / 2, z0[0] - hz / 2, z0[-1] + hz / 2]),
                (self.axes[2], grid[:, :, kx], [y0[0] - hy / 2, y0[-1] + hy / 2, z0[0] - hz / 2, z0[-1] + hz / 2]),
            ]
            for ax, img, extent in overlays:
                mappable = ax.imshow(
                    np.ma.masked_invalid(img),
                    cmap=self._state.colormap,
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    alpha=self._state.overlay_alpha,
                    extent=extent,
                    interpolation="nearest",
                )
            self.cax.set_visible(True)
            self._cbar = self.figure.colorbar(mappable, cax=self.cax)
            self._cbar.set_label(label, color=COLORS.TEXT_SECONDARY, fontsize=8)
            self._cbar.ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=7)
        # cursor lines showing the other two slice positions
        self.axes[0].axhline(iy, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[0].axvline(ix, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[1].axhline(iz, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[1].axvline(ix, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[2].axhline(iz, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[2].axvline(iy, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        # every pane shows its whole slice: the last imshow (the field overlay) must not zoom the view to itself
        for ax, _img, (w, h), *_rest in panes:
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(-0.5, h - 0.5)
        self.canvas.draw_idle()

    def _draw_mask(self, iz: int, iy: int, ix: int) -> None:
        """Tint the excluded (False) region of the mask on the three slices."""
        if not self._state.show_mask:
            return
        mask = self._state.current_mask()
        if mask is None or self._volume is None or mask.shape != self._volume.shape:
            return
        alpha = float(self._state.mask_alpha)
        for ax, m2d in ((self.axes[0], mask[iz]), (self.axes[1], mask[:, iy, :]), (self.axes[2], mask[:, :, ix])):
            h, w = m2d.shape
            excluded = np.ma.masked_where(m2d, np.ones_like(m2d, dtype=np.float32))
            ax.imshow(
                excluded,
                cmap=MASK_TINT,
                origin="lower",
                alpha=alpha,
                extent=[-0.5, w - 0.5, -0.5, h - 0.5],
                interpolation="nearest",
            )
            if m2d.any() and not m2d.all():
                ax.contour(m2d.astype(np.float32), levels=[0.5], colors=[MASK_EDGE], linewidths=0.6)

    # ------------------------------------------------------------------ mouse gestures
    def _plane_at(self, event) -> str | None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return None
        for ax, plane in zip(self.axes, PLANE_OF_AXIS):
            if event.inaxes is ax:
                return plane
        return None

    def _depth_for(self, plane: str) -> tuple[int, int] | None:
        s = self.mask_tools.settings()
        if s.depth == "all":
            return None
        if s.depth == "current":
            iz, iy, ix = self.slice_indices()
            i = {"z": iz, "y": iy, "x": ix}[NORMAL_OF_PLANE[plane]]
            return (i, i)
        first, last = sorted(s.depth_range)
        return (first, last)

    def _on_press(self, event) -> None:
        if self._volume is None:
            return
        s = self.mask_tools.settings()
        plane = self._plane_at(event)
        if s.tool == "none" or plane is None:
            return
        try:
            self._state.ensure_mask_editor()
        except ValueError as exc:
            self._state.log(str(exc), "warning")
            return
        p = _voxel_point(event)
        if s.tool == "polygon":
            if event.button == RIGHT or getattr(event, "dblclick", False):
                self._finish_polygon(plane)
                return
            if event.button != LEFT:
                return
            if self._gesture is None or self._gesture["plane"] != plane:
                self._cancel_gesture()
                self._gesture = {"plane": plane, "shape": "polygon", "points": [p], "artists": []}
            else:
                self._gesture["points"].append(p)
            self._update_preview()
            return
        if event.button != LEFT:
            return
        self._cancel_gesture()
        self._gesture = {"plane": plane, "shape": s.tool, "points": [p], "artists": []}
        self._update_preview()

    def _on_motion(self, event) -> None:
        g = self._gesture
        if g is None or g["shape"] == "polygon":
            return
        if self._plane_at(event) != g["plane"]:
            return
        p = _voxel_point(event)
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
        if self._plane_at(event) == g["plane"]:
            p = _voxel_point(event)
            if g["shape"] == "brush":
                g["points"].append(p)
            else:
                g["points"] = [g["points"][0], p]
        self._commit_gesture()

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
        s = self.mask_tools.settings()
        pts = g["points"]
        if g["shape"] in ("rectangle", "ellipse") and len(pts) < 2:
            self.canvas.draw_idle()
            return
        try:
            op = MaskOp(
                shape=g["shape"],
                plane=g["plane"],
                points=tuple(pts),
                depth=self._depth_for(g["plane"]),
                mode=s.mode,
                radius=float(s.radius),
            )
        except ValueError as exc:
            self._state.log(f"ignored shape: {exc}", "warning")
            self.canvas.draw_idle()
            return
        self._state.apply_mask_op(op)  # emits mask_changed -> redraw

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
            lw = max(1.0, 2.0 * self.mask_tools.settings().radius * self._points_per_voxel(ax))
            g["artists"].extend(ax.plot(pts[:, 0], pts[:, 1], "-", color=PREVIEW_COLOR, lw=lw, alpha=0.5, solid_capstyle="round"))
        self.canvas.draw_idle()

    def _points_per_voxel(self, ax) -> float:
        """Display points per data unit along x (line widths are in points)."""
        try:
            p0 = ax.transData.transform((0.0, 0.0))
            p1 = ax.transData.transform((1.0, 0.0))
            return float(abs(p1[0] - p0[0])) * 72.0 / float(self.figure.dpi)
        except Exception:
            return 1.0

    def retranslate_ui(self) -> None:
        self._empty.setText(self.tr("No volume loaded. Use 'Add volumes...' to start."))
        self._layout_label.setText(self.tr("Layout"))
        for i, text in enumerate((self.tr("Row"), self.tr("Column"), self.tr("2 x 2"))):
            self.layout_combo.setItemText(i, text)
        self.layout_combo.setToolTip(self.tr("Arrangement of the XY / XZ / YZ slices"))
        self.mask_tools.retranslate_ui()
