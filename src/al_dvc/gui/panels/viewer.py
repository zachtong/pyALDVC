"""Three-plane slice viewer: the current volume, result overlays, and mask drawing."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from al_dvc.export.export_utils import field_array
from al_dvc.export.slice_plots import DISPLACEMENT_LIKE, apply_equal_scale, build_axes, ordered_limits, restore_cells

from ..app_state import AppState
from ..lattice_preview import describe, layer_segments, nearest_node, plan_from_result, plan_lattice, subset_rect
from ..mask_editor import MaskOp
from ..names import field_name
from ..theme import COLORS
from .mask_tools import MaskToolbar

PLANE_OF_AXIS = ("xy", "xz", "yz")  # axes[0], axes[1], axes[2]
NORMAL_OF_PLANE = {"xy": "z", "xz": "y", "yz": "x"}
MASK_TINT = ListedColormap([[1.0, 0.25, 0.25, 1.0]])
MASK_EDGE = "#ff5555"
PREVIEW_COLOR = "#ffd166"
RANGE_COLOR = "#fbbf24"  # the texture analysis range: a dashed amber box on the three slices
LATTICE_COLOR = "#7dd3fc"  # the node grid: thin, pale and translucent, so the image stays readable through it
LATTICE_WIDTH = 0.7
LATTICE_ALPHA = 0.55
HOVER_COLOR = "#f8fafc"  # the subset of the node under the pointer
LATTICE_ON_PLANE = 0.5  # voxels: a lattice layer this close to the slice is drawn at full strength
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
        self.equal_scale = QCheckBox()
        self.equal_scale.setChecked(bool(getattr(state, "slice_equal_scale", False)))
        self.show_mesh = QCheckBox()  # the node grid, like pyALDIC's Show grid
        self.show_mesh.setChecked(bool(getattr(state, "show_mesh", True)))
        self.show_subset = QCheckBox()  # the subset of the crosshair node and of the node under the pointer
        self.show_subset.setChecked(bool(getattr(state, "show_subset_window", False)))
        self._lattice_label = QLabel()
        self._lattice_label.setObjectName("hint")
        self._plan = None  # LatticePlan drawn on the slices, None when hidden or not computable
        self._hover_key: tuple | None = None
        self._hover_artist = None
        self.sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        self.mask_tools = MaskToolbar(state)
        # drawing gesture in progress: plane, shape, points (h, v), preview artists
        self._gesture: dict | None = None
        self._range_box = None  # ((x0, x1), (y0, y1), (z0, z1)) drawn on the slices (texture analysis range)
        self._capture = None  # callback(plane, (h0, h1), (v0, v1)) receiving the next rectangle drags

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout_row = QHBoxLayout()
        layout_row.addWidget(self.show_mesh)
        layout_row.addWidget(self.show_subset)
        layout_row.addSpacing(8)
        layout_row.addWidget(self._lattice_label)
        layout_row.addStretch(1)
        layout_row.addWidget(self.equal_scale)
        layout_row.addSpacing(12)
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
        self.canvas.mpl_connect("axes_leave_event", lambda _e: self._set_hover(None, None))
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.layout_combo.currentIndexChanged.connect(lambda i: self.set_layout(LAYOUTS[i]))
        self.equal_scale.toggled.connect(self._on_equal_scale)
        self.show_mesh.toggled.connect(self._on_show_mesh)
        self.show_subset.toggled.connect(self._on_show_subset)
        self.mask_tools.settings_changed.connect(self._on_draw_settings_changed)
        self.canvas.mpl_connect("resize_event", lambda _e: self._on_canvas_resized())

        self._state.volumes_changed.connect(self._on_volumes_changed)
        self._state.current_frame_changed.connect(lambda _i: self._on_volumes_changed())
        self._state.results_changed.connect(self.redraw)
        self._state.display_changed.connect(self.redraw)
        self._state.mask_changed.connect(self.redraw)
        self._state.params_changed.connect(self.redraw)
        self.retranslate_ui()
        self._on_volumes_changed()

    # ------------------------------------------------------------------ layout
    def _build_axes(self, key: str) -> None:
        """Create the three image axes (XY, XZ, YZ) and the colorbar axes for one of :data:`LAYOUTS`.

        The colorbar lives in its own axes so drawing it never steals space from the images.
        """
        self.axes, self.cax = build_axes(self.figure, key)
        self._cbar = None
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

    def sync_from_state(self) -> None:
        """Layout, scale and lattice toggles from the state (after a session load)."""
        for box, value in (
            (self.equal_scale, self._state.slice_equal_scale),
            (self.show_mesh, getattr(self._state, "show_mesh", True)),
            (self.show_subset, getattr(self._state, "show_subset_window", False)),
        ):
            box.blockSignals(True)
            box.setChecked(bool(value))
            box.blockSignals(False)
        self.set_layout(self._state.slice_layout)
        self.redraw()

    def _on_draw_settings_changed(self) -> None:
        """A tool, mode, depth or target change ends the gesture in progress: it was started under other settings."""
        if self._gesture is not None:
            self._cancel_gesture()
            self.canvas.draw_idle()

    def _on_equal_scale(self, on: bool) -> None:
        self._state.slice_equal_scale = bool(on)
        self.redraw()

    def _on_show_mesh(self, on: bool) -> None:
        self._state.show_mesh = bool(on)
        self.redraw()

    def _on_show_subset(self, on: bool) -> None:
        self._state.show_subset_window = bool(on)
        self.redraw()

    def _on_canvas_resized(self) -> None:
        """Equal scale depends on the pixel size of the panes: recompute the limits after a resize."""
        if self.equal_scale.isChecked() and self._volume is not None:
            self.redraw()

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
        frame = self._state.result_frame()
        if res is None or frame is None or not self._state.show_overlay:
            return None
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
        self._plan = None
        self._hover_key = None
        self._hover_artist = None  # gone with the cleared axes
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
        res = self._state.results
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
                vmin, vmax = ordered_limits(float(np.percentile(finite, 1)), float(np.percentile(finite, 99)))
            else:
                vmin, vmax = ordered_limits(self._state.color_min, self._state.color_max)
            kz = int(np.argmin(np.abs(z0 - iz)))
            ky = int(np.argmin(np.abs(y0 - iy)))
            kx = int(np.argmin(np.abs(x0 - ix)))
            overlays = [
                (self.axes[0], grid[kz], [x0[0] - hx / 2, x0[-1] + hx / 2, y0[0] - hy / 2, y0[-1] + hy / 2]),
                (self.axes[1], grid[:, ky, :], [x0[0] - hx / 2, x0[-1] + hx / 2, z0[0] - hz / 2, z0[-1] + hz / 2]),
                (self.axes[2], grid[:, :, kx], [y0[0] - hy / 2, y0[-1] + hy / 2, z0[0] - hz / 2, z0[-1] + hz / 2]),
            ]
            for ax, img, extent in overlays:
                ax.imshow(
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
            # the bar is drawn from its own mappable: the overlay's alpha would wash the colours out
            bar_mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=self._state.colormap)
            self.cax.set_axes_locator(None)  # every colorbar wraps the locator again: nested wrappers recurse
            self._cbar = self.figure.colorbar(bar_mappable, cax=self.cax)
            # the unit belongs to the result (its values were scaled with the result's voxel size), not to the
            # editable parameters of the next run
            units = getattr(res.dvc_para, "units", None) or getattr(self._state.para, "units", "voxel")
            text = field_name(label) + (f" [{units}]" if label in DISPLACEMENT_LIKE else "")
            self._cbar.set_label(text, color=COLORS.TEXT_SECONDARY, fontsize=8)
            self._cbar.ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=7)
        self._draw_lattice(iz, iy, ix)  # on top of the field, so the grid is legible either way
        self._draw_range_box()
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
        if self.equal_scale.isChecked():
            apply_equal_scale(self.figure, self.axes, [size for _ax, _img, size, *_rest in panes])
        else:
            restore_cells(self.axes)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ analysis range (texture window)
    def set_range_box(self, box) -> None:
        """Show ``((x0, x1), (y0, y1), (z0, z1))`` as a dashed box on the three slices (``None`` hides it)."""
        self._range_box = None if box is None else tuple((int(a), int(b)) for a, b in box)
        self.redraw()

    def capture_box(self, callback) -> None:
        """Route the next rectangle drags to ``callback(plane, (h0, h1), (v0, v1))`` instead of the mask
        (``None`` ends the capture). The drag on a plane sets the two axes of that plane."""
        self._capture = callback
        if callback is None:
            self._cancel_gesture()
            self.canvas.draw_idle()

    @property
    def capturing(self) -> bool:
        return self._capture is not None

    def _draw_range_box(self) -> None:
        box = self._range_box
        if box is None:
            return
        (x0, x1), (y0, y1), (z0, z1) = box
        spans = {"xy": ((x0, x1), (y0, y1)), "xz": ((x0, x1), (z0, z1)), "yz": ((y0, y1), (z0, z1))}
        for plane, ((h0, h1), (v0, v1)) in spans.items():
            ax = self.axes[PLANE_OF_AXIS.index(plane)]
            ax.add_patch(
                Rectangle((h0 - 0.5, v0 - 0.5), h1 - h0, v1 - v0, fill=False, ec=RANGE_COLOR, lw=1.4, ls="--", alpha=0.9)
            )

    def _draw_lattice(self, iz: int, iy: int, ix: int) -> None:
        """The node grid (layer nearest to every slice) and the subset of the crosshair node.

        Before a run the grid is the lattice the parameters would place, once a region of interest
        exists; after a run it is the mesh the run used, with a node valid where its displacement is
        finite. "Show grid" and "Show subset" are independent, like pyALDIC's toggles; the label with
        the node count is always filled in.
        """
        self._lattice_label.setText("")
        self._lattice_label.setToolTip("")
        if self._volume is None:
            return
        res = self._state.results
        frame = self._state.result_frame()
        if res is not None and frame is not None and tuple(res.volume_shape) == tuple(self._volume.shape):
            plan = plan_from_result(res, frame)
            grid_ready = True
        else:
            para = self._state.para
            try:
                plan = plan_lattice(
                    self._volume.shape, para.winsize, para.winstepsize, self._state.effective_voi(), self._state.current_mask()
                )
            except ValueError as exc:
                self._lattice_label.setText(str(exc))
                self._lattice_label.setToolTip(str(exc))
                return
            grid_ready = plan.centre_valid is not None  # no region of interest yet: nothing to judge the grid against
        text = describe(plan)
        self._lattice_label.setText(text)
        self._lattice_label.setToolTip(text)  # the toolbar may not have room for the whole line
        want_grid = self.show_mesh.isChecked() and grid_ready
        want_subset = self.show_subset.isChecked()
        if not (want_grid or want_subset):
            return
        self._plan = plan
        panes = (
            (self.axes[0], "xy", iz, (float(ix), float(iy))),
            (self.axes[1], "xz", iy, (float(ix), float(iz))),
            (self.axes[2], "yz", ix, (float(iy), float(iz))),
        )
        for ax, plane, index, crosshair in panes:
            segments, dist = layer_segments(plan, plane, index)
            on_plane = dist <= LATTICE_ON_PLANE
            if want_grid and len(segments):
                alpha = LATTICE_ALPHA if on_plane else 0.5 * LATTICE_ALPHA
                ax.add_collection(LineCollection(segments, colors=LATTICE_COLOR, linewidths=LATTICE_WIDTH, alpha=alpha))
            if not want_subset:
                continue
            node = nearest_node(plan, plane, crosshair[0], crosshair[1], index)
            if node is None:
                continue
            alpha = 0.95 if on_plane else 0.5
            left, bottom, w, h = subset_rect(plan, plane, node)
            ax.add_patch(Rectangle((left, bottom), w, h, fill=False, ec=PREVIEW_COLOR, lw=1.4, alpha=alpha))
            # the next node along the horizontal axis shows how much neighbouring subsets overlap
            step_h = plan.winstepsize[{"xy": 0, "xz": 0, "yz": 1}[plane]]
            neighbour = nearest_node(plan, plane, node[0] + step_h, node[1], index)
            if neighbour is not None and neighbour != node:
                left, bottom, w, h = subset_rect(plan, plane, neighbour)
                ax.add_patch(Rectangle((left, bottom), w, h, fill=False, ec=PREVIEW_COLOR, lw=0.8, ls="--", alpha=0.5 * alpha))

    def _set_hover(self, plane: str | None, node: tuple[float, float] | None) -> None:
        """Outline the subset of the node under the pointer (one artist, replaced as the pointer moves)."""
        key = (plane, node) if node is not None else None
        if key == self._hover_key:
            return
        self._hover_key = key
        if self._hover_artist is not None:
            try:
                self._hover_artist.remove()
            except Exception:
                pass
            self._hover_artist = None
        if key is not None and self._plan is not None:
            ax = self.axes[PLANE_OF_AXIS.index(plane)]
            left, bottom, w, h = subset_rect(self._plan, plane, node)
            self._hover_artist = ax.add_patch(Rectangle((left, bottom), w, h, fill=False, ec=HOVER_COLOR, lw=1.0))
        self.canvas.draw_idle()

    def _hover_lattice(self, event) -> None:
        if self._plan is None or not self.show_subset.isChecked():
            return
        plane = self._plane_at(event)
        if plane is None:
            self._set_hover(None, None)
            return
        iz, iy, ix = self.slice_indices()
        index = {"xy": iz, "xz": iy, "yz": ix}[plane]
        self._set_hover(plane, nearest_node(self._plan, plane, float(event.xdata), float(event.ydata), index))

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

    def _gesture_context(self, plane: str) -> dict:
        """What a gesture commits with, frozen when it starts.

        The slice, depth range, mode, brush radius and target are read now; a later change of any
        of them cancels the gesture (:meth:`_on_draw_settings_changed`) instead of being applied
        to a shape drawn under other settings.
        """
        s = self.mask_tools.settings()
        return {
            "tool": s.tool,
            "mode": s.mode,
            "radius": float(s.radius),
            "depth": self._depth_for(plane),
            "target": getattr(self._state, "mask_target", "current"),
        }

    def _on_press(self, event) -> None:
        if self._volume is None:
            return
        plane = self._plane_at(event)
        if self._capture is not None:  # the texture window is asking for a box: a plain rectangle drag
            if plane is None or event.button != LEFT:
                return
            self._cancel_gesture()
            self._gesture = {
                "plane": plane,
                "shape": "rectangle",
                "points": [_voxel_point(event)],
                "artists": [],
                "capture": True,
            }
            self._update_preview()
            return
        s = self.mask_tools.settings()
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
                self._gesture = {
                    "plane": plane,
                    "shape": "polygon",
                    "points": [p],
                    "artists": [],
                    "context": self._gesture_context(plane),
                }
            else:
                self._gesture["points"].append(p)
            self._update_preview()
            return
        if event.button != LEFT:
            return
        self._cancel_gesture()
        self._gesture = {"plane": plane, "shape": s.tool, "points": [p], "artists": [], "context": self._gesture_context(plane)}
        self._update_preview()

    def _on_motion(self, event) -> None:
        g = self._gesture
        if g is None:
            self._hover_lattice(event)
            return
        if g["shape"] == "polygon":
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
        """``(width, height)`` in voxels of the slice shown on ``plane``."""
        nz, ny, nx = self._volume.shape if self._volume is not None else (1, 1, 1)
        return {"xy": (nx, ny), "xz": (nx, nz), "yz": (ny, nz)}[plane]

    def _gesture_point(self, event) -> tuple[float, float] | None:
        """Voxel coordinates of the pointer for the gesture in progress, even outside its axes.

        A drag that leaves the image is clamped to the slice edge, so a rectangle can be made to
        hug the border without aiming at the last voxel; an event on another plane's axes is
        still mapped onto the gesture's plane through its pixel position.
        """
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
        if g.get("capture"):
            self.canvas.draw_idle()
            if len(pts) == 2 and self._capture is not None:
                (h1, v1), (h2, v2) = pts
                self._capture(g["plane"], (min(h1, h2), max(h1, h2)), (min(v1, v2), max(v1, v2)))
            return
        ctx = g.get("context") or self._gesture_context(g["plane"])  # the settings the gesture was started with
        if g["shape"] in ("rectangle", "ellipse") and len(pts) < 2:
            self.canvas.draw_idle()
            return
        try:
            op = MaskOp(
                shape=g["shape"],
                plane=g["plane"],
                points=tuple(pts),
                depth=ctx["depth"],
                mode=ctx["mode"],
                radius=float(ctx["radius"]),
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
        self.equal_scale.setText(self.tr("Same scale"))
        self.show_mesh.setText(self.tr("Show grid"))
        self.show_mesh.setToolTip(
            self.tr(
                "The node grid on the slices: before a run the lattice the parameters would place inside the region "
                "of interest (layer nearest to each slice), after a run the mesh the run used."
            )
        )
        self.show_subset.setText(self.tr("Show subset"))
        self.show_subset.setToolTip(
            self.tr(
                "Outline the subset of the node at the crosshair (dashed: its neighbour, showing the overlap) and of "
                "the node under the pointer, so the subset size can be judged against the texture."
            )
        )
        self.equal_scale.setToolTip(
            self.tr("Draw the three planes with the same voxels-per-pixel scale (smaller slices get padding)")
        )
        for i, text in enumerate((self.tr("Row"), self.tr("Column"), self.tr("2 x 2"))):
            self.layout_combo.setItemText(i, text)
        self.layout_combo.setToolTip(self.tr("Arrangement of the XY / XZ / YZ slices"))
        self.mask_tools.retranslate_ui()
