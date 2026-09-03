"""Three-plane slice viewer: the current volume with an optional result field overlay."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from al_dvc.export.export_utils import field_array

from ..app_state import AppState
from ..theme import COLORS


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
        self.axes = [self.figure.add_subplot(1, 3, k + 1) for k in range(3)]
        self._cbar = None
        self.sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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

        self._state.volumes_changed.connect(self._on_volumes_changed)
        self._state.current_frame_changed.connect(lambda _i: self._on_volumes_changed())
        self._state.results_changed.connect(self.redraw)
        self._state.display_changed.connect(self.redraw)
        self.retranslate_ui()
        self._on_volumes_changed()

    # ------------------------------------------------------------------ data
    def _on_volumes_changed(self) -> None:
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
        self.redraw()

    def _on_slider(self, axis: str, value: int) -> None:
        self._state.slice_index[axis] = int(value)
        self.redraw()

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
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        if self._volume is None:
            self._empty.setVisible(True)
            self.canvas.draw_idle()
            return
        self._empty.setVisible(False)
        vol = self._volume
        nz, ny, nx = vol.shape
        iz = int(np.clip(self._state.slice_index.get("z") or nz // 2, 0, nz - 1))
        iy = int(np.clip(self._state.slice_index.get("y") or ny // 2, 0, ny - 1))
        ix = int(np.clip(self._state.slice_index.get("x") or nx // 2, 0, nx - 1))
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
            self._cbar = self.figure.colorbar(mappable, ax=self.axes, fraction=0.025, pad=0.02)
            self._cbar.set_label(label, color=COLORS.TEXT_SECONDARY, fontsize=8)
            self._cbar.ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=7)
        # cursor lines showing the other two slice positions
        self.axes[0].axhline(iy, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[0].axvline(ix, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[1].axhline(iz, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[1].axvline(ix, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[2].axhline(iz, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.axes[2].axvline(iy, color=COLORS.ACCENT, lw=0.5, alpha=0.6)
        self.canvas.draw_idle()

    def retranslate_ui(self) -> None:
        self._empty.setText(self.tr("No volume loaded. Use 'Add volumes...' to start."))
