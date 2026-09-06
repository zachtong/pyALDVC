"""A self-contained three-plane field canvas (matplotlib in Qt) with its own display state.

Used by windows that must not touch the main window's display settings (the strain
post-processing window, following pyALDIC's private ``VizController``). The drawing itself is
:func:`al_dvc.export.slice_plots.draw_field_planes`, shared with the image export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from al_dvc.export.slice_plots import LAYOUTS, PlaneStyle, build_axes, draw_field_planes, ordered_limits

from .theme import COLORS

__all__ = ["FieldSliceCanvas", "LAYOUTS"]


class FieldSliceCanvas(QWidget):
    """XY / XZ / YZ planes of one node-grid field with sliders for the three positions."""

    slices_changed = Signal(int, int, int)  # (iz, iy, ix)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result = None
        self._background: np.ndarray | None = None
        self._shape: tuple[int, int, int] = (1, 1, 1)
        self._frame = 0
        self._field = "disp_magnitude"
        self._cmap = "turbo"
        self._clim: tuple[float, float] | None = None
        self._alpha = 0.85
        self._layout = "grid"
        self._equal_scale = False
        self._indices: dict[str, int | None] = {"z": None, "y": None, "x": None}
        self.last_clim: tuple[float, float] | None = None
        self._style = PlaneStyle(
            background=COLORS.BG_CANVAS, text=COLORS.TEXT_SECONDARY, border=COLORS.BORDER, cursor=COLORS.ACCENT
        )

        self.figure = Figure(figsize=(9, 3.4), facecolor=COLORS.BG_CANVAS)
        self.canvas = FigureCanvas(self.figure)
        self.axes, self.cax = build_axes(self.figure, self._layout)
        self.sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("resize_event", lambda _e: self._equal_scale and self.redraw())
        rows = QHBoxLayout()
        for axis in ("z", "y", "x"):
            col = QVBoxLayout()
            lab = QLabel(f"{axis} = -")
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
        self._empty.setObjectName("hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty)
        self.redraw()

    # ------------------------------------------------------------------ data
    def set_data(self, result, background: np.ndarray | None = None) -> None:
        """The result to draw and an optional volume shown under the field."""
        self._result = result
        self._background = None if background is None else np.asarray(background)
        if result is not None:
            self._shape = tuple(int(s) for s in result.volume_shape)
        nz, ny, nx = self._shape
        for axis, n in (("z", nz), ("y", ny), ("x", nx)):
            s = self.sliders[axis]
            s.blockSignals(True)
            s.setRange(0, max(0, n - 1))
            cur = self._indices.get(axis)
            s.setValue(min(cur, n - 1) if cur is not None else n // 2)
            s.blockSignals(False)
            self._indices[axis] = s.value()
        self.redraw()

    def set_view(
        self,
        frame: int | None = None,
        field: str | None = None,
        cmap: str | None = None,
        clim: tuple[float, float] | None | str = "keep",
        layout: str | None = None,
        alpha: float | None = None,
        equal_scale: bool | None = None,
    ) -> None:
        """Change what is drawn; ``clim=None`` means automatic, ``"keep"`` leaves it unchanged."""
        if frame is not None:
            self._frame = int(frame)
        if field is not None:
            self._field = str(field)
        if cmap is not None:
            self._cmap = str(cmap)
        if clim != "keep":
            self._clim = None if clim is None else ordered_limits(float(clim[0]), float(clim[1]))
        if alpha is not None:
            self._alpha = float(alpha)
        if equal_scale is not None:
            self._equal_scale = bool(equal_scale)
        if layout is not None and layout != self._layout:
            self._layout = layout
            self.axes, self.cax = build_axes(self.figure, layout)
        self.redraw()

    def set_slices(self, iz: int | None = None, iy: int | None = None, ix: int | None = None) -> None:
        for axis, v in (("z", iz), ("y", iy), ("x", ix)):
            if v is not None:
                s = self.sliders[axis]
                s.blockSignals(True)
                s.setValue(int(v))
                s.blockSignals(False)
                self._indices[axis] = s.value()
        self.redraw()

    @property
    def indices(self) -> tuple[int, int, int]:
        nz, ny, nx = self._shape
        z = self._indices.get("z")
        y = self._indices.get("y")
        x = self._indices.get("x")
        return (nz // 2 if z is None else z, ny // 2 if y is None else y, nx // 2 if x is None else x)

    @property
    def layout_key(self) -> str:
        return self._layout

    @property
    def field(self) -> str:
        return self._field

    @property
    def frame(self) -> int:
        return self._frame

    def _on_slider(self, axis: str, value: int) -> None:
        self._indices[axis] = int(value)
        self.redraw()
        self.slices_changed.emit(*self.indices)

    # ------------------------------------------------------------------ drawing
    def has_field(self) -> bool:
        res = self._result
        if res is None or not res.result_disp:
            return False
        try:
            from al_dvc.export.export_utils import field_array

            field_array(res, min(self._frame, len(res.result_disp) - 1), self._field)
        except (ValueError, IndexError):
            return False
        return True

    def redraw(self) -> None:
        iz, iy, ix = self.indices
        self._slider_labels["z"].setText(f"z = {iz}")
        self._slider_labels["y"].setText(f"y = {iy}")
        self._slider_labels["x"].setText(f"x = {ix}")
        if not self.has_field():
            for ax in self.axes:
                ax.clear()
                ax.set_facecolor(self._style.background)
                ax.tick_params(colors=self._style.text, labelsize=7)
            self.cax.clear()
            self.cax.set_visible(False)
            self._empty.setVisible(True)
            self.last_clim = None
            self.canvas.draw_idle()
            return
        self._empty.setVisible(False)
        res = self._result
        frame = min(self._frame, len(res.result_disp) - 1)
        info = draw_field_planes(
            self.axes,
            self.cax,
            res,
            frame,
            self._field,
            {"z": iz, "y": iy, "x": ix},
            self._cmap,
            self._clim,
            self._background,
            alpha=self._alpha,
            style=self._style,
            volume_shape=self._shape,
            equal_scale=self._equal_scale,
        )
        self.last_clim = info["clim"]
        self.canvas.draw_idle()

    def save_png(self, path: str | Path, dpi: int = 150) -> Path:
        out = Path(path)
        if out.suffix.lower() != ".png":
            out = out.with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(out, dpi=dpi, facecolor=self.figure.get_facecolor())
        return out

    def set_empty_text(self, text: str) -> None:
        self._empty.setText(text)
