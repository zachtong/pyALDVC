"""Toolbar for drawing masks on the slice viewer.

The toolbar owns the drawing settings (tool, add/cut, depth rule, brush
radius) and the edit buttons (undo, redo, invert, fill, clear, save); the
viewer reads :meth:`MaskToolbar.settings` when the mouse touches a slice
and turns the gesture into a :class:`~al_dvc.gui.mask_editor.MaskOp`.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..app_state import AppState
from ..mask_editor import MaskOp
from ..widgets import COMBO_WIDTH, guard_wheel, make_form

ROI_LABEL_WIDTH = 64  # px: short labels, the column is only 400 px wide

TOOLS = ("none", "rectangle", "ellipse", "polygon", "brush")
DEPTHS = ("all", "current", "range")
TARGETS = ("current", "all")
MASK_FILTER = "Mask volumes (*.tif *.tiff *.npy);;All files (*)"


@dataclass(frozen=True)
class DrawSettings:
    tool: str
    mode: str
    depth: str
    depth_range: tuple[int, int]
    radius: int


class MaskToolbar(QWidget):
    """Tool / mode / depth selection and the edit buttons."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self.tool = QComboBox()
        for key in TOOLS:
            self.tool.addItem(key, key)
        self.mode = QComboBox()
        for key in ("add", "cut"):
            self.mode.addItem(key, key)
        self.depth = QComboBox()
        for key in DEPTHS:
            self.depth.addItem(key, key)
        self.depth_from = QSpinBox()
        self.depth_to = QSpinBox()
        for s in (self.depth_from, self.depth_to):
            s.setRange(0, 100000)
        self.radius = QSpinBox()
        self.radius.setRange(1, 200)
        self.radius.setValue(4)
        self.target = QComboBox()
        for key in TARGETS:
            self.target.addItem(key, key)
        self.show_mask = QCheckBox()
        self.show_mask.setChecked(state.show_mask)
        self._btn = {k: QPushButton() for k in ("undo", "redo", "invert", "fill", "clear", "remove", "save")}
        self._labels = {k: QLabel() for k in ("tool", "mode", "depth", "to", "radius", "target")}
        self._status = QLabel()
        self._status.setObjectName("hint")

        # compact vertical form (the section lives in the left column, pyALDIC style)
        form = make_form()
        for key, w in [("tool", self.tool), ("mode", self.mode)]:
            form.addRow(self._labels[key], w)
        depth_widget = QWidget()
        depth_row = QHBoxLayout(depth_widget)
        depth_row.setContentsMargins(0, 0, 0, 0)
        depth_row.setSpacing(4)
        depth_row.addWidget(self.depth)
        depth_row.addWidget(self.depth_from)
        depth_row.addWidget(self._labels["to"])
        depth_row.addWidget(self.depth_to)
        form.addRow(self._labels["depth"], depth_widget)
        form.addRow(self._labels["radius"], self.radius)
        form.addRow(self._labels["target"], self.target)
        for w in (self.tool, self.mode, self.target):
            w.setMinimumWidth(COMBO_WIDTH)
        self.depth.setMinimumWidth(100)
        for w in (self.depth_from, self.depth_to, self.radius):
            w.setFixedWidth(56)
        for key, label in self._labels.items():
            if key != "to":
                label.setFixedWidth(ROI_LABEL_WIDTH)
        buttons = QGridLayout()
        buttons.setSpacing(4)
        for i, key in enumerate(("undo", "redo", "invert", "fill", "clear", "remove")):
            self._btn[key].setMinimumWidth(0)
            buttons.addWidget(self._btn[key], i // 2, i % 2)
        buttons.addWidget(self._btn["save"], 3, 0, 1, 2)
        status_row = QHBoxLayout()
        status_row.addWidget(self.show_mask)
        status_row.addWidget(self._status, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addLayout(status_row)
        guard_wheel(self)

        self._btn["undo"].clicked.connect(self._state.undo_mask)
        self._btn["redo"].clicked.connect(self._state.redo_mask)
        self._btn["invert"].clicked.connect(lambda: self._state.apply_mask_op(MaskOp("invert")))
        self._btn["fill"].clicked.connect(lambda: self._state.apply_mask_op(MaskOp("fill")))
        self._btn["clear"].clicked.connect(lambda: self._state.apply_mask_op(MaskOp("empty")))
        self._btn["remove"].clicked.connect(lambda: self._state.remove_mask())
        self._btn["save"].clicked.connect(self._on_save)
        self.target.currentIndexChanged.connect(lambda i: self._state.set_mask_display(target=TARGETS[i]))
        self.show_mask.toggled.connect(lambda v: self._state.set_mask_display(show=bool(v)))
        self.depth.currentIndexChanged.connect(lambda _i: self._update_enabled())
        self.tool.currentIndexChanged.connect(lambda _i: self._update_enabled())
        self._state.mask_changed.connect(self.refresh)
        self._state.volumes_changed.connect(self.refresh)
        self._state.current_frame_changed.connect(lambda _i: self.refresh())
        self.retranslate_ui()
        self.refresh()

    # ------------------------------------------------------------------ settings
    def settings(self) -> DrawSettings:
        return DrawSettings(
            tool=str(self.tool.currentData()),
            mode=str(self.mode.currentData()),
            depth=str(self.depth.currentData()),
            depth_range=(int(self.depth_from.value()), int(self.depth_to.value())),
            radius=int(self.radius.value()),
        )

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(f"tool must be one of {TOOLS}, got {tool!r}")
        self.tool.setCurrentIndex(TOOLS.index(tool))

    def set_depth(self, kind: str, first: int | None = None, last: int | None = None) -> None:
        if kind not in DEPTHS:
            raise ValueError(f"depth must be one of {DEPTHS}, got {kind!r}")
        self.depth.setCurrentIndex(DEPTHS.index(kind))
        if first is not None:
            self.depth_from.setValue(int(first))
        if last is not None:
            self.depth_to.setValue(int(last))

    # ------------------------------------------------------------------ actions
    def _on_save(self) -> None:
        if self._state.current_mask() is None:
            return
        entry = self._state.volumes[self._state.current_frame] if self._state.volumes else None
        default = (entry.path.rsplit(".", 1)[0] + "_mask.tif") if entry is not None and entry.path else "mask.tif"
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Save mask"), default, MASK_FILTER)
        if path:
            try:
                out = self._state.save_mask(path)
            except Exception as exc:
                self._state.log(f"saving the mask failed: {exc}", "error")
                return
            self._state.log(self.tr("Mask saved: {path}").format(path=out))

    # ------------------------------------------------------------------ view
    def refresh(self) -> None:
        ed = self._state.mask_editor
        has_volume = bool(self._state.volumes)
        self._btn["undo"].setEnabled(ed is not None and ed.can_undo)
        self._btn["redo"].setEnabled(ed is not None and ed.can_redo)
        for key in ("invert", "fill", "clear"):
            self._btn[key].setEnabled(has_volume)
        mask = self._state.current_mask() if has_volume else None
        self._btn["save"].setEnabled(mask is not None)
        self._btn["remove"].setEnabled(mask is not None)
        if mask is None:
            self._status.setText(self.tr("no mask") if has_volume else "")
        else:
            cov = 100.0 * float(mask.mean())
            n_ops = len(ed.ops) if ed is not None else 0
            self._status.setText(self.tr("material {cov:.1f} %, {n} operation(s)").format(cov=cov, n=n_ops))
        self._update_enabled()

    def _update_enabled(self) -> None:
        has_volume = bool(self._state.volumes)
        drawing = has_volume and self.tool.currentData() != "none"
        for w in (self.tool, self.target, self.show_mask):
            w.setEnabled(has_volume)
        self.mode.setEnabled(drawing)
        self.depth.setEnabled(drawing)
        rng = drawing and self.depth.currentData() == "range"
        self.depth_from.setEnabled(rng)
        self.depth_to.setEnabled(rng)
        self.radius.setEnabled(drawing and self.tool.currentData() == "brush")

    def retranslate_ui(self) -> None:
        self._labels["tool"].setText(self.tr("Draw"))
        self._labels["mode"].setText(self.tr("Mode"))
        self._labels["depth"].setText(self.tr("Depth"))
        self._labels["to"].setText(self.tr("to"))
        self._labels["radius"].setText(self.tr("Brush"))
        self._labels["target"].setText(self.tr("Mask for"))
        self.target.setToolTip(
            self.tr(
                "Which frames receive the drawn mask. The reference mask (frame 0) defines the region of interest\n"
                "and the analysed box; a mask on a deformed frame only excludes its own voxels. Choose all frames\n"
                "when the same region applies to every scan."
            )
        )
        tools = {
            "none": self.tr("(off)"),
            "rectangle": self.tr("Rectangle"),
            "ellipse": self.tr("Ellipse"),
            "polygon": self.tr("Polygon"),
            "brush": self.tr("Brush"),
        }
        for i, key in enumerate(TOOLS):
            self.tool.setItemText(i, tools[key])
        self.mode.setItemText(0, self.tr("Add"))
        self.mode.setItemText(1, self.tr("Cut"))
        depths = {"all": self.tr("All slices"), "current": self.tr("Current slice"), "range": self.tr("Range")}
        for i, key in enumerate(DEPTHS):
            self.depth.setItemText(i, depths[key])
        self.target.setItemText(0, self.tr("This frame"))
        self.target.setItemText(1, self.tr("All frames"))
        texts = {
            "undo": self.tr("Undo"),
            "redo": self.tr("Redo"),
            "invert": self.tr("Invert"),
            "fill": self.tr("Fill"),
            "clear": self.tr("Clear"),
            "remove": self.tr("Remove mask"),
            "save": self.tr("Save mask..."),
        }
        for key, text in texts.items():
            self._btn[key].setText(text)
        self.show_mask.setText(self.tr("Show mask"))
        self.tool.setToolTip(
            self.tr(
                "Draw on any of the three slices. Rectangle / ellipse: drag. Polygon: click the vertices, "
                "right-click to close, Esc to cancel. Brush: drag. True (material) is kept; the excluded region is tinted red."
            )
        )
        self.depth.setToolTip(self.tr("Slices along the normal of the plane you draw on that the shape is applied to"))
        self.refresh()
