"""Region-of-interest tools: an icon toolbar for drawing masks on the slice viewer.

Row 1: shape tools (rectangle, ellipse, polygon, brush; click the active one again to stop
drawing) and the brush size. Row 2: how a shape combines with the mask (replace, add, cut) and
which slices it reaches (depth). Row 3: automatic mask from the intensity, undo / redo, invert,
fill, clear, remove, save. Row 4: which frames receive the mask, visibility, coverage.

The toolbar owns the drawing settings; the viewer reads :meth:`MaskToolbar.settings` when the
mouse touches a slice and turns the gesture into a :class:`~al_dvc.gui.mask_editor.MaskOp`.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..app_state import AppState
from ..icons import tool_button
from ..mask_editor import MaskOp
from ..widgets import guard_wheel

TOOLS = ("none", "rectangle", "ellipse", "polygon", "brush")
MODES = ("add", "cut", "replace")  # combo order kept for callers that address modes by index
DEPTHS = ("all", "current", "range")
TARGETS = ("current", "all")
MASK_FILTER = "Mask volumes (*.tif *.tiff *.npy);;All files (*)"
EDIT_BUTTONS = ("auto", "undo", "redo", "invert", "fill", "clear", "remove", "save")


@dataclass(frozen=True)
class DrawSettings:
    tool: str
    mode: str
    depth: str
    depth_range: tuple[int, int]
    radius: int


class MaskToolbar(QWidget):
    """Tool / mode / depth selection and the edit buttons (icons)."""

    settings_changed = Signal()  # tool, mode, depth, brush or target changed: a gesture in progress is void

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        # hidden combos hold the current tool and mode so that callers can address them by name / index
        self.tool = QComboBox()
        for key in TOOLS:
            self.tool.addItem(key, key)
        self.mode = QComboBox()
        for key in MODES:
            self.mode.addItem(key, key)
        self.tool.hide()
        self.mode.hide()
        self.tool_buttons = {k: tool_button(k, checkable=True) for k in TOOLS if k != "none"}
        self.mode_buttons = {k: tool_button(k, checkable=True) for k in MODES}
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(False)  # exclusivity with "none" is handled by hand
        for k, b in self.tool_buttons.items():
            self._tool_group.addButton(b)
            b.clicked.connect(lambda checked, key=k: self._on_tool_button(key, checked))
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for k, b in self.mode_buttons.items():
            self._mode_group.addButton(b)
            b.clicked.connect(lambda _c, key=k: self.mode.setCurrentIndex(MODES.index(key)))
        self.mode_buttons["add"].setChecked(True)
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
        self.target = QComboBox()
        for key in TARGETS:
            self.target.addItem(key, key)
        self.target.setMinimumWidth(110)
        # the selector only says where future drawing goes; copying the mask to every frame is an explicit,
        # undoable action of its own
        self._btn_copy_all = tool_button("copy")  # icon only: the row must fit beside "Show mask"
        self.show_mask = QCheckBox()
        self.show_mask.setChecked(state.show_mask)
        self._btn = {k: tool_button(k) for k in EDIT_BUTTONS}
        self._labels = {k: QLabel() for k in ("tool", "mode", "depth", "to", "radius", "target")}
        self._labels["to"].setText("-")
        self._status = QLabel()
        self._status.setObjectName("hint")
        # the row must never widen the column: the hint takes what is left and is elided by the tooltip
        self._status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row1 = QHBoxLayout()
        row1.setSpacing(2)
        row1.addWidget(self._labels["tool"])
        for k in ("rectangle", "ellipse", "polygon", "brush"):
            row1.addWidget(self.tool_buttons[k])
        row1.addSpacing(8)
        row1.addWidget(self._labels["radius"])
        row1.addWidget(self.radius)
        row1.addStretch(1)
        row2 = QHBoxLayout()
        row2.setSpacing(2)
        row2.addWidget(self._labels["mode"])
        for k in ("replace", "add", "cut"):
            row2.addWidget(self.mode_buttons[k])
        row2.addStretch(1)
        row_depth = QHBoxLayout()  # its own row: the depth controls are wider than the mode buttons
        row_depth.setSpacing(2)
        row_depth.addWidget(self._labels["depth"])
        row_depth.addWidget(self.depth)
        row_depth.addWidget(self.depth_from)
        row_depth.addWidget(self._labels["to"])
        row_depth.addWidget(self.depth_to)
        row_depth.addStretch(1)
        row3 = QHBoxLayout()
        row3.setSpacing(2)
        for k in EDIT_BUTTONS:
            row3.addWidget(self._btn[k])
            if k in ("auto", "redo", "clear"):
                row3.addSpacing(8)
        row3.addStretch(1)
        row4 = QHBoxLayout()
        row4.setSpacing(6)
        row4.addWidget(self._labels["target"])
        row4.addWidget(self.target)
        row4.addWidget(self._btn_copy_all)
        row4.addWidget(self.show_mask)
        row4.addWidget(self._status, 1)
        for row in (row1, row2, row_depth, row3, row4):
            layout.addLayout(row)
        for key in ("tool", "mode", "depth", "target"):
            self._labels[key].setFixedWidth(58)
        guard_wheel(self)

        self._btn["auto"].clicked.connect(self._on_auto)
        self._btn["undo"].clicked.connect(self._state.undo_mask)
        self._btn["redo"].clicked.connect(self._state.redo_mask)
        self._btn["invert"].clicked.connect(lambda: self._state.apply_mask_op(MaskOp("invert")))
        self._btn["fill"].clicked.connect(lambda: self._state.apply_mask_op(MaskOp("fill")))
        self._btn["clear"].clicked.connect(lambda: self._state.apply_mask_op(MaskOp("empty")))
        self._btn["remove"].clicked.connect(lambda: self._state.remove_mask())
        self._btn["save"].clicked.connect(self._on_save)
        self.target.currentIndexChanged.connect(self._on_target_index)
        self._btn_copy_all.clicked.connect(self._on_copy_all)
        self.show_mask.toggled.connect(lambda v: self._state.set_mask_display(show=bool(v)))
        self.depth.currentIndexChanged.connect(self._on_depth_changed)
        for s in (self.depth_from, self.depth_to, self.radius):
            s.valueChanged.connect(lambda _v: self.settings_changed.emit())
        self.tool.currentIndexChanged.connect(self._on_tool_index)
        self.mode.currentIndexChanged.connect(self._on_mode_index)
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

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode.setCurrentIndex(MODES.index(mode))

    def set_depth(self, kind: str, first: int | None = None, last: int | None = None) -> None:
        if kind not in DEPTHS:
            raise ValueError(f"depth must be one of {DEPTHS}, got {kind!r}")
        self.depth.setCurrentIndex(DEPTHS.index(kind))
        if first is not None:
            self.depth_from.setValue(int(first))
        if last is not None:
            self.depth_to.setValue(int(last))

    def _on_tool_button(self, key: str, checked: bool) -> None:
        self.set_tool(key if checked else "none")

    def _on_tool_index(self, index: int) -> None:
        current = TOOLS[index]
        for k, b in self.tool_buttons.items():
            b.blockSignals(True)
            b.setChecked(k == current)
            b.blockSignals(False)
        self._update_enabled()
        self.settings_changed.emit()

    def _on_mode_index(self, index: int) -> None:
        current = MODES[index]
        for k, b in self.mode_buttons.items():
            b.blockSignals(True)
            b.setChecked(k == current)
            b.blockSignals(False)
        self.settings_changed.emit()

    def _on_depth_changed(self, _index: int) -> None:
        self._update_enabled()
        self.settings_changed.emit()

    def _on_target_index(self, index: int) -> None:
        self._state.set_mask_display(target=TARGETS[index])
        self.settings_changed.emit()

    def _on_copy_all(self) -> None:
        """Every frame receives a copy of this frame's mask (one undo step)."""
        copy_all = getattr(self._state, "copy_mask_to_all_frames", None)
        if copy_all is None:
            self._state.log("copying the mask to every frame is not available in this build", "warning")
            return
        copy_all()
        self._state.log(self.tr("Mask copied to all {n} frames (undo reverses it).").format(n=len(self._state.volumes)))

    # ------------------------------------------------------------------ actions
    def _on_auto(self) -> None:
        """Mask from the intensity of the current frame: Otsu threshold, holes filled, largest component kept."""
        if not self._state.volumes:
            return
        try:
            self._state.apply_mask_op(MaskOp("threshold", mode="replace"))
        except Exception as exc:
            self._state.log(self.tr("Automatic mask failed: {error}").format(error=exc), "error")
            return
        mask = self._state.current_mask()
        if mask is not None:
            self._state.log(self.tr("Automatic mask: material {pct:.1f}% of the volume").format(pct=100.0 * float(mask.mean())))

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
        if self.show_mask.isChecked() != bool(self._state.show_mask):  # the state may hide the mask (results)
            self.show_mask.blockSignals(True)
            self.show_mask.setChecked(bool(self._state.show_mask))
            self.show_mask.blockSignals(False)
        ed = self._state.mask_editor
        has_volume = bool(self._state.volumes)
        self._btn["undo"].setEnabled(ed is not None and ed.can_undo)
        self._btn["redo"].setEnabled(ed is not None and ed.can_redo)
        for key in ("auto", "invert", "fill", "clear"):
            self._btn[key].setEnabled(has_volume)
        mask = self._state.current_mask() if has_volume else None
        self._btn["save"].setEnabled(mask is not None)
        self._btn["remove"].setEnabled(mask is not None)
        self._btn_copy_all.setEnabled(mask is not None and len(self._state.volumes) > 1)
        if mask is None:
            self._status.setText(self.tr("no mask") if has_volume else "")
        else:
            cov = 100.0 * float(mask.mean())
            n_ops = len(ed.ops) if ed is not None else 0
            self._status.setText(self.tr("material {cov:.1f} %, {n} operation(s)").format(cov=cov, n=n_ops))
        self._status.setToolTip(self._status.text())
        self._update_enabled()

    def _update_enabled(self) -> None:
        has_volume = bool(self._state.volumes)
        drawing = has_volume and self.tool.currentData() != "none"
        for b in self.tool_buttons.values():
            b.setEnabled(has_volume)
        for w in (self.target, self.show_mask):
            w.setEnabled(has_volume)
        for b in self.mode_buttons.values():
            b.setEnabled(drawing)
        self.depth.setEnabled(drawing)
        rng = drawing and self.depth.currentData() == "range"
        self.depth_from.setEnabled(rng)
        self.depth_to.setEnabled(rng)
        brush = drawing and self.tool.currentData() == "brush"
        self.radius.setEnabled(brush)
        self.radius.setVisible(brush)
        self._labels["radius"].setVisible(brush)

    def retranslate_ui(self) -> None:
        self._labels["tool"].setText(self.tr("Draw"))
        self._labels["mode"].setText(self.tr("Mode"))
        self._labels["depth"].setText(self.tr("Depth"))
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
            "rectangle": self.tr("Rectangle"),
            "ellipse": self.tr("Ellipse"),
            "polygon": self.tr("Polygon"),
            "brush": self.tr("Brush"),
        }
        for k, b in self.tool_buttons.items():
            b.setToolTip(tools[k] + self.tr(" (click again to stop drawing)"))
        modes = {
            "replace": self.tr("Replace: the shape becomes the mask"),
            "add": self.tr("Add: the shape joins the mask"),
            "cut": self.tr("Cut: the shape is removed from the mask"),
        }
        for k, b in self.mode_buttons.items():
            b.setToolTip(modes[k])
        depths = {"all": self.tr("All slices"), "current": self.tr("Current slice"), "range": self.tr("Range")}
        for i, key in enumerate(DEPTHS):
            self.depth.setItemText(i, depths[key])
        self.depth.setToolTip(self.tr("Slices along the normal of the plane you draw on that the shape is applied to"))
        self.target.setItemText(0, self.tr("This frame"))
        self.target.setItemText(1, self.tr("All frames"))
        self._btn_copy_all.setText(self.tr("Copy to all frames"))
        self._btn_copy_all.setToolTip(
            self.tr("Copy to all frames: give every frame this frame's mask (the undo button reverses it)")
        )
        tips = {
            "auto": self.tr("Automatic mask: Otsu threshold on the intensity, holes filled, largest component kept"),
            "undo": self.tr("Undo"),
            "redo": self.tr("Redo"),
            "invert": self.tr("Invert"),
            "fill": self.tr("Select the entire volume"),
            "clear": self.tr("Empty mask: nothing is analysed until you draw"),
            "remove": self.tr("No mask: the whole volume is analysed"),
            "save": self.tr("Save mask..."),
        }
        for key, text in tips.items():
            self._btn[key].setToolTip(text)
        self.show_mask.setText(self.tr("Show mask"))
        self.refresh()
