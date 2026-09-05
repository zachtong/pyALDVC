"""3-D view of the result on the node lattice (pyvista).

Two backends behind one panel:

* ``interactive`` -- a ``pyvistaqt.QtInteractor`` embedded in the panel
  (rotate / zoom with the mouse);
* ``static`` -- the same scene rendered off-screen into an image shown in a
  label, with preset cameras. Used when no OpenGL context is available to
  Qt (the offscreen platform of tests and the self-test, some remote
  desktops) or when the interactor fails to initialise.

Without pyvista the panel shows how to install it and does nothing else.
The scene itself is built by :mod:`al_dvc.gui.view3d_scene`, so the two
backends cannot drift apart. The controls shown depend on the mode: slice
positions for ``slices`` (shared with the Slices tab), the iso level for
``surface``, the warp scale for ``warped``; arrows, outline, volume slices,
background and camera are always available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..app_state import AppState
from ..icons import tool_button
from ..view3d_scene import (
    BACKGROUND,
    BACKGROUNDS,
    CAMERAS,
    MODES,
    SceneInfo,
    SceneOptions,
    available,
    build_scene,
    import_error,
    render_image,
)
from ..widgets import guard_wheel

logger = logging.getLogger(__name__)

STATIC_SIZE = (900, 640)
INSTALL_HINT = "pip install pyvista pyvistaqt"
SLICE_AXES = ("z", "y", "x")


class View3DPanel(QWidget):
    """Result fields on the node lattice: orthogonal slices, points, iso-surface or warped grid, with arrows."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._dirty = True
        self._camera = "iso"
        self._camera_reset_pending = True
        self._last_info: SceneInfo | None = None
        self._last_image: np.ndarray | None = None
        self._updating = False
        self.backend = "unavailable"
        self._interactor = None

        # controls -------------------------------------------------------
        self.mode = QComboBox()
        for key in MODES:
            self.mode.addItem(key, key)
        self.slice_spins: dict[str, QSpinBox] = {}
        for axis in SLICE_AXES:
            s = QSpinBox()
            s.setRange(0, 0)
            s.setFixedWidth(64)
            self.slice_spins[axis] = s
        self.iso = QDoubleSpinBox()
        self.iso.setRange(0.0, 1.0)
        self.iso.setSingleStep(0.05)
        self.iso.setValue(0.5)
        self.warp_scale = QDoubleSpinBox()
        self.warp_scale.setRange(0.0, 1000.0)
        self.warp_scale.setValue(1.0)
        self.warp_scale.setSingleStep(1.0)
        self.arrows = QCheckBox()
        self.stride = QSpinBox()
        self.stride.setRange(1, 20)
        self.stride.setValue(2)
        self.stride.setFixedWidth(56)
        self.arrow_scale = QDoubleSpinBox()
        self.arrow_scale.setRange(0.05, 100.0)
        self.arrow_scale.setValue(1.0)
        self.arrow_scale.setSingleStep(0.5)
        self.arrow_scale.setFixedWidth(72)
        self.volume_slices = QCheckBox()
        self.outline = QCheckBox()
        self.outline.setChecked(True)
        self.background = QComboBox()
        for key in BACKGROUNDS:
            self.background.addItem(key, key)
        self.camera = QComboBox()
        for key in CAMERAS:
            self.camera.addItem(key, key)
        self._btn_refresh = tool_button("refresh")
        self._btn_shot = tool_button("camera")
        self._labels: dict[str, QLabel] = {
            k: QLabel()
            for k in (
                "mode",
                "slice_z",
                "slice_y",
                "slice_x",
                "iso",
                "warp_scale",
                "stride",
                "arrow_scale",
                "background",
                "camera",
            )
        }

        # row 1: mode and the mode-specific controls
        top = QHBoxLayout()
        top.addWidget(self._labels["mode"])
        top.addWidget(self.mode)
        for axis in SLICE_AXES:
            top.addWidget(self._labels[f"slice_{axis}"])
            top.addWidget(self.slice_spins[axis])
        top.addWidget(self._labels["iso"])
        top.addWidget(self.iso)
        top.addWidget(self._labels["warp_scale"])
        top.addWidget(self.warp_scale)
        top.addStretch(1)
        # row 2: scene-wide options; row 3: toggles and actions (icons), so the rows fit a narrow centre column
        bottom = QHBoxLayout()
        bottom.addWidget(self._labels["background"])
        bottom.addWidget(self.background)
        bottom.addSpacing(8)
        bottom.addWidget(self._labels["camera"])
        bottom.addWidget(self.camera)
        bottom.addStretch(1)
        actions = QHBoxLayout()
        actions.addWidget(self.volume_slices)
        actions.addWidget(self.outline)
        actions.addWidget(self.arrows)
        actions.addStretch(1)
        actions.addWidget(self._btn_refresh)
        actions.addWidget(self._btn_shot)
        self._arrow_row = QWidget()
        arrow_row = QHBoxLayout(self._arrow_row)
        arrow_row.setContentsMargins(0, 0, 0, 0)
        arrow_row.addWidget(self._labels["stride"])
        arrow_row.addWidget(self.stride)
        arrow_row.addSpacing(8)
        arrow_row.addWidget(self._labels["arrow_scale"])
        arrow_row.addWidget(self.arrow_scale)
        arrow_row.addStretch(1)

        # body -----------------------------------------------------------
        self._stack = QStackedWidget()
        self._hint = QLabel()
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)
        self._hint.setObjectName("hint")
        self._image = QLabel()
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setStyleSheet(f"background: {BACKGROUND};")
        self._image.setMinimumSize(200, 150)
        self._stack.addWidget(self._hint)
        self._stack.addWidget(self._image)
        self._status = QLabel()
        self._status.setObjectName("hint")
        self._status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addLayout(bottom)
        layout.addLayout(actions)
        layout.addWidget(self._arrow_row)
        layout.addWidget(self._stack, stretch=1)
        self.slice_sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        sliders = QHBoxLayout()
        for axis in SLICE_AXES:
            col = QVBoxLayout()
            lab = QLabel(f"{axis} = -")
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 0)
            s.valueChanged.connect(lambda v, a=axis: self._on_slice_spin(a, v))
            col.addWidget(lab)
            col.addWidget(s)
            sliders.addLayout(col)
            self.slice_sliders[axis] = s
            self._slider_labels[axis] = lab
        self._slider_box = QWidget()
        self._slider_box.setLayout(sliders)
        layout.addWidget(self._slider_box)
        layout.addWidget(self._status)

        self._init_backend()
        guard_wheel(self)

        self.mode.currentIndexChanged.connect(lambda _i: self._on_control_changed())
        for w in (self.arrows, self.volume_slices, self.outline):
            w.toggled.connect(lambda _v: self._on_control_changed())
        for w in (self.stride, self.arrow_scale, self.warp_scale, self.iso):
            w.valueChanged.connect(lambda _v: self._on_control_changed())
        for axis, s in self.slice_spins.items():
            s.valueChanged.connect(lambda v, a=axis: self._on_slice_spin(a, v))
        self.background.currentIndexChanged.connect(lambda _i: self._on_background())
        self.camera.currentIndexChanged.connect(lambda i: self._on_camera(CAMERAS[i]))
        self._btn_refresh.clicked.connect(self.refresh)
        self._btn_shot.clicked.connect(self._on_screenshot)
        self._state.results_changed.connect(self._on_results_changed)
        self._state.display_changed.connect(self._on_display_changed)
        self._state.volumes_changed.connect(self._on_volumes_changed)
        self._state.current_frame_changed.connect(lambda _i: self.invalidate())
        self.retranslate_ui()
        self._sync_slice_spins()
        self._update_enabled()

    # ------------------------------------------------------------------ backends
    def _init_backend(self) -> None:
        if not available():
            self.backend = "unavailable"
            self._stack.setCurrentWidget(self._hint)
            return
        app = QApplication.instance()
        if app is not None and app.platformName() == "offscreen":
            self.backend = "static"
        else:
            try:
                from pyvistaqt import QtInteractor

                self._interactor = QtInteractor(self)
                self._interactor.set_background(BACKGROUND)
                self._stack.addWidget(self._interactor)
                self.backend = "interactive"
            except Exception as exc:  # no OpenGL context, missing pyvistaqt, ...
                logger.warning("3-D view: interactive backend unavailable (%s); using static rendering", exc)
                self._interactor = None
                self.backend = "static"
        self._stack.setCurrentWidget(self._hint)

    # ------------------------------------------------------------------ options
    def mode_key(self) -> str:
        return str(self.mode.currentData() or MODES[max(0, self.mode.currentIndex())])

    def background_key(self) -> str:
        return str(self.background.currentData() or "dark")

    def options(self) -> SceneOptions:
        st = self._state
        return SceneOptions(
            field=st.display_field,
            frame=st.display_frame,
            mode=self.mode_key(),
            colormap=st.colormap,
            clim=None if st.color_auto else (float(st.color_min), float(st.color_max)),
            opacity=float(st.overlay_alpha),
            warp_scale=float(self.warp_scale.value()),
            show_arrows=self.arrows.isChecked(),
            arrow_stride=int(self.stride.value()),
            arrow_scale=float(self.arrow_scale.value()),
            show_outline=self.outline.isChecked(),
            show_volume_slices=self.volume_slices.isChecked(),
            iso_fraction=float(self.iso.value()),
            slice_index=dict(st.slice_index),
            background=BACKGROUNDS.get(self.background_key(), BACKGROUND),
        )

    def _volume_for_scene(self):
        if not self.volume_slices.isChecked() or not self._state.volumes:
            return None
        try:
            return self._state.volume_array(min(self._state.current_frame, len(self._state.volumes) - 1))
        except Exception as exc:
            self._state.log(f"3-D view: cannot load the volume for the slices: {exc}", "warning")
            return None

    # ------------------------------------------------------------------ slice positions (shared with the Slices tab)
    def _sync_slice_spins(self) -> None:
        """Ranges from the loaded volume, values from ``state.slice_index`` (without re-emitting)."""
        shape = self._state.volume_shape() if self._state.volumes else None
        self._updating = True
        try:
            for axis, n in zip(SLICE_AXES, shape if shape is not None else (1, 1, 1)):
                cur = self._state.slice_index.get(axis)
                value = int(cur) if cur is not None else int(n) // 2
                for w in (self.slice_spins[axis], self.slice_sliders[axis]):
                    w.setRange(0, max(0, int(n) - 1))
                    w.setValue(value)
                self._slider_labels[axis].setText(f"{axis} = {value}")
        finally:
            self._updating = False

    def _on_slice_spin(self, axis: str, value: int) -> None:
        if self._updating:
            return
        self._state.set_slice(axis, int(value))  # display_changed -> the Slices tab and this view redraw

    def _on_display_changed(self) -> None:
        self._sync_slice_spins()
        self.invalidate()

    def _on_volumes_changed(self) -> None:
        self._sync_slice_spins()
        self.invalidate()

    # ------------------------------------------------------------------ drawing
    def invalidate(self) -> None:
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def _on_results_changed(self) -> None:
        self._camera_reset_pending = True
        self._update_enabled()
        self.invalidate()

    def _on_control_changed(self) -> None:
        self._update_enabled()
        self.invalidate()

    def _on_background(self) -> None:
        colour = BACKGROUNDS.get(self.background_key(), BACKGROUND)
        self._image.setStyleSheet(f"background: {colour};")
        if self._interactor is not None:
            self._interactor.set_background(colour)
        self.invalidate()

    def _on_camera(self, camera: str) -> None:
        self._camera = camera
        self._camera_reset_pending = True
        self.invalidate()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def refresh(self) -> None:
        """Rebuild the scene from the state (no-op without results or pyvista)."""
        self._dirty = False
        res = self._state.results
        if self.backend == "unavailable" or res is None or not res.result_disp:
            self._last_info = None
            self._stack.setCurrentWidget(self._hint)
            self._hint.setText(self._hint_text())
            return
        opts = self.options()
        volume = self._volume_for_scene()
        try:
            if self.backend == "interactive":
                info = build_scene(self._interactor, res, opts, volume)
                if self._camera_reset_pending:
                    self._apply_interactor_camera()
                self._interactor.render()
                self._stack.setCurrentWidget(self._interactor)
            else:
                img, info = render_image(res, opts, volume, window_size=STATIC_SIZE, camera=self._camera)
                self._last_image = img
                self._show_image(img)
                self._stack.setCurrentWidget(self._image)
        except Exception as exc:
            self._state.log(f"3-D view failed: {exc}", "error")
            self._stack.setCurrentWidget(self._hint)
            self._hint.setText(str(exc))
            return
        self._last_info = info
        lo, hi = info.clim
        parts = [f"{info.field}: {info.n_finite}/{info.n_nodes} nodes, [{lo:.4g}, {hi:.4g}]"]
        if info.n_arrows:
            parts.append(self.tr("{n} arrows").format(n=info.n_arrows))
        if self.backend == "static":
            parts.append(self.tr("static rendering; use the camera presets to rotate"))
        self._status.setText("   ".join(parts))

    def _apply_interactor_camera(self) -> None:
        if self._camera == "iso":
            self._interactor.view_isometric()
        else:
            getattr(self._interactor, f"view_{self._camera}")()
        self._interactor.reset_camera()
        self._camera_reset_pending = False

    def _show_image(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        target = self._image.size()
        if target.width() > 50 and target.height() > 50:
            pix = pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._image.setPixmap(pix)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.backend == "static" and self._last_image is not None and self._stack.currentWidget() is self._image:
            self._show_image(self._last_image)

    # ------------------------------------------------------------------ screenshots
    def screenshot(self, path: str | Path) -> Path | None:
        """Save the current view as PNG (the static backend re-renders at full size)."""
        res = self._state.results
        if self.backend == "unavailable" or res is None:
            return None
        out = Path(path)
        if out.suffix.lower() != ".png":
            out = out.with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.backend == "interactive":
                self._interactor.screenshot(str(out))
            else:
                render_image(
                    res, self.options(), self._volume_for_scene(), window_size=(1600, 1200), camera=self._camera, path=out
                )
        except Exception as exc:
            self._state.log(f"screenshot failed: {exc}", "error")
            return None
        self._state.log(self.tr("Screenshot saved: {path}").format(path=out))
        return out

    def _on_screenshot(self) -> None:
        if self._state.results is None:
            return
        default = str(self._state.output_dir / f"view3d_{self._state.display_field}.png")
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Save screenshot"), default, "PNG (*.png)")
        if path:
            self.screenshot(path)

    # ------------------------------------------------------------------ misc
    def _update_enabled(self) -> None:
        """Enable what makes sense and show only the controls of the current mode."""
        has = self.backend != "unavailable" and self._state.results is not None
        mode = self.mode_key()
        for w in (
            self.mode,
            self.arrows,
            self.outline,
            self.volume_slices,
            self.background,
            self.camera,
            self._btn_refresh,
            self._btn_shot,
        ):
            w.setEnabled(has)
        show_slices = mode == "slices"
        self._slider_box.setVisible(show_slices or self.volume_slices.isChecked())
        for s in self.slice_sliders.values():
            s.setEnabled(has)
        for axis in SLICE_AXES:
            self._labels[f"slice_{axis}"].setVisible(False)
            self.slice_spins[axis].setVisible(False)
            self.slice_spins[axis].setEnabled(has)
        self._labels["iso"].setVisible(mode == "surface")
        self.iso.setVisible(mode == "surface")
        self.iso.setEnabled(has)
        self._labels["warp_scale"].setVisible(mode == "warped")
        self.warp_scale.setVisible(mode == "warped")
        self.warp_scale.setEnabled(has)
        arrows_on = has and self.arrows.isChecked()
        self._arrow_row.setVisible(self.arrows.isChecked())
        for w in (self.stride, self.arrow_scale):
            w.setEnabled(arrows_on)

    def visible_controls(self) -> set[str]:
        """Names of the mode-specific controls currently shown (tests)."""
        names = set()
        if self._slider_box.isVisibleTo(self) and self.mode_key() == "slices":
            names.add("slices")
        if self.iso.isVisibleTo(self):
            names.add("iso")
        if self.warp_scale.isVisibleTo(self):
            names.add("warp_scale")
        if self.stride.isVisibleTo(self):
            names.add("arrows")
        return names

    def _hint_text(self) -> str:
        if self.backend == "unavailable":
            reason = import_error() or ""
            return self.tr("The 3-D view needs pyvista and pyvistaqt:\n{cmd}").format(cmd=INSTALL_HINT) + (
                f"\n\n{reason}" if reason else ""
            )
        return self.tr("No results to show. Run an analysis first.")

    def retranslate_ui(self) -> None:
        self._labels["mode"].setText(self.tr("Mode"))
        self._labels["slice_z"].setText(self.tr("Slice z"))
        self._labels["slice_y"].setText(self.tr("Slice y"))
        self._labels["slice_x"].setText(self.tr("Slice x"))
        self._labels["stride"].setText(self.tr("Stride"))
        self._labels["arrow_scale"].setText(self.tr("Arrow scale"))
        self._labels["warp_scale"].setText(self.tr("Warp scale"))
        self._labels["iso"].setText(self.tr("Iso level"))
        self._labels["background"].setText(self.tr("Background"))
        self._labels["camera"].setText(self.tr("Camera"))
        self.arrows.setText(self.tr("Arrows"))
        self.volume_slices.setText(self.tr("Volume slices"))
        self.outline.setText(self.tr("Outline"))
        self._btn_refresh.setToolTip(self.tr("Refresh"))
        self._btn_shot.setToolTip(self.tr("Screenshot..."))
        names = {
            "slices": self.tr("Slices"),
            "points": self.tr("Points"),
            "surface": self.tr("Iso-surface"),
            "warped": self.tr("Deformed lattice"),
        }
        for i, key in enumerate(MODES):
            self.mode.setItemText(i, names[key])
        cams = {"iso": self.tr("Isometric"), "xy": "XY", "xz": "XZ", "yz": "YZ"}
        for i, key in enumerate(CAMERAS):
            self.camera.setItemText(i, cams[key])
        bgs = {"dark": self.tr("Dark"), "black": self.tr("Black"), "grey": self.tr("Grey"), "white": self.tr("White")}
        for i, key in enumerate(BACKGROUNDS):
            self.background.setItemText(i, bgs[key])
        self.mode.setToolTip(
            self.tr(
                "Orthogonal slices of the field, node points, an iso-surface, "
                "or the node lattice moved by the displacement (valid cells only)"
            )
        )
        for axis in SLICE_AXES:
            self.slice_spins[axis].setToolTip(self.tr("Position of the three field slices (shared with the Slices tab)"))
        self.volume_slices.setToolTip(self.tr("Show the current volume's XY / XZ / YZ slices at the slider positions"))
        self._hint.setText(self._hint_text())
