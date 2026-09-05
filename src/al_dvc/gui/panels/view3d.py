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
background and camera are always available. The camera row turns and zooms the preset view; the
animation row plays an orbit, the result frames, a slice sweep or the growing deformed lattice
live, and records the same sequence off-screen as GIF, MP4 or PNG frames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QElapsedTimer, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..app_state import AppState
from ..icons import tool_button
from ..names import field_name, fill_combo, retranslate_combo
from ..view3d_animation import (
    DEFAULT_SPEEDS,
    FORMATS,
    SIZES,
    SPEED_RANGES,
    AnimationSpec,
    frame_at,
    mp4_available,
    record_animation,
)
from ..view3d_scene import (
    BACKGROUND,
    BACKGROUNDS,
    CAMERAS,
    MODES,
    CameraSpec,
    SceneInfo,
    SceneOptions,
    apply_camera,
    available,
    build_scene,
    import_error,
    render_image,
)
from ..widgets import guard_wheel, headless

logger = logging.getLogger(__name__)

STATIC_SIZE = (900, 640)
INSTALL_HINT = "pip install pyvista pyvistaqt"
SLICE_AXES = ("z", "y", "x")
PLAY_INTERVAL_MS = {"interactive": 33, "static": 150}  # the static backend re-renders every tick
RECORD_SIZE = (1280, 960)  # frames rendered off-screen while the live view keeps its own size


class _RecordWorker(QThread):
    """Render and write an animation off the UI thread."""

    progress = Signal(float, str)
    finished_record = Signal(object)  # Path | None
    failed = Signal(str)

    def __init__(self, result, volume, spec, camera, options, path, window_size, parent=None) -> None:
        super().__init__(parent)
        self._args = (result, volume, spec, camera, options, path, window_size)
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            out = record_animation(*self._args, progress=self.progress.emit, stop=lambda: self._stop)
        except Exception as exc:
            logger.exception("Recording failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished_record.emit(out)


class RecordDialog(QDialog):
    """Format, length, rate and size of the recording; the animation itself comes from the panel."""

    def __init__(self, spec: AnimationSpec, parent=None) -> None:
        super().__init__(parent)
        self.setModal(True)
        form = QFormLayout(self)
        self.format = QComboBox()
        for key in FORMATS:
            self.format.addItem(key.upper(), key)
        if not mp4_available():
            self.format.model().item(FORMATS.index("mp4")).setEnabled(False)
        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(spec.fps)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.5, 600.0)
        self.duration.setDecimals(1)
        self.duration.setValue(spec.duration)
        self.size = QComboBox()
        for key in SIZES:
            self.size.addItem(key, key)
        self.loop = QCheckBox()
        self.loop.setChecked(spec.loop)
        self._labels = {k: QLabel() for k in ("format", "fps", "duration", "size")}
        form.addRow(self._labels["format"], self.format)
        form.addRow(self._labels["fps"], self.fps)
        form.addRow(self._labels["duration"], self.duration)
        form.addRow(self._labels["size"], self.size)
        form.addRow(self.loop)
        self._hint = QLabel()
        self._hint.setObjectName("hint")
        self._hint.setWordWrap(True)
        form.addRow(self._hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.retranslate_ui()

    def values(self) -> dict:
        return {
            "format": str(self.format.currentData() or "gif"),
            "fps": int(self.fps.value()),
            "duration": float(self.duration.value()),
            "size": str(self.size.currentData() or "view"),
            "loop": bool(self.loop.isChecked()),
        }

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("Record animation"))
        self._labels["format"].setText(self.tr("Format"))
        self._labels["fps"].setText(self.tr("Frames per second"))
        self._labels["duration"].setText(self.tr("Duration [s]"))
        self._labels["size"].setText(self.tr("Size"))
        for i, text in enumerate((self.tr("Current view"), "1280 x 960", "1920 x 1440")):
            self.size.setItemText(i, text)
        self.loop.setText(self.tr("Loop the GIF"))
        self._hint.setText(
            self.tr(
                "GIF plays everywhere; MP4 needs imageio-ffmpeg (pip install imageio imageio-ffmpeg); "
                "PNG writes one file per frame."
            )
        )


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
        self.camera.setMinimumWidth(100)  # room for 'Isometric' next to the camera row
        self._btn_refresh = tool_button("refresh")
        self._btn_shot = tool_button("camera")
        # camera row: turn and zoom the preset view
        self.azimuth = QSpinBox()
        self.azimuth.setRange(-180, 180)
        self.azimuth.setSingleStep(5)
        self.azimuth.setWrapping(True)
        self.azimuth.setFixedWidth(64)
        self.elevation = QSpinBox()
        self.elevation.setRange(-89, 89)
        self.elevation.setSingleStep(5)
        self.elevation.setFixedWidth(64)
        self.zoom = QDoubleSpinBox()
        self.zoom.setRange(0.2, 5.0)
        self.zoom.setSingleStep(0.1)
        self.zoom.setValue(1.0)
        self.zoom.setFixedWidth(64)
        self._btn_reset_camera = QPushButton()
        # animation row: kind, axis, direction, speed, play / pause / stop, record
        self.anim_kind = QComboBox()
        fill_combo(self.anim_kind, "animation")
        self.anim_axis = QComboBox()
        for axis in ("z", "y", "x"):
            self.anim_axis.addItem(axis, axis)
        self.anim_direction = QComboBox()
        fill_combo(self.anim_direction, "direction")
        self.anim_speed = QDoubleSpinBox()
        self.anim_speed.setDecimals(1)
        self.anim_speed.setFixedWidth(72)
        self._btn_play = tool_button("play")
        self._btn_stop = tool_button("stop")
        self._btn_record = tool_button("record")
        self._record_progress = QProgressBar()
        self._record_progress.setRange(0, 1000)
        self._record_progress.setTextVisible(False)
        self._record_progress.setFixedWidth(120)
        self._record_progress.setVisible(False)
        self._play_timer = QTimer(self)
        self._play_clock = QElapsedTimer()
        self._play_offset = 0.0  # seconds already played before the current pause
        self._play_base: tuple | None = None  # (CameraSpec, SceneOptions) the animation started from
        self._recorder: _RecordWorker | None = None
        self._playing = False
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
                "azimuth",
                "elevation",
                "zoom",
                "anim_kind",
                "anim_speed",
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
        bottom.addSpacing(8)
        bottom.addWidget(self._labels["azimuth"])
        bottom.addWidget(self.azimuth)
        bottom.addWidget(self._labels["elevation"])
        bottom.addWidget(self.elevation)
        bottom.addWidget(self._labels["zoom"])
        bottom.addWidget(self.zoom)
        bottom.addWidget(self._btn_reset_camera)
        bottom.addStretch(1)
        anim = QHBoxLayout()
        anim.addWidget(self._labels["anim_kind"])
        anim.addWidget(self.anim_kind)
        anim.addWidget(self.anim_axis)
        anim.addWidget(self.anim_direction)
        anim.addWidget(self._labels["anim_speed"])
        anim.addWidget(self.anim_speed)
        anim.addSpacing(6)
        anim.addWidget(self._btn_play)
        anim.addWidget(self._btn_stop)
        anim.addWidget(self._btn_record)
        anim.addWidget(self._record_progress)
        anim.addStretch(1)
        self._anim_row = QWidget()
        self._anim_row.setLayout(anim)
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
        layout.addWidget(self._anim_row)
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
        for w in (self.azimuth, self.elevation, self.zoom):
            w.valueChanged.connect(lambda _v: self._on_camera_tweak())
        self._btn_reset_camera.clicked.connect(self.reset_camera)
        self.anim_kind.currentIndexChanged.connect(lambda _i: self._on_anim_kind())
        self._btn_play.clicked.connect(self.toggle_play)
        self._btn_stop.clicked.connect(self.stop_animation)
        self._btn_record.clicked.connect(self._on_record)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._on_anim_kind()
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
            title=field_name(st.display_field),
        )

    def camera_spec(self) -> CameraSpec:
        """The preset turned and zoomed as the camera row says."""
        return CameraSpec(
            preset=self._camera,
            azimuth=float(self.azimuth.value()),
            elevation=float(self.elevation.value()),
            zoom=float(self.zoom.value()),
            view_up=str(self.anim_axis.currentData() or "z") if self.anim_kind_key() == "orbit" else "z",
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
        mode = self.mode_key()
        if mode != getattr(self, "_last_mode", None):  # a new kind of geometry: frame it again
            self._last_mode = mode
            self._camera_reset_pending = True
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

    def _on_camera_tweak(self) -> None:
        if self._updating:
            return
        self._camera_reset_pending = True
        self.invalidate()

    def reset_camera(self) -> None:
        """Back to the untouched preset."""
        self._updating = True
        try:
            self.azimuth.setValue(0)
            self.elevation.setValue(0)
            self.zoom.setValue(1.0)
        finally:
            self._updating = False
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
                img, info = render_image(res, opts, volume, window_size=STATIC_SIZE, camera=self.camera_spec())
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
        parts = [f"{field_name(info.field)}: {info.n_finite}/{info.n_nodes} nodes, [{lo:.4g}, {hi:.4g}]"]
        if info.n_arrows:
            parts.append(self.tr("{n} arrows").format(n=info.n_arrows))
        if info.note == "nodes_only":
            parts.append(self.tr("no cell has 8 valid nodes: the valid nodes are drawn instead of the lattice"))
        if self.backend == "static":
            parts.append(self.tr("static rendering; use the camera presets to rotate"))
        self._status.setText("   ".join(parts))

    def _apply_interactor_camera(self, spec: CameraSpec | None = None) -> None:
        apply_camera(self._interactor, spec or self.camera_spec())
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
                    res, self.options(), self._volume_for_scene(), window_size=(1600, 1200), camera=self.camera_spec(), path=out
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

    # ------------------------------------------------------------------ animation
    def anim_kind_key(self) -> str:
        return str(self.anim_kind.currentData() or "orbit")

    def animation_spec(self, **overrides) -> AnimationSpec:
        """The animation the controls describe (``overrides`` for the recording dialog)."""
        kind = self.anim_kind_key()
        values = dict(
            kind=kind,
            axis=str(self.anim_axis.currentData() or "z"),
            direction=1 if (self.anim_direction.currentData() or "ccw") == "ccw" else -1,
            speed=float(self.anim_speed.value()),
        )
        if kind == "orbit" and "duration" not in overrides:
            values["duration"] = 360.0 / float(self.anim_speed.value())  # one full turn
        values.update(overrides)
        return AnimationSpec(**values)

    def _on_anim_kind(self) -> None:
        kind = self.anim_kind_key()
        lo, hi = SPEED_RANGES[kind]
        self._updating = True
        try:
            self.anim_speed.setRange(lo, hi)
            self.anim_speed.setValue(DEFAULT_SPEEDS[kind])
        finally:
            self._updating = False
        self.anim_axis.setVisible(kind in ("orbit", "slice"))
        self.anim_direction.setVisible(kind in ("orbit", "frames", "slice"))
        self.anim_speed.setSuffix({"orbit": " \u00b0/s", "frames": " f/s", "slice": " vx/s", "warp": " /s"}[kind])
        if self._playing:
            self.stop_animation()

    def _play_frame(self, t: float):
        base_cam, base_opts = self._play_base
        res = self._state.results
        return frame_at(self.animation_spec(), t, base_cam, base_opts, len(res.result_disp), tuple(res.volume_shape))

    def toggle_play(self) -> None:
        """Play or pause the animation described by the controls."""
        if self._playing:
            self._play_offset += self._play_clock.elapsed() / 1000.0
            self._play_timer.stop()
            self._playing = False
        else:
            if self._state.results is None or self.backend == "unavailable":
                return
            if self._play_base is None:
                self._play_base = (self.camera_spec(), self.options())
            self._play_clock.start()
            self._play_timer.start(PLAY_INTERVAL_MS.get(self.backend, 100))
            self._playing = True
        self._btn_play.setToolTip(self.tr("Pause") if self._playing else self.tr("Play"))

    def stop_animation(self) -> None:
        """Stop and return to where the animation started."""
        self._play_timer.stop()
        self._playing = False
        self._play_offset = 0.0
        base = self._play_base
        self._play_base = None
        self._btn_play.setToolTip(self.tr("Play"))
        if base is not None:
            self._camera_reset_pending = True
            self.invalidate()

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def play_time(self) -> float:
        """Seconds into the animation (paused time included)."""
        return self._play_offset + (self._play_clock.elapsed() / 1000.0 if self._playing else 0.0)

    def _on_play_tick(self) -> None:
        res = self._state.results
        if res is None or self._play_base is None:
            self.stop_animation()
            return
        frame = self._play_frame(self.play_time)
        try:
            if self.backend == "interactive":
                if frame.options != self._play_base[1]:
                    build_scene(self._interactor, res, frame.options, self._volume_for_scene())
                apply_camera(self._interactor, frame.camera)
                self._interactor.render()
            else:
                img, _info = render_image(
                    res, frame.options, self._volume_for_scene(), window_size=STATIC_SIZE, camera=frame.camera
                )
                self._last_image = img
                self._show_image(img)
        except Exception as exc:
            self._state.log(f"animation: {exc}", "error")
            self.stop_animation()

    def record(self, path, **overrides):
        """Record the animation to ``path`` on a worker thread (``overrides``: format, fps, duration, size, loop)."""
        res = self._state.results
        if res is None or self.backend == "unavailable" or (self._recorder is not None and self._recorder.isRunning()):
            return None
        was_playing = self._playing
        if was_playing:
            self.toggle_play()
        base_cam, base_opts = self._play_base or (self.camera_spec(), self.options())
        spec = self.animation_spec(**overrides)
        size = STATIC_SIZE if self.backend == "static" else RECORD_SIZE
        self._recorder = _RecordWorker(res, self._volume_for_scene(), spec, base_cam, base_opts, Path(path), size, parent=self)
        self._recorder.progress.connect(self._on_record_progress)
        self._recorder.finished_record.connect(self._on_record_finished)
        self._recorder.failed.connect(self._on_record_failed)
        self._record_progress.setValue(0)
        self._record_progress.setVisible(True)
        self._btn_record.setEnabled(False)
        self._state.log(
            self.tr("Recording {kind} animation: {n} frames").format(kind=self.anim_kind.currentText(), n=spec.n_frames)
        )
        self._recorder.start()
        return self._recorder

    def wait_recording(self, timeout_ms: int = 600_000) -> bool:
        return self._recorder.wait(timeout_ms) if self._recorder is not None else True

    def _on_record_progress(self, fraction: float, message: str) -> None:
        self._record_progress.setValue(int(round(1000 * fraction)))
        self._record_progress.setToolTip(message)

    def _on_record_finished(self, out) -> None:
        self._record_progress.setVisible(False)
        self._btn_record.setEnabled(True)
        if out is None:
            self._state.log(self.tr("Recording cancelled."), "warning")
        else:
            self._state.log(self.tr("Animation saved: {path}").format(path=out), "success")

    def _on_record_failed(self, message: str) -> None:
        self._record_progress.setVisible(False)
        self._btn_record.setEnabled(True)
        self._state.log(self.tr("Recording failed: {msg}").format(msg=message), "error")

    def _on_record(self) -> None:
        if self._state.results is None:
            return
        dialog = RecordDialog(self.animation_spec(), self)
        if not headless() and dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        default = str(self._state.output_dir / f"view3d_{self.anim_kind_key()}_{self._state.display_field}.{values['format']}")
        if headless():
            path = default
        else:
            filters = {"gif": "GIF (*.gif)", "mp4": "MP4 (*.mp4)", "png": "Folder"}
            if values["format"] == "png":
                path = QFileDialog.getExistingDirectory(self, self.tr("Folder for the PNG frames"), str(self._state.output_dir))
            else:
                path, _ = QFileDialog.getSaveFileName(self, self.tr("Save animation"), default, filters[values["format"]])
        if path:
            self.record(path, **values)

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
            self.azimuth,
            self.elevation,
            self.zoom,
            self._btn_reset_camera,
            self.anim_kind,
            self.anim_axis,
            self.anim_direction,
            self.anim_speed,
            self._btn_play,
            self._btn_stop,
        ):
            w.setEnabled(has)
        self._btn_record.setEnabled(has and (self._recorder is None or not self._recorder.isRunning()))
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
        self._labels["azimuth"].setText(self.tr("Turn"))
        self._labels["elevation"].setText(self.tr("Tilt"))
        self._labels["zoom"].setText(self.tr("Zoom"))
        self._labels["anim_kind"].setText(self.tr("Animate"))
        self._labels["anim_speed"].setText(self.tr("Speed"))
        self._btn_reset_camera.setText(self.tr("Reset"))
        self._btn_reset_camera.setToolTip(self.tr("Back to the untouched camera preset"))
        self.azimuth.setToolTip(self.tr("Turn the camera about the vertical axis [degrees]"))
        self.elevation.setToolTip(self.tr("Tilt the camera up or down [degrees]"))
        self.zoom.setToolTip(self.tr("Zoom factor on the preset framing"))
        retranslate_combo(self.anim_kind, "animation")
        retranslate_combo(self.anim_direction, "direction")
        self.anim_kind.setToolTip(
            self.tr(
                "Orbit: the camera turns about the chosen axis. Frames: the result frames play in sequence. "
                "Slice sweep: one field slice moves through the volume. "
                "Deformed lattice: the lattice grows to the warp scale and back."
            )
        )
        self.anim_axis.setToolTip(self.tr("Axis turned about (orbit) or swept along (slice)"))
        self.anim_speed.setToolTip(self.tr("Degrees, frames, voxels or cycles per second"))
        self._btn_play.setToolTip(self.tr("Pause") if self._playing else self.tr("Play"))
        self._btn_stop.setToolTip(self.tr("Stop and return to the start"))
        self._btn_record.setToolTip(self.tr("Record the animation as GIF, MP4 or PNG frames..."))
        self.arrows.setText(self.tr("Arrows"))
        self.volume_slices.setText(self.tr("Volume slices"))
        self.outline.setText(self.tr("Outline"))
        self._btn_refresh.setToolTip(self.tr("Refresh"))
        self._btn_shot.setToolTip(self.tr("Screenshot..."))
        retranslate_combo(self.mode, "view3d_mode")
        retranslate_combo(self.camera, "camera")
        retranslate_combo(self.background, "background")
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
