"""Camera specs, animation frames and recordings of the 3-D view."""

import os

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402

from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.gui.view3d_animation import (  # noqa: E402
    KINDS,
    AnimationSpec,
    frame_at,
    frames,
    mp4_available,
    record_animation,
)
from al_dvc.gui.view3d_scene import CameraSpec, SceneOptions, apply_camera, build_scene, render_image, ui_font_file  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402


@pytest.fixture(scope="module")
def result():
    shape = (36, 40, 44)
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=8)
    d1 = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, 0.0, 0.0]), (0.3, 0.0, 0.0), centre))
    d2 = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.02, 0.0, 0.0]), (0.6, 0.0, 0.0), centre))
    para = dvcpara_default(winsize=12, winstepsize=6, search_radius=3, admm_max_iter=1, verbose=False)
    return run_aldvc(para, [ref, d1, d2], compute_strain=False), ref


def test_camera_spec_turns_the_view(result):
    res, _ref = result
    with pytest.raises(ValueError):
        CameraSpec(preset="top")
    with pytest.raises(ValueError):
        CameraSpec(view_up="w")
    with pytest.raises(ValueError):
        CameraSpec(zoom=0)
    base, _ = render_image(res, SceneOptions(mode="points"), window_size=(200, 160), camera=CameraSpec())
    turned, _ = render_image(res, SceneOptions(mode="points"), window_size=(200, 160), camera=CameraSpec(azimuth=90))
    same, _ = render_image(res, SceneOptions(mode="points"), window_size=(200, 160), camera="iso")
    assert np.abs(base.astype(int) - turned.astype(int)).mean() > 1.0
    assert np.array_equal(base, same)  # the plain preset equals the default spec
    tilted, _ = render_image(res, SceneOptions(mode="points"), window_size=(200, 160), camera=CameraSpec(elevation=40, zoom=1.5))
    assert np.abs(base.astype(int) - tilted.astype(int)).mean() > 1.0
    pl = pv.Plotter(off_screen=True, window_size=(120, 90))
    build_scene(pl, res, SceneOptions(mode="points"), None)
    apply_camera(pl, CameraSpec(view_up="y", azimuth=30))
    assert tuple(np.round(pl.camera.up, 3)) != (0.0, 0.0, 1.0)
    pl.close()


def test_scalar_bar_uses_a_real_font_file(result):
    res, _ref = result
    font = ui_font_file()
    assert font is not None and os.path.isfile(font)
    pl = pv.Plotter(off_screen=True, window_size=(200, 160))
    build_scene(pl, res, SceneOptions(mode="slices", title="Displacement magnitude"), None)
    (_name, bar), *_ = pl.scalar_bars.items()
    from vtkmodules.vtkCommonCore import VTK_FONT_FILE

    assert bar.GetTitleTextProperty().GetFontFamily() == VTK_FONT_FILE
    assert bar.GetLabelTextProperty().GetFontFile() == font
    pl.close()


def test_animation_spec_and_frames(result):
    res, _ref = result
    with pytest.raises(ValueError):
        AnimationSpec(kind="spin")
    with pytest.raises(ValueError):
        AnimationSpec(kind="orbit", speed=0.0)
    with pytest.raises(ValueError):
        AnimationSpec(direction=2)
    with pytest.raises(ValueError):
        AnimationSpec(format="avi")
    turn = AnimationSpec.one_turn(90.0, fps=10)
    assert turn.duration == 4.0 and turn.n_frames == 40
    base_cam = CameraSpec(azimuth=10.0)
    base = SceneOptions(mode="slices", frame=1, slice_index={"z": 10, "y": None, "x": 5}, warp_scale=4.0)
    shape = res.volume_shape
    n = len(res.result_disp)
    # orbit: the azimuth advances at the speed, in the chosen direction, about the chosen axis
    f = frame_at(AnimationSpec(kind="orbit", axis="y", speed=90.0, direction=-1), 1.0, base_cam, base, n, shape)
    assert f.camera.azimuth == pytest.approx((10.0 - 90.0) % 360) and f.camera.view_up == "y" and f.options == base
    # frames: wraps around the result frames
    f = frame_at(AnimationSpec(kind="frames", speed=2.0), 1.0, base_cam, base, n, shape)
    assert f.options.frame == (1 + 2) % n and f.camera == base_cam
    # slice: sweeps the chosen axis through the volume and wraps
    f = frame_at(AnimationSpec(kind="slice", axis="z", speed=20.0), 1.0, base_cam, base, n, shape)
    assert f.options.slice_index["z"] == (10 + 20) % shape[0] and f.options.slice_index["x"] == 5
    f = frame_at(AnimationSpec(kind="slice", axis="y", speed=20.0), 0.5, base_cam, base, n, shape)
    assert f.options.slice_index["y"] == (shape[1] // 2 + 10) % shape[1]  # an unset slice starts in the middle
    # warp: a triangle wave from 0 to the scale and back
    spec = AnimationSpec(kind="warp", speed=1.0)
    assert frame_at(spec, 0.0, base_cam, base, n, shape).options.warp_scale == 0.0
    assert frame_at(spec, 0.5, base_cam, base, n, shape).options.warp_scale == pytest.approx(4.0)
    assert frame_at(spec, 0.25, base_cam, base, n, shape).options.warp_scale == pytest.approx(2.0)
    seq = list(frames(AnimationSpec(kind="orbit", speed=60.0, fps=5, duration=2.0), base_cam, base, n, shape))
    assert len(seq) == 10 and seq[0].time == 0.0 and seq[-1].time == pytest.approx(1.8)
    assert all(k in KINDS for k in ("orbit", "frames", "slice", "warp"))


def test_record_gif_and_png(result, tmp_path):
    res, ref = result
    spec = AnimationSpec(kind="orbit", speed=120.0, fps=4, duration=1.0, size="view", format="gif")
    calls = []
    out = record_animation(
        res,
        None,
        spec,
        CameraSpec(),
        SceneOptions(mode="points"),
        tmp_path / "orbit",
        window_size=(160, 120),
        progress=lambda f, m: calls.append(f),
    )
    assert out is not None and out.suffix == ".gif" and out.is_file()
    with Image.open(out) as im:
        assert im.n_frames == spec.n_frames == 4 and im.size == (160, 120)
    assert calls[-1] == pytest.approx(1.0)
    stopped = record_animation(
        res, None, spec, CameraSpec(), SceneOptions(mode="points"), tmp_path / "x", window_size=(160, 120), stop=lambda: True
    )
    assert stopped is None
    png = AnimationSpec(kind="slice", axis="z", speed=40.0, fps=3, duration=1.0, format="png")
    folder = record_animation(
        res,
        ref,
        png,
        CameraSpec(),
        SceneOptions(mode="slices", show_volume_slices=True),
        tmp_path / "slices",
        window_size=(160, 120),
    )
    assert folder is not None and folder.is_dir() and len(list(folder.glob("frame_*.png"))) == 3


@pytest.mark.skipif(not mp4_available(), reason="imageio-ffmpeg is not installed")
def test_record_mp4(result, tmp_path):
    res, _ref = result
    spec = AnimationSpec(kind="frames", speed=2.0, fps=4, duration=1.0, format="mp4")
    out = record_animation(
        res, None, spec, CameraSpec(), SceneOptions(mode="points"), tmp_path / "frames", window_size=(160, 120)
    )
    assert out is not None and out.suffix == ".mp4" and out.stat().st_size > 1000


# --------------------------------------------------------------------------- the panel (static backend, offscreen)
def _pump(n=10):
    from PySide6.QtWidgets import QApplication

    for _ in range(n):
        QApplication.processEvents()


def test_panel_camera_controls_playback_and_recording(result, tmp_path):
    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow, create_application

    create_application(["pytest"])
    res, ref = result
    window = MainWindow()
    window.show()
    window.state.set_volume_arrays([ref, ref, ref], ["a", "b", "c"])
    window.state.set_output_dir(tmp_path)
    window.state.set_results(res)
    window.center_tabs.setCurrentIndex(1)
    panel = window.view3d
    _pump()
    assert panel.backend == "static" and panel._last_image is not None
    before = panel._last_image.copy()
    # the camera row turns the static render
    panel.azimuth.setValue(60)
    _pump()
    assert panel.camera_spec().azimuth == 60.0
    assert np.abs(panel._last_image.astype(int) - before.astype(int)).mean() > 0.5
    panel.reset_camera()
    _pump()
    assert panel.camera_spec() == CameraSpec(preset="iso")
    assert np.array_equal(panel._last_image, before)
    # animation controls follow the kind
    from al_dvc.gui.names import select_key

    assert select_key(panel.anim_kind, "orbit") and panel.anim_axis.isVisibleTo(panel)
    spec = panel.animation_spec()
    assert spec.kind == "orbit" and spec.duration == pytest.approx(360.0 / spec.speed)
    assert select_key(panel.anim_kind, "warp") and not panel.anim_axis.isVisibleTo(panel)
    assert panel.animation_spec().speed == 1.0
    select_key(panel.anim_kind, "orbit")
    panel.anim_speed.setValue(180.0)
    # play, pause, stop: playing advances the animation clock and renders frames; stop restores the view
    panel.toggle_play()
    assert panel.playing
    QApplication.processEvents()
    panel._play_timer.stop()  # drive the ticks by hand so the test does not depend on wall-clock timing
    panel._play_offset = 0.5
    panel._on_play_tick()
    _pump()
    assert np.abs(panel._last_image.astype(int) - before.astype(int)).mean() > 0.5
    panel.toggle_play()
    assert not panel.playing and panel.play_time >= 0.5
    panel.stop_animation()
    _pump()
    assert not panel.playing and panel.play_time == 0.0 and panel._play_base is None
    assert np.array_equal(panel._last_image, before)
    # recording (headless: no dialogs, default path) writes a GIF of the requested length
    worker = panel.record(tmp_path / "turn", format="gif", fps=4, duration=1.0, size="view")
    assert worker is not None and panel.wait_recording(300_000)
    _pump()
    out = tmp_path / "turn.gif"
    assert out.is_file()
    with Image.open(out) as im:
        assert im.n_frames == 4
    assert panel._btn_record.isEnabled() and not panel._record_progress.isVisible()
    # the dialog-driven path (headless) uses the default file name
    panel._on_record()
    assert panel.wait_recording(300_000)
    _pump()
    assert any(p.suffix == ".gif" and p.name.startswith("view3d_orbit") for p in tmp_path.iterdir())
    window.close()
