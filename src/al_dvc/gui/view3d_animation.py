"""Animations of the 3-D view: what changes from frame to frame, and how the frames are written.

Four kinds, all described by an :class:`AnimationSpec` so the live playback in the panel and the
off-screen recording produce the same sequence:

``orbit``
    the camera turns about the view-up axis (``axis``) at ``speed`` degrees per second;
``frames``
    the reference state (no displacement) and the result frames play one after another at
    ``speed`` frames per second; with ``smooth`` the displacement and the field are interpolated
    between consecutive frames, so the deformed lattice moves like the real deformation;
``slice``
    one of the three field slices sweeps through the volume at ``speed`` voxels per second.

Recording renders every frame off-screen at the requested size and writes a GIF (always), an
MP4 (when ``imageio`` with ffmpeg is installed) or a folder of PNGs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .view3d_scene import CameraState, SceneOptions, render_image

KINDS = ("orbit", "frames", "slice")
FORMATS = ("gif", "mp4", "png")
MAX_FRAMES = {"gif": 600, "mp4": 3600, "png": 3600}  # a GIF is assembled in memory; MP4 and PNG stream to disk
SIZES = {"view": None, "hd": (1280, 960), "full": (1920, 1440)}
DEFAULT_SPEEDS = {"orbit": 30.0, "frames": 2.0, "slice": 20.0}  # degrees/s, frames/s, voxels/s
SPEED_RANGES = {"orbit": (1.0, 360.0), "frames": (0.05, 30.0), "slice": (1.0, 500.0)}
SPEED_STEPS = {"orbit": 5.0, "frames": 0.1, "slice": 5.0}  # spin-box increments; a slow-motion frame takes 1 / speed seconds


def mp4_available() -> bool:
    """True when ``imageio`` can write MP4 through ffmpeg."""
    try:
        import imageio.v3  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class AnimationSpec:
    """One animation: its kind, axis, direction, speed and, for recording, length, rate, size and format."""

    kind: str = "orbit"
    axis: str = "z"  # orbit: the view-up axis turned about; slice: the slice moved
    direction: int = 1  # +1 or -1
    speed: float = 30.0
    fps: int = 20
    duration: float = 6.0  # seconds; an orbit of one full turn takes 360 / speed seconds
    size: str = "view"
    format: str = "gif"
    loop: bool = True
    smooth: bool = False  # frames: interpolate displacement and field between consecutive frames

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be x, y or z, got {self.axis!r}")
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")
        lo, hi = SPEED_RANGES[self.kind]
        if not lo <= self.speed <= hi:
            raise ValueError(f"speed for {self.kind} must be in [{lo}, {hi}], got {self.speed}")
        if self.fps < 1 or self.duration <= 0:
            raise ValueError("fps must be >= 1 and duration positive")
        if self.format not in FORMATS or self.size not in SIZES:
            raise ValueError(f"format must be one of {FORMATS} and size one of {tuple(SIZES)}")
        if self.n_frames > MAX_FRAMES[self.format]:
            raise ValueError(
                f"{self.n_frames} frames: a {self.format.upper()} recording is limited to {MAX_FRAMES[self.format]} frames"
            )

    @property
    def n_frames(self) -> int:
        return max(2, int(round(self.fps * self.duration)))

    @staticmethod
    def max_duration(fmt: str, fps: int) -> float:
        """The longest recording of ``fmt`` at ``fps`` frames per second."""
        return MAX_FRAMES[fmt] / max(1, int(fps))

    @staticmethod
    def one_turn(speed: float, **kw) -> AnimationSpec:
        """An orbit whose recording is exactly one full turn."""
        return AnimationSpec(kind="orbit", speed=speed, duration=360.0 / speed, **kw)


@dataclass(frozen=True)
class Frame:
    """What one frame of an animation shows: the camera and the scene options."""

    index: int
    time: float
    camera: object  # CameraSpec or CameraState
    options: SceneOptions


def frame_at(spec: AnimationSpec, t: float, base_camera, base_options: SceneOptions, n_result_frames: int, shape) -> Frame:
    """The frame of ``spec`` at time ``t`` (seconds), starting from ``base_camera`` (a :class:`CameraSpec` or the
    live :class:`CameraState`) and ``base_options``.

    ``shape`` is the volume shape ``(nz, ny, nx)`` for slice sweeps; ``n_result_frames`` bounds the
    frame animation. Everything wraps, so playback can run for as long as the user likes.
    """
    camera = base_camera
    options = base_options
    if spec.kind == "orbit":
        angle = spec.direction * spec.speed * t
        if isinstance(base_camera, CameraState):
            camera = base_camera.rotated(azimuth=angle, view_up=spec.axis)
        else:
            camera = replace(base_camera, view_up=spec.axis, azimuth=(base_camera.azimuth + angle) % 360.0)
    elif spec.kind == "frames":
        n = max(1, int(n_result_frames))
        length = n + 1  # the reference state (frame -1, no displacement) followed by the result frames
        start = int(base_options.frame) + 1  # sequence index of the frame the animation started from
        pos = (start + spec.direction * spec.speed * t) % length
        k = int(np.floor(pos))
        # fraction of the way to the next frame; the last frame is held, then the loop restarts at the
        # reference state (no interpolation from the final deformation back to zero)
        blend = float(pos - k) if spec.smooth and k < length - 1 else 0.0
        options = replace(base_options, frame=k - 1, blend=blend)
    else:  # slice
        nz, ny, nx = (int(v) for v in shape)
        length = {"x": nx, "y": ny, "z": nz}[spec.axis]
        start = base_options.slice_index.get(spec.axis)
        start = int(start) if start is not None else length // 2
        pos = int(np.floor(start + spec.direction * spec.speed * t)) % max(1, length)
        options = replace(base_options, slice_index={**base_options.slice_index, spec.axis: pos})
    return Frame(int(round(t * spec.fps)), float(t), camera, options)


def frames(spec: AnimationSpec, base_camera, base_options: SceneOptions, n_result_frames: int, shape) -> Iterator[Frame]:
    """The recorded sequence: ``n_frames`` frames from ``t = 0`` to just before ``duration``."""
    n = spec.n_frames
    for i in range(n):
        yield frame_at(spec, i / spec.fps, base_camera, base_options, n_result_frames, shape)


def frame_bytes(spec: AnimationSpec, window_size: tuple[int, int]) -> int:
    """Raw RGB bytes of every frame of ``spec`` (what a GIF holds in memory before it is written)."""
    w, h = SIZES[spec.size] or window_size
    return int(spec.n_frames) * int(w) * int(h) * 3


def record_animation(
    result,
    volume,
    spec: AnimationSpec,
    base_camera,
    base_options: SceneOptions,
    path: str | Path,
    window_size: tuple[int, int] = (900, 700),
    progress: Callable[[float, str], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> Path | None:
    """Render ``spec`` off-screen and write it to ``path`` (extension follows ``spec.format``).

    ``volume`` is the array shown as volume slices, or one array per result frame (a sequence) so a
    frames animation shows every field on its own volume. Returns the written path, or ``None`` when
    stopped. PNG writes ``path`` as a folder of ``frame_0001.png`` ... files; MP4 is streamed frame by
    frame; a GIF is assembled in memory (hence its frame limit).
    """
    from PIL import Image

    out = Path(path)
    size = SIZES[spec.size] or window_size
    shape = tuple(result.volume_shape)
    n_result_frames = len(result.result_disp)
    images: list[np.ndarray] = []
    total = spec.n_frames
    writer = None
    if spec.format == "png":
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = out.with_suffix(f".{spec.format}")
        out.parent.mkdir(parents=True, exist_ok=True)
    if spec.format == "mp4":
        import imageio.v2 as iio2  # checked by mp4_available() before the dialog offers it

        writer = iio2.get_writer(str(out), fps=spec.fps, codec="libx264", macro_block_size=1)
    try:
        for frame in frames(spec, base_camera, base_options, n_result_frames, shape):
            if stop is not None and stop():
                return None
            vol = volume[frame.options.frame] if isinstance(volume, (list, tuple)) else volume
            img, _info = render_image(result, frame.options, vol, window_size=size, camera=frame.camera)
            if spec.format == "png":
                Image.fromarray(img).save(out / f"frame_{frame.index + 1:04d}.png")
            elif writer is not None:
                writer.append_data(np.ascontiguousarray(img[..., :3]))
            else:
                images.append(img)
            if progress is not None:
                progress((frame.index + 1) / total, f"{frame.index + 1}/{total}")
    finally:
        if writer is not None:
            writer.close()
    if spec.format in ("png", "mp4"):
        return out
    pil = [Image.fromarray(im).convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for im in images]
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=int(round(1000 / spec.fps)), loop=0 if spec.loop else 1)
    return out
