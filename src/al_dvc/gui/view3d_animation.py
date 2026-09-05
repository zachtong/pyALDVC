"""Animations of the 3-D view: what changes from frame to frame, and how the frames are written.

Four kinds, all described by an :class:`AnimationSpec` so the live playback in the panel and the
off-screen recording produce the same sequence:

``orbit``
    the camera turns about the view-up axis (``axis``) at ``speed`` degrees per second;
``frames``
    the result frames play one after another at ``speed`` frames per second;
``slice``
    one of the three field slices sweeps through the volume at ``speed`` voxels per second;
``warp``
    the deformed lattice grows from an undeformed lattice to the chosen warp scale and back.

Recording renders every frame off-screen at the requested size and writes a GIF (always), an
MP4 (when ``imageio`` with ffmpeg is installed) or a folder of PNGs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .view3d_scene import CameraState, SceneOptions, render_image

KINDS = ("orbit", "frames", "slice", "warp")
FORMATS = ("gif", "mp4", "png")
SIZES = {"view": None, "hd": (1280, 960), "full": (1920, 1440)}
DEFAULT_SPEEDS = {"orbit": 30.0, "frames": 2.0, "slice": 20.0, "warp": 1.0}  # deg/s, frames/s, voxel/s, cycles/s
SPEED_RANGES = {"orbit": (1.0, 360.0), "frames": (0.2, 30.0), "slice": (1.0, 500.0), "warp": (0.05, 5.0)}


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

    @property
    def n_frames(self) -> int:
        return max(2, int(round(self.fps * self.duration)))

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
        k = int(np.floor(spec.speed * t)) % n
        options = replace(base_options, frame=(base_options.frame + spec.direction * k) % n)
    elif spec.kind == "slice":
        nz, ny, nx = (int(v) for v in shape)
        length = {"x": nx, "y": ny, "z": nz}[spec.axis]
        start = base_options.slice_index.get(spec.axis)
        start = int(start) if start is not None else length // 2
        pos = int(np.floor(start + spec.direction * spec.speed * t)) % max(1, length)
        options = replace(base_options, slice_index={**base_options.slice_index, spec.axis: pos})
    else:  # warp: a triangle wave between 0 and the chosen scale
        phase = (spec.speed * t) % 1.0
        f = 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)
        options = replace(base_options, warp_scale=float(base_options.warp_scale) * f)
    return Frame(int(round(t * spec.fps)), float(t), camera, options)


def frames(spec: AnimationSpec, base_camera, base_options: SceneOptions, n_result_frames: int, shape) -> Iterator[Frame]:
    """The recorded sequence: ``n_frames`` frames from ``t = 0`` to just before ``duration``."""
    n = spec.n_frames
    for i in range(n):
        yield frame_at(spec, i / spec.fps, base_camera, base_options, n_result_frames, shape)


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

    Returns the written path, or ``None`` when stopped. PNG writes ``path`` as a folder of
    ``frame_0001.png`` ... files.
    """
    from PIL import Image

    out = Path(path)
    size = SIZES[spec.size] or window_size
    shape = tuple(result.volume_shape)
    n_result_frames = len(result.result_disp)
    images: list[np.ndarray] = []
    total = spec.n_frames
    if spec.format == "png":
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = out.with_suffix(f".{spec.format}")
        out.parent.mkdir(parents=True, exist_ok=True)
    for frame in frames(spec, base_camera, base_options, n_result_frames, shape):
        if stop is not None and stop():
            return None
        img, _info = render_image(result, frame.options, volume, window_size=size, camera=frame.camera)
        if spec.format == "png":
            Image.fromarray(img).save(out / f"frame_{frame.index + 1:04d}.png")
        else:
            images.append(img)
        if progress is not None:
            progress((frame.index + 1) / total, f"{frame.index + 1}/{total}")
    if spec.format == "png":
        return out
    if spec.format == "gif":
        pil = [Image.fromarray(im).convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for im in images]
        pil[0].save(out, save_all=True, append_images=pil[1:], duration=int(round(1000 / spec.fps)), loop=0 if spec.loop else 1)
        return out
    import imageio.v3 as iio  # mp4: checked by mp4_available() before the dialog offers it

    iio.imwrite(out, np.stack(images), fps=spec.fps, codec="libx264", macro_block_size=1)
    return out
