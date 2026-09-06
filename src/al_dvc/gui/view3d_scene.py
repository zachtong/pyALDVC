"""Build pyvista scenes from DVC results (no Qt: usable for tests, reports and scripts).

The node grid of a result is a regular lattice, so it maps one-to-one onto a
``pyvista.ImageData`` with the node spacing as voxel spacing and the first
node as origin. Node ordering ``n = iz*ny*nx + iy*nx + ix`` is exactly VTK's
point ordering (x fastest), so per-node arrays attach without reordering.

``pyvista`` is imported lazily: the module can be imported (and
:func:`available` queried) on installations without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from al_dvc.core.data_structures import PipelineResult
from al_dvc.export.export_utils import displacement_physical, field_array

if TYPE_CHECKING:  # pragma: no cover
    import pyvista as pv

MODES = ("slices", "points", "surface", "warped")
DISPLACEMENT_ARRAY = "displacement"
MAX_ARROWS = 20_000
ARROW_SCALE_FRACTION = 0.6  # longest arrow = this fraction of the node spacing at unit arrow_scale
VOLUME_SLICE_MAX_PIXELS = 1_000_000  # subsample larger image slices before turning them into textures
VOLUME_SLICE_OPACITY = 0.9  # the field slices behind a volume slice stay faintly visible
VOLUME_SLICE_PERCENTILES = (0.5, 99.5)  # grey-level window of the volume slices
PLANE_NORMAL = {"xy": "z", "xz": "y", "yz": "x"}  # plane name -> the axis it is normal to (the slice's axis)
CAMERAS = ("iso", "xy", "xz", "yz")
VIEW_UPS = {"z": (0.0, 0.0, 1.0), "y": (0.0, 1.0, 0.0), "x": (1.0, 0.0, 0.0)}
SCENE_COLUMNS = (6, 1)  # relative widths of the scene renderer and the colour-bar renderer
BACKGROUND = "#1b1d23"
BACKGROUNDS = {"dark": BACKGROUND, "black": "#000000", "grey": "#808080", "white": "#ffffff"}


@dataclass(frozen=True)
class CameraSpec:
    """A camera as the panel and the animations describe it: a preset, then turned and zoomed.

    ``azimuth`` turns the camera about the view-up axis (degrees), ``elevation`` tilts it, ``zoom``
    multiplies the preset's framing; ``view_up`` is the axis kept vertical (the axis an orbit
    turns about).
    """

    preset: str = "iso"
    azimuth: float = 0.0
    elevation: float = 0.0
    zoom: float = 1.0
    view_up: str = "z"

    def __post_init__(self) -> None:
        if self.preset not in CAMERAS:
            raise ValueError(f"preset must be one of {CAMERAS}, got {self.preset!r}")
        if self.view_up not in VIEW_UPS:
            raise ValueError(f"view_up must be one of {tuple(VIEW_UPS)}, got {self.view_up!r}")
        if self.zoom <= 0:
            raise ValueError("zoom must be positive")


@dataclass(frozen=True)
class CameraState:
    """A camera as it is: position, focal point, view-up and projection (what a mouse drag leaves behind).

    Animations rotate it about its focal point and recordings start from it, so a recording begins
    exactly where the user turned the view to.
    """

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    view_angle: float = 30.0
    parallel_scale: float = 1.0
    parallel: bool = False

    @classmethod
    def from_camera(cls, cam) -> "CameraState":
        return cls(
            tuple(float(v) for v in cam.position),
            tuple(float(v) for v in cam.focal_point),
            tuple(float(v) for v in cam.up),
            float(cam.view_angle),
            float(cam.parallel_scale),
            bool(cam.parallel_projection),
        )

    def apply_to(self, cam) -> None:
        cam.position = self.position
        cam.focal_point = self.focal_point
        cam.up = self.view_up
        cam.view_angle = self.view_angle
        cam.parallel_projection = self.parallel
        cam.parallel_scale = self.parallel_scale

    def rotated(
        self, azimuth: float = 0.0, elevation: float = 0.0, view_up: str | None = None, dolly: float = 1.0
    ) -> "CameraState":
        """The same camera turned about its focal point (``view_up`` names the axis kept vertical)."""
        from vtkmodules.vtkRenderingCore import vtkCamera

        cam = vtkCamera()
        cam.SetPosition(*self.position)
        cam.SetFocalPoint(*self.focal_point)
        cam.SetViewUp(*(VIEW_UPS[view_up] if view_up is not None else self.view_up))
        cam.SetViewAngle(self.view_angle)
        cam.SetParallelProjection(self.parallel)
        cam.SetParallelScale(self.parallel_scale)
        if azimuth:
            cam.Azimuth(float(azimuth))
        if elevation:
            cam.Elevation(float(elevation))
            cam.OrthogonalizeViewUp()
        if dolly != 1.0:
            cam.Dolly(float(dolly))
        return CameraState(
            tuple(cam.GetPosition()),
            tuple(cam.GetFocalPoint()),
            tuple(cam.GetViewUp()),
            float(cam.GetViewAngle()),
            float(cam.GetParallelScale()),
            bool(cam.GetParallelProjection()),
        )

    def relative_to(self, preset: "CameraState") -> tuple[float, float, float]:
        """``(azimuth, elevation, zoom)`` that turn ``preset`` into this camera (roll is ignored).

        Used to show a mouse-dragged camera in the Turn / Tilt / Zoom boxes.
        """
        up = np.asarray(preset.view_up, dtype=float)
        up /= np.linalg.norm(up) or 1.0
        d0 = np.asarray(preset.position) - np.asarray(preset.focal_point)
        d1 = np.asarray(self.position) - np.asarray(self.focal_point)
        n0, n1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if n0 == 0 or n1 == 0:
            return 0.0, 0.0, 1.0
        e0 = float(np.degrees(np.arcsin(np.clip(np.dot(d0, up) / n0, -1.0, 1.0))))
        e1 = float(np.degrees(np.arcsin(np.clip(np.dot(d1, up) / n1, -1.0, 1.0))))
        a0 = d0 - np.dot(d0, up) * up
        a1 = d1 - np.dot(d1, up) * up
        azimuth = 0.0
        if np.linalg.norm(a0) > 1e-9 and np.linalg.norm(a1) > 1e-9:
            a0 /= np.linalg.norm(a0)
            a1 /= np.linalg.norm(a1)
            azimuth = float(np.degrees(np.arctan2(np.dot(np.cross(a0, a1), up), np.dot(a0, a1))))
        zoom = float(preset.parallel_scale / self.parallel_scale) if self.parallel and self.parallel_scale else float(n0 / n1)
        return azimuth, e1 - e0, zoom


def ui_font_file() -> str | None:
    """A TrueType file of the interface font for VTK text (VTK's built-in "arial" is not the system font).

    Segoe UI or Arial on Windows, Arial on macOS, else matplotlib's DejaVu Sans, which is always shipped.
    """
    import os
    import sys

    candidates = []
    if sys.platform == "win32":
        fonts = os.path.join(os.environ.get("WINDIR", r"C:\\Windows"), "Fonts")
        candidates += [os.path.join(fonts, "segoeui.ttf"), os.path.join(fonts, "arial.ttf")]
    elif sys.platform == "darwin":
        candidates += ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]
    try:
        import matplotlib

        candidates.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"))
    except Exception:  # matplotlib is a dependency, but stay safe
        pass
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


BAR_TITLE_WIDTH = 13  # characters per line of the colour bar title


def _wrap_title(title: str) -> str:
    """Break a long title at spaces so it fits over the narrow colour bar."""
    words = str(title).split()
    lines: list[str] = []
    for word in words:
        if lines and len(lines[-1]) + 1 + len(word) <= BAR_TITLE_WIDTH:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return chr(10).join(lines) if lines else str(title)


def scene_plotter(window_size: tuple[int, int], off_screen: bool = True):
    """An off-screen plotter with the scene renderer and the narrow colour-bar renderer side by side."""
    import pyvista as pv

    pl = pv.Plotter(off_screen=off_screen, shape=(1, 2), col_weights=list(SCENE_COLUMNS), border=False, window_size=window_size)
    pl.subplot(0, 0)
    return pl


def _use_ui_font(text_property, font_file: str | None) -> None:
    if font_file is None:
        return
    from vtkmodules.vtkCommonCore import VTK_FONT_FILE

    text_property.SetFontFamily(VTK_FONT_FILE)
    text_property.SetFontFile(font_file)


def foreground_for(background: str) -> str:
    """White text on a dark background, black on a light one (scalar bar, axes triad)."""
    try:
        h = background.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return "white"
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "white"


def import_error() -> str | None:
    """Why pyvista cannot be imported (``None`` when it can): shown by the self-test and the panel."""
    try:
        import pyvista  # noqa: F401
    except Exception as exc:  # ImportError, or a broken VTK install raising something else
        return f"{type(exc).__name__}: {exc}"
    return None


def available() -> bool:
    """True when pyvista can be imported."""
    return import_error() is None


@dataclass(frozen=True)
class SceneOptions:
    """What to draw. Defaults match the slice viewer's overlay."""

    field: str = "disp_magnitude"
    frame: int = 0  # result frame; -1 is the reference state (no displacement)
    blend: float = 0.0  # fraction of the way from ``frame`` to the next one (displacement and field interpolated)
    mode: str = "slices"
    colormap: str = "turbo"
    clim: tuple[float, float] | None = None
    opacity: float = 1.0
    warp_scale: float = 1.0
    show_arrows: bool = False
    arrow_stride: int = 2
    arrow_scale: float = 1.0
    show_outline: bool = True
    show_volume_slices: bool = False
    iso_fraction: float = 0.5
    slice_index: dict[str, int | None] = dc_field(default_factory=dict)
    slice_visible: dict[str, bool] = dc_field(default_factory=dict)  # per normal axis (z: XY, y: XZ, x: YZ); missing = shown
    background: str = BACKGROUND
    title: str | None = None  # scalar bar title; the field key when None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.arrow_stride < 1:
            raise ValueError("arrow_stride must be >= 1")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be within [0, 1]")
        if not 0.0 <= self.iso_fraction <= 1.0:
            raise ValueError("iso_fraction must be within [0, 1]")


@dataclass
class SceneInfo:
    """What :func:`build_scene` drew, for labels and tests."""

    field: str
    clim: tuple[float, float]
    n_nodes: int
    n_finite: int
    n_arrows: int = 0
    note: str = ""  # something the viewer should tell the user, e.g. a fallback drawing
    actors: dict[str, Any] = dc_field(default_factory=dict)


# ----------------------------------------------------------------------------- datasets
def _frame_arrays(result: PipelineResult, frame: int, fields: tuple[str, ...]):
    """Per-node field values and physical displacement of ``frame``; ``-1`` is the reference state (zeros at the
    valid nodes, NaN elsewhere)."""
    if frame < 0:
        valid = np.asarray(result.dvc_mesh.node_valid, dtype=bool)
        zeros = np.where(valid, 0.0, np.nan)
        return {name: zeros.copy() for name in fields}, np.zeros((valid.size, 3), dtype=np.float64)
    fr = result.result_disp[frame]
    values = {name: np.asarray(field_array(result, frame, name), dtype=np.float64) for name in fields}
    return values, np.asarray(displacement_physical(fr, result.dvc_para.voxel_size), dtype=np.float64)


def node_grid(
    result: PipelineResult, frame: int, fields: tuple[str, ...] = ("disp_magnitude",), blend: float = 0.0
) -> "pv.ImageData":
    """``ImageData`` over the node lattice with the requested per-node fields and the displacement vectors.

    ``frame`` -1 is the reference state; ``blend`` in (0, 1] interpolates linearly towards the next frame
    (after the last frame: back to the reference state), so a frames animation can deform continuously.
    """
    import pyvista as pv

    mesh = result.dvc_mesh
    nz, ny, nx = mesh.grid_shape
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=tuple(float(s) for s in mesh.spacing),
        origin=(float(mesh.x0[0]), float(mesh.y0[0]), float(mesh.z0[0])),
    )
    if grid.n_points != mesh.n_nodes:
        raise ValueError(f"node grid {mesh.grid_shape} has {grid.n_points} lattice points but the mesh has {mesh.n_nodes} nodes")
    values, U = _frame_arrays(result, frame, fields)
    if blend > 0.0:
        nxt = frame + 1 if frame + 1 < len(result.result_disp) else -1
        values2, U2 = _frame_arrays(result, nxt, fields)
        values = {name: (1.0 - blend) * values[name] + blend * values2[name] for name in fields}
        U = (1.0 - blend) * U + blend * U2
    for name in fields:
        grid.point_data[name] = values[name]
    grid.point_data[DISPLACEMENT_ARRAY] = np.nan_to_num(U)
    return grid


def volume_slice_planes(
    volume: NDArray, slice_index: dict[str, int | None], voxel_size=(1.0, 1.0, 1.0)
) -> dict[str, tuple["pv.PolyData", "pv.Texture"]]:
    """The XY, XZ and YZ planes of an image volume as textured quads: ``(quad, texture)`` per plane.

    A quad has four vertices whatever the image size; the grey levels travel as one texture, so
    a 1024 x 1024 slice costs the renderer a single image upload instead of a million-point
    surface (which was heavy enough to crash the interactive window on large scans). The window
    of grey levels is shared by the three planes (percentiles of their pixels).
    """
    import pyvista as pv

    vol = np.asarray(volume)
    nz, ny, nx = vol.shape
    sx, sy, sz = (float(v) for v in voxel_size)
    iz = int(np.clip(slice_index.get("z") if slice_index.get("z") is not None else nz // 2, 0, nz - 1))
    iy = int(np.clip(slice_index.get("y") if slice_index.get("y") is not None else ny // 2, 0, ny - 1))
    ix = int(np.clip(slice_index.get("x") if slice_index.get("x") is not None else nx // 2, 0, nx - 1))
    # (rows, cols) images: rows run along the plane's second world axis, columns along its first
    images = {"xy": vol[iz], "xz": vol[:, iy, :], "yz": vol[:, :, ix]}
    corners = {  # origin, end of the first (column) axis, end of the second (row) axis, in world units
        "xy": ((0.0, 0.0, iz * sz), ((nx - 1) * sx, 0.0, iz * sz), (0.0, (ny - 1) * sy, iz * sz)),
        "xz": ((0.0, iy * sy, 0.0), ((nx - 1) * sx, iy * sy, 0.0), (0.0, iy * sy, (nz - 1) * sz)),
        "yz": ((ix * sx, 0.0, 0.0), (ix * sx, (ny - 1) * sy, 0.0), (ix * sx, 0.0, (nz - 1) * sz)),
    }
    lo, hi = _grey_window(images.values())
    planes = {}
    for key, img in images.items():
        img = np.asarray(img, dtype=np.float32)
        step = int(np.ceil(np.sqrt(img.size / VOLUME_SLICE_MAX_PIXELS))) if img.size > VOLUME_SLICE_MAX_PIXELS else 1
        if step > 1:
            img = img[::step, ::step]
        grey = np.clip((img - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
        rgb = np.ascontiguousarray(np.repeat(grey[:, :, None], 3, axis=2))
        origin, point_u, point_v = (np.asarray(c, dtype=np.float64) for c in corners[key])
        quad = pv.PolyData(np.vstack([origin, point_u, point_u + point_v - origin, point_v]), faces=[4, 0, 1, 2, 3])
        quad.texture_map_to_plane(origin=origin, point_u=point_u, point_v=point_v, inplace=True)
        texture = pv.numpy_to_texture(rgb[::-1])  # pyvista puts row 0 at the top: flipped, row 0 sits at the origin
        texture.interpolate = True
        planes[key] = (quad, texture)
    return planes


def _grey_window(images) -> tuple[float, float]:
    """Shared grey-level window of the slices: percentiles of a sample of their pixels."""
    samples = []
    for img in images:
        flat = np.asarray(img, dtype=np.float32).ravel()
        samples.append(flat[:: max(1, flat.size // 200_000)])
    values = np.concatenate(samples)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(values, VOLUME_SLICE_PERCENTILES))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def auto_clim(values: NDArray) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


# ----------------------------------------------------------------------------- scene
def build_scene(plotter, result: PipelineResult, opts: SceneOptions, volume: NDArray | None = None) -> SceneInfo:
    """Clear ``plotter`` and draw the result according to ``opts``; returns what was drawn."""
    import pyvista as pv

    plotter.clear()
    frame = int(np.clip(opts.frame, -1, len(result.result_disp) - 1))
    grid = node_grid(result, frame, (opts.field,), blend=opts.blend)
    values = np.asarray(grid.point_data[opts.field])
    finite = np.isfinite(values)
    clim = opts.clim if opts.clim is not None else auto_clim(values)
    info = SceneInfo(field=opts.field, clim=clim, n_nodes=int(values.size), n_finite=int(finite.sum()))
    common = dict(scalars=opts.field, cmap=opts.colormap, clim=clim, nan_opacity=0.0, show_scalar_bar=False)
    fg = foreground_for(opts.background)
    plotter.set_background(opts.background)
    title = _wrap_title(opts.title or opts.field)
    own_renderer = len(plotter.renderers) > 1  # the bar lives in the narrow right renderer: it never overlaps the scene
    bar_x = 0.18 if own_renderer else min(0.90, max(0.60, 1.0 - 0.035 - 0.0085 * len(title)))
    scalar_bar = dict(
        title=title + chr(10),  # the newline keeps the title clear of the top label
        vertical=True,
        n_labels=5,
        fmt="%.3g",
        width=0.34 if own_renderer else 0.07,
        height=0.62 if own_renderer else 0.55,
        position_x=bar_x,
        position_y=0.16,
        title_font_size=12,
        label_font_size=10,
        unconstrained_font_size=True,
        font_family="arial",
        color=fg,
        shadow=False,
        italic=False,
        bold=False,
    )

    if opts.mode == "slices":
        x, y, z = _slice_positions(result, opts.slice_index)
        parts = [
            grid.slice(normal=normal, origin=(x, y, z))
            for axis, normal in (("z", (0.0, 0.0, 1.0)), ("y", (0.0, 1.0, 0.0)), ("x", (1.0, 0.0, 0.0)))
            if opts.slice_visible.get(axis, True)
        ]
        parts = [part for part in parts if part.n_points]  # a cut at the lattice edge can be empty: merging it drops the arrays
        if parts:
            sliced = parts[0] if len(parts) == 1 else parts[0].merge(parts[1:])
            info.actors["field"] = plotter.add_mesh(sliced, opacity=opts.opacity, **common)
    elif opts.mode == "points":
        idx = np.flatnonzero(finite)
        if idx.size:
            cloud = pv.PolyData(np.asarray(grid.points)[idx])
            cloud.point_data[opts.field] = values[idx]
            info.actors["field"] = plotter.add_mesh(
                cloud, style="points", point_size=8.0, render_points_as_spheres=True, opacity=opts.opacity, **common
            )
    elif opts.mode == "surface":
        lo, hi = clim
        level = lo + opts.iso_fraction * (hi - lo)
        filled = grid.copy()
        filled.point_data[opts.field] = np.where(finite, values, lo - 1.0)
        surf = filled.contour(isosurfaces=[level], scalars=opts.field)
        if surf.n_points:
            surf = surf.compute_normals(auto_orient_normals=True, consistent_normals=True)
            info.actors["field"] = plotter.add_mesh(surf, opacity=opts.opacity, smooth_shading=True, specular=0.3, **common)
        info.actors["iso_level"] = level
    else:  # warped: the lattice of the valid nodes moved by the displacement, drawn with its cell edges
        grid.point_data["_valid"] = finite.astype(np.float32)
        # only cells whose 8 nodes are valid: a cell with one NaN corner would be drawn fully transparent
        cells = grid.threshold(0.5, scalars="_valid", preference="point", all_scalars=True)
        if cells.n_cells:
            warped = cells.warp_by_vector(DISPLACEMENT_ARRAY, factor=opts.warp_scale)
            info.actors["field"] = plotter.add_mesh(
                warped, opacity=opts.opacity, show_edges=True, edge_color=fg, line_width=1, **common
            )
        elif finite.any():  # no complete cell (a thin region, a sparse field): the valid nodes, moved
            pts = np.asarray(grid.points)[finite] + opts.warp_scale * np.asarray(grid.point_data[DISPLACEMENT_ARRAY])[finite]
            import pyvista as pv

            cloud = pv.PolyData(pts)
            cloud.point_data[opts.field] = values[finite]
            info.actors["field"] = plotter.add_mesh(
                cloud, style="points", point_size=8.0, render_points_as_spheres=True, opacity=opts.opacity, **common
            )
            info.note = "nodes_only"
        if opts.show_outline:
            info.actors["original"] = plotter.add_mesh(grid.outline(), color="gray", line_width=1)

    if opts.show_arrows:
        info.n_arrows = _add_arrows(plotter, grid, finite, result, opts, info)
    if opts.show_outline:
        info.actors["outline"] = plotter.add_mesh(_volume_outline(result), color=fg, line_width=1, opacity=0.6)
    if opts.show_volume_slices and volume is not None:
        planes = volume_slice_planes(volume, opts.slice_index, result.dvc_para.voxel_size)
        for key, (quad, texture) in planes.items():
            if not opts.slice_visible.get(PLANE_NORMAL[key], True):
                continue
            info.actors[f"volume_{key}"] = plotter.add_mesh(
                quad, texture=texture, lighting=False, show_scalar_bar=False, opacity=VOLUME_SLICE_OPACITY
            )
    if "field" in info.actors:
        # bind the bar to the field's mapper explicitly: by default pyvista takes the last added mesh (outline, arrows)
        if own_renderer:
            plotter.subplot(0, 1)
        bar = plotter.add_scalar_bar(mapper=info.actors["field"].mapper, **scalar_bar)
        font_file = ui_font_file()
        _use_ui_font(bar.GetTitleTextProperty(), font_file)
        _use_ui_font(bar.GetLabelTextProperty(), font_file)
        if own_renderer:
            plotter.subplot(0, 0)
    _orientation_axes(plotter, fg)
    return info


def _orientation_axes(plotter, fg) -> None:
    """The orientation marker, created once per plotter and recoloured afterwards.

    ``plotter.add_axes`` destroys and recreates the marker widget; done on every rebuild of the
    interactive window (each animation frame, each control change) that left the render window with
    a dangling widget renderer now and then, and the next paint crashed inside VTK. The widget
    survives ``plotter.clear()``, so it is only created when the renderer has none."""
    import pyvista as pv

    renderer = plotter.renderer
    if getattr(renderer, "axes_widget", None) is None:
        plotter.add_axes(color=fg)
        return
    actor = getattr(renderer, "axes_actor", None)
    if actor is None:
        return
    rgb = pv.Color(fg).float_rgb
    for caption in (actor.GetXAxisCaptionActor2D(), actor.GetYAxisCaptionActor2D(), actor.GetZAxisCaptionActor2D()):
        caption.GetCaptionTextProperty().SetColor(*rgb)


def _slice_positions(result: PipelineResult, slice_index: dict[str, int | None]) -> tuple[float, float, float]:
    """World coordinates of the three orthogonal slices (physical units via voxel_size)."""
    mesh = result.dvc_mesh
    nz, ny, nx = result.volume_shape
    vs = np.asarray(result.dvc_para.voxel_size, dtype=np.float64)
    ix = slice_index.get("x") if slice_index.get("x") is not None else nx // 2
    iy = slice_index.get("y") if slice_index.get("y") is not None else ny // 2
    iz = slice_index.get("z") if slice_index.get("z") is not None else nz // 2
    x = float(np.clip(ix, mesh.x0[0], mesh.x0[-1]))
    y = float(np.clip(iy, mesh.y0[0], mesh.y0[-1]))
    z = float(np.clip(iz, mesh.z0[0], mesh.z0[-1]))
    return x * vs[0], y * vs[1], z * vs[2]


def _volume_outline(result: PipelineResult) -> "pv.PolyData":
    import pyvista as pv

    nz, ny, nx = result.volume_shape
    sx, sy, sz = (float(v) for v in result.dvc_para.voxel_size)
    box = pv.ImageData(dimensions=(2, 2, 2), spacing=((nx - 1) * sx, (ny - 1) * sy, (nz - 1) * sz), origin=(0.0, 0.0, 0.0))
    return box.outline()


def _add_arrows(plotter, grid, finite: NDArray[np.bool_], result: PipelineResult, opts: SceneOptions, info: SceneInfo) -> int:
    import pyvista as pv

    nz, ny, nx = result.dvc_mesh.grid_shape
    keep = np.zeros((nz, ny, nx), dtype=bool)
    s = opts.arrow_stride
    keep[::s, ::s, ::s] = True
    idx = np.flatnonzero(keep.ravel() & finite)
    if idx.size > MAX_ARROWS:
        idx = idx[:: int(np.ceil(idx.size / MAX_ARROWS))]
    if idx.size == 0:
        return 0
    vec = np.asarray(grid.point_data[DISPLACEMENT_ARRAY])[idx]
    mag = np.linalg.norm(vec, axis=1)
    longest = float(mag.max()) if mag.size else 0.0
    if longest <= 0.0:
        return 0
    spacing = float(min(result.dvc_mesh.spacing)) * float(min(result.dvc_para.voxel_size))
    factor = ARROW_SCALE_FRACTION * spacing / longest * opts.arrow_scale
    cloud = pv.PolyData(np.asarray(grid.points)[idx])
    cloud.point_data["vec"] = vec
    cloud.point_data["mag"] = mag
    arrows = cloud.glyph(orient="vec", scale="mag", factor=factor)
    info.actors["arrows"] = plotter.add_mesh(arrows, scalars="mag", cmap=opts.colormap, show_scalar_bar=False)
    return int(idx.size)


def render_image(
    result: PipelineResult,
    opts: SceneOptions | None = None,
    volume: NDArray | None = None,
    window_size: tuple[int, int] = (900, 700),
    camera: str | CameraSpec = "iso",
    path=None,
) -> tuple[NDArray[np.uint8], SceneInfo]:
    """Render a scene off-screen; returns ``(rgb image (h, w, 3), info)`` and writes ``path`` when given.

    ``camera`` is a preset name or a :class:`CameraSpec`.
    """
    pl = scene_plotter(window_size)
    opts = opts or SceneOptions()
    pl.set_background(opts.background, all_renderers=True)
    info = build_scene(pl, result, opts, volume)
    _apply_camera(pl, camera)
    img = np.asarray(pl.screenshot(str(path) if path is not None else None, return_img=True))
    pl.close()
    return img[..., :3].copy(), info


def render_png(
    result: PipelineResult,
    path,
    opts: SceneOptions | None = None,
    volume: NDArray | None = None,
    window_size: tuple[int, int] = (900, 700),
    camera: str | CameraSpec = "iso",
) -> SceneInfo:
    """Render a scene off-screen to ``path``; the same code the panel's static backend uses."""
    return render_image(result, opts, volume, window_size, camera, path=path)[1]


def apply_camera(plotter, camera) -> None:
    """Point ``plotter``'s camera as ``camera`` says.

    A preset name or a :class:`CameraSpec` starts from the preset and turns, tilts and dollies it; a
    :class:`CameraState` is restored as it is.
    """
    if isinstance(camera, CameraState):
        camera.apply_to(plotter.camera)
        plotter.renderer.ResetCameraClippingRange()
        return
    spec = camera if isinstance(camera, CameraSpec) else CameraSpec(preset=str(camera))
    if spec.preset == "iso":
        plotter.view_isometric()
    else:
        getattr(plotter, f"view_{spec.preset}")()
    plotter.reset_camera()
    cam = plotter.camera
    if spec.view_up != "z":
        cam.up = VIEW_UPS[spec.view_up]
    if spec.azimuth:
        cam.Azimuth(float(spec.azimuth))
    if spec.elevation:
        cam.Elevation(float(spec.elevation))
        cam.OrthogonalizeViewUp()
    if spec.zoom != 1.0:
        cam.Dolly(float(spec.zoom))  # what the mouse wheel does: move the camera, not the view angle
    plotter.renderer.ResetCameraClippingRange()


_apply_camera = apply_camera  # the old private name
