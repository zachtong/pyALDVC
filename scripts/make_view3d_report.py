"""Visual report of the 3-D view: every mode, arrows, volume slices, cameras, timings.

    python scripts/make_view3d_report.py [--out reports/view3d.pdf] [--quick]

The scenes are rendered off-screen with the same code the GUI panel uses
(``al_dvc.gui.view3d_scene``), on a synthetic speckle pair with a known
affine deformation, so the pictures can be checked against the imposed
field (u grows along x, the warped grid stretches along x and shrinks
along y). The iso-surface pages use a rigid rotation about z, whose |u|
grows with the distance from the axis: its iso-surfaces are nested cylinders.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc import __version__  # noqa: E402
from al_dvc.core.config import dvcpara_default  # noqa: E402
from al_dvc.core.pipeline import run_aldvc  # noqa: E402
from al_dvc.gui.view3d_scene import (  # noqa: E402
    CAMERAS,
    MODES,
    CameraSpec,
    SceneOptions,
    build_scene,
    facing_quadrant,
    node_grid,
    render_image,
)
from al_dvc.synthetic import (  # noqa: E402
    affine_displacement,
    generate_speckle_volume,
    rotation_displacement,
    warp_volume_lagrangian,
)

SHAPE_FULL = (96, 104, 112)
SHAPE_QUICK = (48, 52, 56)
F = np.array([[0.03, 0.005, 0.0], [0.0, -0.015, 0.003], [0.002, 0.0, 0.01]])
T = (1.2, -0.8, 0.5)
ROTATION_DEG = 8.0  # the iso-surface pages: rigid rotation about the z axis through the volume centre
ISO_LEVELS = 5


@contextmanager
def transparent_nan_everywhere():
    """The drawing of 1.0.1, for the before / after page: every field mesh with a transparent NaN colour, which put
    even a NaN-free mesh into VTK's translucent pass at opacity 1."""
    import pyvista as pv

    original = pv.Plotter.add_mesh

    def add_mesh(self, mesh, *args, **kwargs):
        if "clim" in kwargs:  # the field meshes (arrows, outlines and volume slices pass no colour range)
            kwargs.setdefault("nan_opacity", 0.0)
        return original(self, mesh, *args, **kwargs)

    pv.Plotter.add_mesh = add_mesh
    try:
        yield
    finally:
        pv.Plotter.add_mesh = original


def translucent_pass(result, opts: SceneOptions) -> bool:
    """Whether VTK draws the field of ``opts`` in its translucent pass (decided once the mapper has run)."""
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=(160, 120))
    try:
        actor = build_scene(pl, result, opts, None).actors["field"]
        actor.mapper.Update()
        return bool(actor.HasTranslucentPolygonalGeometry())
    finally:
        pl.close()


def _page_text(pdf, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, title, fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.90, "\n".join(lines), fontsize=9.5, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _page_images(pdf, title: str, items: list[tuple[str, np.ndarray]], ncols: int = 2) -> None:
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 3.6 * nrows + 0.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (label, img) in zip(axes, items):
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "view3d.pdf"))
    ap.add_argument("--quick", action="store_true", help="smaller volume (fast iteration)")
    args = ap.parse_args(argv)
    shape = SHAPE_QUICK if args.quick else SHAPE_FULL
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=21)
    dfm = warp_volume_lagrangian(ref, affine_displacement(F, T, centre))
    para = dvcpara_default(winsize=16 if args.quick else 24, winstepsize=8 if args.quick else 12, search_radius=6, verbose=False)
    t0 = time.perf_counter()
    res = run_aldvc(para, [ref, dfm])
    run_time = time.perf_counter() - t0
    mid = {"z": shape[0] // 2, "y": shape[1] // 2, "x": shape[2] // 2}
    size = (640, 480)

    timings: list[tuple[str, float]] = []

    def render(label: str, opts: SceneOptions, camera="iso", volume=None, result=None):
        t = time.perf_counter()
        img, info = render_image(res if result is None else result, opts, volume, window_size=size, camera=camera)
        timings.append((label, time.perf_counter() - t))
        text = f"{label}\n{info.field}: {info.n_finite}/{info.n_nodes} nodes, [{info.clim[0]:.3g}, {info.clim[1]:.3g}]"
        if len(info.iso_levels) > 1:
            text += "\nlevels " + ", ".join(f"{v:.3g}" for v in info.iso_levels)
        return text, img

    modes = [render(f"mode = {m}", SceneOptions(mode=m, field="disp_u", slice_index=mid, warp_scale=5.0)) for m in MODES]
    extras = [
        render(
            "arrows (stride 2) on slices", SceneOptions(field="disp_magnitude", show_arrows=True, arrow_stride=2, slice_index=mid)
        ),
        render(
            "volume slices + field slices", SceneOptions(field="disp_v", show_volume_slices=True, slice_index=mid), volume=ref
        ),
        render("exx iso-surface at 50 %", SceneOptions(mode="surface", field="exx", iso_fraction=0.5)),
        render(
            "warped x5 with arrows, points",
            SceneOptions(mode="points", field="disp_w", show_arrows=True, arrow_stride=3, slice_index=mid),
        ),
    ]
    cams = [
        render(f"camera = {c}", SceneOptions(field="disp_u", show_arrows=True, arrow_stride=3, slice_index=mid), camera=c)
        for c in CAMERAS
    ]

    # animation frames: an orbit and a slice sweep, as the panel plays and records them
    from al_dvc.gui.view3d_animation import AnimationSpec, frames

    anim_pages = []
    for spec, label, opts in (
        (
            AnimationSpec(kind="orbit", speed=60.0, fps=1, duration=6.0),
            "orbit about z, 60 deg/s",
            SceneOptions(field="disp_u", mode="slices", slice_index=mid, show_arrows=True, arrow_stride=3),
        ),
        (
            AnimationSpec(kind="slice", axis="z", speed=shape[0] / 6.0, fps=1, duration=6.0),
            "slice sweep along z",
            SceneOptions(field="disp_u", mode="slices", slice_index={**mid, "z": 0}),
        ),
        (
            AnimationSpec(kind="frames", speed=0.5, smooth=True, fps=1, duration=6.0),
            "frames with smooth deformation (reference state to the last frame, x5)",
            SceneOptions(field="disp_u", mode="warped", warp_scale=5.0),
        ),
    ):
        items = []
        t = time.perf_counter()
        for fr in frames(spec, CameraSpec(), opts, len(res.result_disp), shape):
            img, _info = render_image(res, fr.options, None, window_size=(320, 240), camera=fr.camera)
            items.append((f"{label}, t = {fr.time:.0f} s", img))
        timings.append((f"animation: {label} (6 frames, 320x240)", time.perf_counter() - t))
        anim_pages.append((label, items))

    # several iso-surfaces and the cut-away, on a rigid rotation about z (nested cylinders of |u|), on a denser grid
    rotated = warp_volume_lagrangian(ref, rotation_displacement(ROTATION_DEG, "z", centre))
    para_rot = dvcpara_default(
        winsize=16 if args.quick else 24, winstepsize=4 if args.quick else 6, search_radius=6, verbose=False
    )
    t0 = time.perf_counter()
    res_rot = run_aldvc(para_rot, [ref, rotated], compute_strain=False)
    rot_time = time.perf_counter() - t0
    xmin, xmax, ymin, ymax, _zmin, _zmax = node_grid(res_rot, 0).bounds
    u_in = 2.0 * 0.5 * min(xmax - xmin, ymax - ymin) * np.sin(np.radians(ROTATION_DEG) / 2.0)  # |u| at the grid's edge
    iso_clim = (0.0, 0.95 * u_in * (ISO_LEVELS + 1) / ISO_LEVELS)  # the outermost level just inside the node grid
    iso = SceneOptions(mode="surface", field="disp_magnitude", clim=iso_clim, title="|u| [voxel]")
    nested = replace(iso, iso_levels=ISO_LEVELS)
    cut = replace(nested, iso_cutaway=True)
    turned = CameraSpec(azimuth=180.0)
    paper = CameraSpec(azimuth=-95.0, elevation=-5.0)
    iso_items = [
        render("Surfaces = 1 (level at 50 %)", iso, result=res_rot),
        render(f"Surfaces = {ISO_LEVELS}", nested, result=res_rot),
        render("cut (1, 1), isometric camera", cut, result=res_rot),
        render("same cut, camera turned 180 deg", cut, camera=turned, result=res_rot),
        render(
            f"cut facing that camera {facing_quadrant(turned)}",
            replace(cut, cutaway_quadrant=facing_quadrant(turned)),
            camera=turned,
            result=res_rot,
        ),
        render(
            f"turn -95, tilt -5: cut {facing_quadrant(paper)}",
            replace(cut, cutaway_quadrant=facing_quadrant(paper)),
            camera=paper,
            result=res_rot,
        ),
    ]
    # opacity 1 before and after the fix: 1.0.1 gave every field a transparent NaN colour
    lattice = SceneOptions(field="disp_u", mode="warped", warp_scale=5.0)
    with transparent_nan_everywhere():
        before = [
            render("1.0.1: deformed lattice x5", lattice),
            render(f"1.0.1: {ISO_LEVELS} surfaces", nested, result=res_rot),
        ]
        passes_before = {m: translucent_pass(res, SceneOptions(mode=m)) for m in MODES}
    after = [render("now: deformed lattice x5", lattice), render(f"now: {ISO_LEVELS} surfaces", nested, result=res_rot)]
    passes_after = {m: translucent_pass(res, SceneOptions(mode=m)) for m in MODES}
    iso_lines = (
        [
            f"Rigid rotation by {ROTATION_DEG:g} degrees about the z axis through the volume centre:",
            "|u| = 2 r sin(a/2) grows with the distance r from the axis, so its iso-surfaces are nested",
            f"cylinders. Subset {para_rot.winsize[0]}, step {para_rot.winstepsize[0]}: {res_rot.dvc_mesh.n_nodes} nodes,"
            f" run {rot_time:.1f} s; colour range 0 to {iso_clim[1]:.3g} voxel.",
            "",
            "Surfaces = n (SceneOptions.iso_levels, 1 to 10): one surface lies at the iso level, a fraction",
            "of the colour range; n > 1 surfaces lie at lo + k/(n+1) (hi - lo), k = 1..n, in one mesh. The",
            "contour carries each level as its field value, so every surface takes the colour of its level",
            "on the colour bar.",
            "",
            "Cut away a quarter (iso_cutaway): the quarter sign(x - cx), sign(y - cy) = cutaway_quadrant",
            "about the centre of the node grid is clipped from every surface, exactly at the two planes.",
            "The quarter is an option, not read from the plotter (the scene is built before the camera is",
            "pointed): (1, 1) faces the isometric preset, facing_quadrant(camera) the one facing any other",
            "camera. The panel takes the camera row (or the mouse-turned camera, when the mouse is",
            "released); an animation keeps its quarter, so the cut turns with the object in an orbit.",
            "",
            "Opaque fields: 1.0.1 drew every field with a transparent NaN colour, which puts the whole",
            "mesh into VTK's translucent pass, NaN or not, even at opacity 1: nested surfaces and the far",
            "faces of the deformed lattice blended through the near ones. Only the field slices keep it",
            "(they hold NaN outside the region of interest; their pictures are unchanged, pixel for pixel).",
            "",
            f"{'Translucent pass at opacity 1, mode':<40s} {'1.0.1':<7s} now",
        ]
        + [f"  {m:<38s} {str(passes_before[m]):<7s} {passes_after[m]}" for m in MODES]
        + [
            "",
            "Limitations: the levels follow the colour range, not the data (a level outside the field draws",
            "nothing; Auto range is the 1st to 99th percentile). The cut splits at the centre of the node",
            "grid in x and y only (an axis elsewhere is not followed; z is never cut). The surfaces come",
            "from the node lattice (coarse for large steps). Below opacity 1 the surfaces are translucent",
            "and VTK blends them in drawing order. The view is unlit (pyvista's clear() removes the lights):",
            "each surface shows exactly its colour-bar colour; depth shows through occlusion and the cut.",
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        lines = (
            [
                f"pyALDVC {__version__} -- 3-D view (al_dvc.gui.view3d_scene, pyvista)",
                "",
                "The result's node lattice is a pyvista ImageData (node spacing = voxel spacing,",
                "first node = origin); node order n = iz*ny*nx + iy*nx + ix is VTK point order,",
                "so per-node arrays attach without reordering. Non-converged nodes carry NaN: the",
                "slices leave them transparent, the other modes leave them out. Modes: orthogonal",
                "slices of the field at the slice-viewer positions, node points, one or more",
                "iso-surfaces over the colour range (a quarter can be cut away; see the last pages),",
                "and the lattice warped by the displacement (scaled). Arrows are displacement glyphs on a",
                f"strided subset (cap {20000} arrows). Volume slices show the image data at the",
                "same positions as grey planes.",
                "",
                f"Synthetic pair {shape[::-1]} (x,y,z), F = diag-ish {np.diag(F).round(3).tolist()}, t = {T}.",
                f"Subset {para.winsize[0]}, step {para.winstepsize[0]}: {res.dvc_mesh.n_nodes} nodes, run {run_time:.1f} s.",
                "",
                "Expected pictures: u (disp_u) increases along x (colour ramps along x on the XY",
                "and XZ slices); the warped grid (x5) stretches along x and shrinks along y;",
                "arrows point mostly along +x on the right half and -x on the left half.",
                "",
                "Render timings (off-screen, 640x480):",
            ]
            + [f"  {label:<42s} {dt * 1000:7.0f} ms" for label, dt in timings]
            + [
                "",
                "In the application the same scenes are drawn in an embedded pyvistaqt interactor",
                "(mouse rotation); without an OpenGL context (offscreen tests, some remote",
                "desktops) the panel falls back to this off-screen rendering with camera presets.",
                "",
                "Limitations: fields are shown on the node lattice, not interpolated into the",
                "volume; the iso-surface uses the lattice too (coarse for large steps); the",
                "volume slices are subsampled above 4 Mpixel per plane; huge node counts (>1e6)",
                "make the warped-grid mode slow.",
            ]
        )
        _page_text(pdf, "3-D view", lines)
        _page_images(pdf, "Modes (field disp_u, warp x5)", modes)
        _page_images(pdf, "Arrows, volume slices, iso-surface, points", extras)
        _page_images(pdf, "Camera presets (disp_u with arrows)", cams)
        for label, items in anim_pages:
            _page_images(pdf, f"Animation frames: {label}", items, ncols=3)
        _page_text(pdf, "Several iso-surfaces, cut-away, opaque fields", iso_lines)
        _page_images(pdf, "Iso-surfaces of |u| of a rotation about z", iso_items, ncols=3)
        _page_images(pdf, "Opacity 1: 1.0.1 (left) and now (right)", [before[0], after[0], before[1], after[1]])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
