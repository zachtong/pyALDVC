"""Visual report of the 3-D view: every mode, arrows, volume slices, cameras, timings.

    python scripts/make_view3d_report.py [--out reports/view3d.pdf] [--quick]

The scenes are rendered off-screen with the same code the GUI panel uses
(``al_dvc.gui.view3d_scene``), on a synthetic speckle pair with a known
affine deformation, so the pictures can be checked against the imposed
field (u grows along x, the warped grid stretches along x and shrinks
along y).
"""

from __future__ import annotations

import argparse
import sys
import time
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
from al_dvc.gui.view3d_scene import CAMERAS, MODES, SceneOptions, render_image  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE_FULL = (96, 104, 112)
SHAPE_QUICK = (48, 52, 56)
F = np.array([[0.03, 0.005, 0.0], [0.0, -0.015, 0.003], [0.002, 0.0, 0.01]])
T = (1.2, -0.8, 0.5)


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

    def render(label: str, opts: SceneOptions, camera: str = "iso", volume=None):
        t = time.perf_counter()
        img, info = render_image(res, opts, volume, window_size=size, camera=camera)
        timings.append((label, time.perf_counter() - t))
        return f"{label}\n{info.field}: {info.n_finite}/{info.n_nodes} nodes, [{info.clim[0]:.3g}, {info.clim[1]:.3g}]", img

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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        lines = (
            [
                f"pyALDVC {__version__} -- 3-D view (al_dvc.gui.view3d_scene, pyvista)",
                "",
                "The result's node lattice is a pyvista ImageData (node spacing = voxel spacing,",
                "first node = origin); node order n = iz*ny*nx + iy*nx + ix is VTK point order,",
                "so per-node arrays attach without reordering. Non-converged nodes carry NaN and",
                "are transparent. Modes: orthogonal slices of the field at the slice-viewer",
                "positions, node points, iso-surface at a fraction of the colour range, and the",
                "lattice warped by the displacement (scaled). Arrows are displacement glyphs on a",
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
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
