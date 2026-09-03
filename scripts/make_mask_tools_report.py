"""Visual report of mask drawing on the slice viewer: shapes, depth rules, timings, effect on a run.

    python scripts/make_mask_tools_report.py [--out reports/mask_tools.pdf] [--quick]

Pages: (1) description and rasterisation timings at several volume sizes;
(2) the viewer after a sequence of drawing operations (captured offscreen);
(3) a DVC run on a synthetic pair with a drawn mask: node validity and the
displacement error inside the material.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from al_dvc import __version__  # noqa: E402
from al_dvc.gui.mask_editor import MaskOp, rasterise  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE_FULL = (96, 104, 112)
SHAPE_QUICK = (48, 52, 56)
F = np.diag([0.02, -0.01, 0.01])
T = (0.8, -0.5, 0.3)


def _timings(sizes) -> list[str]:
    lines = []
    for n in sizes:
        shape = (n, n, n)
        ops = [
            ("rectangle, all slices", MaskOp("rectangle", "xy", ((n * 0.1, n * 0.1), (n * 0.8, n * 0.7)))),
            (
                "ellipse, 10-slice range",
                MaskOp("ellipse", "xz", ((n * 0.2, n * 0.2), (n * 0.9, n * 0.8)), depth=(n // 2, n // 2 + 9)),
            ),
            (
                "polygon (6 vertices), all slices",
                MaskOp(
                    "polygon",
                    "yz",
                    tuple((n * a, n * b) for a, b in [(0.1, 0.1), (0.5, 0.05), (0.9, 0.3), (0.8, 0.9), (0.4, 0.95), (0.05, 0.5)]),
                ),
            ),
            (
                "brush (40 points, r=4), one slice",
                MaskOp(
                    "brush",
                    "xy",
                    tuple((n * 0.1 + k * n * 0.02, n * 0.5 + 3 * np.sin(k / 3)) for k in range(40)),
                    depth=(n // 2, n // 2),
                    radius=4.0,
                ),
            ),
        ]
        for label, op in ops:
            t = time.perf_counter()
            rasterise(op, shape)
            dt = (time.perf_counter() - t) * 1000
            lines.append(f"  {n}^3 voxels  {label:<34s} {dt:8.1f} ms")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "mask_tools.pdf"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    shape = SHAPE_QUICK if args.quick else SHAPE_FULL
    nz, ny, nx = shape
    centre = tuple((s - 1) / 2 for s in shape[::-1])
    ref = generate_speckle_volume(shape, sigma=2.0, seed=31)
    dfm = warp_volume_lagrangian(ref, affine_displacement(F, T, centre))

    from PySide6.QtWidgets import QApplication

    from al_dvc.gui.app import MainWindow, create_application

    _app = create_application([sys.argv[0]])  # keeps Qt alive for the window
    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    window.state.set_volume_arrays([ref, dfm], ["reference", "deformed"])
    window.viewer.canvas.draw()
    shots: list[tuple[str, np.ndarray]] = []

    def grab(title: str) -> None:
        for _ in range(10):
            QApplication.processEvents()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            name = fh.name
        window.viewer.grab().save(name)
        shots.append((title, mpimg.imread(name)))
        os.unlink(name)

    st = window.state
    st.set_mask_display(target="all")
    ops = [
        ("1. ellipse (add, all slices) on XY", MaskOp("ellipse", "xy", ((nx * 0.08, ny * 0.1), (nx * 0.92, ny * 0.9)))),
        (
            "2. rectangle (cut, all slices) on XY",
            MaskOp("rectangle", "xy", ((nx * 0.4, ny * 0.4), (nx * 0.6, ny * 0.6)), mode="cut"),
        ),
        (
            "3. polygon (cut, z range) on XZ",
            MaskOp(
                "polygon",
                "xz",
                ((nx * 0.1, nz * 0.1), (nx * 0.5, nz * 0.05), (nx * 0.3, nz * 0.45)),
                depth=(ny // 3, 2 * ny // 3),
                mode="cut",
            ),
        ),
        (
            "4. brush (add, current slice) on YZ",
            MaskOp(
                "brush",
                "yz",
                tuple((ny * 0.2 + k * ny * 0.03, nz * 0.8 + 2 * np.sin(k / 2)) for k in range(20)),
                depth=(nx // 2, nx // 2),
                radius=3.0,
            ),
        ),
    ]
    for title, op in ops:
        st.apply_mask_op(op)
        grab(title)
    coverage = st.mask_editor.coverage
    window.viewer.mask_tools.set_tool("polygon")  # the toolbar as the user sees it while drawing
    grab("5. toolbar with the polygon tool selected")

    # run with the drawn mask
    st.set_params(winsize=16 if args.quick else 24, winstepsize=8 if args.quick else 12, search_radius=6, verbose=False)
    with tempfile.TemporaryDirectory(prefix="pyaldvc_mask_report_") as tmp:
        st.set_output_dir(tmp)
        st.write_checkpoints = False
        t0 = time.perf_counter()
        window.run_panel.start()
        window.run_panel.wait(600_000)
        run_time = time.perf_counter() - t0
        for _ in range(20):
            QApplication.processEvents()
        st.set_display(display_field="disp_u")
        grab("6. result overlay (disp_u) inside the drawn mask")
    res = st.results
    mask = st.current_mask().copy()
    window.close()

    mesh = res.dvc_mesh
    fr = res.result_disp[0]
    fn = affine_displacement(F, T, centre)
    truth = fn(mesh.coordinates[:, 0], mesh.coordinates[:, 1], mesh.coordinates[:, 2])
    truth = np.stack(truth, axis=-1) if isinstance(truth, (tuple, list)) else np.asarray(truth)
    truth = truth.reshape(mesh.n_nodes, 3)
    valid = mesh.node_valid
    err = np.linalg.norm(fr.U[valid] - truth[valid], axis=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        lines = [
            f"pyALDVC {__version__} -- mask drawing on the slice viewer (al_dvc.gui.mask_editor, panels/mask_tools.py)",
            "",
            "A mask (True = material) is built from 2-D shapes drawn on the XY, XZ or YZ slice and",
            "extruded along the plane's normal through all slices, the current slice or a range.",
            "Shapes: rectangle, ellipse, polygon, brush; each in add (union) or cut (subtract) mode;",
            "plus invert / fill / clear. The mask is a base plus a replayable list of operations:",
            "undo drops the last one and replays; sessions store the operations (and the saved",
            "mask file) so drawings survive reloads. The excluded region is tinted red on the",
            "slices; 'Apply to' targets the current frame or every frame; 'Save mask...' writes a",
            "uint8 volume the pipeline accepts like any mask file.",
            "",
            f"Synthetic pair {shape[::-1]} (x,y,z); four operations -> material {100 * coverage:.1f} % of the voxels.",
            f"Run with the drawn mask on both frames: {mesh.n_nodes} nodes, {int(valid.sum())} valid, {run_time:.1f} s.",
        ]
        if True:
            lines.append(
                f"Displacement error at the valid nodes: median {np.median(err):.4f}, 95 % {np.percentile(err, 95):.4f} voxel."
            )
        lines += ["", "Rasterisation timings (one operation, full volume):"] + _timings(
            [128, 256] if args.quick else [128, 256, 512]
        )
        lines += [
            "",
            "Limitations: shapes are prisms along one axis (no free-form 3-D surfaces); the brush",
            "paints the plane it is drawn on (extruded like the other shapes); very large volumes",
            "replay every operation on undo (bounded by folding old operations into the base).",
        ]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, "Mask drawing", fontsize=15, weight="bold", va="top")
        fig.text(0.05, 0.90, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        for i in range(0, len(shots), 2):
            chunk = shots[i : i + 2]
            fig, axes = plt.subplots(len(chunk), 1, figsize=(8.5, 4.2 * len(chunk) + 0.5))
            for ax, (title, img) in zip(np.atleast_1d(axes), chunk):
                ax.imshow(img)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        # node validity vs mask on the mid XY plane
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
        iz = nz // 2
        axes[0].imshow(ref[iz], cmap="gray", origin="lower")
        axes[0].imshow(
            np.ma.masked_where(mask[iz], np.ones_like(mask[iz], dtype=float)), cmap="autumn", alpha=0.35, origin="lower"
        )
        kz = int(np.argmin(np.abs(mesh.z0 - iz)))
        vg = mesh.to_grid(valid)[kz]
        xx, yy = np.meshgrid(mesh.x0, mesh.y0)
        axes[0].scatter(xx[vg], yy[vg], s=6, c="lime", label="valid node")
        axes[0].scatter(xx[~vg], yy[~vg], s=6, c="red", label="dropped node")
        axes[0].set_title(f"XY z = {iz}: mask (red tint) and node validity", fontsize=9)
        axes[0].legend(fontsize=7, loc="lower right")
        if True:
            axes[1].hist(err, bins=40, color="#4c72b0")
            axes[1].set_title("displacement error at valid nodes (voxel)", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
