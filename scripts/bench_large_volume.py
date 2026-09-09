"""Measure what a large volume costs: mask editing, viewer redraws, node counting, solver memory.

Every number here is wall-clock on the calling thread, which for the GUI cases is the thread Qt would
be blocking. Peak allocations are ``tracemalloc`` peaks of the measured call alone, reported per voxel
so they extrapolate: multiply by the voxel count of the scan you care about.

Run from the repository root::

    python scripts/bench_large_volume.py --out reports/large_volume_before.json
    python scripts/bench_large_volume.py --sizes 128 192 256 --quick

The JSON it writes is the input of ``scripts/make_large_volume_report.py``; run it once before a
change and once after, and the report draws both.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DEFAULT_SIZES = (128, 192, 256, 320)
REPEATS = 3


# ----------------------------------------------------------------------------- helpers
def timed(fn, repeats: int = REPEATS) -> float:
    """Seconds per call, best of ``repeats`` (the best run is the one least disturbed by the OS)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def peak_bytes(fn) -> tuple[float, int]:
    """``(seconds, peak NumPy allocation in bytes)`` of one call."""
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dt, int(peak)


def speckle(shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """A cheap textured volume; the content does not matter for any timing here, only the size."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 4000, size=shape, dtype=np.uint16)
    return np.ascontiguousarray(v)


def sphere_mask(shape: tuple[int, int, int]) -> np.ndarray:
    """An inscribed sphere: a mask whose bounding box is not the whole volume."""
    nz, ny, nx = shape
    z, y, x = np.ogrid[:nz, :ny, :nx]
    r = min(shape) * 0.45
    return ((z - nz / 2) ** 2 + (y - ny / 2) ** 2 + (x - nx / 2) ** 2) <= r * r


# ----------------------------------------------------------------------------- benchmarks
def bench_mask_editor(shape: tuple[int, int, int]) -> dict:
    """The drawing model itself: construction, one operation of each kind, undo, and the reductions."""
    from al_dvc.gui.mask_editor import MaskEditor, MaskOp
    from al_dvc.texture.boxes import box_of_mask

    nz, ny, nx = shape
    base = np.ones(shape, dtype=bool)
    out: dict = {}
    out["construct_full_base_s"], peak = peak_bytes(lambda: MaskEditor(shape, base=base))
    out["construct_peak_bytes_per_voxel"] = peak / float(np.prod(shape))
    ed = MaskEditor(shape, base=base)
    whole = MaskOp("rectangle", plane="xy", points=((nx * 0.1, ny * 0.1), (nx * 0.9, ny * 0.9)), mode="add")
    one = MaskOp(
        "rectangle", plane="xy", points=((nx * 0.1, ny * 0.1), (nx * 0.9, ny * 0.9)), mode="add", depth=(nz // 2, nz // 2)
    )
    stroke = MaskOp(
        "brush",
        plane="xy",
        points=tuple((nx * 0.2 + k, ny * 0.5) for k in range(40)),
        radius=6.0,
        mode="add",
        depth=(nz // 2, nz // 2),
    )
    out["apply_rectangle_all_slices_s"] = timed(lambda: ed.apply(whole))
    out["apply_rectangle_one_slice_s"] = timed(lambda: ed.apply(one))
    out["apply_brush_one_slice_s"] = timed(lambda: ed.apply(stroke))
    out["undo_s"] = timed(lambda: (ed.undo(), ed.redo()))
    m = ed.mask
    out["box_of_mask_s"] = timed(lambda: box_of_mask(m))
    out["count_nonzero_s"] = timed(lambda: int(np.count_nonzero(m)))
    out["mask_sum_s"] = timed(lambda: float(m.sum()))
    out["mask_copy_s"] = timed(lambda: m.copy())
    return out


def bench_region_viewer(shape: tuple[int, int, int]) -> dict:
    """The texture window's slice viewer end to end: what the user feels while drawing the region."""
    from al_dvc.gui.app import create_application
    from al_dvc.gui.mask_editor import MaskOp
    from al_dvc.gui.region_viewer import RegionViewer

    app = create_application(["bench"])
    nz, ny, nx = shape
    vol = speckle(shape)
    viewer = RegionViewer()
    viewer.resize(1200, 400)
    viewer.show()
    for _ in range(5):
        app.processEvents()
    out: dict = {}
    out["set_volume_s"] = timed(lambda: viewer.set_volume(vol), repeats=1)
    for _ in range(5):
        app.processEvents()

    def redraw():
        viewer.redraw()
        viewer.canvas.draw()

    out["redraw_s"] = timed(redraw)
    rect = MaskOp("rectangle", plane="xy", points=((nx * 0.1, ny * 0.1), (nx * 0.9, ny * 0.9)), mode="replace")
    stroke = MaskOp(
        "brush",
        plane="xy",
        points=tuple((nx * 0.2 + k, ny * 0.5) for k in range(40)),
        radius=6.0,
        mode="add",
        depth=(nz // 2, nz // 2),
    )

    def apply_and_pump(op):
        viewer.apply(op)
        for _ in range(3):
            app.processEvents()

    out["draw_rectangle_s"] = timed(lambda: apply_and_pump(rect))
    out["draw_brush_one_slice_s"] = timed(lambda: apply_and_pump(stroke))

    def slide():
        viewer.set_slices(nz // 3, ny // 3, nx // 3)
        viewer.set_slices(nz // 2, ny // 2, nx // 2)

    out["slider_move_s"] = timed(slide)
    viewer.close()
    return out


def bench_texture_window(shape: tuple[int, int, int]) -> dict:
    """One drawing operation with the whole texture window attached (the case the user reported)."""
    from al_dvc.gui.app import MainWindow, create_application
    from al_dvc.gui.mask_editor import MaskOp

    app = create_application(["bench"])
    nz, ny, nx = shape
    vol = speckle(shape).astype(np.float32)
    window = MainWindow()
    tw = window.open_texture_window()
    tw.resize(1300, 850)
    tw.show()
    window.state.set_volume_arrays([vol], ["ref"])
    for _ in range(40):
        app.processEvents()
    tw.go_to_step(tw.TAB_SWEEP)
    for _ in range(20):  # let the one-off backend probe finish before timing anything
        app.processEvents()
    out: dict = {}
    calls = {"n": 0}
    from al_dvc.gui.region_viewer import RegionViewer

    real_box = RegionViewer.box

    def counting_box(self):
        calls["n"] += 1
        return real_box(self)

    RegionViewer.box = counting_box
    try:
        rects = [
            MaskOp("rectangle", plane="xy", points=((nx * 0.1 + i, ny * 0.1 + i), (nx * 0.9, ny * 0.9)), mode="replace")
            for i in range(REPEATS)
        ]

        def draw(i):
            tw.region.apply(rects[i])
            for _ in range(4):
                app.processEvents()

        best = float("inf")
        for i in range(REPEATS):
            calls["n"] = 0
            t0 = time.perf_counter()
            draw(i)
            best = min(best, time.perf_counter() - t0)
        out["draw_rectangle_s"] = best
        out["box_calls_per_edit"] = calls["n"]
    finally:
        RegionViewer.box = real_box
    tw.close()
    window.close()
    return out


def bench_node_counting(shape: tuple[int, int, int]) -> dict:
    """subset_valid_fraction: the per-node valid count that decides whether a masked run fits in memory."""
    from al_dvc.core.data_structures import VOIRange
    from al_dvc.mesh.grid_mesh import build_grid_axes, mesh_setup, subset_valid_fraction

    nz, ny, nx = shape
    mask = sphere_mask(shape).astype(np.uint8)
    winsize = (32, 32, 32)
    step = (16, 16, 16)
    voi = VOIRange(x=(0, nx - 1), y=(0, ny - 1), z=(0, nz - 1))
    x0, y0, z0 = build_grid_axes(voi, shape, winsize, step)
    mesh = mesh_setup(x0, y0, z0)
    out: dict = {}
    dt, peak = peak_bytes(lambda: subset_valid_fraction(mask, mesh.coordinates, winsize))
    out["subset_valid_fraction_s"] = dt
    out["subset_valid_fraction_peak_bytes_per_voxel"] = peak / float(np.prod(shape))
    out["n_nodes"] = int(mesh.n_nodes)
    return out


def bench_voi_and_lattice(shape: tuple[int, int, int]) -> dict:
    """The two full-volume passes the main viewer repeats on every redraw."""
    from al_dvc.core.data_structures import voi_from_mask
    from al_dvc.gui.lattice_preview import plan_lattice

    mask = sphere_mask(shape)
    out: dict = {}
    out["voi_from_mask_s"] = timed(lambda: voi_from_mask(mask, (32, 32, 32), 8))
    out["plan_lattice_cut_s"] = timed(lambda: plan_lattice(shape, (32, 32, 32), (16, 16, 16), None, mask))
    out["plan_lattice_nocut_s"] = timed(lambda: plan_lattice(shape, (32, 32, 32), (16, 16, 16), None, mask, cut=False))
    return out


def bench_reference_bundle(shape: tuple[int, int, int]) -> dict:
    """What one reference frame costs the solver, per voxel, in both gradient modes."""
    from al_dvc.io.volume_ops import build_reference_bundle, normalize_volume

    vol = normalize_volume(speckle(shape))
    n = float(np.prod(shape))
    out: dict = {}
    for mode in ("stored", "on_the_fly"):
        dt, peak = peak_bytes(lambda m=mode: build_reference_bundle(vol, None, gradient_mode=m))
        out[f"build_{mode}_s"] = dt
        out[f"build_{mode}_peak_bytes_per_voxel"] = peak / n
        bundle = build_reference_bundle(vol, None, gradient_mode=mode)
        resident = sum(int(getattr(bundle, k).nbytes) for k in ("f", "gx", "gy", "gz", "mask"))
        out[f"resident_{mode}_bytes_per_voxel"] = resident / n
        del bundle
    return out


BENCHMARKS = {
    "mask_editor": bench_mask_editor,
    "region_viewer": bench_region_viewer,
    "texture_window": bench_texture_window,
    "node_counting": bench_node_counting,
    "voi_and_lattice": bench_voi_and_lattice,
    "reference_bundle": bench_reference_bundle,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES), help="cubic edge lengths to measure")
    ap.add_argument("--only", nargs="+", choices=sorted(BENCHMARKS), help="run only these benchmarks")
    ap.add_argument("--quick", action="store_true", help="one repeat instead of three")
    ap.add_argument("--label", default="", help="a name for this run, e.g. 'before' or 'after'")
    ap.add_argument("--out", default=str(ROOT / "reports" / "large_volume_bench.json"))
    args = ap.parse_args(argv)
    global REPEATS
    if args.quick:
        REPEATS = 1
    names = args.only or list(BENCHMARKS)
    results: dict = {
        "label": args.label,
        "platform": f"{platform.system()} {platform.machine()} python {platform.python_version()} numpy {np.__version__}",
        "repeats": REPEATS,
        "sizes": {},
    }
    for edge in args.sizes:
        shape = (edge, edge, edge)
        n = int(np.prod(shape))
        print(f"\n{edge}^3 = {n / 1e6:.0f} M voxels ({2 * n / 1e9:.2f} GB as uint16)")
        entry: dict = {"voxels": n}
        for name in names:
            t0 = time.perf_counter()
            try:
                entry[name] = BENCHMARKS[name](shape)
            except Exception as exc:  # a benchmark that cannot run must not lose the others
                entry[name] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {name:<18s} FAILED: {type(exc).__name__}: {exc}")
                continue
            print(f"  {name:<18s} ({time.perf_counter() - t0:.1f} s)")
            for k, v in entry[name].items():
                if k.endswith("_s"):
                    print(f"      {k[:-2]:<40s} {1000 * v:9.2f} ms")
                elif "per_voxel" in k:
                    print(f"      {k:<40s} {v:9.2f} B/voxel  ({v * 1024**3 / 1e9:.0f} GB at 1024^3)")
                else:
                    print(f"      {k:<40s} {v:9}")
        results["sizes"][str(edge)] = entry
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
