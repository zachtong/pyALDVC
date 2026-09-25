"""Command-line interface: ``al-dvc run|synth|info|plot``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: top level must be a mapping")
    return cfg


def _progress_printer(frac: float, msg: str) -> None:
    sys.stdout.write(f"\r[{100 * frac:5.1f}%] {msg:<70}")
    sys.stdout.flush()
    if frac >= 1.0:
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from .core.config import dvcpara_default
    from .core.pipeline import run_aldvc
    from .export import (
        export_csv,
        export_mat,
        export_npz,
        export_report,
        export_run_summary,
        export_vtk,
    )
    from .io.volume_io import FileVolumeProvider, resolve_volume_paths

    cfg: dict = {}
    if args.config:
        cfg = _load_config(Path(args.config))
    para_kwargs = dict(cfg.get("para", {}))
    if args.winsize is not None:
        para_kwargs["winsize"] = args.winsize
    if args.step is not None:
        para_kwargs["winstepsize"] = args.step
    if args.no_global:
        para_kwargs["use_global_step"] = False
    for kv in args.set or []:
        key, _, val = kv.partition("=")
        para_kwargs[key.strip()] = json.loads(val) if val.strip() else True
    para = dvcpara_default(**para_kwargs)
    from .core.config import units_problem

    if units_problem(para):
        print(f"warning: {units_problem(para)}", file=sys.stderr)

    volumes = args.volumes or cfg.get("volumes")
    if not volumes:
        raise SystemExit("No volumes given (use --volumes or 'volumes:' in the config).")
    paths = (
        resolve_volume_paths(volumes if len(volumes) > 1 else volumes[0])
        if isinstance(volumes, list)
        else resolve_volume_paths(volumes)
    )
    masks = args.masks or cfg.get("masks")
    mask_paths = None
    if masks:
        mask_paths = (
            resolve_volume_paths(masks if isinstance(masks, str) or len(masks) > 1 else masks[0])
            if not isinstance(masks, list) or len(masks) != 1
            else resolve_volume_paths(masks[0])
        )
        if len(mask_paths) == 1 and len(paths) > 1:
            mask_paths = mask_paths * len(paths)
    out_dir = Path(args.output or cfg.get("output", "aldvc_results"))
    exports = args.export or cfg.get("export", ["npz", "report", "summary"])
    basename = cfg.get("basename", "aldvc")

    print(f"Volumes ({len(paths)}): {[p.name for p in paths]}")
    provider = FileVolumeProvider(paths, para.voi, mask_paths, load_kwargs=cfg.get("load_kwargs", {}))
    t0 = time.perf_counter()
    checkpoint = args.checkpoint or cfg.get("checkpoint")
    result = run_aldvc(
        para,
        provider,
        progress_fn=_progress_printer,
        compute_strain=not args.no_strain,
        checkpoint_dir=checkpoint,
        resume=not args.restart,
    )
    print(f"Done in {time.perf_counter() - t0:.1f}s; exporting to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if "npz" in exports:
        print("  ", export_npz(result, out_dir / f"{basename}.npz"))
    if "mat" in exports:
        print("  ", export_mat(result, out_dir / f"{basename}.mat"))
    if "csv" in exports:
        for p in export_csv(result, out_dir / "csv", basename):
            print("  ", p)
    if "vtk" in exports:
        for p in export_vtk(result, out_dir / "vtk", basename):
            print("  ", p)
    if "report" in exports:
        print("  ", export_report(result, out_dir / f"{basename}_report.pdf"))
    if "summary" in exports:
        print("  ", export_run_summary(result, out_dir / f"{basename}_summary.json"))
    return 0


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> int:
    from .io.volume_io import save_volume
    from .synthetic import (
        add_noise,
        affine_displacement,
        generate_speckle_volume,
        sinusoidal_displacement,
        warp_volume_lagrangian,
    )

    shape = tuple(int(s) for s in args.shape)
    ref = generate_speckle_volume(shape, sigma=args.sigma, seed=args.seed)
    c = tuple((s - 1) / 2 for s in shape[::-1])
    if args.mode == "translation":
        fn = affine_displacement(None, tuple(args.value), c)
    elif args.mode == "stretch":
        e = args.value[0]
        fn = affine_displacement(np.diag([e, -0.3 * e, -0.3 * e]), (0, 0, 0), c)
    elif args.mode == "shear":
        Fm = np.zeros((3, 3))
        Fm[0, 1] = args.value[0]
        fn = affine_displacement(Fm, (0, 0, 0), c)
    else:
        fn = sinusoidal_displacement(args.value[0], args.value[1] if len(args.value) > 1 else shape[-1] / 2, c)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frames = [ref]
    for k in range(1, args.frames + 1):
        scale = k / args.frames

        def scaled(x, y, z, fn=fn, scale=scale):
            u, v, w = fn(x, y, z)
            return scale * u, scale * v, scale * w

        g = warp_volume_lagrangian(ref, scaled)
        frames.append(g)
    for i, vol in enumerate(frames):
        if args.noise > 0:
            vol = add_noise(vol, args.noise, seed=100 + i)
        arr = np.clip(vol, 0, 1)
        if args.dtype == "uint8":
            arr = (arr * 255).astype(np.uint8)
        elif args.dtype == "uint16":
            arr = (arr * 65535).astype(np.uint16)
        save_volume(out / f"synth_{i:03d}.tif", arr)
    print(f"Wrote {len(frames)} volumes of shape {shape} to {out}")
    return 0


# ---------------------------------------------------------------------------
# info / plot
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    from .io.volume_io import volume_info

    for p in args.paths:
        info = volume_info(p)
        print(json.dumps(info, indent=2))
    return 0


def _sweep_edge(sweep) -> int | None:
    """The edge the sweep settled on: the first threshold that converged, else None."""
    if sweep is None or not sweep.levels:
        return None
    for d in sweep.decisions.values():
        if d.converged and d.start_index is not None:
            return int(max(sweep.levels[d.start_index].size))
    return None


def cmd_texture(args: argparse.Namespace) -> int:
    """Correlation lengths (and optionally the size sweep) of a volume, written as CSV, JSON and PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .gui.texture_window import write_profiles_csv, write_summary_json
    from .io.volume_io import load_volume
    from .texture import (
        THRESHOLD_LABELS,
        THRESHOLDS,
        analyse_cube,
        box_centre,
        box_of_mask,
        cube_box,
        cube_limits,
        normalise_box,
        recommend_parameters,
        sweep_concentric,
        whole_box,
    )

    vol = np.asarray(load_volume(args.volume))
    box = whole_box(vol.shape)
    mask = None
    if args.roi:
        mask = np.asarray(load_volume(args.roi)) > 0
        if mask.shape != vol.shape:
            print(f"error: region shape {mask.shape} does not match the volume shape {vol.shape}", file=sys.stderr)
            return 2
        box = box_of_mask(mask)  # the cubes stay inside the region's bounding box
        if mask.all():
            mask = None
    if args.region:
        x0, x1, y0, y1, z0, z1 = args.region
        box = ((x0, x1), (y0, y1), (z0, z1))
        mask = None
    try:
        box = normalise_box(box, vol.shape)
        centre = tuple(int(v) for v in args.centre) if args.centre else box_centre(box)
        limits = cube_limits(centre, box)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    spacing = tuple(args.spacing) if args.spacing else 1.0
    t0 = time.perf_counter()
    sweep = None
    if args.sweep:
        sweep = sweep_concentric(
            vol,
            centre,
            box,
            args.sweep_start,
            args.sweep_step,
            args.sweep_count,
            spacing=spacing,
            mask=mask,
            estimator=args.estimator,
            criterion=args.rve_criterion.replace("-", "_"),
            cv_window=args.cv_window,
        )
    edge = args.size if args.size else _sweep_edge(sweep) or min(limits)
    size = tuple(min(int(edge), int(limit)) for limit in limits)  # clipped per axis, as the window does
    try:
        result = analyse_cube(vol, cube_box(centre, size), spacing, mask, estimator=args.estimator)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rec = recommend_parameters(result) if result.status == "ok" else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_profiles_csv(result, out / "texture_profiles.csv")
    write_summary_json(result, sweep, rec, out / "texture_summary.json")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for axis, p in result.profiles.items():
        ax.plot(p.lag, p.mean, label=axis)
    for t in THRESHOLDS:
        ax.axhline(t, color="gray", lw=0.6)
    ax.set_xlabel("lag [voxel]")
    ax.set_ylabel("autocorrelation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "texture_profiles.png", dpi=150)
    plt.close(fig)
    size = " x ".join(str(v) for v in result.settings["size"])
    print(
        f"status: {result.status}  ({time.perf_counter() - t0:.1f} s, {size} voxel about "
        f"{centre}, {result.acf.n_voxels:,} voxels)"
    )
    for axis, table in result.lengths.items():
        cells = [
            f"{THRESHOLD_LABELS.get(t, t)}: {c.value:.2f}" if c.found else f"{THRESHOLD_LABELS.get(t, t)}: {c.status}"
            for t, c in table.items()
        ]
        print(f"  {axis:6s} " + "  ".join(cells))
    if rec is not None:
        print("suggested subset", " x ".join(str(e + 1) for e in rec.subset), "step", " x ".join(str(s) for s in rec.step))
        for note in rec.notes:
            print("  note:", note)
    if sweep is not None:
        for t, d in sweep.decisions.items():
            where = f"from size {sweep.levels[d.start_index].size}" if d.converged else f"not converged: {d.reason}"
            print(f"sweep {THRESHOLD_LABELS.get(t, t)}: {where}")
    print("written to", out)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .core.data_structures import DVCMesh
    from .export.export_npz import load_npz_result
    from .viz.slices import plot_field_slices

    d = load_npz_result(args.npz)
    grid = tuple(int(s) for s in d["grid_shape"])
    mesh = DVCMesh(
        coordinates=d["coordinates"],
        elements=d["elements"],
        grid_shape=grid,
        x0=d["x0"],
        y0=d["y0"],
        z0=d["z0"],
        spacing=tuple(float(s) for s in d["spacing"]),
        node_valid=d["node_valid"],
    )
    key = f"{args.field}_{args.frame}"
    if key not in d:
        if args.field in ("disp_u", "disp_v", "disp_w"):
            comp = "uvw".index(args.field[-1])
            arr = d[f"disp_phys_{args.frame}"][:, comp] if f"disp_phys_{args.frame}" in d else d[f"U_accum_{args.frame}"][:, comp]
        else:
            raise SystemExit(
                f"field '{args.field}' not found in {args.npz}; keys: {sorted(k for k in d if k.endswith(f'_{args.frame}'))}"
            )
    else:
        arr = d[key]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    plot_field_slices(arr.reshape(grid), mesh, title=f"{args.field} frame {args.frame}", fig=fig, axes=axes)
    out = args.output or f"{Path(args.npz).stem}_{args.field}_{args.frame}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)
    return 0


# ---------------------------------------------------------------------------


def _batch_progress(index: int, n: int, frac: float, msg: str) -> None:
    sys.stdout.write(f"\r[{index + 1}/{n} {100 * frac:5.1f}%] {msg:<64}")
    sys.stdout.flush()
    if frac >= 1.0:
        sys.stdout.write("\n")


def cmd_batch(args: argparse.Namespace) -> int:
    """Run several session files one after another (exit code 1 when any job failed)."""
    from .gui.batch import BatchRunner

    runner = BatchRunner(
        args.sessions,
        exports=tuple(args.export),
        checkpoints=not args.no_checkpoints,
        compute_strain=not args.no_strain,
        progress_fn=None if args.quiet else _batch_progress,
    )
    jobs = runner.run()
    print(BatchRunner.summary_table(jobs))
    for job in jobs:
        if job.traceback and args.verbose:
            print(job.traceback)
    return 0 if all(job.status == "done" for job in jobs) else 1


def _load_regions(path: str) -> list:
    """Regions from a JSON file: a list of regions, a statistics summary (``regions``) or a session (``analysis``)."""
    from .analysis.regions import regions_from_dicts
    from .io.session_bundle import is_bundle, read_session_document

    try:  # a session file (a zip bundle since format 3, JSON before) or a JSON file
        doc = read_session_document(path) if is_bundle(path) else json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read the regions in {path}: {exc}") from exc
    if isinstance(doc, dict):
        doc = (doc.get("analysis") or {}).get("regions", doc.get("regions")) if "analysis" in doc else doc.get("regions")
    try:
        return regions_from_dicts(doc if doc is not None else [])
    except ValueError as exc:
        raise SystemExit(f"invalid region in {path}: {exc}") from exc


def _named_region(regions: list, name: str | None, what: str):
    if name is None:
        return None
    for r in regions:
        if r.name == name:
            return r
    known = ", ".join(r.name for r in regions) or "none (use --regions)"
    raise SystemExit(f"{what}: no region named {name!r}; regions: {known}")


def cmd_stats(args: argparse.Namespace) -> int:
    """Statistics of an exported result: one row per frame (CSV), and a JSON summary."""
    from .analysis import MOTION_KINDS, NodeFilter, available_groups, frame_series, homogeneous, noise_floor
    from .analysis.export_stats import stats_metadata, write_region_series_csv, write_series_csv, write_summary_json
    from .analysis.fields import FIELD_GROUPS
    from .export.export_npz import result_from_npz

    if args.motion not in MOTION_KINDS:
        raise SystemExit(f"--motion must be one of {MOTION_KINDS}")
    result = result_from_npz(args.result)
    groups = available_groups(result)
    fields = list(args.fields) if args.fields else [f for g in ("displacement", "strain") if g in groups for f in FIELD_GROUPS[g]]
    known = {f for g in groups for f in FIELD_GROUPS[g]}
    unknown = [f for f in fields if f not in known]
    if unknown:
        raise SystemExit(f"not available in this result: {unknown} (available: {sorted(known)})")
    n_frames = len(result.result_disp)
    frames = [k - 1 for k in args.frames] if args.frames else list(range(n_frames))
    if any(not 0 <= k < n_frames for k in frames):
        raise SystemExit(f"frames are 1 to {n_frames}")
    nf = NodeFilter(
        converged_only=not args.all_nodes,
        drop_outliers=not args.all_nodes,
        min_zncc=args.min_zncc,
        edge_layers=args.edge_layers,
        drop_cut=args.drop_cut,
    )
    regions = _load_regions(args.regions) if args.regions else []
    region = _named_region(regions, args.region, "--region")
    fit_region = _named_region(regions, args.fit_region, "--fit-region")
    if args.compare and not regions:
        raise SystemExit("--compare needs regions (--regions FILE)")
    mask = None if region is None else region.node_mask(result)
    fit_mask = None if fit_region is None else fit_region.node_mask(result)
    targets = [None, *regions] if args.compare else [region]
    runs = []
    for r in targets:
        runs.append(
            (
                r,
                frame_series(
                    result,
                    fields,
                    nf,
                    args.motion,
                    frames,
                    region=None if r is None else r.node_mask(result),
                    fit_region=fit_mask,
                    with_ci=args.ci,
                ),
            )
        )
    meta = stats_metadata(
        result,
        nf,
        args.motion,
        region="every region" if args.compare else ("whole" if region is None else region.name),
        fit_region="whole" if fit_region is None else fit_region.name,
    )
    out = Path(args.out)
    if args.compare:
        named = [("All nodes" if r is None else r.name, s) for r, s in runs]
        csv_path = write_region_series_csv(out / "statistics.csv", named, fields, meta)
    else:
        csv_path = write_series_csv(out / "statistics.csv", runs[0][1], fields, meta)
    series = runs[0][1]
    payload: dict = {"meta": meta, "regions": [r.as_dict() for r in regions]}
    if args.compare:
        payload["frames_by_region"] = {("All nodes" if r is None else r.name): s for r, s in runs}
    else:
        payload["frames"] = series
    if args.noise_floor:
        payload["noise_floor"] = noise_floor(result, frames[0], nf, nominal=args.nominal, region=mask)
    fits = []
    for k in frames:  # a frame without enough nodes for an affine fit loses its own fit, not the others'
        try:
            fits.append(homogeneous(result, k, nf, region=mask))
        except ValueError as exc:
            fits.append(None)
            print(f"warning: frame {k + 1}: no homogeneous fit ({exc})", file=sys.stderr)
    payload["homogeneous"] = fits
    json_path = write_summary_json(out / "statistics.json", payload)
    unit = meta["unit"]
    for r, run in runs:
        prefix = "" if len(runs) == 1 and r is None else f"{'All nodes' if r is None else r.name}, "
        for fs in run:
            cells = []
            for name, st in fs.stats.items():
                ci = f" +- {st.ci95:.2g}" if args.ci and st.ci95 == st.ci95 else ""
                cells.append(f"{name} mean {st.mean:.4g}{ci} std {st.std:.4g}")
            print(f"{prefix}frame {fs.frame + 1}: {fs.selection.n} nodes ({unit}); " + "; ".join(cells))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


def _display_available() -> bool:
    """Whether a window can open here: not on Linux without X11 or Wayland (an SSH session on a cluster), where Qt
    would abort instead of starting."""
    if sys.platform.startswith("linux"):
        return any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"))
    return True


def _opens_window(argv: list[str]) -> bool:
    """``al-dvc`` alone, ``al-dvc session.aldvc`` and ``al-dvc --self-test`` belong to the application, as
    pyALDIC's ``al-dic`` opens its window; every command (``run``, ``batch``, ...) goes to the parser."""
    return not argv or argv[0] == "--self-test" or argv[0].lower().endswith(".aldvc")


def _launch_window(argv: list[str]) -> int:
    """Open the application with ``argv`` (a session file, or ``--self-test [report]``)."""
    if argv[:1] != ["--self-test"] and not _display_available():  # the self-test runs offscreen anywhere
        build_parser().print_help()
        print("\nThere is no display here, so the application cannot open: use one of the commands above.", file=sys.stderr)
        return 2
    try:
        from .gui.app import main as gui_main
    except ImportError as exc:  # PySide6 missing
        raise SystemExit(f"the GUI needs PySide6: pip install al-dvc ({exc})") from exc
    return int(gui_main([sys.argv[0], *argv]) or 0)


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the graphical application (``al-dvc gui`` is ``al-dvc`` alone, kept for the old spelling)."""
    return _launch_window([args.session] if args.session else [])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="al-dvc",
        description="Augmented Lagrangian Digital Volume Correlation",
        epilog="Without a command, al-dvc opens the application; al-dvc SESSION.aldvc opens a saved session in it, "
        "and al-dvc --self-test checks the installation.",
    )
    ap.add_argument("-q", "--quiet", action="store_true", help="only warnings")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline on a volume sequence")
    r.add_argument("config", nargs="?", help="YAML/JSON config (keys: volumes, masks, output, export, para)")
    r.add_argument("--volumes", nargs="+", help="volume files, a glob, or a folder")
    r.add_argument("--masks", nargs="+", help="mask volume(s)")
    r.add_argument("-o", "--output", help="output folder")
    r.add_argument("--winsize", type=int)
    r.add_argument("--step", type=int)
    r.add_argument("--no-global", action="store_true", help="local subset DVC only")
    r.add_argument("--no-strain", action="store_true")
    r.add_argument("--export", nargs="+", choices=["npz", "mat", "csv", "vtk", "report", "summary"])
    r.add_argument("--set", nargs="*", metavar="KEY=JSON", help="override any DVCPara field, e.g. mu=1e-3")
    r.add_argument("--checkpoint", metavar="DIR", help="write per-frame checkpoints here and resume from them")
    r.add_argument("--restart", action="store_true", help="ignore existing checkpoints in --checkpoint")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("synth", help="generate a synthetic speckle volume sequence")
    s.add_argument("output")
    s.add_argument("--shape", nargs=3, type=int, default=[96, 96, 96], metavar=("NZ", "NY", "NX"))
    s.add_argument("--mode", choices=["translation", "stretch", "shear", "sinusoid"], default="stretch")
    s.add_argument("--value", nargs="+", type=float, default=[0.02])
    s.add_argument("--frames", type=int, default=1)
    s.add_argument("--sigma", type=float, default=2.0)
    s.add_argument("--noise", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dtype", choices=["float32", "uint8", "uint16"], default="uint16")
    s.set_defaults(func=cmd_synth)

    i = sub.add_parser("info", help="print volume metadata")
    i.add_argument("paths", nargs="+")
    i.set_defaults(func=cmd_info)

    from .texture.concentric import DEFAULT_COUNT, DEFAULT_START, DEFAULT_STEP
    from .texture.rve import DEFAULT_CV_WINDOW

    t = sub.add_parser("texture", help="correlation lengths of a volume and a subset suggestion")
    t.add_argument("volume", help="volume file (any supported format)")
    t.add_argument("--roi", help="region of interest volume: the cubes stay inside it")
    t.add_argument(
        "--region",
        type=int,
        nargs=6,
        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
        help="analysis region (half-open voxel indices); the cubes stay inside it",
    )
    t.add_argument("--centre", type=int, nargs=3, metavar=("X", "Y", "Z"), help="centre of the cubes (default: the region's)")
    t.add_argument("--spacing", type=float, nargs=3, metavar=("DX", "DY", "DZ"), help="voxel size")
    t.add_argument("--size", type=int, help="edge of the analysed cube (default: the RVE size, else the largest that fits)")
    t.add_argument("--sweep", action="store_true", help="also run the RVE analysis (concentric cubes)")
    t.add_argument("--sweep-start", type=int, default=DEFAULT_START, help="edge of the smallest cube of the sweep")
    t.add_argument("--sweep-step", type=int, default=DEFAULT_STEP, help="growth of the cube edge per size")
    t.add_argument("--sweep-count", type=int, default=DEFAULT_COUNT, help="number of cube sizes")
    t.add_argument(
        "--estimator",
        choices=("overlap", "window"),
        default="overlap",
        help="autocorrelation estimator: overlap-corrected (default), or the raw window estimator of DVC Challenge 2.0",
    )
    t.add_argument(
        "--rve-criterion",
        choices=("plateau", "cv-window"),
        default="plateau",
        help="how the sweep decides a length is stable: the plateau test (default), or the sliding-window CV test "
        "of DVC Challenge 2.0",
    )
    t.add_argument("--cv-window", type=int, default=DEFAULT_CV_WINDOW, help="consecutive sizes per window of the CV test")
    t.add_argument("-o", "--out", default="texture", help="output directory")
    t.set_defaults(func=cmd_texture)

    p = sub.add_parser("plot", help="plot a field from an exported .npz")
    p.add_argument("npz")
    p.add_argument("--field", default="exx")
    p.add_argument("--frame", type=int, default=1)
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_plot)
    b = sub.add_parser("batch", help="run several .aldvc session files one after another")
    b.add_argument("sessions", nargs="+", help="session files (.aldvc)")
    b.add_argument("--export", nargs="+", default=["npz", "summary"], choices=["npz", "summary", "report", "vtk", "mat", "csv"])
    b.add_argument("--no-checkpoints", action="store_true", help="do not write per-frame checkpoints")
    b.add_argument("--no-strain", action="store_true", help="skip the strain computation")
    b.add_argument("--quiet", action="store_true", help="no progress line")
    b.add_argument("--verbose", action="store_true", help="print tracebacks of failed jobs")
    b.set_defaults(func=cmd_batch)
    st = sub.add_parser("stats", help="statistics of an exported result (.npz), frame by frame")
    st.add_argument("result", help="result archive written by the npz export")
    st.add_argument("--fields", nargs="+", help="fields (default: displacement, and the strain tensor when computed)")
    st.add_argument("--frames", nargs="+", type=int, help="frames, 1-based (default: all)")
    st.add_argument("--motion", default="none", help="remove a motion first: none, translation, rigid or affine")
    st.add_argument("--all-nodes", action="store_true", help="also use nodes that did not converge or were rejected")
    st.add_argument("--min-zncc", type=float, default=0.0, help="drop nodes below this ZNCC")
    st.add_argument("--edge-layers", type=int, default=0, help="drop this many node layers at the edges")
    st.add_argument("--drop-cut", action="store_true", help="drop nodes whose subset was cut at a boundary")
    st.add_argument("--noise-floor", action="store_true", help="add the noise floor (static or known-translation pair)")
    st.add_argument("--nominal", type=float, nargs=3, metavar=("DX", "DY", "DZ"), help="applied displacement for the bias")
    st.add_argument("--regions", help="regions: a session (.aldvc), a statistics summary (.json) or a JSON list of regions")
    st.add_argument("--region", help="take the statistics over this region (its name)")
    st.add_argument("--fit-region", help="fit the removed motion over this region (its name), e.g. a grip")
    st.add_argument("--compare", action="store_true", help="one series per region, and one for every node")
    st.add_argument("--ci", action="store_true", help="add the 95 %% confidence interval of each mean (n_eff, ci95)")
    st.add_argument("-o", "--out", default="statistics", help="output folder")
    st.set_defaults(func=cmd_stats)
    g = sub.add_parser("gui", help="open the application (the same as al-dvc alone)")
    g.add_argument("session", nargs="?", help="session file (.aldvc) to open")
    g.set_defaults(func=cmd_gui)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if _opens_window(argv):
        return _launch_window(argv)
    ap = build_parser()
    args = ap.parse_args(argv)
    _setup_logging(not args.quiet)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
