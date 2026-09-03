"""Command-line interface: ``al-dvc run|synth|info|plot``."""

from __future__ import annotations

import argparse
import json
import logging
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


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the graphical application (needs the ``gui`` extra: ``pip install al-dvc[gui]``)."""
    try:
        from .gui.app import main as gui_main
    except ImportError as exc:  # PySide6 missing
        raise SystemExit(f"the GUI needs PySide6: pip install al-dvc[gui] ({exc})") from exc
    argv = [sys.argv[0]] + ([args.session] if args.session else [])
    return int(gui_main(argv) or 0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="al-dvc", description="Augmented Lagrangian Digital Volume Correlation")
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

    p = sub.add_parser("plot", help="plot a field from an exported .npz")
    p.add_argument("npz")
    p.add_argument("--field", default="exx")
    p.add_argument("--frame", type=int, default=1)
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_plot)
    g = sub.add_parser("gui", help="launch the graphical application")
    g.add_argument("session", nargs="?", help="session file (.aldvc) to open")
    g.set_defaults(func=cmd_gui)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    _setup_logging(not args.quiet)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
