"""Statistics to files: a CSV with one row per frame and a JSON summary, both self-describing.

The CSV follows pyALDIC's analysis export: frames are 1-based, a value that does not exist is an empty cell
(never ``nan``), and a comment header records the software, the unit, the node filter, the motion correction
and the definitions, so the file can be read without the session that produced it.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..core.config import length_unit
from ..core.data_structures import PipelineResult
from .selection import NodeFilter
from .series import FrameStats
from .stats import CI_NAMES, STAT_NAMES, FieldStats

COLUMNS = STAT_NAMES + CI_NAMES

DEFINITIONS = {
    "std": "population standard deviation (ddof 0), as the DVC Challenge 2.0 noise floor (Eq. 2)",
    "robust_std": "1.4826 x median absolute deviation from the median",
    "p05/p95": "5th and 95th percentiles (linear interpolation)",
    "rms": "root mean square of the values",
    "bias": "mean of (measured - nominal) displacement per component (DVC Challenge 2.0 Eq. 1)",
    "noise": "population std of (measured - nominal) after removing the bias, i.e. the rigid translation (Eq. 2)",
    "u_rms": "sqrt(|bias|^2 + |noise|^2) (DVC Challenge 2.0 SI Eq. S10)",
    "MAER/SDER": "mean / std over nodes of (1/6) sum |strain component| over the six tensor components",
    "VSG": "(window - 1) x step + subset span, voxels (iDICs GPG Eq. 7.3)",
    "rigid": "motion x = Xc + t + R (X - Xc) fitted by weighted Kabsch; removed as u' = R^T (X + u - Xc - t) + Xc - X",
    "affine": "x = Xc + t + F (X - Xc) by least squares; removed as u' = u - t - (F - I)(X - Xc)",
    "shear": "tensorial components (exy = 0.5 (du/dy + dv/dx))",
    "n_eff": "effective number of independent nodes, n^2 / sum_ij rho_ij: the autocorrelation rho of the field minus "
    "its least-squares plane, summed out to the first lag shell where it falls below 0.05 (corrected for the "
    "removed plane), then a fitted tail exp(-(r/L)^q) beyond",
    "ci95": "half width of the 95 % confidence interval of the mean from the fluctuations about a linear trend: "
    "t(n_eff - 4) std_residual / sqrt(n_eff - 4)",
    "profile": "mean and population std of the nodes of each node layer along the axis",
    "line": "trilinear interpolation between the nodes; empty where a neighbouring node has no value",
    "extensometer": "strain = L / L0 - 1, L the distance between two material points displaced by the "
    "trilinearly interpolated displacement",
}


def stats_metadata(
    result: PipelineResult, node_filter: NodeFilter | None, motion: str, region: str = "whole", fit_region: str = "whole"
) -> dict:
    """What a statistics file must say about how its numbers were made: ``region`` is where the statistics were
    taken, ``fit_region`` where the removed motion was fitted."""
    from al_dvc import __version__

    para = result.dvc_para
    return {
        "software": f"pyALDVC {__version__}",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "unit": length_unit(para),
        "voxel_size": [float(v) for v in para.voxel_size],
        "strain_type": para.strain_type if result.result_strain else None,
        "strain_method": para.strain_method if result.result_strain else None,
        "motion": motion,
        "node_filter": asdict(node_filter or NodeFilter()),
        "region": region,
        "fit_region": fit_region,
        "definitions": DEFINITIONS,
    }


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else repr(float(value))
    return str(value)


def _write_header(fh, meta: dict, what: str) -> None:
    fh.write(f"# {meta.get('software', 'pyALDVC')} {what}, {meta.get('created', '')}\n")
    fh.write(f"# unit: {meta.get('unit', '')} (lengths); strain dimensionless; rotation in degrees\n")
    fh.write(
        f"# motion removed: {meta.get('motion', 'none')} (fitted over: {meta.get('fit_region', 'whole')}); "
        f"region: {meta.get('region', 'whole')}\n"
    )
    fh.write(f"# nodes: {json.dumps(meta.get('node_filter', {}))}\n")
    for key, text in meta.get("definitions", DEFINITIONS).items():
        fh.write(f"# {key}: {text}\n")


def _stat_cells(st: FieldStats | None) -> list[str]:
    return [_cell(getattr(st, s)) if st is not None else "" for s in COLUMNS]


def _write_series(path, named, fields: Sequence[str], meta: dict, with_region: bool) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = (["region"] if with_region else []) + ["frame", "nodes_used", "valid_fraction", "flag"]
    header += ["motion_tx", "motion_ty", "motion_tz", "motion_rotation_deg", "motion_residual_rms"]
    header += [f"{f}_{s}" for f in fields for s in COLUMNS]
    with p.open("w", newline="", encoding="utf-8") as fh:
        _write_header(fh, meta, "statistics")
        writer = csv.writer(fh)
        writer.writerow(header)
        for name, series in named:
            for fs in series:
                fit = fs.fit
                row = [name] if with_region else []
                row += [fs.frame + 1, fs.selection.n, _cell(fs.selection.valid_fraction), fs.flag]
                if fit is None:
                    row += [""] * 5
                else:
                    row += [_cell(v) for v in fit.translation] + [_cell(fit.rotation_deg), _cell(fit.residual_rms)]
                for f in fields:
                    row += _stat_cells(fs.stats.get(f))
                writer.writerow(row)
    return p


def write_series_csv(path: str | Path, series: Sequence[FrameStats], fields: Sequence[str], meta: dict) -> Path:
    """One row per frame: nodes used, the motion fitted, and every statistic of every field (``n_eff`` and
    ``ci95`` are empty when the confidence interval was not computed)."""
    return _write_series(path, [("", series)], fields, meta, with_region=False)


def write_region_series_csv(
    path: str | Path, named: Sequence[tuple[str, Sequence[FrameStats]]], fields: Sequence[str], meta: dict
) -> Path:
    """The series of several regions in one file: a ``region`` column, then one row per region and frame."""
    return _write_series(path, named, fields, meta, with_region=True)


def write_regions_csv(path: str | Path, rows: Sequence[tuple[str, FieldStats]], field: str, frame: int, meta: dict) -> Path:
    """The statistics of ``field`` in each region for one frame (``frame`` 0-based, written 1-based)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        _write_header(fh, meta, f"statistics per region, frame {frame + 1}")
        writer = csv.writer(fh)
        writer.writerow(["region", "field", *COLUMNS])
        for name, st in rows:
            writer.writerow([name, field, *_stat_cells(st)])
    return p


def write_profile_csv(path: str | Path, profiles: Sequence[tuple[int, object]], field: str, meta: dict) -> Path:
    """Layer profiles ``(frame, AxisProfile)`` (``frame`` 0-based): one row per frame and node layer."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        _write_header(fh, meta, f"profile of {field}")
        writer = csv.writer(fh)
        writer.writerow(["frame", "axis", "position", "mean", "std", "n"])
        for frame, prof in profiles:
            for pos, m, sd, n in zip(prof.positions, prof.mean, prof.std, prof.n):
                writer.writerow([frame + 1, prof.axis, _cell(pos), _cell(m), _cell(sd), int(n)])
    return p


def write_line_csv(path: str | Path, lines: Sequence[tuple[int, object, object]], field: str, p0, p1, meta: dict) -> Path:
    """Values of ``field`` along the segment ``p0``-``p1`` (voxels): ``(frame, distance, values)`` per frame."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        _write_header(fh, meta, f"line profile of {field}")
        fh.write(f"# from {[float(v) for v in p0]} to {[float(v) for v in p1]} (voxels [x, y, z])\n")
        writer = csv.writer(fh)
        writer.writerow(["frame", "distance", field])
        for frame, distance, values in lines:
            for d, v in zip(distance, values):
                writer.writerow([frame + 1, _cell(d), _cell(v)])
    return p


def write_extensometer_csv(path: str | Path, ext, meta: dict, field: str | None = None, ends=None) -> Path:
    """The virtual extensometer over the frames: length and strain ``L / L0 - 1``; with ``ends`` ``(frames, 2)``, also
    ``field`` at the two end points."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        _write_header(fh, meta, "virtual extensometer")
        fh.write(f"# from {ext.p0.tolist()} to {ext.p1.tolist()} (voxels [x, y, z]); L0 = {ext.L0!r} {meta.get('unit', '')}\n")
        writer = csv.writer(fh)
        extra = [] if ends is None else [f"{field}_first_point", f"{field}_second_point"]
        writer.writerow(["frame", "length", "strain", *extra])
        for i, (k, length, strain) in enumerate(zip(ext.frames, ext.length, ext.strain)):
            cells = [] if ends is None else [_cell(ends[i][0]), _cell(ends[i][1])]
            writer.writerow([int(k) + 1, _cell(length), _cell(strain), *cells])
    return p


def _jsonable(value):
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_summary_json(path: str | Path, payload: dict) -> Path:
    """``payload`` (statistics objects, arrays, plain values) as JSON; a missing number is ``null``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return p
