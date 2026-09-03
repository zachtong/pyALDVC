"""Per-frame checkpoints so that long sequences can be resumed.

A checkpoint directory holds ``meta.json`` (parameters, volume shape, frame
schedule, node grid) and one ``frame_<k>.npz`` per solved frame pair with
everything :class:`FrameResult` carries (displacements, gradients, initial
guess, ZNCC, status, uncertainty, ADMM diagnostics) plus the node validity
of the reference mesh. :func:`run_aldvc` writes a file when a frame completes
and, on a later call with the same directory, loads the finished frames
instead of recomputing them. A directory written with different parameters,
volumes or a different node grid is rejected.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import para_to_dict
from .data_structures import ADMMInfo, DVCMesh, FrameResult, FrameSchedule, LocalSolveInfo

META_NAME = "meta.json"
FORMAT_VERSION = 1

__all__ = ["Checkpoint", "CheckpointMismatch", "META_NAME"]


class CheckpointMismatch(ValueError):
    """The checkpoint directory belongs to a different run."""


def _meta_for(para, volume_shape, schedule: FrameSchedule, mesh: DVCMesh, n_frames: int) -> dict[str, Any]:
    d = para_to_dict(para)
    d.pop("verbose", None)
    d.pop("n_threads", None)
    return {
        "format": FORMAT_VERSION,
        "para": d,
        "volume_shape": [int(s) for s in volume_shape],
        "n_frames": int(n_frames),
        "ref_indices": [int(r) for r in schedule.ref_indices],
        "grid_shape": [int(s) for s in mesh.grid_shape],
        "x0": [float(v) for v in mesh.x0],
        "y0": [float(v) for v in mesh.y0],
        "z0": [float(v) for v in mesh.z0],
    }


class Checkpoint:
    """Reader/writer of a checkpoint directory (see module docstring)."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def frame_path(self, k: int) -> Path:
        return self.dir / f"frame_{k:04d}.npz"

    # ------------------------------------------------------------------ meta
    def prepare(self, para, volume_shape, schedule: FrameSchedule, mesh: DVCMesh, n_frames: int, resume: bool) -> None:
        """Create the directory and ``meta.json``; with ``resume`` verify an existing meta."""
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = _meta_for(para, volume_shape, schedule, mesh, n_frames)
        meta_path = self.dir / META_NAME
        if meta_path.exists() and resume:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            diff = _meta_diff(existing, meta)
            if diff:
                raise CheckpointMismatch(
                    f"checkpoint directory {self.dir} was written by a different run ({', '.join(diff)}); "
                    "use resume=False (or another directory) to start over"
                )
            return
        for p in self.dir.glob("frame_*.npz"):
            p.unlink()
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ frames
    def has(self, k: int) -> bool:
        return self.frame_path(k).is_file()

    def completed_frames(self) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in self.dir.glob("frame_*.npz"))

    def save(self, k: int, fr: FrameResult, mesh: DVCMesh) -> Path:
        arrays: dict[str, Any] = {
            "U": fr.U,
            "F": fr.F,
            "ref_frame": np.int64(fr.ref_frame),
            "node_valid": np.asarray(mesh.node_valid, dtype=bool),
            "elements": np.asarray(mesh.elements, dtype=np.int64),
            "boundary_nodes": np.asarray(mesh.boundary_nodes, dtype=np.int64),
        }
        for name in ("U_accum", "U_local", "F_local", "U0", "zncc", "status", "U_std"):
            val = getattr(fr, name)
            if val is not None:
                arrays[name] = np.asarray(val)
        if fr.admm is not None:
            a = fr.admm
            arrays["admm_scalars"] = np.array([a.beta, a.mu, a.n_steps, a.subpb2_time], dtype=np.float64)
            for name in ("update_global", "update_local", "primal_residual_u", "primal_residual_f"):
                arrays["admm_" + name] = np.asarray(getattr(a, name), dtype=np.float64)
            for i, li in enumerate(a.local_info):
                arrays[f"admm_local_{i}_n_iter"] = np.asarray(li.n_iter)
                arrays[f"admm_local_{i}_status"] = np.asarray(li.status)
                arrays[f"admm_local_{i}_zncc"] = np.asarray(li.zncc)
                arrays[f"admm_local_{i}_scalars"] = np.array([li.solve_time, li.n_bad], dtype=np.float64)
            if a.beta_sweep is not None:
                for key, val in a.beta_sweep.items():
                    arrays["admm_sweep_" + key] = np.asarray(val)
        path = self.frame_path(k)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        return path

    def load(self, k: int, base_mesh: DVCMesh) -> tuple[FrameResult, DVCMesh]:
        with np.load(self.frame_path(k), allow_pickle=False) as z:
            d = {key: z[key] for key in z.files}
        mesh = replace(
            base_mesh,
            node_valid=d["node_valid"].astype(bool),
            elements=d["elements"].astype(np.int64),
            boundary_nodes=d["boundary_nodes"].astype(np.int64),
        )
        admm = None
        if "admm_scalars" in d:
            beta, mu, n_steps, subpb2_time = (float(v) for v in d["admm_scalars"])
            local_info = []
            i = 0
            while f"admm_local_{i}_n_iter" in d:
                st, nb = d[f"admm_local_{i}_scalars"]
                local_info.append(
                    LocalSolveInfo(
                        n_iter=d[f"admm_local_{i}_n_iter"],
                        status=d[f"admm_local_{i}_status"],
                        zncc=d[f"admm_local_{i}_zncc"],
                        solve_time=float(st),
                        n_bad=int(nb),
                    )
                )
                i += 1
            sweep_keys = [key for key in d if key.startswith("admm_sweep_")]
            sweep = {key[len("admm_sweep_") :]: d[key] for key in sweep_keys} if sweep_keys else None
            if sweep is not None:
                for key in ("k_best",):
                    if key in sweep:
                        sweep[key] = int(sweep[key])
                for key in ("beta", "criterion"):
                    if key in sweep and np.ndim(sweep[key]) == 0:
                        sweep[key] = sweep[key].item()
            admm = ADMMInfo(
                beta=beta,
                mu=mu,
                n_steps=int(n_steps),
                update_global=tuple(float(v) for v in d["admm_update_global"]),
                update_local=tuple(float(v) for v in d["admm_update_local"]),
                primal_residual_u=tuple(float(v) for v in d["admm_primal_residual_u"]),
                primal_residual_f=tuple(float(v) for v in d["admm_primal_residual_f"]),
                local_info=tuple(local_info),
                subpb2_time=subpb2_time,
                beta_sweep=sweep,
            )
        fr = FrameResult(
            U=d["U"],
            F=d["F"],
            ref_frame=int(d["ref_frame"]),
            U_accum=d.get("U_accum"),
            U_local=d.get("U_local"),
            F_local=d.get("F_local"),
            U0=d.get("U0"),
            zncc=d.get("zncc"),
            U_std=d.get("U_std"),
            status=d.get("status"),
            admm=admm,
        )
        return fr, mesh


def _meta_diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Names of the top-level entries that differ (parameters compared key by key)."""
    diff = []
    for key in sorted(set(a) | set(b)):
        if key == "para":
            pa, pb = a.get("para", {}), b.get("para", {})
            for pk in sorted(set(pa) | set(pb)):
                if _norm(pa.get(pk)) != _norm(pb.get(pk)):
                    diff.append(f"para.{pk}")
        elif _norm(a.get(key)) != _norm(b.get(key)):
            diff.append(key)
    return diff


def _norm(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, float):
        return round(value, 12)
    return value


def load_checkpoint_frames(directory: str | Path, base_mesh: DVCMesh) -> dict[int, tuple[FrameResult, DVCMesh]]:
    """All completed frames of a checkpoint directory, keyed by frame index."""
    ck = Checkpoint(directory)
    return {k: ck.load(k, base_mesh) for k in ck.completed_frames()}
