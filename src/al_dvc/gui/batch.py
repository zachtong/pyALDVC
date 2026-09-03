"""Run several sessions one after another (the GUI's batch dialog and ``al-dvc batch``).

A batch is a list of ``.aldvc`` session files. Each one is loaded, its
volumes and masks (files and drawn operations) are read, the pipeline runs
with the session's parameters, and the chosen exports are written into the
session's output folder. Jobs are independent: a failure is recorded in its
:class:`BatchJob` and the next session starts. No Qt widgets here; the
dialog wraps :class:`BatchRunner` in a thread.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from al_dvc.core.data_structures import STATUS_CONVERGED, PipelineResult
from al_dvc.core.pipeline import run_aldvc

from .session import SessionData, load_session

EXPORT_KINDS = ("npz", "summary", "report", "vtk", "mat", "csv")
DEFAULT_EXPORTS = ("npz", "summary")
STATUSES = ("pending", "running", "done", "stopped", "failed", "skipped")
CHECKPOINT_SUBDIR = "checkpoints"
MAX_TABLE_MESSAGE = 72

ProgressFn = Callable[[float, str], None]
BatchProgressFn = Callable[[int, int, float, str], None]  # (job index, n jobs, fraction, message)
JobFn = Callable[[int, "BatchJob"], None]


@dataclass
class BatchJob:
    """One session of a batch and what happened to it."""

    session: Path
    status: str = "pending"
    message: str = ""
    elapsed: float = 0.0
    n_nodes: int = 0
    n_frames: int = 0
    converged: float = float("nan")
    output_dir: Path | None = None
    outputs: list[Path] = field(default_factory=list)
    traceback: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "done"


def load_session_inputs(data: SessionData) -> tuple[list, list | None]:
    """Volumes and masks of a session (mask files, then drawn operations on top)."""
    from al_dvc.io.volume_io import load_volume

    from .mask_editor import MaskEditor

    volumes: list = []
    masks: list = []
    for v in data.volumes:
        path = v.get("path")
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"volume not found: {path}")
        vol = load_volume(path)
        volumes.append(vol)
        mask = None
        if v.get("mask"):
            if not Path(v["mask"]).exists():
                raise FileNotFoundError(f"mask not found: {v['mask']}")
            mask = np.asarray(load_volume(v["mask"])) > 0
        if v.get("mask_ops"):
            mask = MaskEditor.from_dict(v["mask_ops"], base=mask).mask
        masks.append(mask)
    if len(volumes) < 2:
        raise ValueError(f"a session needs at least two volumes, this one has {len(volumes)}")
    if all(m is None for m in masks):
        return volumes, None
    # the pipeline takes one mask per frame: frames without one are fully material
    masks = [np.ones(np.asarray(vol).shape, dtype=bool) if m is None else m for vol, m in zip(volumes, masks)]
    return volumes, masks


def export_results(result: PipelineResult, out_dir: str | Path, basename: str, kinds=DEFAULT_EXPORTS) -> list[Path]:
    """Write the requested export kinds into ``out_dir``; returns the paths written."""
    from al_dvc.export import export_csv, export_mat, export_npz, export_report, export_run_summary, export_vtk

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for kind in kinds:
        if kind not in EXPORT_KINDS:
            raise ValueError(f"unknown export kind {kind!r}; choose from {EXPORT_KINDS}")
        if kind == "npz":
            paths.append(Path(export_npz(result, out / f"{basename}.npz")))
        elif kind == "summary":
            paths.append(Path(export_run_summary(result, out / f"{basename}_summary.json")))
        elif kind == "report":
            paths.append(Path(export_report(result, out / f"{basename}_report.pdf")))
        elif kind == "vtk":
            paths.append(Path(export_vtk(result, out / "vtk")[0]).parent)
        elif kind == "mat":
            paths.append(Path(export_mat(result, out / f"{basename}.mat")))
        elif kind == "csv":
            paths.append(Path(export_csv(result, out / "csv")[0]).parent)
    return paths


def run_session_file(
    path: str | Path,
    exports=DEFAULT_EXPORTS,
    progress_fn: ProgressFn | None = None,
    stop_fn: Callable[[], bool] | None = None,
    checkpoints: bool = True,
    compute_strain: bool = True,
) -> BatchJob:
    """Run one session file to completion; every failure ends up in the returned job, never raised."""
    job = BatchJob(session=Path(path))
    t0 = time.perf_counter()
    try:
        data = load_session(path)
        volumes, masks = load_session_inputs(data)
        out_dir = Path(data.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = out_dir
        job.status = "running"
        result = run_aldvc(
            data.para,
            volumes,
            masks,
            progress_fn=progress_fn,
            stop_fn=stop_fn,
            compute_strain=compute_strain,
            checkpoint_dir=out_dir / CHECKPOINT_SUBDIR if checkpoints else None,
        )
        job.n_nodes = int(result.dvc_mesh.n_nodes)
        job.n_frames = int(result.n_frames)
        statuses = [np.asarray(fr.status) for fr in result.result_disp if fr.status is not None]
        if statuses:
            job.converged = float(np.mean(np.concatenate(statuses) == STATUS_CONVERGED))
        job.outputs = export_results(result, out_dir, job.session.stem, exports)
        if result.stopped_early:
            job.status = "stopped"
            job.message = result.stop_reason or f"stopped after {result.n_frames} frame(s)"
        else:
            job.status = "done"
            job.message = f"{result.n_frames} frame(s), {job.n_nodes} nodes"
    except Exception as exc:  # recorded per job; the batch goes on
        job.status = "failed"
        job.message = f"{type(exc).__name__}: {exc}"
        job.traceback = traceback.format_exc()
    job.elapsed = time.perf_counter() - t0
    return job


class BatchRunner:
    """Run session files in order, reporting per-job and overall progress."""

    def __init__(
        self,
        sessions,
        exports=DEFAULT_EXPORTS,
        checkpoints: bool = True,
        compute_strain: bool = True,
        progress_fn: BatchProgressFn | None = None,
        job_fn: JobFn | None = None,
        stop_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.sessions = [Path(s) for s in sessions]
        if not self.sessions:
            raise ValueError("the batch has no sessions")
        for kind in exports:
            if kind not in EXPORT_KINDS:
                raise ValueError(f"unknown export kind {kind!r}; choose from {EXPORT_KINDS}")
        self.exports = tuple(exports)
        self.checkpoints = checkpoints
        self.compute_strain = compute_strain
        self.progress_fn = progress_fn
        self.job_fn = job_fn
        self.stop_fn = stop_fn
        self.jobs: list[BatchJob] = [BatchJob(session=s) for s in self.sessions]

    def _stopped(self) -> bool:
        return bool(self.stop_fn and self.stop_fn())

    def run(self) -> list[BatchJob]:
        n = len(self.jobs)
        for i, job in enumerate(self.jobs):
            if self._stopped():
                job.status = "skipped"
                job.message = "batch stopped"
                if self.job_fn:
                    self.job_fn(i, job)
                continue
            job.status = "running"
            if self.job_fn:
                self.job_fn(i, job)

            def progress(frac: float, msg: str, _i: int = i) -> None:
                if self.progress_fn:
                    self.progress_fn(_i, n, float(frac), msg)

            done = run_session_file(
                job.session,
                exports=self.exports,
                progress_fn=progress,
                stop_fn=self.stop_fn,
                checkpoints=self.checkpoints,
                compute_strain=self.compute_strain,
            )
            self.jobs[i] = done
            if self.job_fn:
                self.job_fn(i, done)
        return self.jobs

    @staticmethod
    def summary_table(jobs: list[BatchJob]) -> str:
        """Fixed-width text table (CLI output, dialog log)."""
        rows = [("session", "status", "nodes", "frames", "converged", "time", "message")]
        for j in jobs:
            conv = "" if np.isnan(j.converged) else f"{100 * j.converged:.1f} %"
            msg = j.message if len(j.message) <= MAX_TABLE_MESSAGE else j.message[: MAX_TABLE_MESSAGE - 3] + "..."
            rows.append((j.session.name, j.status, str(j.n_nodes or ""), str(j.n_frames or ""), conv, f"{j.elapsed:.1f} s", msg))
        widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
        lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines)
