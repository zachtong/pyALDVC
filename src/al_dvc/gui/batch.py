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

from al_dvc.core.data_structures import PipelineResult
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
    outputs: list[Path] = field(default_factory=list)  # written even when a later export kind failed
    results_path: Path | None = None  # the NumPy archive of the job, when one was written
    traceback: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "done"


def session_provider(data: SessionData):
    """A session's frames as a streaming provider: the run reads a frame when it needs it.

    The batch used to load every volume and every mask of a session into lists before the run
    started, so a sequence of N frames cost N frames of memory before the first correlation, where
    the interactive run had long been reading them through ``FileVolumeProvider`` with two normalised
    frames resident. Mask files stream the same way. A mask that was *drawn* has to be rebuilt from
    its operations, and a threshold operation needs the intensities, so that frame's volume is read
    once here and dropped again; only the boolean mask stays.
    """
    from al_dvc.io.volume_io import FileVolumeProvider, load_volume

    from .mask_editor import MaskEditor

    paths: list[str] = []
    mask_paths: list[str | None] = []
    drawn: list = []
    for v in data.volumes:
        path = v.get("path")
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"volume not found: {path}")
        mask_file = v.get("mask") or None
        if mask_file and not Path(mask_file).exists():
            raise FileNotFoundError(f"mask not found: {mask_file}")
        mask = None
        if v.get("mask_ops"):
            vol = load_volume(path)  # transient: the drawing replays on it, exactly as the GUI rebuilds a session
            base = (np.asarray(load_volume(mask_file)) > 0) if mask_file else None
            mask = MaskEditor.from_dict(v["mask_ops"], base=base, volume=vol).mask
            del vol
        paths.append(path)
        mask_paths.append(mask_file)
        drawn.append(mask)
    if len(paths) < 2:
        raise ValueError(f"a session needs at least two volumes, this one has {len(paths)}")
    any_mask = any(m is not None for m in drawn) or any(p is not None for p in mask_paths)
    return FileVolumeProvider(
        paths,
        data.para.voi,
        mask_paths=mask_paths if any_mask else None,
        masks=drawn if any_mask else None,
    )


def load_session_inputs(data: SessionData) -> tuple[list, list | None]:
    """Volumes and masks of a session as lists (mask files, then drawn operations on top).

    Materialises every frame: kept for callers that want arrays. The batch run itself uses
    :func:`session_provider` and never holds more than the provider's cache.
    """
    provider = session_provider(data)
    volumes = [provider.get_normalized(i) for i in range(len(provider))]
    if not provider.has_masks:
        return volumes, None
    masks = [provider.get_mask(i) for i in range(len(provider))]
    # the pipeline takes one mask per frame: frames without one are fully material
    return volumes, [np.ones(provider.shape, dtype=bool) if m is None else m for m in masks]


def export_results_checked(
    result: PipelineResult, out_dir: str | Path, basename: str, kinds=DEFAULT_EXPORTS
) -> tuple[list[Path], dict[str, str]]:
    """Write the requested export kinds into ``out_dir``: ``(paths written, {kind: error})``.

    Every kind is attempted; a failure is recorded and the next kind is written. The CSV and
    ParaView files carry ``basename`` too, so sessions sharing an output folder never overwrite
    each other's files.
    """
    from al_dvc.export import export_csv, export_mat, export_npz, export_report, export_run_summary, export_vtk

    for kind in kinds:
        if kind not in EXPORT_KINDS:
            raise ValueError(f"unknown export kind {kind!r}; choose from {EXPORT_KINDS}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    errors: dict[str, str] = {}
    for kind in kinds:
        try:
            if kind == "npz":
                paths.append(Path(export_npz(result, out / f"{basename}.npz")))
            elif kind == "summary":
                paths.append(Path(export_run_summary(result, out / f"{basename}_summary.json")))
            elif kind == "report":
                paths.append(Path(export_report(result, out / f"{basename}_report.pdf")))
            elif kind == "vtk":
                paths.append(Path(export_vtk(result, out / "vtk", basename)[0]).parent)
            elif kind == "mat":
                paths.append(Path(export_mat(result, out / f"{basename}.mat")))
            elif kind == "csv":
                paths.append(Path(export_csv(result, out / "csv", basename)[0]).parent)
        except Exception as exc:  # the other kinds are still written; the caller reports the failure
            errors[kind] = f"{type(exc).__name__}: {exc}"
    return paths, errors


def export_results(result: PipelineResult, out_dir: str | Path, basename: str, kinds=DEFAULT_EXPORTS) -> list[Path]:
    """Write the requested export kinds into ``out_dir``; returns the paths written, raises when a kind failed."""
    paths, errors = export_results_checked(result, out_dir, basename, kinds)
    if errors:
        raise RuntimeError("export failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
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
        provider = session_provider(data)  # streams the frames; nothing but the provider's cache is resident
        out_dir = Path(data.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = out_dir
        job.status = "running"
        result = run_aldvc(
            data.para,
            provider,
            None,
            progress_fn=progress_fn,
            stop_fn=stop_fn,
            compute_strain=compute_strain,
            checkpoint_dir=out_dir / CHECKPOINT_SUBDIR if checkpoints else None,
        )
        job.n_nodes = int(result.dvc_mesh.n_nodes)
        job.n_frames = int(result.n_frames)
        from al_dvc.export.export_utils import converged_fraction

        fractions = [converged_fraction(result, fr) for fr in result.result_disp]
        fractions = [f for f in fractions if f is not None]  # the reference's valid nodes only
        if fractions:
            job.converged = float(np.mean(fractions))
        job.outputs, export_errors = export_results_checked(result, out_dir, job.session.stem, exports)
        job.results_path = next((p for p in job.outputs if p.suffix == ".npz"), None)
        if export_errors:
            raise RuntimeError("export failed: " + "; ".join(f"{k}: {v}" for k, v in export_errors.items()))
        if result.stopped_early:
            job.status = "stopped"
            job.message = result.stop_reason or f"stopped after {result.n_frames} frame(s)"
        else:
            job.status = "done"
            job.message = f"{result.n_frames} frame(s), {job.n_nodes} nodes"
        if data.notes:
            job.message += " -- " + " ".join(data.notes)
    except Exception as exc:  # recorded per job; the batch goes on
        from al_dvc.core.checkpoint import CheckpointMismatch

        job.status = "failed"
        job.message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, CheckpointMismatch):
            job.message += (
                " -- the checkpoints were written with other parameters: delete that folder, or run the batch "
                "without checkpoints, to start over"
            )
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
