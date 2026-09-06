"""Batch runs: session runner, stop / failure handling, the CLI, and the dialog (offscreen)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from al_dvc.cli import main as cli_main  # noqa: E402
from al_dvc.gui.app import MainWindow, create_application  # noqa: E402
from al_dvc.gui.app_state import AppState  # noqa: E402
from al_dvc.gui.batch import EXPORT_KINDS, BatchRunner, export_results, run_session_file  # noqa: E402
from al_dvc.gui.mask_editor import MaskOp  # noqa: E402
from al_dvc.gui.session import save_session  # noqa: E402
from al_dvc.io.volume_io import save_volume  # noqa: E402
from al_dvc.synthetic import affine_displacement, generate_speckle_volume, warp_volume_lagrangian  # noqa: E402

SHAPE = (40, 44, 48)


@pytest.fixture(scope="module")
def qapp():
    return create_application(["pytest"])


@pytest.fixture(scope="module")
def sessions(qapp, tmp_path_factory):
    """Two valid sessions (different subset sizes) and one with a missing volume."""
    root = tmp_path_factory.mktemp("batch")
    centre = tuple((s - 1) / 2 for s in SHAPE[::-1])
    ref = generate_speckle_volume(SHAPE, sigma=2.0, seed=13)
    dfm = warp_volume_lagrangian(ref, affine_displacement(np.diag([0.01, -0.005, 0.005]), (0.4, -0.3, 0.2), centre))
    p0, p1 = root / "ref.npy", root / "def.npy"
    save_volume(p0, ref)
    save_volume(p1, dfm)
    paths = []
    for name, ws in (("a", 16), ("b", 20)):
        state = AppState()
        state.add_volume_paths([str(p0), str(p1)])
        state.set_params(winsize=ws, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
        state.set_output_dir(root / f"out_{name}")
        if name == "b":  # a drawn mask travels with the session (copied to every frame explicitly)
            state.apply_mask_op(MaskOp("rectangle", "xy", ((0, 0), (31, 43))))
            state.copy_mask_to_all_frames()
        paths.append(save_session(state, root / f"{name}.aldvc"))
    broken = AppState()
    broken.add_volume_paths([str(p0), str(root / "missing.npy")])
    broken.set_output_dir(root / "out_broken")
    paths.append(save_session(broken, root / "broken.aldvc"))
    return paths


def test_run_session_file_exports_and_summary(sessions):
    job = run_session_file(sessions[0], exports=("npz", "summary"), checkpoints=False)
    assert job.status == "done", job.message
    assert job.n_nodes > 0 and job.n_frames == 1 and job.converged > 0.9
    assert job.output_dir is not None and job.output_dir.name == "out_a"
    names = {p.name for p in job.outputs}
    assert names == {"a.npz", "a_summary.json"}
    summary = json.loads((job.output_dir / "a_summary.json").read_text(encoding="utf-8"))
    assert "summary" in summary or "n_nodes" in json.dumps(summary)


def test_drawn_mask_is_applied_in_batch(sessions):
    job = run_session_file(sessions[1], exports=("npz",), checkpoints=False)
    assert job.status == "done", job.message
    d = np.load(job.outputs[0])
    valid = d["node_valid"]
    x = d["coordinates"][:, 0]
    assert not valid[x > 31 + 10].any() and valid[x < 31 - 10].all()


def test_missing_volume_fails_the_job_only(sessions):
    runner = BatchRunner(sessions[::-1], exports=("summary",), checkpoints=False)  # broken first
    jobs = runner.run()
    assert [j.status for j in jobs] == ["failed", "done", "done"]
    assert "not found" in jobs[0].message and jobs[0].traceback
    table = BatchRunner.summary_table(jobs)
    assert "broken.aldvc" in table and "failed" in table and "done" in table


def test_stop_skips_the_remaining_sessions(sessions):
    started: list[int] = []
    stop = {"flag": False}

    def job_fn(i, job):
        if job.status == "running":
            started.append(i)
        if job.status in ("done", "stopped"):
            stop["flag"] = True  # stop after the first job finishes

    jobs = BatchRunner(sessions[:2], exports=(), checkpoints=False, job_fn=job_fn, stop_fn=lambda: stop["flag"]).run()
    assert started == [0]
    assert jobs[0].status in ("done", "stopped") and jobs[1].status == "skipped"


def test_runner_validation(sessions):
    with pytest.raises(ValueError):
        BatchRunner([], exports=("npz",))
    with pytest.raises(ValueError):
        BatchRunner(sessions[:1], exports=("xlsx",))
    assert set(EXPORT_KINDS) >= {"npz", "summary", "report", "vtk", "mat", "csv"}


def test_export_results_all_kinds(sessions, tmp_path):
    job = run_session_file(sessions[0], exports=(), checkpoints=False)
    assert job.status == "done"
    from al_dvc.core.config import dvcpara_default
    from al_dvc.core.pipeline import run_aldvc

    ref = np.load(sessions[0].parent / "ref.npy")
    dfm = np.load(sessions[0].parent / "def.npy")
    res = run_aldvc(dvcpara_default(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False), [ref, dfm])
    paths = export_results(res, tmp_path, "case", EXPORT_KINDS)
    assert len(paths) == len(EXPORT_KINDS) and all(p.exists() for p in paths)
    with pytest.raises(ValueError):
        export_results(res, tmp_path, "case", ("nope",))


def test_cli_batch(sessions, capsys):
    code = cli_main(["batch", str(sessions[0]), str(sessions[2]), "--export", "npz", "--no-checkpoints", "--quiet"])
    out = capsys.readouterr().out
    assert code == 1  # one job failed
    assert "a.aldvc" in out and "broken.aldvc" in out and "failed" in out
    assert cli_main(["batch", str(sessions[0]), "--export", "summary", "--no-checkpoints", "--quiet"]) == 0


def test_dialog_runs_a_queue(qapp, sessions):
    window = MainWindow()
    window.show()
    dialog = window.open_batch_dialog()
    assert window.open_batch_dialog() is dialog  # single instance
    assert dialog.add_sessions([str(sessions[0]), str(sessions[2]), str(sessions[0]), "nope.aldvc"]) == 2
    assert dialog.sessions() == [str(sessions[0]), str(sessions[2])]
    dialog.exports["report"].setChecked(False)
    dialog.checkpoints.setChecked(False)
    assert dialog.selected_exports() == ("npz", "summary")
    assert dialog.start()
    assert not dialog.start()  # already running
    assert dialog.wait(300_000)
    for _ in range(30):
        QApplication.processEvents()
    assert [j.status for j in dialog.jobs] == ["done", "failed"]
    assert dialog.table.item(0, 1).text() == "done" and dialog.table.item(1, 1).text() == "failed"
    assert "1 done, 1 failed" in dialog._summary.text()
    assert "broken.aldvc" in dialog.log.toPlainText()
    dialog.table.selectRow(0)
    dialog._on_open_selected()  # loads session a into the main window
    assert window.state.para.winsize == (16, 16, 16) and len(window.state.volumes) == 2
    dialog.clear()
    assert dialog.sessions() == [] and not dialog._btn["start"].isEnabled()
    window.close()
