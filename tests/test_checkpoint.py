"""Per-frame checkpoints: interrupted sequences resume with identical results."""

import json

import numpy as np
import pytest

from al_dvc.core.checkpoint import META_NAME, Checkpoint, CheckpointMismatch, load_checkpoint_frames
from al_dvc.core.config import dvcpara_default
from al_dvc.core.pipeline import run_aldvc
from al_dvc.synthetic import affine_displacement, warp_volume_lagrangian
from tests.conftest import CENTRE, F_AFFINE


@pytest.fixture(scope="module")
def sequence(speckle):
    frames = [speckle]
    for k in (1, 2, 3):
        disp = affine_displacement(F_AFFINE * k * 0.5, (0.6 * k, -0.3 * k, 0.4 * k), CENTRE)
        frames.append(warp_volume_lagrangian(speckle, disp))
    return frames


def _para(**kw):
    base = dict(winsize=16, winstepsize=8, search_radius=5, verbose=False, admm_max_iter=2)
    base.update(kw)
    return dvcpara_default(**base)


def _same(a, b, skip=()):
    for name in ("U", "F", "U_accum", "U_local", "F_local", "U0", "zncc", "status", "U_std"):
        if name in skip:
            continue
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            assert x is None and y is None, name
        else:
            assert np.allclose(x, y, equal_nan=True), name


def test_resume_matches_uninterrupted_run(sequence, tmp_path):
    para = _para(reference_mode="incremental")
    ck = tmp_path / "ck"
    calls = {"n": 0}

    def stop_after_first_frame():
        calls["n"] += 1
        return calls["n"] > 3  # the pipeline polls before each frame and after the initial guess

    partial = run_aldvc(para, sequence, stop_fn=stop_after_first_frame, checkpoint_dir=ck)
    assert partial.stopped_early and partial.n_frames < 3
    done = Checkpoint(ck).completed_frames()
    assert done == list(range(1, partial.n_frames + 1))
    assert (ck / META_NAME).is_file()

    resumed = run_aldvc(para, sequence, checkpoint_dir=ck)
    straight = run_aldvc(para, sequence)
    assert not resumed.stopped_early and resumed.n_frames == 3
    assert all(resumed.timings[f"frame_{k}"] == 0.0 for k in done)
    for a, b in zip(resumed.result_disp, straight.result_disp):
        _same(a, b)
        assert a.ref_frame == b.ref_frame
        assert a.admm is not None and b.admm is not None
        assert a.admm.beta == pytest.approx(b.admm.beta)
        assert len(a.admm.local_info) == len(b.admm.local_info)
    for sa, sb in zip(resumed.result_strain, straight.result_strain):
        assert np.allclose(sa.exx, sb.exx, equal_nan=True)
    assert Checkpoint(ck).completed_frames() == [1, 2, 3]
    # a third call reuses everything
    again = run_aldvc(para, sequence, checkpoint_dir=ck)
    assert all(again.timings[f"frame_{k}"] == 0.0 for k in (1, 2, 3))
    frames = load_checkpoint_frames(ck, resumed.dvc_mesh)
    assert sorted(frames) == [1, 2, 3]
    _same(frames[2][0], straight.result_disp[1], skip=("U_accum",))  # composed only inside run_aldvc


def test_checkpoint_rejects_other_runs(sequence, tmp_path):
    ck = tmp_path / "ck"
    para = _para()
    run_aldvc(para, sequence[:2], checkpoint_dir=ck, compute_strain=False)
    with pytest.raises(CheckpointMismatch):
        run_aldvc(_para(winsize=24), sequence[:2], checkpoint_dir=ck, compute_strain=False)
    with pytest.raises(CheckpointMismatch):
        run_aldvc(para, sequence[:3], checkpoint_dir=ck, compute_strain=False)  # different number of frames
    meta = json.loads((ck / META_NAME).read_text(encoding="utf-8"))
    assert meta["n_frames"] == 2 and meta["para"]["winsize"] == [16, 16, 16]
    # resume=False starts over and rewrites the meta
    r = run_aldvc(_para(winsize=24), sequence[:2], checkpoint_dir=ck, resume=False, compute_strain=False)
    meta = json.loads((ck / META_NAME).read_text(encoding="utf-8"))
    assert meta["para"]["winsize"] == [24, 24, 24] and r.n_frames == 1
