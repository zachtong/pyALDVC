import numpy as np
import pytest

from al_dvc.core.config import DVCPara, dvcpara_default, para_from_dict, para_to_dict
from al_dvc.core.data_structures import (
    F_to_matlab_order,
    FrameSchedule,
    P_from_UF,
    UF_from_P,
    VOIRange,
)


def test_defaults_are_valid():
    p = dvcpara_default()
    assert isinstance(p, DVCPara)
    assert p.winsize == (32, 32, 32)
    assert p.winstepsize == (16, 16, 16)


def test_scalar_broadcast():
    p = dvcpara_default(winsize=24, winstepsize=8, search_radius=3, voxel_size=2.5)
    assert p.winsize == (24, 24, 24)
    assert p.search_radius == (3, 3, 3)
    assert p.voxel_size == (2.5, 2.5, 2.5)


def test_replace_broadcasts_and_validates():
    from dataclasses import replace

    p = dvcpara_default(winsize=16)
    q = replace(p, search_radius=4, winsize=24, voi={"x": (0, 5), "y": (0, 5), "z": (0, 5)})
    assert q.search_radius == (4, 4, 4) and q.winsize == (24, 24, 24)
    assert isinstance(q.voi, VOIRange)
    with pytest.raises(ValueError):
        replace(p, mu=-1.0)
    assert DVCPara(winsize=20).winsize == (20, 20, 20)


def test_tuple_and_dict_inputs():
    p = dvcpara_default(winsize=[16, 20, 24], voi={"x": (0, 10), "y": (0, 20), "z": (0, 30)})
    assert p.winsize == (16, 20, 24)
    assert p.voi.z == (0, 30)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(winsize=15),
        dict(winsize=2),
        dict(winstepsize=0),
        dict(mu=0),
        dict(icgn_tol=2),
        dict(icgn_dp_tol=0),
        dict(admm_max_iter=0),
        dict(subpb2_method="spectral"),
        dict(strain_type="log"),
        dict(interp_method="quintic"),
        dict(min_valid_ratio=0),
        dict(voxel_size=(1, 0, 1)),
        dict(gauss_pt_order=4),
        dict(init_guess_method="fft"),
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises((ValueError, TypeError)):
        dvcpara_default(**kwargs)


def test_unknown_field_raises():
    with pytest.raises(TypeError):
        dvcpara_default(window_size=32)


def test_para_roundtrip():
    p = dvcpara_default(winsize=20, beta=1e-3, frame_schedule=FrameSchedule((0, 1, 2)))
    d = para_to_dict(p)
    q = para_from_dict(d)
    assert q.winsize == p.winsize
    assert q.beta == p.beta
    assert q.frame_schedule.ref_indices == (0, 1, 2)


def test_voi_clamp():
    v = VOIRange(x=(-5, 999), y=(3, 10), z=(0, -1)).clamp((20, 30, 40))
    assert v.x == (0, 39)
    assert v.y == (3, 10)
    assert v.z == (0, 19)
    assert v.extent == (20, 8, 40)


def test_frame_schedule_modes():
    acc = FrameSchedule.from_mode("accumulative", 4)
    inc = FrameSchedule.from_mode("incremental", 4)
    assert acc.ref_indices == (0, 0, 0)
    assert inc.ref_indices == (0, 1, 2)
    assert inc.path_to_root(3) == [3, 2, 1, 0]
    with pytest.raises(ValueError):
        FrameSchedule((0, 2))
    every2 = FrameSchedule.from_every_n(2, 6)
    assert every2.ref_indices == (0, 0, 2, 2, 4)


def test_P_UF_roundtrip_and_matlab_order():
    rng = np.random.default_rng(0)
    U = rng.standard_normal((5, 3))
    F = rng.standard_normal((5, 3, 3))
    P = P_from_UF(U, F)
    U2, F2 = UF_from_P(P)
    assert np.allclose(U, U2) and np.allclose(F, F2)
    m = F_to_matlab_order(F)
    # MATLAB order per node: F11 F21 F31 F12 F22 F32 F13 F23 F33
    assert np.allclose(
        m[:9], [F[0, 0, 0], F[0, 1, 0], F[0, 2, 0], F[0, 0, 1], F[0, 1, 1], F[0, 2, 1], F[0, 0, 2], F[0, 1, 2], F[0, 2, 2]]
    )
