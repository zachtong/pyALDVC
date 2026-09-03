"""Tests for reading MATLAB ALDVC result files."""

import numpy as np
import pytest
from scipy.io import savemat

from al_dvc.core.data_structures import F_to_matlab_order
from al_dvc.io.matlab_results import (
    interleaved_to_F,
    interleaved_to_U,
    load_matlab_results,
    match_nodes,
)


def _fake_results(path, n_frames=2):
    rng = np.random.default_rng(0)
    x0 = np.arange(20, 60, 8, dtype=float)  # 1-based MATLAB coordinates
    y0 = np.arange(16, 40, 8, dtype=float)
    z0 = np.arange(12, 36, 8, dtype=float)
    Z, Y, X = np.meshgrid(z0, y0, x0, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    n = coords.shape[0]
    U = [rng.normal(size=(n, 3)) for _ in range(n_frames)]
    F = [rng.normal(size=(n, 3, 3)) for _ in range(n_frames)]
    disp = np.empty((n_frames,), dtype=object)
    grad = np.empty((n_frames,), dtype=object)
    mubeta = np.empty((n_frames,), dtype=object)
    conv = np.empty((n_frames,), dtype=object)
    for k in range(n_frames):
        disp[k] = {
            "U": U[k].reshape(-1, 1),
            "U_local_ICGN": (U[k] + 0.1).reshape(-1, 1),
            "U0_crosscorr": np.round(U[k]).reshape(-1, 1),
        }
        grad[k] = {"F": F_to_matlab_order(F[k]).reshape(-1, 1), "F_local_ICGN": F_to_matlab_order(F[k] + 0.01).reshape(-1, 1)}
        mubeta[k] = {"ALVarBeta": 0.02 * (k + 1), "ALVarMu": 1e-3}
        conv[k] = {"ConvItPerEle": np.full((n, 6), 5.0 + k)}
    savemat(
        str(path),
        {
            "DVCpara": {
                "winsize": np.array([32, 32, 32]),
                "winstepsize": 8,
                "gridRange": {"gridxRange": np.array([10, 70]), "gridyRange": np.array([8, 48]), "gridzRange": np.array([4, 44])},
                "Subpb2FDOrFEM": "finiteDifference",
                "interpMethod": "cubic",
            },
            "DVCmesh": {"coordinatesFEM": coords, "elementsFEM": np.arange(1, 9, dtype=float).reshape(1, 8)},
            "ResultDisp": disp,
            "ResultDefGrad": grad,
            "ResultMuBeta": mubeta,
            "ResultConvItPerEle": conv,
            "fileNameAll": np.array(["a_01.mat", "a_02.mat"], dtype=object),
        },
    )
    return coords, U, F


def test_interleave_roundtrip():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(7, 3))
    assert np.array_equal(interleaved_to_U(U.reshape(-1)), U)
    F = rng.normal(size=(7, 3, 3))
    assert np.array_equal(interleaved_to_F(F_to_matlab_order(F)), F)
    # MATLAB per-node order is column-major: [F11, F21, F31, F12, ...]
    vec = np.arange(1, 10, dtype=float)
    F1 = interleaved_to_F(vec)[0]
    assert F1[0, 0] == 1 and F1[1, 0] == 2 and F1[2, 0] == 3 and F1[0, 1] == 4 and F1[2, 2] == 9
    with pytest.raises(ValueError):
        interleaved_to_U(np.zeros(4))
    with pytest.raises(ValueError):
        interleaved_to_F(np.zeros(10))


def test_load_matlab_results(tmp_path):
    p = tmp_path / "results_ws32_st8.mat"
    coords, U, F = _fake_results(p)
    res = load_matlab_results(p)
    assert res.winsize == (32, 32, 32) and res.winstepsize == (8, 8, 8)
    assert res.file_names == ["a_01.mat", "a_02.mat"]
    assert res.grid_range == {"x": (9, 69), "y": (7, 47), "z": (3, 43)}
    assert res.para["Subpb2FDOrFEM"] == "finiteDifference"
    assert np.array_equal(res.coordinates, coords - 1)  # 0-based
    assert res.elements.min() == 0
    assert len(res.frames) == 2
    for k, fr in enumerate(res.frames):
        assert np.allclose(fr.U, U[k])
        assert np.allclose(fr.U_local, U[k] + 0.1)
        assert np.allclose(fr.U0, np.round(U[k]))
        assert np.allclose(fr.F, F[k])
        assert np.allclose(fr.F_local, F[k] + 0.01)
        assert fr.beta == pytest.approx(0.02 * (k + 1)) and fr.mu == pytest.approx(1e-3)
        assert fr.conv_iter.shape == (coords.shape[0], 6) and fr.conv_iter[0, 0] == 5 + k
    with pytest.raises(FileNotFoundError):
        load_matlab_results(tmp_path / "missing.mat")
    savemat(str(tmp_path / "other.mat"), {"vol": np.zeros((2, 2, 2))})
    with pytest.raises(KeyError):
        load_matlab_results(tmp_path / "other.mat")


def test_match_nodes():
    rng = np.random.default_rng(2)
    b = rng.integers(0, 50, size=(200, 3)).astype(float)
    b = np.unique(b, axis=0)
    perm = rng.permutation(b.shape[0])
    a = np.vstack([b[perm[:50]] + 1e-6, [[999.0, 999.0, 999.0]]])
    idx = match_nodes(a, b)
    assert np.array_equal(idx[:50], perm[:50])
    assert idx[50] == -1
