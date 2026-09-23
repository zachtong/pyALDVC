"""What a result records for its statistics: the median-test outliers, and a full round trip through .npz."""

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import STATUS_CONVERGED
from al_dvc.core.pipeline import run_aldvc


@pytest.fixture(scope="module")
def pair(affine_pair):
    f, g, _ = affine_pair
    return f, g


def _flag_one_node(monkeypatch, target: dict):
    """Make the median test reject exactly one converged node, the centre of the grid."""
    import importlib

    local_icgn = importlib.import_module("al_dvc.solver.local_icgn")  # the package re-exports a function of that name
    subpb1_solver = importlib.import_module("al_dvc.solver.subpb1_solver")

    def fake_median_test(U_grid, good_grid, threshold, edge_ok=None):
        flag = np.zeros(good_grid.shape, dtype=bool)
        idx = tuple(s // 2 for s in good_grid.shape)
        flag[idx] = True
        target["index"] = int(np.ravel_multi_index(idx, good_grid.shape))
        return flag

    monkeypatch.setattr(local_icgn, "universal_median_test", fake_median_test)
    monkeypatch.setattr(subpb1_solver, "universal_median_test", fake_median_test)


def test_the_median_test_outliers_are_recorded(pair, monkeypatch):
    """A node the median test rejected carries a replaced value, not a measurement; the result now says which."""
    target: dict = {}
    _flag_one_node(monkeypatch, target)
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False, admm_max_iter=2)
    res = run_aldvc(para, list(pair), compute_strain=False)
    fr = res.result_disp[0]
    assert fr.outlier is not None and fr.outlier.dtype == bool and fr.outlier.shape == (res.dvc_mesh.n_nodes,)
    assert fr.outlier[target["index"]]
    assert not np.any(fr.outlier & (fr.status != STATUS_CONVERGED))  # an outlier is a converged node that was rejected
    assert int(fr.outlier.sum()) == 1


def test_a_result_round_trips_through_npz(pair, tmp_path):
    from al_dvc.export.export_npz import export_npz, result_from_npz
    from al_dvc.export.export_utils import available_fields, field_array

    para = dvcpara_default(
        winsize=16, winstepsize=8, search_radius=5, verbose=False, admm_max_iter=2, voxel_size=(2.0, 2.0, 2.0), units="um"
    )
    res = run_aldvc(para, list(pair))
    back = result_from_npz(export_npz(res, tmp_path / "r.npz"))
    assert back.dvc_para == res.dvc_para
    assert back.frame_schedule.ref_indices == res.frame_schedule.ref_indices
    assert tuple(back.volume_shape) == tuple(res.volume_shape)
    m0, m1 = res.dvc_mesh, back.dvc_mesh
    np.testing.assert_array_equal(m0.coordinates, m1.coordinates)
    np.testing.assert_array_equal(m0.node_valid, m1.node_valid)
    np.testing.assert_array_equal(np.sort(m0.boundary_nodes), np.sort(m1.boundary_nodes))
    assert tuple(m0.grid_shape) == tuple(m1.grid_shape) and tuple(m0.spacing) == tuple(m1.spacing)
    f0, f1 = res.result_disp[0], back.result_disp[0]
    for name in ("U", "F", "U_accum", "zncc", "status", "U_std", "outlier"):
        np.testing.assert_array_equal(getattr(f0, name), getattr(f1, name), err_msg=name)
    assert available_fields(back) == available_fields(res)
    for name in available_fields(res):
        np.testing.assert_array_equal(field_array(res, 0, name), field_array(back, 0, name), err_msg=name)


def test_the_checkpoint_keeps_the_outliers(pair, tmp_path):
    from al_dvc.core.checkpoint import Checkpoint

    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False, admm_max_iter=2)
    res = run_aldvc(para, list(pair), compute_strain=False)
    (tmp_path / "ck").mkdir()
    ckpt = Checkpoint(tmp_path / "ck")
    ckpt.save(1, res.result_disp[0], res.dvc_mesh)
    fr, _mesh = ckpt.load(1, res.dvc_mesh)
    np.testing.assert_array_equal(fr.outlier, res.result_disp[0].outlier)
