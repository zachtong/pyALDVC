"""Per-node displacement uncertainty: predicted standard deviation vs empirical error."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import gaussian_filter

from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import STATUS_CONVERGED
from al_dvc.core.pipeline import run_aldvc
from al_dvc.export import export_npz, export_vtk, field_array, load_npz_result
from al_dvc.synthetic import add_noise, warp_volume_lagrangian
from tests.conftest import gt_at, interior_mask


def _run_local(f, g, noise, seeds=(1, 2), **kw):
    fn = add_noise(f, noise, seed=seeds[0]) if noise > 0 else f
    gn = add_noise(g, noise, seed=seeds[1]) if noise > 0 else g
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False, use_global_step=False, **kw)
    return run_aldvc(para, [fn, gn], compute_strain=False)


def test_predicted_std_tracks_empirical_error(affine_pair):
    f, g, disp = affine_pair
    mean_pred = []
    for noise in (0.005, 0.01, 0.02):
        r = _run_local(f, g, noise)
        fr = r.result_disp[0]
        mesh = r.dvc_mesh
        assert fr.U_std is not None and fr.U_std.shape == (mesh.n_nodes, 3)
        U_gt, _ = gt_at(mesh, disp)
        ok = interior_mask(mesh) & (fr.status == STATUS_CONVERGED) & np.all(np.isfinite(fr.U_std), axis=1)
        assert ok.sum() > 40
        emp = np.sqrt(np.mean((fr.U - U_gt)[ok] ** 2, axis=0))
        pred = np.sqrt(np.mean(fr.U_std[ok] ** 2, axis=0))
        ratio = pred / emp
        assert np.all((0.6 < ratio) & (ratio < 1.5)), (noise, emp, pred)
        mean_pred.append(float(pred.mean()))
    assert mean_pred[0] < mean_pred[1] < mean_pred[2]
    # at SNR ~ 1.5 the linearised theory underestimates the error by about 2x (documented limitation)
    r = _run_local(f, g, 0.04)
    fr = r.result_disp[0]
    U_gt, _ = gt_at(r.dvc_mesh, disp)
    ok = interior_mask(r.dvc_mesh) & (fr.status == STATUS_CONVERGED) & np.all(np.isfinite(fr.U_std), axis=1)
    ratio = np.sqrt(np.mean(fr.U_std[ok] ** 2, axis=0)) / np.sqrt(np.mean((fr.U - U_gt)[ok] ** 2, axis=0))
    assert np.all((0.3 < ratio) & (ratio < 1.5)), ratio
    # nodes that did not converge carry no estimate
    r = _run_local(f, g, 0.0)
    fr = r.result_disp[0]
    assert np.all(np.isnan(fr.U_std[fr.status != STATUS_CONVERGED]))


def test_uncertainty_reflects_texture_anisotropy(affine_pair):
    f, g, disp = affine_pair
    f_b = gaussian_filter(f, sigma=(3.5, 0.0, 0.0), mode="nearest")  # weak texture along z
    g_b = warp_volume_lagrangian(f_b, disp)
    r = _run_local(f_b, g_b, 0.02)
    fr = r.result_disp[0]
    ok = interior_mask(r.dvc_mesh) & (fr.status == STATUS_CONVERGED)
    std = fr.U_std[ok]
    assert np.nanmedian(std[:, 2]) > 1.25 * np.nanmedian(std[:, 0])
    U_gt, _ = gt_at(r.dvc_mesh, disp)
    emp = np.sqrt(np.mean((fr.U - U_gt)[ok] ** 2, axis=0))
    assert emp[2] > 1.25 * emp[0]


def test_uncertainty_exports(affine_pair, tmp_path):
    f, g, _ = affine_pair
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False, admm_max_iter=2, voxel_size=(2.0, 2.0, 2.0))
    r = run_aldvc(para, [f, g], compute_strain=False)
    fr = r.result_disp[0]
    assert fr.U_std is not None and np.isfinite(fr.U_std).any()
    s = field_array(r, 0, "disp_std_u")
    assert np.allclose(s[np.isfinite(s)], 2.0 * fr.U_std[np.isfinite(s), 0])
    mag = field_array(r, 0, "disp_std")
    assert np.all(mag[np.isfinite(mag)] >= 0)
    d = load_npz_result(export_npz(r, tmp_path / "r.npz"))
    assert np.allclose(d["U_std_1"], fr.U_std, equal_nan=True)
    vtks = export_vtk(r, tmp_path / "vtk", fields=["disp_std"])
    names = {da.attrib["Name"] for da in ET.parse(vtks[0]).getroot().iter("DataArray")}
    assert {"displacement_std", "disp_std"} <= names
