"""al_dvc.analysis: summary statistics, rigid-body motion, node selection, noise floor, homogeneous deformation.

Most tests build a result whose displacement is known exactly (a rigid motion, an affine field, a field plus
known noise), so every number is checked against its closed form.
"""

import json
import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from al_dvc.analysis import (
    MotionFit,
    NodeFilter,
    fit_motion,
    frame_series,
    frame_stats,
    frame_view,
    homogeneous,
    noise_floor,
    remove_motion,
    select_nodes,
    spatial_temporal_std,
    strain_precision,
    summarize,
    vector_stats,
    vsg_size,
)
from al_dvc.analysis.motion import corrected_gradient
from al_dvc.core.config import dvcpara_default
from al_dvc.core.data_structures import FrameResult, FrameSchedule, PipelineResult
from al_dvc.mesh.grid_mesh import grid_surface_nodes, mesh_setup
from al_dvc.strain.compute_strain import compute_strain
from al_dvc.strain.strain_types import strain_tensor

STEP = 8


def make_result(
    disp_fns,
    *,
    voxel_size=(1.0, 1.0, 1.0),
    strain=True,
    strain_type="infinitesimal",
    status=None,
    zncc=None,
    outlier=None,
    node_valid=None,
    split=None,
    ref_indices=None,
    n_axis=(9, 10, 11),
):
    """A result on a regular node grid whose cumulative displacement (voxels) is ``fn(x, y, z)`` per frame."""
    x0 = 20.0 + STEP * np.arange(n_axis[0])
    y0 = 24.0 + STEP * np.arange(n_axis[1])
    z0 = 28.0 + STEP * np.arange(n_axis[2])
    mesh = mesh_setup(x0, y0, z0)
    n = mesh.n_nodes
    mesh = replace(
        mesh,
        node_valid=np.ones(n, dtype=bool) if node_valid is None else np.asarray(node_valid, dtype=bool),
        boundary_nodes=grid_surface_nodes(mesh.grid_shape),
    )
    scaled = any(abs(v - 1.0) > 1e-12 for v in voxel_size)
    para = dvcpara_default(
        winsize=16,
        winstepsize=STEP,
        voxel_size=voxel_size,
        units="um" if scaled else "voxel",
        strain_type=strain_type,
        verbose=False,
    )
    ref_indices = tuple(ref_indices) if ref_indices is not None else tuple(0 for _ in disp_fns)
    frames = []
    for k, fn in enumerate(disp_fns):
        X = mesh.coordinates
        U = np.column_stack(fn(X[:, 0], X[:, 1], X[:, 2])).astype(np.float64)
        frames.append(
            FrameResult(
                U=U,
                F=np.zeros((n, 3, 3)),
                ref_frame=ref_indices[k],
                U_accum=U,
                zncc=np.full(n, 0.95) if zncc is None else np.asarray(zncc[k] if np.ndim(zncc) == 2 else zncc, dtype=float),
                status=np.zeros(n, dtype=np.int8)
                if status is None
                else np.asarray(status[k] if np.ndim(status) == 2 else status, dtype=np.int8),
                outlier=np.zeros(n, dtype=bool) if outlier is None else np.asarray(outlier, dtype=bool),
                split_fraction=split,
            )
        )
    strains = [compute_strain(mesh, para, fr.U_accum) for fr in frames] if strain else []
    return PipelineResult(
        dvc_para=para,
        dvc_mesh=mesh,
        result_disp=frames,
        result_strain=strains,
        frame_schedule=FrameSchedule(ref_indices=ref_indices),
        volume_shape=(200, 200, 200),
    )


def rigid_in_physical(rotvec_deg, t_phys, centre_phys, voxel_size):
    """Displacement function (voxels) of a rigid motion defined in physical space."""
    R = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    vs = np.asarray(voxel_size, dtype=float)
    c = np.asarray(centre_phys, dtype=float)

    def fn(x, y, z):
        Xp = np.column_stack([x, y, z]) * vs
        up = (Xp - c) @ R.T + c + np.asarray(t_phys) - Xp
        uv = up / vs
        return uv[:, 0], uv[:, 1], uv[:, 2]

    return fn, R


# ------------------------------------------------------------------ summary statistics
def test_summary_statistics_by_hand():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0, np.nan])
    s = summarize(x)
    v = x[:5]
    assert s.n == 5
    assert s.mean == pytest.approx(22.0)
    assert s.std == pytest.approx(np.sqrt(np.mean((v - 22.0) ** 2)))  # population, as DVC Challenge 2.0 Eq. 2
    assert s.median == 3.0 and s.min == 1.0 and s.max == 100.0
    assert s.robust_std == pytest.approx(1.4826 * 1.0)  # median |x - 3| = 1: the wild 100 does not move it
    assert s.p05 == pytest.approx(np.percentile(v, 5)) and s.p95 == pytest.approx(np.percentile(v, 95))
    assert s.rms == pytest.approx(np.sqrt(np.mean(v**2)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an empty selection is a normal case, not a warning storm
        empty = summarize(np.array([np.nan, np.nan]))
    assert empty.n == 0 and np.isnan(empty.mean) and np.isnan(empty.std)


def test_bias_noise_floor_and_rms_follow_the_paper():
    rng = np.random.default_rng(1)
    nominal = np.array([0.5, 0.0, -1.0])
    U = nominal + rng.normal([0.04, -0.02, 0.01], [0.02, 0.03, 0.05], size=(400, 3))
    U[7] = np.nan  # a node without a value is not counted
    s = vector_stats(U, nominal)
    E = np.delete(U, 7, axis=0) - nominal
    assert s.n == 399
    np.testing.assert_allclose(s.bias, E.mean(axis=0))  # Eq. 1
    np.testing.assert_allclose(s.noise, E.std(axis=0, ddof=0))  # Eq. 2
    assert s.u_rms == pytest.approx(np.sqrt(np.sum(s.bias**2) + np.sum(s.noise**2)))  # SI Eq. S10


def test_maer_and_sder():
    E6 = np.array([[1e-4, -2e-4, 3e-4, 0.0, 6e-4, 0.0], [-1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], [np.nan, 0, 0, 0, 0, 0]])
    maer, sder = strain_precision(E6)
    e = np.abs(E6[:2]).mean(axis=1)
    assert maer == pytest.approx(e.mean()) and sder == pytest.approx(e.std())


def test_spatial_and_temporal_standard_deviations():
    """iDICs GPG: spatial = std over the region, averaged over images; temporal = std over time, averaged over points."""
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 0.03, size=(200, 3))  # fixed spatial pattern
    b = rng.normal(0.0, 0.01, size=(5, 3))  # frame-to-frame drift
    U = a[None, :, :] + b[:, None, :]
    spatial, temporal = spatial_temporal_std(U)
    np.testing.assert_allclose(spatial, a.std(axis=0))
    np.testing.assert_allclose(temporal, b.std(axis=0))


# ------------------------------------------------------------------ rigid-body motion
def _points(n=300, seed=3, planar=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 500, size=(n, 3))
    if planar:
        X[:, 2] = 250.0
    return X


def test_a_rigid_motion_is_recovered_exactly():
    X = _points()
    R = Rotation.from_rotvec(np.radians([3.0, -12.0, 25.0])).as_matrix()
    c, t = X.mean(axis=0), np.array([4.0, -2.5, 80.0])
    U = (X - c) @ R.T + c + t - X
    fit = fit_motion(X, U, "rigid")
    assert isinstance(fit, MotionFit) and fit.n == 300
    np.testing.assert_allclose(fit.matrix, R, atol=1e-12)
    np.testing.assert_allclose(fit.translation, t, atol=1e-9)
    assert fit.residual_rms < 1e-9
    assert fit.rotation_deg == pytest.approx(np.degrees(np.linalg.norm(np.radians([3.0, -12.0, 25.0]))))
    assert np.abs(remove_motion(X, U, fit)).max() < 1e-9


def test_translation_is_the_mean_and_affine_recovers_the_gradient():
    X = _points()
    rng = np.random.default_rng(4)
    U = np.array([1.0, 2.0, 3.0]) + rng.normal(0, 0.1, size=X.shape)
    fit = fit_motion(X, U, "translation")
    np.testing.assert_allclose(fit.translation, U.mean(axis=0))
    np.testing.assert_allclose(remove_motion(X, U, fit), U - U.mean(axis=0))
    H = np.array([[0.02, 0.003, 0.0], [0.001, -0.01, 0.002], [0.0, 0.004, 0.005]])
    c = X.mean(axis=0)
    U = np.array([0.5, -0.3, 0.2]) + (X - c) @ H.T
    fit = fit_motion(X, U, "affine")
    np.testing.assert_allclose(fit.matrix, np.eye(3) + H, atol=1e-12)
    assert np.abs(remove_motion(X, U, fit)).max() < 1e-9


def test_a_planar_node_set_never_returns_a_reflection():
    X = _points(planar=True)
    R = Rotation.from_rotvec(np.radians([0.0, 0.0, 30.0])).as_matrix()
    U = X @ R.T - X
    fit = fit_motion(X, U, "rigid")
    assert np.linalg.det(fit.matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(fit.matrix, R, atol=1e-10)


def test_the_rigid_fit_agrees_with_scipy_under_noise_and_honours_weights():
    X = _points()
    R = Rotation.from_rotvec(np.radians([10.0, 5.0, -8.0])).as_matrix()
    rng = np.random.default_rng(5)
    U = X @ R.T - X + rng.normal(0, 0.5, size=X.shape)
    fit = fit_motion(X, U, "rigid")
    P, Q = X - X.mean(axis=0), X + U - (X + U).mean(axis=0)
    oracle = Rotation.align_vectors(Q, P)[0].as_matrix()
    np.testing.assert_allclose(fit.matrix, oracle, atol=1e-9)
    U_bad = X @ R.T - X
    U_bad[:20] += 50.0  # wild nodes
    w = np.ones(len(X))
    w[:20] = 0.0
    np.testing.assert_allclose(fit_motion(X, U_bad, "rigid", weights=w).matrix, R, atol=1e-10)


def test_removing_a_rotation_keeps_green_lagrange_and_corrects_infinitesimal_strain():
    rng = np.random.default_rng(6)
    H = rng.normal(0, 0.01, size=(50, 3, 3))
    R0 = Rotation.from_rotvec(np.radians([0.0, 0.0, 2.0])).as_matrix()
    H_rot = np.einsum("ij,njk->nik", R0, np.eye(3) + H) - np.eye(3)  # the same deformation, rotated by 2 deg
    fit = MotionFit(kind="rigid", n=50, centre=np.zeros(3), translation=np.zeros(3), matrix=R0, residual_rms=0.0)
    np.testing.assert_allclose(corrected_gradient(H_rot, fit), H, atol=1e-12)
    np.testing.assert_allclose(strain_tensor(H_rot, "green_lagrange"), strain_tensor(H, "green_lagrange"), atol=1e-12)
    # a pure 2-degree rotation reads as a normal strain of cos(2 deg) - 1 = -6.1e-4 in infinitesimal strain
    pure = strain_tensor((R0 - np.eye(3))[None], "infinitesimal")[0]
    assert pure[0, 0] == pytest.approx(np.cos(np.radians(2.0)) - 1.0)


# ------------------------------------------------------------------ node selection
def test_each_filter_removes_its_nodes_and_the_counts_add_up():
    zero = lambda x, y, z: (0 * x, 0 * y, 0 * z)  # noqa: E731
    res = make_result([zero], strain=False)
    n = res.dvc_mesh.n_nodes
    valid = np.ones(n, dtype=bool)
    valid[:40] = False
    status = np.zeros(n, dtype=np.int8)
    status[40:70] = 1
    outlier = np.zeros(n, dtype=bool)
    outlier[70:75] = True
    zncc = np.full(n, 0.95)
    zncc[75:90] = 0.5
    split = np.ones(n, dtype=np.float32)
    split[90:96] = 0.6
    res = make_result([zero], strain=False, node_valid=valid, status=status, outlier=outlier, zncc=zncc, split=split)
    default = select_nodes(res, 0)
    assert default.candidates == n
    assert default.removed == {"invalid": 40, "not_converged": 30, "outlier": 5, "low_zncc": 0, "edge": 0, "cut": 0}
    assert default.n == n - 75 and default.valid_fraction == pytest.approx((n - 75) / n)
    strict = select_nodes(res, 0, NodeFilter(min_zncc=0.8, drop_cut=True))
    assert strict.removed["low_zncc"] == 15 and strict.removed["cut"] == 6 and strict.n == n - 96
    loose = select_nodes(res, 0, NodeFilter(converged_only=False, drop_outliers=False))
    assert loose.n == n - 40  # invalid nodes are never used: nothing was measured there
    edge = select_nodes(make_result([zero], strain=False), 0, NodeFilter(edge_layers=1))
    nz, ny, nx = res.dvc_mesh.grid_shape
    assert edge.n == (nz - 2) * (ny - 2) * (nx - 2) and edge.removed["edge"] == n - edge.n
    region = np.zeros(n, dtype=bool)
    region[100:200] = True
    assert select_nodes(res, 0, NodeFilter(), region=region).candidates == 100


def test_incremental_frames_use_the_nodes_every_pair_measured():
    zero = lambda x, y, z: (0 * x, 0 * y, 0 * z)  # noqa: E731
    res = make_result([zero, zero], strain=False, ref_indices=(0, 1))
    n = res.dvc_mesh.n_nodes
    status = np.zeros((2, n), dtype=np.int8)
    status[0, 5] = 1  # the first pair lost node 5; frame 2 is composed through it
    res = make_result([zero, zero], strain=False, ref_indices=(0, 1), status=status)
    assert not select_nodes(res, 0).mask[5] and not select_nodes(res, 1).mask[5]


# ------------------------------------------------------------------ frames, corrections, series
def test_rigid_removal_leaves_the_deformation_and_the_objective_strain():
    """A 5-degree rigid rotation on top of a 1 % stretch: the rotation is found, the stretch is what remains."""
    centre = (60.0, 60.0, 70.0)
    rigid, R = rigid_in_physical([0.0, 0.0, 5.0], [3.0, -1.0, 8.0], centre, (1.0, 1.0, 1.0))

    def fn(x, y, z):  # stretch first (about the centre), then the rigid motion
        xs = centre[0] + 1.01 * (x - centre[0])
        u, v, w = rigid(xs, y, z)
        return u + (xs - x), v, w

    for measure in ("infinitesimal", "green_lagrange"):
        res = make_result([fn], strain_type=measure)
        raw = frame_view(res, 0)
        view = frame_view(res, 0, motion="rigid")
        np.testing.assert_allclose(view.fit.matrix, R, atol=2e-3)  # the stretch is not symmetric about every axis
        stats = frame_stats(res, 0, ("exx", "eyy", "disp_v"), motion="rigid").stats
        if measure == "green_lagrange":  # objective: the rotation never changed it
            np.testing.assert_allclose(view.values("exx"), raw.values("exx"), atol=1e-12)
            assert stats["exx"].median == pytest.approx(0.01 + 0.5 * 0.01**2, abs=1e-6)
        else:  # not objective: 5 degrees read as cos(5 deg) - 1 = -3.8e-3 before the correction
            assert np.nanmedian(raw.values("eyy")) == pytest.approx(np.cos(np.radians(5.0)) - 1.0, abs=2e-4)
            assert stats["eyy"].median == pytest.approx(0.0, abs=2e-4)
            assert stats["exx"].median == pytest.approx(0.01, abs=3e-4)
        assert abs(stats["disp_v"].mean) < 0.05  # the rigid translation and rotation are gone from v


def test_the_corrected_strain_is_the_strain_of_the_corrected_displacement():
    rigid, _R = rigid_in_physical([4.0, -3.0, 6.0], [1.0, 2.0, 3.0], (60.0, 60.0, 70.0), (1.0, 1.0, 1.0))
    rng = np.random.default_rng(7)

    def fn(x, y, z):
        u, v, w = rigid(x, y, z)
        return u + 0.002 * np.sin(y / 9.0), v + 0.003 * np.cos(x / 7.0), w + 1e-3 * rng.normal(size=x.shape)

    res = make_result([fn])
    view = frame_view(res, 0, motion="rigid")
    vs = np.asarray(res.dvc_para.voxel_size)
    direct = compute_strain(res.dvc_mesh, res.dvc_para, view.displacement() / vs)
    for name in ("exx", "eyy", "ezz", "exy", "exz", "eyz", "rotation_deg"):
        np.testing.assert_allclose(view.values(name), direct.field(name), atol=1e-10, err_msg=name)


def test_the_fit_is_made_in_physical_units():
    """Anisotropic voxels: a rigid rotation in micrometres is not a rotation in voxel indices."""
    vs = (1.0, 1.0, 2.5)
    fn, R = rigid_in_physical([6.0, 4.0, -9.0], [2.0, 0.0, -5.0], (60.0, 60.0, 150.0), vs)
    res = make_result([fn], voxel_size=vs, strain=False)
    view = frame_view(res, 0, motion="rigid")
    np.testing.assert_allclose(view.fit.matrix, R, atol=1e-10)
    assert view.fit.residual_rms < 1e-8
    assert np.nanmax(np.abs(view.displacement())) < 1e-8


def test_frame_stats_series_and_the_valid_fraction_flag():
    fns = [lambda x, y, z, a=a: (a + 0 * x, 0 * y, 0.01 * (z - 70.0)) for a in (0.1, 0.2, 0.3)]
    res = make_result(fns)
    series = frame_series(res, ("disp_u", "ezz"))
    assert [fs.frame for fs in series] == [0, 1, 2]
    assert [fs.stats["disp_u"].mean for fs in series] == pytest.approx([0.1, 0.2, 0.3])
    assert series[0].stats["ezz"].median == pytest.approx(0.01)
    assert all(fs.flag == "ok" for fs in series)
    seen = []
    assert len(frame_series(res, ("disp_u",), stop=lambda: len(seen) >= 2, progress=lambda i, n: seen.append(i))) == 2
    n = res.dvc_mesh.n_nodes
    status = np.ones(n, dtype=np.int8)
    status[:10] = 0
    few = make_result(fns[:1], status=status)
    assert frame_stats(few, 0, ("disp_u",), min_valid_fraction=0.5).flag == "few_nodes"


# ------------------------------------------------------------------ noise floor and homogeneous deformation
def test_the_noise_floor_of_a_static_pair():
    rng = np.random.default_rng(8)
    n_axis = (9, 10, 11)
    n = int(np.prod(n_axis))
    noise = rng.normal([0.03, -0.01, 0.02], [0.02, 0.025, 0.03], size=(n, 3))
    nominal = np.array([1.0, 0.0, 0.0])

    def fn(x, y, z):
        return nominal[0] + noise[:, 0], nominal[1] + noise[:, 1], nominal[2] + noise[:, 2]

    res = make_result([fn, fn], n_axis=n_axis)
    nf = noise_floor(res, 0, nominal=nominal)
    np.testing.assert_allclose(nf.displacement.bias, noise.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(nf.displacement.noise, noise.std(axis=0), atol=1e-12)
    assert nf.rigid is not None and np.all(np.abs(nf.rigid.bias) < 1e-9)  # the rigid fit takes the mean away
    assert set(nf.strain) == {"exx", "eyy", "ezz", "exy", "exz", "eyz"} and nf.maer > 0
    np.testing.assert_allclose(nf.vsg, vsg_size(res.dvc_para))
    assert nf.temporal_std is not None and np.allclose(nf.temporal_std, 0.0)  # two identical frames


def test_the_vsg_size_follows_the_good_practices_guide():
    para = dvcpara_default(winsize=32, winstepsize=8, strain_plane_fit_halfwidth=2)
    np.testing.assert_allclose(vsg_size(para), [4 * 8 + 33] * 3)  # (window - 1) step + subset, window 5, span 33
    para = dvcpara_default(winsize=32, winstepsize=8, strain_method="fd")
    np.testing.assert_allclose(vsg_size(para), [2 * 8 + 33] * 3)


def test_the_homogeneous_deformation_of_an_affine_field():
    H = np.array([[0.02, 0.004, 0.0], [0.0, -0.01, 0.003], [0.001, 0.0, 0.006]])
    c = np.array([60.0, 60.0, 70.0])

    def fn(x, y, z):
        d = np.column_stack([x, y, z]) - c
        u = np.array([0.4, -0.2, 1.0]) + d @ H.T
        return u[:, 0], u[:, 1], u[:, 2]

    res = make_result([fn])
    hom = homogeneous(res, 0)
    np.testing.assert_allclose(hom.F, np.eye(3) + H, atol=1e-10)
    np.testing.assert_allclose(hom.infinitesimal, 0.5 * (H + H.T), atol=1e-10)
    np.testing.assert_allclose(hom.green_lagrange, 0.5 * ((np.eye(3) + H).T @ (np.eye(3) + H) - np.eye(3)), atol=1e-10)
    np.testing.assert_allclose(hom.rotation @ hom.stretch, hom.F, atol=1e-12)
    assert hom.nodal_mean["exx"] == pytest.approx(0.02, abs=1e-9)


# ------------------------------------------------------------------ exports and the command line
def test_the_series_csv_and_the_summary_json(tmp_path):
    from al_dvc.analysis.export_stats import stats_metadata, write_series_csv, write_summary_json

    fns = [lambda x, y, z, a=a: (a + 0 * x, 0 * y, 0 * z) for a in (0.1, 0.2)]
    res = make_result(fns)
    series = frame_series(res, ("disp_u", "exx"), motion="translation")
    meta = stats_metadata(res, NodeFilter(), "translation")
    path = write_series_csv(tmp_path / "s.csv", series, ("disp_u", "exx"), meta)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# pyALDVC") and any("ddof 0" in ln for ln in lines if ln.startswith("#"))
    header = next(ln for ln in lines if not ln.startswith("#")).split(",")
    assert header[:3] == ["frame", "nodes_used", "valid_fraction"] and "disp_u_mean" in header and "exx_std" in header
    rows = [ln.split(",") for ln in lines if not ln.startswith("#")][1:]
    assert [r[0] for r in rows] == ["1", "2"]  # frames are 1-based in files
    out = write_summary_json(tmp_path / "s.json", {"meta": meta, "frame": frame_stats(res, 0, ("disp_u",))})
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["meta"]["motion"] == "translation" and doc["frame"]["stats"]["disp_u"]["n"] == res.dvc_mesh.n_nodes
    assert "NaN" not in out.read_text(encoding="utf-8")


def test_al_dvc_stats_on_an_exported_result(tmp_path):
    from al_dvc import cli
    from al_dvc.export.export_npz import export_npz

    fns = [lambda x, y, z, a=a: (a + 0 * x, 0.001 * (y - 60.0), 0 * z) for a in (0.1, 0.2)]
    res = make_result(fns, voxel_size=(2.0, 2.0, 2.0))
    npz = export_npz(res, tmp_path / "run.npz")
    assert (
        cli.main(
            [
                "stats",
                str(npz),
                "--fields",
                "disp_u",
                "eyy",
                "--motion",
                "translation",
                "--noise-floor",
                "-o",
                str(tmp_path / "st"),
            ]
        )
        == 0
    )
    csv_text = (tmp_path / "st" / "statistics.csv").read_text(encoding="utf-8")
    assert "disp_u_mean" in csv_text and "eyy_median" in csv_text
    doc = json.loads((tmp_path / "st" / "statistics.json").read_text(encoding="utf-8"))
    assert doc["meta"]["unit"] == "um" and "noise_floor" in doc


def test_chain_pairs_falls_back_only_without_a_schedule():
    from al_dvc.analysis.selection import chain_pairs

    zero = lambda x, y, z: (0 * x, 0 * y, 0 * z)  # noqa: E731
    res = make_result([zero, zero], strain=False, ref_indices=(0, 1))
    assert chain_pairs(res, 1) == [1, 0]
    assert chain_pairs(replace(res, frame_schedule=None), 1) == [1]
    with pytest.raises(IndexError):
        chain_pairs(res, 5)  # a wrong frame is a caller's bug, not a reason to guess


def test_al_dvc_stats_keeps_the_homogeneous_fit_of_every_frame_that_has_one(tmp_path):
    """Review: one frame without enough nodes for an affine fit dropped the fits of all frames."""
    from al_dvc import cli
    from al_dvc.export.export_npz import export_npz

    n = 9 * 10 * 11
    status = np.zeros((2, n), dtype=np.int8)
    status[1, 3:] = 1  # frame 2: three converged nodes, one too few for an affine fit
    fns = [lambda x, y, z: (0.01 * (x - 60.0), 0 * y, 0 * z)] * 2
    npz = export_npz(make_result(fns, status=status), tmp_path / "run.npz")
    assert cli.main(["stats", str(npz), "--fields", "disp_u", "-o", str(tmp_path / "st")]) == 0
    doc = json.loads((tmp_path / "st" / "statistics.json").read_text(encoding="utf-8"))
    assert doc["homogeneous"][0] is not None and doc["homogeneous"][1] is None
