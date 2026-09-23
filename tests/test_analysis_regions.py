"""Phase 2 of the result statistics: regions, profiles, the virtual extensometer, confidence intervals of a mean,
the motion fitted over one region and removed everywhere, and the corrected result the main window shows."""

import json
from dataclasses import replace

import numpy as np
import pytest

from al_dvc.analysis import NodeFilter, frame_series, frame_stats, frame_view
from al_dvc.analysis.corrected import Correction, corrected_result
from al_dvc.analysis.profiles import axis_profile, extensometer, sample_line
from al_dvc.analysis.regions import Region, regions_from_dicts, regions_to_dicts
from al_dvc.analysis.stats import effective_sample_size, mean_confidence
from al_dvc.export.export_utils import field_array
from tests.test_analysis import make_result, rigid_in_physical

# node grid of make_result: x = 20 + 8 i (9 nodes), y = 24 + 8 j (10), z = 28 + 8 k (11)


def _coords():
    return make_result([lambda x, y, z: (0 * x, 0 * y, 0 * z)], strain=False).dvc_mesh.coordinates


# ------------------------------------------------------------------ regions
def test_every_region_shape_selects_the_nodes_inside_it():
    X = _coords()
    box = Region(1, "box", "box", {"lo": [30, 30, 40], "hi": [60, 50, 80]})
    inside = (X[:, 0] >= 30) & (X[:, 0] <= 60) & (X[:, 1] >= 30) & (X[:, 1] <= 50) & (X[:, 2] >= 40) & (X[:, 2] <= 80)
    np.testing.assert_array_equal(box.contains(X), inside)
    sphere = Region(2, "ball", "sphere", {"centre": [52, 60, 68], "radius": 17.0})
    np.testing.assert_array_equal(sphere.contains(X), np.linalg.norm(X - [52, 60, 68], axis=1) <= 17.0)
    cyl = Region(3, "rod", "cylinder", {"centre": [52, 60, 0], "radius": 12.0, "axis": "z", "lo": 40, "hi": 90})
    expect = (np.hypot(X[:, 0] - 52, X[:, 1] - 60) <= 12.0) & (X[:, 2] >= 40) & (X[:, 2] <= 90)
    np.testing.assert_array_equal(cyl.contains(X), expect)
    slab = Region(4, "layer", "slab", {"axis": "y", "lo": 40, "hi": 56})
    np.testing.assert_array_equal(slab.contains(X), (X[:, 1] >= 40) & (X[:, 1] <= 56))
    # drawn on a plane and extruded along its normal: a triangle on XZ, y from 30 to 70
    tri_params = {"plane": "xz", "outline": "polygon", "points": [[20, 28], [84, 28], [20, 108]], "lo": 30, "hi": 70}
    tri = Region(5, "tri", "prism", tri_params)
    a, b = X[:, 0] - 20, X[:, 2] - 28
    expect = (a / 64 + b / 80 < 1 - 1e-9) & (a > 1e-9) & (b > 1e-9) & (X[:, 1] >= 30) & (X[:, 1] <= 70)
    got = tri.contains(X)
    strictly = expect
    assert np.all(got[strictly]) and got.sum() >= strictly.sum()
    ell = Region(6, "e", "prism", {"plane": "xy", "outline": "ellipse", "points": [[30, 40], [70, 80]], "lo": 0, "hi": 1000})
    np.testing.assert_array_equal(ell.contains(X), ((X[:, 0] - 50) / 20) ** 2 + ((X[:, 1] - 60) / 20) ** 2 <= 1.0)
    rect = Region(7, "r", "prism", {"plane": "yz", "outline": "rect", "points": [[50, 90], [30, 40]], "lo": 44, "hi": 44})
    np.testing.assert_array_equal(
        rect.contains(X), (X[:, 1] >= 30) & (X[:, 1] <= 50) & (X[:, 2] >= 40) & (X[:, 2] <= 90) & (X[:, 0] == 44)
    )


def test_regions_round_trip_and_bad_ones_are_refused():
    regions = [
        Region(1, "box", "box", {"lo": [0, 0, 0], "hi": [10, 10, 10]}, color="#ff0000"),
        Region(2, "tri", "prism", {"plane": "xy", "outline": "polygon", "points": [[0, 0], [5, 0], [0, 5]], "lo": 0, "hi": 9}),
    ]
    back = regions_from_dicts(json.loads(json.dumps(regions_to_dicts(regions))))
    assert [r.as_dict() for r in back] == [r.as_dict() for r in regions]
    for bad in (
        {"id": 1, "name": "x", "shape": "cone", "params": {}},
        {"id": 1, "name": "x", "shape": "box", "params": {"lo": [0, 0, 0]}},
        {"id": 1, "name": "x", "shape": "sphere", "params": {"centre": [0, 0, 0], "radius": -1}},
        {
            "id": 1,
            "name": "x",
            "shape": "prism",
            "params": {"plane": "xy", "outline": "polygon", "points": [[0, 0], [1, 1]], "lo": 0, "hi": 1},
        },
        {"id": 1, "name": "x", "shape": "slab", "params": {"axis": "w", "lo": 0, "hi": 1}},
    ):
        with pytest.raises(ValueError):
            regions_from_dicts([bad])


# ------------------------------------------------------------------ one motion, fitted over one region, removed everywhere
def test_the_motion_is_fitted_once_and_removed_everywhere():
    """A rigid 'fixture' region and a deforming rest: fitted over the fixture, the rest keeps its deformation;
    fitted per region, a region's own rotation would be taken away too."""
    rigid, R = rigid_in_physical([0.0, 0.0, 3.0], [2.0, -1.0, 4.0], (60.0, 60.0, 70.0), (1.0, 1.0, 1.0))

    def fn(x, y, z):  # a stretch in x above z = 70 first, then the same rigid motion everywhere
        xs = x + np.where(z > 70, 0.02 * (x - 52.0), 0.0)
        u, v, w = rigid(xs, y, z)
        return u + (xs - x), v, w

    res = make_result([fn], strain=False)
    X = res.dvc_mesh.coordinates
    fixture = X[:, 2] < 60
    view = frame_view(res, 0, "rigid", region=~fixture, fit_region=fixture)
    np.testing.assert_allclose(view.fit.matrix, R, atol=1e-10)  # the fixture alone is rigid: exact
    u = view.displacement()[:, 0]
    np.testing.assert_allclose(u[fixture], 0.0, atol=1e-9)
    np.testing.assert_allclose(u[X[:, 2] > 70], 0.02 * (X[X[:, 2] > 70, 0] - 52.0), atol=1e-9)
    assert view.selection.n == int((~fixture).sum())  # the statistics stand on the region, the fit on the fixture
    fs = frame_stats(res, 0, ("disp_u",), motion="rigid", region=X[:, 2] > 70, fit_region=fixture)
    assert fs.stats["disp_u"].std == pytest.approx(np.std(0.02 * (X[X[:, 2] > 70, 0] - 52.0)), rel=1e-9)
    series = frame_series(res, ("disp_u",), motion="rigid", region=X[:, 2] > 70, fit_region=fixture)
    assert series[0].stats["disp_u"].std == pytest.approx(fs.stats["disp_u"].std)


# ------------------------------------------------------------------ profiles and the extensometer
def test_the_axis_profile_is_the_mean_of_every_node_layer():
    res = make_result([lambda x, y, z: (0.001 * (z - 28.0) ** 2 + 0.1 * np.sin(x), 0 * y, 0 * z)], voxel_size=(1.0, 1.0, 2.0))
    view = frame_view(res, 0)
    prof = axis_profile(view, "disp_u", "z")
    z0 = res.dvc_mesh.z0
    np.testing.assert_allclose(prof.positions, z0 * 2.0)  # physical units
    grid = res.dvc_mesh.to_grid(view.values("disp_u"))
    np.testing.assert_allclose(prof.mean, grid.mean(axis=(1, 2)))
    np.testing.assert_allclose(prof.std, grid.std(axis=(1, 2)))
    assert np.all(prof.n == 9 * 10)
    mask = np.zeros(res.dvc_mesh.n_nodes, dtype=bool)
    mask[: 9 * 10] = True  # the first z layer only
    only = axis_profile(view, "disp_u", "z", region=mask)
    assert only.n[0] == 90 and np.all(only.n[1:] == 0) and np.all(np.isnan(only.mean[1:]))


def test_a_line_samples_the_field_and_never_invents_values():
    res = make_result([lambda x, y, z: (0.01 * x + 0.02 * y - 0.03 * z, 0 * y, 0 * z)], strain=False)
    view = frame_view(res, 0)
    d, v = sample_line(view, "disp_u", (24.0, 30.0, 40.0), (70.0, 90.0, 100.0), n=25)
    p = np.array([24.0, 30.0, 40.0]) + np.linspace(0, 1, 25)[:, None] * (np.array([70.0, 90.0, 100.0]) - [24.0, 30.0, 40.0])
    np.testing.assert_allclose(v, 0.01 * p[:, 0] + 0.02 * p[:, 1] - 0.03 * p[:, 2], atol=1e-12)  # trilinear is exact on linear
    np.testing.assert_allclose(d, np.linalg.norm(p - p[0], axis=1))
    d, v = sample_line(view, "disp_u", (10.0, 30.0, 40.0), (40.0, 30.0, 40.0), n=31)  # starts outside the grid
    assert np.all(np.isnan(v[d < 10.0 - 1e-9])) and np.all(np.isfinite(v[d > 10.5]))
    n = res.dvc_mesh.n_nodes
    status = np.zeros(n, dtype=np.int8)
    status[res.dvc_mesh.node_index(3, 3, 3)] = 1  # one node not measured: the samples beside it are gaps, not guesses
    gap = frame_view(make_result([lambda x, y, z: (0.01 * x, 0 * y, 0 * z)], strain=False, status=status), 0)
    _d, v = sample_line(gap, "disp_u", (20.0, 48.0, 52.0), (84.0, 48.0, 52.0), n=65)
    xs = np.linspace(20, 84, 65)
    assert np.all(np.isnan(v[(xs > 36) & (xs < 52)])) and np.all(np.isfinite(v[xs < 36]))


def test_the_virtual_extensometer_reads_the_stretch_and_ignores_rotation():
    def frames():
        out = []
        for k, eps in enumerate((0.01, 0.02)):
            rigid, _R = rigid_in_physical([0.0, 0.0, 5.0 * (k + 1)], [1.0, 2.0, 3.0], (52.0, 60.0, 68.0), (1.0, 1.0, 1.0))

            def fn(x, y, z, eps=eps, rigid=rigid):
                xs = 52.0 + (1.0 + eps) * (x - 52.0)
                u, v, w = rigid(xs, y, z)
                return u + (xs - x), v, w

            out.append(fn)
        return out

    res = make_result(frames(), strain=False)
    ext = extensometer(res, (28.0, 60.0, 68.0), (76.0, 60.0, 68.0))
    assert ext.L0 == pytest.approx(48.0)
    np.testing.assert_allclose(ext.strain, [0.01, 0.02], atol=1e-9)
    np.testing.assert_array_equal(ext.frames, [0, 1])


# ------------------------------------------------------------------ confidence interval of a mean
def test_uncorrelated_nodes_are_all_independent():
    rng = np.random.default_rng(1)
    grid = rng.normal(size=(12, 13, 14))
    n_eff = effective_sample_size(grid, np.ones(grid.shape, dtype=bool))
    assert 0.8 * grid.size < n_eff <= grid.size


def test_the_confidence_interval_of_a_correlated_field_covers_the_mean():
    """Smoothed noise (correlation length ~4 nodes): the 95 % interval covers the true mean about 95 % of the
    time; the naive std/sqrt(N) interval misses it most of the time."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(2)
    shape = (24, 24, 24)
    mask = np.ones(shape, dtype=bool)
    mask[:4] = False  # an irregular region: the mask's own autocorrelation must be divided out
    hits = naive_hits = 0
    trials = 120
    for _ in range(trials):
        f = gaussian_filter(rng.normal(size=shape), 2.0, mode="wrap")
        f = 3.0 + f / f.std()
        n_eff, half = mean_confidence(f, mask)
        m = f[mask].mean()
        hits += abs(m - 3.0) <= half
        naive_hits += abs(m - 3.0) <= 1.96 * f[mask].std() / np.sqrt(mask.sum())
        assert n_eff < 0.1 * mask.sum()
    assert 0.85 <= hits / trials <= 1.0
    assert naive_hits / trials < 0.5


def test_frame_stats_can_carry_the_confidence_interval():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.1, size=(9 * 10 * 11, 3))
    res = make_result([lambda x, y, z: (0.5 + noise[:, 0], noise[:, 1], noise[:, 2])], strain=False)
    fs = frame_stats(res, 0, ("disp_u",), with_ci=True)
    st = fs.stats["disp_u"]
    assert 0.5 * st.n < st.n_eff <= st.n and 0 < st.ci95 < 0.05
    assert np.isnan(frame_stats(res, 0, ("disp_u",)).stats["disp_u"].ci95)  # not asked for: not computed


# ------------------------------------------------------------------ the corrected result the main window shows
def test_the_corrected_result_behaves_like_a_result():
    rigid, _R = rigid_in_physical([0.0, 0.0, 4.0], [3.0, -2.0, 5.0], (60.0, 60.0, 70.0), (2.0, 2.0, 2.0))

    def fn(x, y, z):
        u, v, w = rigid(x, y, z)
        return u + 0.01 * (x - 52.0), v, w

    res = make_result([fn, fn], voxel_size=(2.0, 2.0, 2.0), strain_type="infinitesimal")
    corr = corrected_result(res, Correction("rigid", NodeFilter()))
    assert corr.n_frames == 2 and len(corr.result_disp) == 2 and len(corr.result_strain) == 2
    assert [fr.ref_frame for fr in corr.result_disp] == [0, 0]
    view = frame_view(res, 1, "rigid")
    np.testing.assert_allclose(field_array(corr, 1, "disp_u"), view.values("disp_u"), atol=1e-12)
    np.testing.assert_allclose(field_array(corr, 1, "eyy"), view.values("eyy"), atol=1e-12)
    assert abs(np.nanmedian(field_array(corr, 0, "eyy"))) < 1e-6  # the rotation's cos(theta) - 1 is gone
    assert np.nanmedian(field_array(res, 0, "eyy")) < -1e-3  # still in the stored result
    assert corr.result_disp[0] is corr.result_disp[0]  # computed once per frame
    assert corrected_result(res, Correction("none", NodeFilter())) is res
    assert corr.correction.motion == "rigid"
    # a correction that cannot be fitted in a frame leaves that frame as measured and says so
    n = res.dvc_mesh.n_nodes
    status = np.ones((2, n), dtype=np.int8)
    status[0] = 0
    sparse = make_result([fn, fn], status=status, strain=False)
    corr = corrected_result(sparse, Correction("rigid", NodeFilter()))
    np.testing.assert_allclose(field_array(corr, 1, "disp_u"), field_array(sparse, 1, "disp_u"))
    assert corr.uncorrected_frames() == [1]


def test_the_correction_settings_round_trip():
    region = Region(3, "fixture", "box", {"lo": [0, 0, 0], "hi": [100, 100, 40]})
    c = Correction("rigid", NodeFilter(min_zncc=0.8, edge_layers=1), fit_region=region)
    back = Correction.from_dict(json.loads(json.dumps(c.as_dict())))
    assert back == c
    assert replace(c, motion="none").describe().startswith("none")


# ------------------------------------------------------------------ files and the command line
def test_al_dvc_stats_with_regions_from_a_file(tmp_path):
    from al_dvc import cli
    from al_dvc.export.export_npz import export_npz

    # a grip (x <= 36) only translates by 0.5; the rest stretches from it
    fns = [lambda x, y, z, e=e: (0.5 + e * np.maximum(x - 36.0, 0.0), 0 * y, 0 * z) for e in (0.01, 0.02)]
    npz = export_npz(make_result(fns), tmp_path / "run.npz")
    grip = Region(1, "grip", "box", {"lo": [0, 0, 0], "hi": [36, 200, 200]})
    far = Region(2, "far", "slab", {"axis": "x", "lo": 76, "hi": 84})
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps(regions_to_dicts([grip, far])), encoding="utf-8")
    out = tmp_path / "st"
    args = ["stats", str(npz), "--fields", "disp_u", "--motion", "translation", "--regions", str(regions)]
    assert cli.main([*args, "--fit-region", "grip", "--region", "far", "--ci", "-o", str(out)]) == 0
    rows = [ln.split(",") for ln in (out / "statistics.csv").read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    head = rows[0]
    mean = [float(r[head.index("disp_u_mean")]) for r in rows[1:]]
    np.testing.assert_allclose(mean, [0.01 * 44.0, 0.02 * 44.0], atol=1e-9)  # x 76 and 84, mean 80, 80 - 36 = 44
    assert all(r[head.index("disp_u_ci95")] for r in rows[1:])
    doc = json.loads((out / "statistics.json").read_text(encoding="utf-8"))
    assert doc["meta"]["fit_region"] == "grip" and doc["meta"]["region"] == "far" and len(doc["regions"]) == 2
    # every region side by side
    assert cli.main([*args, "--compare", "-o", str(out)]) == 0
    rows = [ln.split(",") for ln in (out / "statistics.csv").read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    assert rows[0][0] == "region" and [r[0] for r in rows[1:]] == ["All nodes", "All nodes", "grip", "grip", "far", "far"]
    # a name that is not there says which ones are
    with pytest.raises(SystemExit, match="grip, far"):
        cli.main([*args, "--region", "nothing", "-o", str(out)])


def test_the_series_leaves_a_gap_where_a_region_has_too_few_nodes():
    n = 9 * 10 * 11
    status = np.zeros((2, n), dtype=np.int8)
    status[1, : int(0.7 * n)] = 1  # frame 2: 70 % of the nodes did not converge
    res = make_result([lambda x, y, z: (0 * x, 0 * y, 0 * z)] * 2, status=status, strain=False)
    series = frame_series(res, ("disp_u",), NodeFilter(), min_valid_fraction=0.5)
    assert [fs.flag for fs in series] == ["ok", "few_nodes"]
    assert [fs.flag for fs in frame_series(res, ("disp_u",), NodeFilter())] == ["ok", "ok"]  # no threshold: no gap


def test_the_csv_files_of_regions_profiles_lines_and_the_extensometer(tmp_path):
    from al_dvc.analysis.export_stats import (
        stats_metadata,
        write_extensometer_csv,
        write_line_csv,
        write_profile_csv,
        write_regions_csv,
    )
    from al_dvc.analysis.stats import summarize

    res = make_result([lambda x, y, z: (0.01 * x, 0 * y, 0 * z)], voxel_size=(2.0, 2.0, 2.0))
    meta = stats_metadata(res, NodeFilter(), "rigid", region="R1", fit_region="grip")
    text = write_regions_csv(tmp_path / "r.csv", [("All nodes", summarize([1.0, 2.0])), ("R1", summarize([]))], "disp_u", 0, meta)
    lines = text.read_text(encoding="utf-8").splitlines()
    assert "fitted over: grip" in lines[2] and "region: R1" in lines[2]
    body = [ln for ln in lines if not ln.startswith("#")]
    assert body[0].startswith("region,field,n,mean") and body[0].endswith("n_eff,ci95")
    assert body[1].startswith("All nodes,disp_u,2,1.5") and body[2] == "R1,disp_u,0" + "," * 11
    view = frame_view(res, 0)
    prof = axis_profile(view, "disp_u", "x")
    body = [
        ln for ln in write_profile_csv(tmp_path / "p.csv", [(0, prof)], "disp_u", meta).read_text().splitlines() if ln[0] != "#"
    ]
    assert len(body) == 1 + 9 and body[1].split(",")[:2] == ["1", "x"]
    d, v = sample_line(view, "disp_u", (20, 60, 68), (84, 60, 68))
    body = write_line_csv(tmp_path / "l.csv", [(0, d, v)], "disp_u", (20, 60, 68), (84, 60, 68), meta).read_text().splitlines()
    assert any(ln.startswith("# from [20.0, 60.0, 68.0]") for ln in body)
    ext = extensometer(res, (20, 60, 68), (84, 60, 68))
    ends = np.array([[0.4, 1.68]])
    body = write_extensometer_csv(tmp_path / "e.csv", ext, meta, field="disp_u", ends=ends).read_text().splitlines()
    rows = [ln.split(",") for ln in body if not ln.startswith("#")]
    assert rows[0] == ["frame", "length", "strain", "disp_u_first_point", "disp_u_second_point"]
    assert float(rows[1][2]) == pytest.approx(0.01) and float(rows[1][4]) == pytest.approx(1.68)


def test_the_effective_sample_size_matches_the_exact_one_of_smoothed_noise():
    """Gaussian-smoothed noise has rho(r) = exp(-r^2 / (4 sigma^2)), so the exact n_eff = n^2 / sum_ij rho_ij is known.
    Before the mean-removal correction and the fitted tail, the estimate was a third too high at sigma = 2."""
    from scipy import fft
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(21)
    shape, sigma = (32, 32, 32), 2.0
    mask = np.ones(shape, dtype=bool)
    mask[:, :, :6] = False
    s2 = tuple(2 * k for k in shape)
    fm = fft.rfftn(mask.astype(float), s=s2)
    pairs = np.rint(fft.irfftn(fm * np.conj(fm), s=s2))
    lag2 = [np.where(np.arange(k) < k // 2, np.arange(k), np.arange(k) - k).astype(float) ** 2 for k in s2]
    r2 = lag2[0][:, None, None] + lag2[1][None, :, None] + lag2[2][None, None, :]
    n = mask.sum()
    exact = n * n / np.sum(pairs * np.exp(-r2 / (4 * sigma**2)))
    estimates = [effective_sample_size(gaussian_filter(rng.normal(size=shape), sigma, mode="wrap"), mask) for _ in range(12)]
    assert np.median(estimates) == pytest.approx(exact, rel=0.2)


def test_a_real_gradient_does_not_widen_the_interval_of_the_mean():
    """A displacement that grows across the region (a stretch) is the field, not uncertainty: the interval of the mean
    is that of the noise on top of it. It used to count the gradient as correlation: 0.03 +- 1.1 voxel."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(4)
    shape = (20, 22, 24)
    noise = 0.05 * gaussian_filter(rng.normal(size=shape), 1.0)
    z, y, x = np.meshgrid(*[np.arange(k, dtype=float) for k in shape], indexing="ij")
    mask = np.ones(shape, dtype=bool)
    n_noise, half_noise = mean_confidence(noise, mask)
    n_grad, half_grad = mean_confidence(noise + 0.01 * (x - 12.0) + 0.004 * z, mask)
    assert n_grad == pytest.approx(n_noise, rel=1e-6) and half_grad == pytest.approx(half_noise, rel=1e-6)
    # a handful of nodes: only the mean is removed, and the interval stays finite
    few = np.zeros(shape, dtype=bool)
    few[0, 0, :5] = True
    n_few, half_few = mean_confidence(noise, few)
    assert 1.0 <= n_few <= 5.0 and np.isfinite(half_few)


def test_the_corrected_result_keeps_only_the_last_frames(monkeypatch):
    """Every corrected frame used to stay in memory: a full strain result per frame of a long series."""
    import al_dvc.analysis.corrected as corrected

    fn, _R = rigid_in_physical([0.0, 0.0, 4.0], [1.0, 2.0, 0.0], (52.0, 60.0, 68.0), (1.0, 1.0, 1.0))
    res = make_result([fn] * (corrected.CACHE_FRAMES + 4))
    calls = []
    real = corrected.frame_view
    monkeypatch.setattr(corrected, "frame_view", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    shown = corrected_result(res, Correction("rigid"))
    first = shown.result_disp[0].U_accum.copy()
    for k in range(len(res.result_disp)):
        np.testing.assert_allclose(shown.result_disp[k].U_accum, 0.0, atol=1e-9)
        assert shown.result_strain[k] is not None
    state = shown.result_disp._state
    assert max(len(state.views), len(state.disp), len(state.strain)) <= corrected.CACHE_FRAMES
    assert shown.uncorrected_frames() == [] and len(calls) == len(res.result_disp)  # nothing fitted twice
    np.testing.assert_array_equal(shown.result_disp[0].U_accum, first)  # recomputed after eviction, the same
