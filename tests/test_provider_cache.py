"""ListVolumeProvider normalises on demand and keeps a bounded number of frames."""

from __future__ import annotations

import numpy as np

from al_dvc.core.config import dvcpara_default
from al_dvc.core.pipeline import run_aldvc
from al_dvc.io.volume_ops import LIST_PROVIDER_CACHE, ListVolumeProvider, normalize_volume
from al_dvc.synthetic import generate_speckle_volume


def test_provider_normalises_lazily_with_lru():
    vols = [generate_speckle_volume((24, 26, 28), sigma=2.0, seed=k) for k in range(6)]
    prov = ListVolumeProvider(vols)
    assert len(prov) == 6 and prov.n_cached == 0
    a = prov.get_normalized(0)
    np.testing.assert_array_equal(a, normalize_volume(vols[0], prov.clamped_voi))
    assert a.dtype == np.float32 and prov.n_cached == 1
    for k in range(1, 6):
        prov.get_normalized(k)
    assert prov.n_cached == LIST_PROVIDER_CACHE
    # a cached frame is returned as the same object; an evicted one is recomputed
    b = prov.get_normalized(5)
    assert prov.get_normalized(5) is b
    c = prov.get_normalized(0)
    np.testing.assert_array_equal(c, a)
    assert c is not a
    small = ListVolumeProvider(vols, cache_size=1)
    small.get_normalized(0)
    small.get_normalized(1)
    assert small.n_cached == 1


def test_pipeline_runs_over_a_sequence_longer_than_the_cache():
    ref = generate_speckle_volume((40, 40, 40), sigma=2.0, seed=3)
    frames = [ref] + [np.roll(ref, k, axis=2) for k in range(1, 5)]  # integer shifts along x
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=4, admm_max_iter=2, verbose=False)
    res = run_aldvc(para, frames)
    assert res.n_frames == 4
    for k, fr in enumerate(res.result_disp, start=1):
        u = fr.U_accum[:, 0] if fr.U_accum is not None else fr.U[:, 0]
        assert abs(np.nanmedian(u) - k) < 0.1
