"""Automatic beta selection: MATLAB score (default) and the normalised score."""

import numpy as np
import pytest

from al_dvc.core.config import dvcpara_default
from al_dvc.core.pipeline import run_aldvc
from al_dvc.solver.beta_tuning import beta_candidates


@pytest.mark.parametrize("criterion", ["matlab", "normalized"])
def test_beta_criteria(affine_pair, criterion):
    f, g, _ = affine_pair
    para = dvcpara_default(winsize=16, winstepsize=8, search_radius=5, verbose=False, admm_max_iter=2, beta_criterion=criterion)
    result = run_aldvc(para, [f, g], compute_strain=False)
    admm = result.result_disp[0].admm
    sweep = admm.beta_sweep
    assert sweep is not None and sweep["criterion"] == criterion
    betas = beta_candidates(para, para.mu)
    assert np.allclose(sweep["betas"], betas)
    assert np.all(np.isfinite(sweep["err1"])) and np.all(np.isfinite(sweep["err2"]))
    h2 = float(np.mean(para.winstepsize)) ** 2
    if criterion == "matlab":
        expected = betas[int(np.argmin(sweep["err1"] + h2 * sweep["err2"]))]
        assert admm.beta == pytest.approx(expected)
        assert np.allclose(sweep["score"], sweep["err1"] + h2 * sweep["err2"])
    else:
        assert betas.min() <= admm.beta <= betas.max()


def test_beta_criterion_validation():
    with pytest.raises(ValueError):
        dvcpara_default(beta_criterion="lcurve")
