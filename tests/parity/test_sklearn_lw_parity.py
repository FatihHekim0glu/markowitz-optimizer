"""Parity tests against ``sklearn.covariance.LedoitWolf`` for the identity target."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from markowitz.estimators import LedoitWolfShrinkage

pytestmark = pytest.mark.parity


def _synthetic_returns(n_obs: int, n_assets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_assets, n_assets))
    spd = a @ a.T + n_assets * np.eye(n_assets)
    chol = np.linalg.cholesky(spd)
    z = rng.normal(size=(n_obs, n_assets))
    return (z @ chol.T) * 0.01


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize(
    ("n_obs", "n_assets"),
    [(100, 5), (252, 10), (1000, 30)],
)
def test_lw_identity_matches_sklearn(seed: int, n_obs: int, n_assets: int) -> None:
    x = _synthetic_returns(n_obs, n_assets, seed)

    sk = LedoitWolf(assume_centered=False).fit(x)
    ours = LedoitWolfShrinkage(target="identity", annualize=False).fit(x)

    assert abs(ours.shrinkage_ - sk.shrinkage_) < 1e-10
    assert np.allclose(ours.covariance_, sk.covariance_, atol=1e-10, rtol=0.0)
    assert ours.n_features_in_ == n_assets
    assert ours.n_samples_ == n_obs


def test_lw_shape_matches_sample_when_no_annualize() -> None:
    x = _synthetic_returns(252, 8, seed=99)
    ours = LedoitWolfShrinkage(target="identity", annualize=False).fit(x)
    assert ours.covariance_.shape == (8, 8)
