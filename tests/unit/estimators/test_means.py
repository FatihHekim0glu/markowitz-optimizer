"""Unit tests for mean-return estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.estimators import (
    CAPMReturns,
    EWMAMean,
    ImpliedReturns,
    JorionBayesStein,
    SampleMean,
)
from markowitz.estimators.exceptions import (
    DimensionMismatchError,
    EstimatorConfigError,
    NotFittedError,
)


@pytest.fixture
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-02", periods=400)
    data = rng.normal(loc=0.0004, scale=0.012, size=(400, 4))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C", "D"])


@pytest.fixture
def long_returns() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    idx = pd.bdate_range("2010-01-04", periods=2000)
    data = rng.normal(loc=0.0003, scale=0.01, size=(2000, 3))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


def _spd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


# ---------------------------------------------------------------- SampleMean


def test_sample_mean_not_fitted_raises() -> None:
    sm = SampleMean()
    with pytest.raises(NotFittedError):
        sm._assert_fitted()


def test_sample_mean_matches_numpy_no_annualize(returns_df: pd.DataFrame) -> None:
    sm = SampleMean(annualize=False).fit(returns_df)
    expected = returns_df.to_numpy().mean(axis=0)
    assert np.allclose(sm.mean_, expected, atol=1e-15)
    assert sm.n_features_in_ == 4
    assert sm.n_samples_ == len(returns_df)
    assert list(sm.feature_names_in_) == ["A", "B", "C", "D"]


def test_sample_mean_annualizes_by_252(returns_df: pd.DataFrame) -> None:
    sm_ann = SampleMean(annualize=True, periods_per_year=252).fit(returns_df)
    sm_raw = SampleMean(annualize=False).fit(returns_df)
    assert np.allclose(sm_ann.mean_, sm_raw.mean_ * 252, atol=1e-15)


def test_sample_mean_infers_business_daily(returns_df: pd.DataFrame) -> None:
    sm = SampleMean(annualize=True).fit(returns_df)
    raw = returns_df.to_numpy().mean(axis=0)
    assert np.allclose(sm.mean_, raw * 252, atol=1e-15)


def test_sample_mean_column_permutation_invariance(returns_df: pd.DataFrame) -> None:
    base = SampleMean(annualize=False).fit(returns_df)
    permuted = returns_df[["D", "B", "A", "C"]]
    perm = SampleMean(annualize=False).fit(permuted)
    base_map = dict(zip(base.feature_names_in_, base.mean_, strict=True))
    perm_map = dict(zip(perm.feature_names_in_, perm.mean_, strict=True))
    for k, v in base_map.items():
        assert pytest.approx(perm_map[k], abs=1e-15) == v


def test_sample_mean_ndarray_input_requires_explicit_ppy() -> None:
    arr = np.random.default_rng(0).normal(size=(100, 3))
    sm = SampleMean(annualize=True, periods_per_year=252).fit(arr)
    assert sm.mean_.shape == (3,)
    assert list(sm.feature_names_in_) == ["x0", "x1", "x2"]


# ---------------------------------------------------------------- EWMAMean


def test_ewma_mean_xor_validation() -> None:
    with pytest.raises(EstimatorConfigError):
        EWMAMean()
    with pytest.raises(EstimatorConfigError):
        EWMAMean(halflife_years=0.5, lam=0.94)


def test_ewma_mean_not_fitted() -> None:
    e = EWMAMean(lam=0.94)
    with pytest.raises(NotFittedError):
        e._assert_fitted()


def test_ewma_mean_lambda_near_one_matches_sample(long_returns: pd.DataFrame) -> None:
    ewma = EWMAMean(lam=0.9999, annualize=False).fit(long_returns)
    sample = SampleMean(annualize=False).fit(long_returns)
    assert np.allclose(ewma.mean_, sample.mean_, atol=1e-4)


def test_ewma_mean_weights_sum_to_one(returns_df: pd.DataFrame) -> None:
    ewma = EWMAMean(lam=0.94, annualize=False).fit(returns_df)
    assert pytest.approx(ewma._weights_.sum()) == 1.0


def test_ewma_mean_halflife_resolves_to_lambda(returns_df: pd.DataFrame) -> None:
    ewma = EWMAMean(halflife_years=0.5, periods_per_year=252, annualize=False).fit(returns_df)
    assert 0.0 < ewma._lambda_ < 1.0
    expected_lam = 0.5 ** (1.0 / (0.5 * 252))
    assert pytest.approx(ewma._lambda_, abs=1e-12) == expected_lam


def test_ewma_mean_annualizes(returns_df: pd.DataFrame) -> None:
    ann = EWMAMean(lam=0.94, annualize=True, periods_per_year=252).fit(returns_df)
    raw = EWMAMean(lam=0.94, annualize=False).fit(returns_df)
    assert np.allclose(ann.mean_, raw.mean_ * 252)


# ---------------------------------------------------------------- JorionBayesStein


def test_jorion_not_fitted() -> None:
    j = JorionBayesStein()
    with pytest.raises(NotFittedError):
        j._assert_fitted()


def test_jorion_phi_in_unit_interval(returns_df: pd.DataFrame) -> None:
    j = JorionBayesStein(annualize=False).fit(returns_df)
    assert 0.0 <= j.shrinkage_ <= 1.0


def test_jorion_identical_means_phi_large() -> None:
    rng = np.random.default_rng(42)
    n_obs, n_assets = 500, 5
    data = rng.normal(scale=0.01, size=(n_obs, n_assets))
    j = JorionBayesStein(annualize=False).fit(data)
    # All true means are zero -> dispersion is small -> phi should be large
    assert j.shrinkage_ > 0.5


def test_jorion_extreme_dispersion_phi_near_zero() -> None:
    rng = np.random.default_rng(3)
    n_obs, n_assets = 200, 4
    base = rng.normal(scale=0.001, size=(n_obs, n_assets))
    shifts = np.array([0.5, -0.5, 0.5, -0.5])
    data = base + shifts
    j = JorionBayesStein(annualize=False).fit(data)
    assert j.shrinkage_ < 0.05


def test_jorion_accepts_cov_kwarg(returns_df: pd.DataFrame) -> None:
    cov = np.cov(returns_df.to_numpy().T, ddof=1)
    j = JorionBayesStein(annualize=False).fit(returns_df, cov=cov)
    assert j.mean_.shape == (returns_df.shape[1],)


def test_jorion_rejects_bad_cov_shape(returns_df: pd.DataFrame) -> None:
    bad = np.eye(returns_df.shape[1] + 1)
    with pytest.raises(DimensionMismatchError):
        JorionBayesStein().fit(returns_df, cov=bad)


def test_jorion_annualizes(returns_df: pd.DataFrame) -> None:
    raw = JorionBayesStein(annualize=False).fit(returns_df)
    ann = JorionBayesStein(annualize=True, periods_per_year=252).fit(returns_df)
    assert np.allclose(ann.mean_, raw.mean_ * 252)


# ---------------------------------------------------------------- CAPMReturns


def test_capm_not_fitted() -> None:
    c = CAPMReturns()
    with pytest.raises(NotFittedError):
        c._assert_fitted()


def test_capm_self_regression_beta_one() -> None:
    rng = np.random.default_rng(5)
    market = rng.normal(scale=0.01, size=500)
    df = pd.DataFrame({"M": market, "Other": market * 2.0})
    c = CAPMReturns(annualize=False).fit(df, market=market)
    assert pytest.approx(c.betas_[0], abs=1e-12) == 1.0
    assert pytest.approx(c.betas_[1], abs=1e-12) == 2.0


def test_capm_zero_beta_gives_risk_free() -> None:
    rng = np.random.default_rng(13)
    market = rng.normal(scale=0.01, size=600)
    idx = pd.bdate_range("2020-01-02", periods=600)
    noise = rng.normal(scale=0.01, size=600)
    df = pd.DataFrame({"Z": noise}, index=idx)
    c = CAPMReturns(
        risk_free_rate=0.02,
        market_premium=0.05,
        annualize=True,
        periods_per_year=252,
    ).fit(df, market=market)
    assert abs(c.betas_[0]) < 0.1
    assert pytest.approx(c.mean_[0], abs=0.01) == 0.02


def test_capm_rejects_mismatched_market_length(returns_df: pd.DataFrame) -> None:
    with pytest.raises(DimensionMismatchError):
        CAPMReturns().fit(returns_df, market=np.zeros(returns_df.shape[0] - 1))


def test_capm_rejects_zero_variance_market(returns_df: pd.DataFrame) -> None:
    with pytest.raises(EstimatorConfigError):
        CAPMReturns().fit(returns_df, market=np.ones(returns_df.shape[0]))


# ---------------------------------------------------------------- ImpliedReturns


def test_implied_not_fitted() -> None:
    i = ImpliedReturns()
    with pytest.raises(NotFittedError):
        i._assert_fitted()


def test_implied_round_trip_known_inputs() -> None:
    n = 4
    sigma = _spd(n, seed=99)
    weights = np.full(n, 1.0 / n)
    delta = 2.5
    rng = np.random.default_rng(0)
    returns = rng.normal(size=(50, n))
    est = ImpliedReturns(delta=delta).fit(returns, weights=weights, cov=sigma)
    assert np.allclose(est.mean_, delta * sigma @ weights, atol=1e-14)


def test_implied_uses_sample_cov_when_omitted() -> None:
    rng = np.random.default_rng(8)
    returns = rng.normal(size=(200, 3))
    weights = np.array([0.5, 0.3, 0.2])
    est = ImpliedReturns(delta=3.0).fit(returns, weights=weights)
    expected = 3.0 * np.cov(returns.T, ddof=1) @ weights
    assert np.allclose(est.mean_, expected, atol=1e-14)


def test_implied_rejects_bad_weights_length() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(size=(50, 3))
    with pytest.raises(DimensionMismatchError):
        ImpliedReturns().fit(returns, weights=np.zeros(2))


def test_implied_rejects_bad_cov_shape() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(size=(50, 3))
    with pytest.raises(DimensionMismatchError):
        ImpliedReturns().fit(returns, weights=np.zeros(3), cov=np.eye(4))


def test_implied_rejects_non_positive_delta() -> None:
    with pytest.raises(EstimatorConfigError):
        ImpliedReturns(delta=0.0)
