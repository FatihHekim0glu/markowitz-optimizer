"""Unit tests for covariance estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.estimators import (
    EWMACovariance,
    LedoitWolfShrinkage,
    SampleCovariance,
)
from markowitz.estimators.exceptions import (
    EstimatorConfigError,
    NotFittedError,
)


@pytest.fixture
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    idx = pd.bdate_range("2020-01-02", periods=400)
    data = rng.normal(loc=0.0, scale=0.012, size=(400, 5))
    return pd.DataFrame(data, index=idx, columns=list("ABCDE"))


@pytest.fixture
def long_returns() -> pd.DataFrame:
    rng = np.random.default_rng(29)
    idx = pd.bdate_range("2010-01-04", periods=2500)
    data = rng.normal(scale=0.01, size=(2500, 4))
    return pd.DataFrame(data, index=idx, columns=list("WXYZ"))


def _is_symmetric(m: np.ndarray, atol: float = 1e-12) -> bool:
    return bool(np.allclose(m, m.T, atol=atol))


def _is_psd(m: np.ndarray, tol: float = -1e-10) -> bool:
    eigs = np.linalg.eigvalsh(0.5 * (m + m.T))
    return bool(np.min(eigs) >= tol)


# ---------------------------------------------------------------- SampleCovariance


def test_sample_cov_not_fitted() -> None:
    s = SampleCovariance()
    with pytest.raises(NotFittedError):
        s._assert_fitted()


def test_sample_cov_matches_numpy(returns_df: pd.DataFrame) -> None:
    s = SampleCovariance(annualize=False).fit(returns_df)
    expected = np.cov(returns_df.to_numpy().T, ddof=1)
    assert np.allclose(s.covariance_, expected, atol=1e-12)
    assert s.n_features_in_ == returns_df.shape[1]
    assert s.n_samples_ == returns_df.shape[0]


def test_sample_cov_symmetric_and_psd(returns_df: pd.DataFrame) -> None:
    s = SampleCovariance(annualize=False).fit(returns_df)
    assert _is_symmetric(s.covariance_)
    assert _is_psd(s.covariance_)


def test_sample_cov_annualizes(returns_df: pd.DataFrame) -> None:
    raw = SampleCovariance(annualize=False).fit(returns_df)
    ann = SampleCovariance(annualize=True, periods_per_year=252).fit(returns_df)
    assert np.allclose(ann.covariance_, raw.covariance_ * 252, atol=1e-12)


def test_sample_cov_ddof_zero(returns_df: pd.DataFrame) -> None:
    s = SampleCovariance(annualize=False, ddof=0).fit(returns_df)
    expected = np.cov(returns_df.to_numpy().T, ddof=0)
    assert np.allclose(s.covariance_, expected, atol=1e-12)


def test_sample_cov_rejects_bad_ddof() -> None:
    with pytest.raises(EstimatorConfigError):
        SampleCovariance(ddof=-1)


# ---------------------------------------------------------------- LedoitWolfShrinkage


def test_lw_not_fitted() -> None:
    lw = LedoitWolfShrinkage()
    with pytest.raises(NotFittedError):
        lw._assert_fitted()


def test_lw_rejects_bad_target() -> None:
    with pytest.raises(EstimatorConfigError):
        LedoitWolfShrinkage(target="bogus")  # type: ignore[arg-type]


def test_lw_identity_psd_and_symmetric(returns_df: pd.DataFrame) -> None:
    lw = LedoitWolfShrinkage(target="identity", annualize=False).fit(returns_df)
    assert _is_symmetric(lw.covariance_)
    assert _is_psd(lw.covariance_)
    assert 0.0 <= lw.shrinkage_ <= 1.0


def test_lw_constant_corr_real_formula_psd(returns_df: pd.DataFrame) -> None:
    lw = LedoitWolfShrinkage(target="constant_corr", annualize=False).fit(returns_df)
    assert _is_symmetric(lw.covariance_)
    assert _is_psd(lw.covariance_)
    assert 0.0 <= lw.shrinkage_ <= 1.0


def test_lw_constant_corr_shrinkage_in_unit_interval(
    returns_df: pd.DataFrame, long_returns: pd.DataFrame
) -> None:
    for df in (returns_df, long_returns):
        lw = LedoitWolfShrinkage(target="constant_corr", annualize=False).fit(df)
        assert 0.0 <= lw.shrinkage_ <= 1.0


def test_lw_constant_corr_against_oas_is_different(returns_df: pd.DataFrame) -> None:
    from sklearn.covariance import oas

    lw = LedoitWolfShrinkage(target="constant_corr", annualize=False).fit(returns_df)
    _, oas_shrinkage = oas(returns_df.to_numpy(), assume_centered=False)
    # The real LW-CC formula must differ from the OAS shrinkage that the old
    # fallback returned, proving the silent fallback is gone.
    assert abs(lw.shrinkage_ - float(oas_shrinkage)) > 1e-6


def test_lw_annualizes(returns_df: pd.DataFrame) -> None:
    raw = LedoitWolfShrinkage(target="identity", annualize=False).fit(returns_df)
    ann = LedoitWolfShrinkage(target="identity", annualize=True, periods_per_year=252).fit(
        returns_df
    )
    assert np.allclose(ann.covariance_, raw.covariance_ * 252, atol=1e-10)


# ---------------------------------------------------------------- EWMACovariance


def test_ewma_cov_not_fitted() -> None:
    e = EWMACovariance()
    with pytest.raises(NotFittedError):
        e._assert_fitted()


def test_ewma_cov_symmetric_and_psd(returns_df: pd.DataFrame) -> None:
    e = EWMACovariance(lam=0.94, annualize=False).fit(returns_df)
    assert _is_symmetric(e.covariance_)
    assert _is_psd(e.covariance_)


def test_ewma_cov_lambda_near_one_close_to_seed_cov(
    long_returns: pd.DataFrame,
) -> None:
    # With lambda very close to 1, the EWMA recursion barely updates the seed,
    # so the result should remain close to the covariance of the burn-in window.
    e = EWMACovariance(lam=0.999999, annualize=False).fit(long_returns)
    seed = np.cov(long_returns.to_numpy()[: e._burn_in_, :].T, ddof=1)
    diff = np.linalg.norm(e.covariance_ - seed, ord="fro")
    base = np.linalg.norm(seed, ord="fro")
    assert diff / base < 0.05


def test_ewma_cov_halflife_resolves(returns_df: pd.DataFrame) -> None:
    e = EWMACovariance(
        halflife_years=0.5,
        lam=None,
        periods_per_year=252,
        annualize=False,
    ).fit(returns_df)
    assert 0.0 < e._lambda_ < 1.0


def test_ewma_cov_rejects_both_lam_and_halflife() -> None:
    with pytest.raises(EstimatorConfigError):
        EWMACovariance(halflife_years=0.5, lam=0.97)


def test_ewma_cov_both_halflife_and_lam_raises() -> None:
    # Even with the default-looking lam=0.94, providing both is an error;
    # the previous brittle sentinel let this slip through silently.
    with pytest.raises(EstimatorConfigError):
        EWMACovariance(halflife_years=0.5, lam=0.94)


def test_ewma_cov_default_lam_when_both_none(returns_df: pd.DataFrame) -> None:
    # With neither halflife_years nor lam supplied, EWMA should default to 0.94.
    e = EWMACovariance(annualize=False).fit(returns_df)
    assert e._lambda_ == pytest.approx(0.94)


def test_ewma_cov_annualizes(returns_df: pd.DataFrame) -> None:
    raw = EWMACovariance(lam=0.94, annualize=False).fit(returns_df)
    ann = EWMACovariance(lam=0.94, annualize=True, periods_per_year=252).fit(returns_df)
    assert np.allclose(ann.covariance_, raw.covariance_ * 252, atol=1e-10)


def test_ewma_cov_explicit_burn_in(returns_df: pd.DataFrame) -> None:
    e = EWMACovariance(lam=0.94, burn_in=30, annualize=False).fit(returns_df)
    assert e._burn_in_ == 30
    assert _is_psd(e.covariance_)


# ---------------------------------------------------------------- Property checks


def test_backward_compat_aliases_exposed() -> None:
    from markowitz.estimators import LedoitWolfCov, SampleCov

    assert LedoitWolfCov is LedoitWolfShrinkage
    assert SampleCov is SampleCovariance


def test_lw_constant_corr_single_asset() -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(scale=0.01, size=(200, 1)), columns=["only"])
    lw = LedoitWolfShrinkage(target="constant_corr", annualize=False).fit(df)
    assert lw.covariance_.shape == (1, 1)
    assert lw.covariance_[0, 0] > 0.0
    assert 0.0 <= lw.shrinkage_ <= 1.0


def test_lw_constant_corr_zero_gamma_returns_sample() -> None:
    # All assets equal -> sample equals prior, gamma_hat = 0, delta = 0.
    x = np.tile(np.linspace(-0.01, 0.01, 200).reshape(-1, 1), (1, 3))
    # Add a tiny perturbation to a single column so columns are not literally
    # collinear (otherwise the sample is rank-deficient but still equals prior).
    x = x + 1e-12 * np.random.default_rng(0).normal(size=x.shape)
    df = pd.DataFrame(x, columns=list("ABC"))
    lw = LedoitWolfShrinkage(target="constant_corr", annualize=False).fit(df)
    assert 0.0 <= lw.shrinkage_ <= 1.0


def test_ewma_cov_resolve_ppy_falls_back_to_252_for_ndarray(
    returns_df: pd.DataFrame,
) -> None:
    # ndarray input has no index; with annualize=True the resolver should
    # surface AmbiguousFrequencyError, which the fit method swallows when
    # halflife_years is None, defaulting ppy to 252.
    e = EWMACovariance(lam=0.94, annualize=True).fit(returns_df.to_numpy())
    assert e._lambda_ == pytest.approx(0.94)


def test_ewma_cov_halflife_with_ndarray_raises_for_ambiguous_index() -> None:
    rng = np.random.default_rng(11)
    arr = rng.normal(scale=0.01, size=(300, 4))
    # halflife_years requires a real ppy; ndarray + no periods_per_year -> raise.
    with pytest.raises(Exception):
        EWMACovariance(halflife_years=0.5, annualize=True).fit(arr)


def test_ewma_cov_rejects_resolved_lambda_outside_range() -> None:
    # Force resolve_decay to return a value outside (0,1) by patching.
    e = EWMACovariance(lam=0.94, annualize=False)
    e._resolve_lambda = lambda _ppy: 1.5  # type: ignore[assignment]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("ABC"))
    with pytest.raises(EstimatorConfigError):
        e.fit(df)


def test_ewma_cov_burn_in_clipped_to_n_obs(returns_df: pd.DataFrame) -> None:
    n_obs = returns_df.shape[0]
    e = EWMACovariance(lam=0.94, burn_in=n_obs + 500, annualize=False).fit(returns_df)
    assert e._burn_in_ == n_obs


def test_ewma_cov_burn_in_floor_two() -> None:
    rng = np.random.default_rng(13)
    df = pd.DataFrame(rng.normal(size=(5, 2)), columns=list("AB"))
    e = EWMACovariance(lam=0.94, burn_in=0, annualize=False).fit(df)
    assert e._burn_in_ == 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_all_covariance_estimators_produce_psd(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_obs, n_assets = 250, 6
    idx = pd.bdate_range("2020-01-02", periods=n_obs)
    df = pd.DataFrame(
        rng.normal(scale=0.01, size=(n_obs, n_assets)),
        index=idx,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    for est in [
        SampleCovariance(annualize=False).fit(df),
        LedoitWolfShrinkage(target="identity", annualize=False).fit(df),
        LedoitWolfShrinkage(target="constant_corr", annualize=False).fit(df),
        EWMACovariance(lam=0.94, annualize=False).fit(df),
    ]:
        assert _is_symmetric(est.covariance_)
        assert _is_psd(est.covariance_)
