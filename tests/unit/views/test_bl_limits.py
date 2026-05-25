"""Limiting-case tests (Omega -> 0 and Omega -> infinity)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.views import AbsoluteView, BlackLitterman, RelativeView, Views

ASSETS = ["A", "B", "C", "D"]


@pytest.fixture
def cov_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    L = np.tril(rng.standard_normal((4, 4)))
    np.fill_diagonal(L, np.abs(np.diag(L)) + 0.5)
    sigma = L @ L.T * 0.01  # daily-ish scale
    return pd.DataFrame(sigma, index=ASSETS, columns=ASSETS)


@pytest.fixture
def market_weights() -> pd.Series:
    return pd.Series([0.30, 0.25, 0.25, 0.20], index=ASSETS)


def test_omega_to_zero_view_is_satisfied(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    view = AbsoluteView("B", 0.10)
    views = Views([view], ASSETS)
    omega = np.array([[1e-12]])
    bl = BlackLitterman(cov_df, market_weights, views=views, omega=omega)
    mu_bl = bl.posterior_returns()
    assert mu_bl.loc["B"] == pytest.approx(0.10, abs=1e-6)


def test_omega_to_infinity_view_is_ignored(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    view = AbsoluteView("B", 0.10)
    views = Views([view], ASSETS)
    omega = np.array([[1e12]])
    bl = BlackLitterman(cov_df, market_weights, views=views, omega=omega)
    pi = bl.implied_returns()
    mu_bl = bl.posterior_returns()
    np.testing.assert_allclose(mu_bl.to_numpy(), pi.to_numpy(), atol=1e-6)


def test_relative_view_omega_to_zero(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    view = RelativeView(long_leg={"A": 1.0}, short_leg={"D": 1.0}, spread=0.03)
    views = Views([view], ASSETS)
    omega = np.array([[1e-12]])
    bl = BlackLitterman(cov_df, market_weights, views=views, omega=omega)
    mu_bl = bl.posterior_returns()
    spread_realised = mu_bl.loc["A"] - mu_bl.loc["D"]
    assert spread_realised == pytest.approx(0.03, abs=1e-6)


def test_supplied_omega_shape_mismatch_raises(
    cov_df: pd.DataFrame, market_weights: pd.Series
) -> None:
    view = AbsoluteView("B", 0.10)
    views = Views([view], ASSETS)
    omega = np.eye(2)
    bl = BlackLitterman(cov_df, market_weights, views=views, omega=omega)
    with pytest.raises(Exception, match="shape"):
        bl.posterior_returns()


def test_supplied_non_symmetric_omega_raises(
    cov_df: pd.DataFrame, market_weights: pd.Series
) -> None:
    v1 = AbsoluteView("A", 0.05)
    v2 = AbsoluteView("B", 0.07)
    views = Views([v1, v2], ASSETS)
    omega = np.array([[1e-3, 1e-3], [0.0, 1e-3]])
    bl = BlackLitterman(cov_df, market_weights, views=views, omega=omega)
    with pytest.raises(Exception, match="symmetric"):
        bl.posterior_returns()


def test_idzorek_approx_requires_confidences(
    cov_df: pd.DataFrame, market_weights: pd.Series
) -> None:
    view = AbsoluteView("B", 0.10)  # no confidence
    views = Views([view], ASSETS)
    bl = BlackLitterman(cov_df, market_weights, views=views, omega_method="idzorek_approx")
    with pytest.raises(Exception, match="confidences"):
        bl.posterior_returns()


def test_idzorek_approx_path_runs(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    view = AbsoluteView("B", 0.10, confidence=0.5)
    views = Views([view], ASSETS)
    bl = BlackLitterman(cov_df, market_weights, views=views, omega_method="idzorek_approx")
    mu_bl = bl.posterior_returns()
    sigma_bl = bl.posterior_covariance()
    assert mu_bl.shape == (4,)
    assert sigma_bl.shape == (4, 4)


def test_idzorek_exact_path_runs(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    v1 = AbsoluteView("A", 0.08, confidence=0.5)
    v2 = AbsoluteView("B", 0.10, confidence=0.7)
    views = Views([v1, v2], ASSETS)
    bl = BlackLitterman(cov_df, market_weights, views=views, omega_method="idzorek_exact")
    mu_bl = bl.posterior_returns()
    assert mu_bl.shape == (4,)


def test_posterior_cov_prior_only_mode(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    view = AbsoluteView("B", 0.10)
    views = Views([view], ASSETS)
    bl = BlackLitterman(
        cov_df,
        market_weights,
        views=views,
        posterior_covariance_mode="prior_only",
    )
    sigma_bl = bl.posterior_covariance()
    pd.testing.assert_frame_equal(sigma_bl, cov_df)


def test_constrained_weights_sum_to_one(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    view = AbsoluteView("B", 0.20)
    views = Views([view], ASSETS)
    bl = BlackLitterman(cov_df, market_weights, views=views)
    w = bl.posterior_weights(constrained=True)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_summary_with_views(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    view = AbsoluteView("B", 0.20)
    views = Views([view], ASSETS)
    bl = BlackLitterman(cov_df, market_weights, views=views)
    df = bl.summary()
    assert df.loc["B", "mu_bl"] != df.loc["B", "pi"]
    assert df.loc["B", "delta_mu"] == pytest.approx(df.loc["B", "mu_bl"] - df.loc["B", "pi"])
