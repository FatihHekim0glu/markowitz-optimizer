"""Unit tests for :mod:`markowitz.backtest.strategies`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.backtest.exceptions import DegenerateWindowError
from markowitz.backtest.strategies import (
    GMVSample,
    MaxSharpeNaive,
    OneOverN,
    RiskParity,
    _safe_inv,
)


@pytest.fixture
def window(rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    return pd.DataFrame(rng.normal(0.005, 0.04, size=(120, 5)), index=idx, columns=list("ABCDE"))


def test_one_over_n(window: pd.DataFrame) -> None:
    w = OneOverN().fit(window)
    assert w.shape == (5,)
    np.testing.assert_allclose(w, np.full(5, 0.2))
    assert w.sum() == pytest.approx(1.0)


def test_gmv_sample_matches_closed_form(window: pd.DataFrame) -> None:
    w = GMVSample().fit(window)
    sigma = np.cov(window.to_numpy(), rowvar=False, ddof=1)
    ones = np.ones(5)
    inv = np.linalg.solve(sigma, np.eye(5))
    expected = inv @ ones / (ones @ inv @ ones)
    np.testing.assert_allclose(w, expected, atol=1e-10)
    assert w.sum() == pytest.approx(1.0)


def test_gmv_sample_has_minimum_variance(window: pd.DataFrame) -> None:
    sigma = np.cov(window.to_numpy(), rowvar=False, ddof=1)
    w = GMVSample().fit(window)
    var_gmv = float(w @ sigma @ w)
    eq = np.full(5, 0.2)
    var_eq = float(eq @ sigma @ eq)
    assert var_gmv <= var_eq + 1e-10


def test_max_sharpe_naive_sums_to_one(window: pd.DataFrame) -> None:
    w = MaxSharpeNaive().fit(window, rf=0.001)
    assert w.sum() == pytest.approx(1.0)


def test_risk_parity_converges_positive(window: pd.DataFrame) -> None:
    w = RiskParity().fit(window)
    assert w.shape == (5,)
    assert np.all(w > 0.0)
    assert w.sum() == pytest.approx(1.0, rel=1e-6)


def test_risk_parity_equal_rc_on_diagonal_cov() -> None:
    # Build an exactly diagonal covariance: ERC weights are proportional to 1/sigma.
    idx = pd.date_range("2010-01-31", periods=400, freq="ME")
    rng = np.random.default_rng(0)
    sd = np.array([0.10, 0.20, 0.05])
    data = rng.standard_normal((400, 3)) * sd
    window = pd.DataFrame(data, index=idx, columns=list("ABC"))
    w = RiskParity(max_iter=2000, tol=1e-10).fit(window)
    sigma = np.cov(window.to_numpy(), rowvar=False, ddof=1)
    rc = w * (sigma @ w)
    # All risk contributions should be roughly equal.
    np.testing.assert_allclose(rc, np.mean(rc), rtol=5e-2)


def test_degenerate_window_raises() -> None:
    bad = pd.DataFrame({"A": [0.01]})
    with pytest.raises(DegenerateWindowError):
        OneOverN().fit(bad)


def test_non_finite_window_raises() -> None:
    bad = pd.DataFrame({"A": [0.01, np.nan], "B": [0.0, 0.02]})
    with pytest.raises(DegenerateWindowError):
        GMVSample().fit(bad)


def test_safe_inv_ridge_fallback_on_singular() -> None:
    # A rank-deficient matrix with non-zero trace -> ridge regularizer is non-zero
    # and the fallback solve produces finite output.
    singular = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    out = _safe_inv(singular)
    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))


def test_max_sharpe_falls_back_to_gmv_when_excess_zero() -> None:
    # Antisymmetric rows produce a sample mean of exactly 0 element-wise; with
    # rf=0 the excess vector is exact-zero, exercising the GMV fallback path.
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    pair = np.array([[0.05, 0.03, 0.04, 0.02], [-0.05, -0.03, -0.04, -0.02]])
    data = np.tile(pair, (60, 1))
    window = pd.DataFrame(data, index=idx, columns=list("ABCD"))
    assert np.all(window.to_numpy().mean(axis=0) == 0.0)
    w = MaxSharpeNaive().fit(window, rf=0.0)
    expected = GMVSample().fit(window, rf=0.0)
    np.testing.assert_allclose(w, expected, atol=1e-10)


def test_risk_parity_nonpositive_diag_raises() -> None:
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(2)
    data = rng.standard_normal((120, 3)) * 0.04
    data[:, 1] = 0.0  # zero variance column -> non-positive diagonal
    window = pd.DataFrame(data, index=idx, columns=list("ABC"))
    with pytest.raises(DegenerateWindowError):
        RiskParity().fit(window)


def test_risk_parity_zero_max_iter_returns_warm_start() -> None:
    # ``max_iter=0`` skips the convergence loop entirely (branch 150->165);
    # the inverse-volatility warm start is normalized and returned as-is.
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(13)
    data = rng.standard_normal((120, 4)) * np.array([0.10, 0.20, 0.05, 0.15])
    window = pd.DataFrame(data, index=idx, columns=list("ABCD"))
    w = RiskParity(max_iter=0).fit(window)
    sigma = np.cov(window.to_numpy(), rowvar=False, ddof=1)
    inv_vol = 1.0 / np.sqrt(np.diag(sigma))
    expected = inv_vol / inv_vol.sum()
    np.testing.assert_allclose(w, expected, atol=1e-12)
    assert w.sum() == pytest.approx(1.0)
