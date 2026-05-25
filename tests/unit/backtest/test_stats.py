"""Unit tests for :mod:`markowitz.backtest.stats`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.backtest.stats import (
    calmar,
    ceq,
    jobson_korkie_memmel,
    ledoit_wolf_bootstrap_pvalue,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


@pytest.fixture
def toy_returns() -> pd.Series:
    return pd.Series(
        [0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002, 0.012, -0.008, 0.018],
        index=pd.date_range("2020-01-31", periods=10, freq="ME"),
    )


def test_sharpe_matches_hand_calculation(toy_returns: pd.Series) -> None:
    expected = float(toy_returns.mean() / toy_returns.std(ddof=1) * np.sqrt(12))
    assert sharpe_ratio(toy_returns, rf=0.0, ann=12) == pytest.approx(expected)


def test_sharpe_zero_variance() -> None:
    flat = pd.Series([0.01] * 12)
    assert sharpe_ratio(flat) == 0.0


def test_sortino_uses_downside_only(toy_returns: pd.Series) -> None:
    down = toy_returns.clip(upper=0.0)
    dd_sd = float(np.sqrt((down**2).mean()))
    expected = float(toy_returns.mean() / dd_sd * np.sqrt(12))
    assert sortino_ratio(toy_returns) == pytest.approx(expected)


def test_sortino_all_positive() -> None:
    all_up = pd.Series([0.01, 0.02, 0.03])
    assert sortino_ratio(all_up) == 0.0


def test_max_drawdown_is_non_positive(toy_returns: pd.Series) -> None:
    assert max_drawdown(toy_returns) <= 0.0


def test_max_drawdown_known_value() -> None:
    r = pd.Series([0.5, -0.5])  # wealth: 1.5 -> 0.75; dd = -0.5
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_monotone_increasing() -> None:
    assert max_drawdown(pd.Series([0.01, 0.02, 0.03])) == pytest.approx(0.0)


def test_max_drawdown_empty() -> None:
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0


def test_calmar_formula(toy_returns: pd.Series) -> None:
    mdd = max_drawdown(toy_returns)
    expected = float(toy_returns.mean()) * 12 / abs(mdd)
    assert calmar(toy_returns) == pytest.approx(expected)


def test_calmar_zero_drawdown() -> None:
    assert calmar(pd.Series([0.01, 0.02])) == 0.0


def test_ceq_formula(toy_returns: pd.Series) -> None:
    gamma = 5.0
    expected = float(toy_returns.mean() - 0.5 * gamma * toy_returns.var(ddof=1))
    assert ceq(toy_returns, gamma=gamma) == pytest.approx(expected)


def test_jkm_identical_series_zero(toy_returns: pd.Series) -> None:
    z, p = jobson_korkie_memmel(toy_returns, toy_returns.copy())
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_jkm_sign_and_pvalue_range() -> None:
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.02, 0.05, 240))
    b = pd.Series(rng.normal(0.005, 0.05, 240))
    z, p = jobson_korkie_memmel(a, b)
    assert z > 0.0
    assert 0.0 <= p <= 1.0


def test_jkm_excess_rf_with_series() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.01, 0.02, 60))
    rf = pd.Series(0.001, index=r.index)
    sr = sharpe_ratio(r, rf=rf, ann=12)
    assert np.isfinite(sr)


def test_lw_bootstrap_returns_valid_pvalue() -> None:
    rng = np.random.default_rng(7)
    a = pd.Series(rng.normal(0.0, 0.02, 80))
    b = pd.Series(rng.normal(0.0, 0.02, 80))
    p = ledoit_wolf_bootstrap_pvalue(a, b, B=199, block_length=4, seed=3)
    assert 0.0 <= p <= 1.0


def test_lw_bootstrap_handles_short_input() -> None:
    a = pd.Series([0.01, 0.02])
    b = pd.Series([0.01, 0.03])
    assert ledoit_wolf_bootstrap_pvalue(a, b, B=10, block_length=5) == 1.0


def test_jkm_short_overlap_returns_neutral() -> None:
    # Fewer than 3 overlapping observations -> early-out (0.0, 1.0).
    a = pd.Series([0.01, 0.02], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    b = pd.Series([0.01, 0.03], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    z, p = jobson_korkie_memmel(a, b)
    assert z == 0.0
    assert p == 1.0


def test_jkm_zero_variance_series_returns_neutral() -> None:
    # When either series has zero std the test is undefined; the loader
    # short-circuits to (0.0, 1.0) rather than dividing by zero.
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    flat = pd.Series([0.01] * 24, index=idx)
    rng = np.random.default_rng(4)
    other = pd.Series(rng.normal(0.0, 0.02, 24), index=idx)
    z, p = jobson_korkie_memmel(flat, other)
    assert z == 0.0
    assert p == 1.0


def test_lw_bootstrap_zero_variance_sharpe_diff_is_zero() -> None:
    # _sharpe_diff short-circuits to 0.0 when either input has sx==0;
    # the resulting bootstrap p-value is still a valid probability.
    idx = pd.date_range("2020-01-31", periods=40, freq="ME")
    flat = pd.Series([0.01] * 40, index=idx)
    other = pd.Series(np.linspace(-0.01, 0.01, 40), index=idx)
    p = ledoit_wolf_bootstrap_pvalue(flat, other, B=50, block_length=4, seed=1)
    assert 0.0 <= p <= 1.0
