"""Tests for compute_returns, align_universe, and summary_stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.data.exceptions import (
    DataIntegrityError,
    EmptyDataError,
    InsufficientDataError,
)
from markowitz.data.loaders import (
    align_universe,
    compute_returns,
    summary_stats,
)


@pytest.fixture
def pinned_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 99.0, 108.9, 120.0],
            "BBB": [50.0, 52.0, 50.96, 53.0, 55.0],
        },
        index=idx,
    )


def test_compute_returns_simple_pinned(pinned_prices: pd.DataFrame) -> None:
    rets = compute_returns(pinned_prices, method="simple")
    # First row dropped; should have 4 rows
    assert rets.shape == (4, 2)
    # Hand-computed return for AAA at t=1: 110/100 - 1 = 0.10
    assert rets["AAA"].iloc[0] == pytest.approx(0.10)
    # t=2: 99/110 - 1 = -0.10
    assert rets["AAA"].iloc[1] == pytest.approx(-0.10)
    # t=3: 108.9/99 - 1 = 0.10
    assert rets["AAA"].iloc[2] == pytest.approx(0.10)


def test_compute_returns_log(pinned_prices: pd.DataFrame) -> None:
    rets = compute_returns(pinned_prices, method="log")
    expected = np.log(110.0 / 100.0)
    assert rets["AAA"].iloc[0] == pytest.approx(expected)
    # Log returns are not equal to simple returns
    simple = compute_returns(pinned_prices, method="simple")
    assert not np.isclose(rets["AAA"].iloc[0], simple["AAA"].iloc[0])


def test_compute_returns_unknown_method_raises(pinned_prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        compute_returns(pinned_prices, method="quadratic")


def test_compute_returns_empty_raises() -> None:
    empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=["A"], dtype="float64")
    with pytest.raises(EmptyDataError):
        compute_returns(empty)


def test_compute_returns_bad_index_raises() -> None:
    bad = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1])
    with pytest.raises(DataIntegrityError):
        compute_returns(bad)


def test_compute_returns_dropna_true_removes_rows() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    p = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [np.nan, 5.0, 6.0]}, index=idx)
    rets = compute_returns(p, dropna=True)
    # B's pct_change at t=1 is NaN (was NaN before); both rows where any NaN removed
    assert not rets.isna().any().any()


def test_compute_returns_dropna_false_keeps_nans() -> None:
    idx = pd.date_range("2024-01-02", periods=4, freq="B")
    p = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [np.nan, 5.0, 6.0, 7.0]}, index=idx)
    rets = compute_returns(p, dropna=False)
    assert rets.isna().any().any()


def test_align_universe_common_window() -> None:
    idx = pd.date_range("2024-01-02", periods=300, freq="B")
    rets = pd.DataFrame(
        {
            "A": np.linspace(0.0, 0.1, 300),
            "B": np.linspace(0.0, 0.1, 300),
        },
        index=idx,
    )
    # Make B ragged for the first 50 obs
    rets.loc[rets.index[:50], "B"] = np.nan
    aligned = align_universe(rets, min_obs_per_ticker=200, common_window=True)
    assert not aligned.isna().any().any()
    assert aligned.shape[0] == 250
    assert set(aligned.columns) == {"A", "B"}


def test_align_universe_drops_sparse_ticker() -> None:
    idx = pd.date_range("2024-01-02", periods=300, freq="B")
    rets = pd.DataFrame(
        {
            "A": np.linspace(0.0, 0.1, 300),
            "B": np.linspace(0.0, 0.1, 300),
            "TINY": np.full(300, np.nan),
        },
        index=idx,
    )
    rets.loc[rets.index[-5:], "TINY"] = 0.01  # only 5 obs
    aligned = align_universe(rets, min_obs_per_ticker=100, common_window=False)
    assert "TINY" not in aligned.columns
    assert set(aligned.columns) == {"A", "B"}


def test_align_universe_raises_insufficient() -> None:
    idx = pd.date_range("2024-01-02", periods=50, freq="B")
    rets = pd.DataFrame({"A": np.zeros(50)}, index=idx)
    with pytest.raises(InsufficientDataError):
        align_universe(rets, min_obs_per_ticker=252)


def test_align_universe_empty_raises() -> None:
    with pytest.raises(EmptyDataError):
        align_universe(pd.DataFrame())


def test_annualization_factor_252_vs_365() -> None:
    # Daily returns of constant 0.001 -> annualized mu should be 0.001 * 252
    idx = pd.date_range("2024-01-02", periods=252, freq="B")
    rets = pd.DataFrame({"X": np.full(252, 0.001)}, index=idx)
    stats = summary_stats(rets, frequency="daily", annualize=True)
    assert stats.loc["X", "mu_hat"] == pytest.approx(0.001 * 252)
    assert stats.loc["X", "mu_hat"] != pytest.approx(0.001 * 365)


def test_summary_stats_columns() -> None:
    idx = pd.date_range("2024-01-02", periods=100, freq="B")
    rng = np.random.default_rng(42)
    rets = pd.DataFrame(
        {"A": rng.normal(0.0, 0.01, 100), "B": rng.normal(0.0, 0.02, 100)},
        index=idx,
    )
    stats = summary_stats(rets, frequency="daily", annualize=True)
    expected_cols = {
        "mu_hat",
        "sigma_hat",
        "sharpe",
        "skew",
        "kurtosis",
        "n_obs",
        "min",
        "max",
    }
    assert expected_cols <= set(stats.columns)
    assert set(stats.index) == {"A", "B"}
    assert stats["n_obs"].dtype.kind == "i"


def test_summary_stats_no_annualization_matches_raw() -> None:
    idx = pd.date_range("2024-01-02", periods=100, freq="B")
    rng = np.random.default_rng(0)
    rets = pd.DataFrame({"X": rng.normal(0.0, 0.01, 100)}, index=idx)
    stats = summary_stats(rets, annualize=False)
    assert stats.loc["X", "mu_hat"] == pytest.approx(rets["X"].mean())


def test_summary_stats_monthly_uses_12() -> None:
    idx = pd.date_range("2024-01-31", periods=24, freq="ME")
    rets = pd.DataFrame({"X": np.full(24, 0.005)}, index=idx)
    stats = summary_stats(rets, frequency="monthly", annualize=True)
    assert stats.loc["X", "mu_hat"] == pytest.approx(0.005 * 12)


def test_summary_stats_empty_raises() -> None:
    with pytest.raises(EmptyDataError):
        summary_stats(pd.DataFrame())


def test_summary_stats_unknown_frequency_raises() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    rets = pd.DataFrame({"X": np.zeros(10)}, index=idx)
    with pytest.raises(ValueError, match="unknown frequency"):
        summary_stats(rets, frequency="hourly")
