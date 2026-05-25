"""Unit tests for :class:`markowitz.backtest.WalkForward`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.backtest import OneOverN, WalkForward
from markowitz.backtest.exceptions import AlignmentError, InsufficientHistoryError


def _make_returns(n_periods: int = 240, n_assets: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n_periods, freq="ME")
    return pd.DataFrame(
        rng.standard_normal((n_periods, n_assets)) * 0.05,
        index=idx,
        columns=[f"A{i}" for i in range(n_assets)],
    )


def test_empty_returns_raises() -> None:
    with pytest.raises(AlignmentError):
        WalkForward(pd.DataFrame(), {"1/N": OneOverN()})


def test_nans_in_returns_raises() -> None:
    returns = _make_returns()
    returns.iloc[10, 0] = np.nan
    with pytest.raises(AlignmentError):
        WalkForward(returns, {"1/N": OneOverN()})


def test_lookback_too_small_raises() -> None:
    with pytest.raises(InsufficientHistoryError):
        WalkForward(_make_returns(), {"1/N": OneOverN()}, lookback=1)


def test_lookback_exceeds_history_raises() -> None:
    with pytest.raises(InsufficientHistoryError):
        WalkForward(_make_returns(n_periods=240), {"1/N": OneOverN()}, lookback=300)


def test_no_strategies_raises() -> None:
    with pytest.raises(AlignmentError):
        WalkForward(_make_returns(), {})


def test_rf_as_series_branch() -> None:
    returns = _make_returns()
    rf = pd.Series(0.0, index=returns.index)
    wf = WalkForward(returns, {"1/N": OneOverN()}, lookback=60, rf=rf)
    result = wf.run()
    assert result.returns_net.shape == (len(returns) - 60, 1)


def test_rf_as_scalar_branch() -> None:
    wf = WalkForward(_make_returns(), {"1/N": OneOverN()}, lookback=60, rf=0.02)
    result = wf.run()
    assert result.returns_net.shape[0] == 180


class _BadShapeStrategy:
    name = "Bad"
    long_only = True

    def fit(self, window: pd.DataFrame, *, rf: float = 0.0) -> np.ndarray:
        return np.zeros(window.shape[1] - 1)


def test_strategy_returns_wrong_shape_raises() -> None:
    wf = WalkForward(_make_returns(), {"Bad": _BadShapeStrategy()}, lookback=60)
    with pytest.raises(AlignmentError):
        wf.run()


def test_quarterly_rebalance_holds_weights_between_rebalances() -> None:
    wf = WalkForward(
        _make_returns(), {"1/N": OneOverN()}, lookback=60, rebalance="Q"
    )
    result = wf.run()
    non_rebal_to = [
        result.turnover["1/N"].iloc[i]
        for i in range(1, len(result.turnover))
        if i % 3 != 0
    ]
    assert any(x == 0.0 for x in non_rebal_to)


def test_drift_falls_back_on_zero_denominator() -> None:
    returns = _make_returns(n_periods=80)
    returns.iloc[65, :] = -1.0
    wf = WalkForward(returns, {"1/N": OneOverN()}, lookback=60, rebalance="Q")
    result = wf.run()
    assert result.returns_net.shape[0] == 20


def test_invalid_rebalance_alias_falls_back_to_monthly() -> None:
    wf = WalkForward(_make_returns(), {"1/N": OneOverN()}, lookback=60, rebalance="ZZZ")
    result = wf.run()
    assert result.returns_net.shape[0] == 180


def test_walk_forward_debug_no_lookahead_passes_on_normal() -> None:
    wf = WalkForward(
        _make_returns(), {"1/N": OneOverN()}, lookback=60, debug_no_lookahead=True
    )
    result = wf.run()
    assert not result.returns_net.isna().any().any()
