"""Regression test guaranteeing no in-sample look-ahead in walk-forward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.backtest import WalkForward


class _LookaheadMockStrategy:
    """Strategy that records the maximum window index it has ever observed."""

    name = "MockLookahead"
    long_only = True

    def __init__(self) -> None:
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fit(self, window: pd.DataFrame, *, rf: float = 0.0) -> np.ndarray:
        self.calls.append((window.index.min(), window.index.max()))
        n = window.shape[1]
        return np.full(n, 1.0 / n, dtype=float)


@pytest.mark.regression
def test_walk_forward_has_no_lookahead() -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    returns = pd.DataFrame(
        rng.standard_normal((240, 4)) * 0.04,
        index=idx,
        columns=list("WXYZ"),
    )

    mock = _LookaheadMockStrategy()
    wf = WalkForward(
        returns,
        {"mock": mock},
        lookback=60,
        cost_bps=0.0,
        debug_no_lookahead=True,
    )
    result = wf.run()

    assert len(mock.calls) > 0
    # For every rebalance, the maximum date seen must be strictly earlier.
    for rebal_date, (_, win_max) in zip(result.rebalance_dates, mock.calls, strict=True):
        assert win_max < rebal_date, (
            f"look-ahead: window ended at {win_max} but rebalance was {rebal_date}"
        )
