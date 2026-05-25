"""Unit tests for :class:`markowitz.backtest.BacktestResult`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from markowitz.backtest import OneOverN, WalkForward


def _make_returns(n_periods: int = 240, n_assets: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n_periods, freq="ME")
    return pd.DataFrame(
        rng.standard_normal((n_periods, n_assets)) * 0.05,
        index=idx,
        columns=[f"A{i}" for i in range(n_assets)],
    )


def _run_small() -> tuple[WalkForward, pd.DataFrame]:
    returns = _make_returns()
    wf = WalkForward(returns, {"1/N": OneOverN()}, lookback=60)
    return wf, returns


def test_summary_returns_dataframe() -> None:
    wf, _ = _run_small()
    result = wf.run()
    summary = result.summary()
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.index) == ["1/N"]


def test_equity_curves() -> None:
    wf, _ = _run_small()
    result = wf.run()
    # Cumulative product of (1 + r) is positive throughout when no -1 returns.
    wealth = (1.0 + result.returns_net["1/N"]).cumprod()
    assert (wealth > 0.0).all()


def test_metrics_table() -> None:
    wf, _ = _run_small()
    result = wf.run()
    summary = result.summary()
    cols_lower = {c.lower() for c in summary.columns}
    assert "sharpe" in cols_lower
    assert any("dd" in c for c in cols_lower)
