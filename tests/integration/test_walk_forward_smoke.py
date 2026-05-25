"""Integration smoke test for :class:`markowitz.backtest.WalkForward`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.backtest import (
    GMVSample,
    MaxSharpeNaive,
    OneOverN,
    RiskParity,
    WalkForward,
)


@pytest.mark.integration
def test_walk_forward_smoke_six_assets() -> None:
    rng = np.random.default_rng(123)
    n_obs = 480
    n_assets = 6
    idx = pd.date_range("1985-01-31", periods=n_obs, freq="ME")
    # Build a non-trivial correlation structure so GMV/MaxSharpe are well-posed.
    mu = np.linspace(0.003, 0.012, n_assets)
    L = np.tril(rng.normal(0.0, 0.02, size=(n_assets, n_assets)))
    np.fill_diagonal(L, np.abs(np.diag(L)) + 0.03)
    eps = rng.standard_normal((n_obs, n_assets))
    data = eps @ L.T + mu
    returns = pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])

    strategies = {
        "1/N": OneOverN(),
        "GMV": GMVSample(),
        "MaxSharpe": MaxSharpeNaive(),
        "RP": RiskParity(),
    }

    wf = WalkForward(returns, strategies, lookback=120, cost_bps=10.0)
    res = wf.run()

    # Sanity: shapes match (n_obs - lookback) rows.
    expected_rows = n_obs - 120
    assert res.returns_gross.shape == (expected_rows, 4)
    assert res.returns_net.shape == (expected_rows, 4)
    assert res.turnover.shape == (expected_rows, 4)

    # All metrics finite.
    assert np.isfinite(res.returns_gross.to_numpy()).all()
    assert np.isfinite(res.returns_net.to_numpy()).all()
    assert np.isfinite(res.turnover.to_numpy()).all()

    # Weights sum to 1 at every rebalance for every strategy.
    for name, frame in res.weights.items():
        assert not frame.empty, f"no rebalance recorded for {name}"
        row_sums = frame.sum(axis=1).to_numpy()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6, err_msg=name)

    # Summary frame is well-formed.
    summary = res.summary()
    assert set(summary.columns) >= {"Sharpe", "Sortino", "MaxDD", "Calmar", "CEQ", "Turnover"}
    assert np.isfinite(summary.to_numpy()).all()
