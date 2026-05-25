"""Unit tests for :func:`markowitz.backtest.costs.apply_transaction_costs`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.backtest.costs import apply_transaction_costs


def test_costs_subtract_expected_amount() -> None:
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    gross = pd.Series([0.01, 0.02, -0.005, 0.0], index=idx)
    turnover = pd.Series([0.2, 0.0, 1.0, 0.5], index=idx)
    bps = 10.0
    net = apply_transaction_costs(gross, turnover, bps=bps)
    expected = gross - (bps / 10_000.0) * turnover
    pd.testing.assert_series_equal(net, expected, check_names=False)


def test_costs_zero_when_no_turnover() -> None:
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    gross = pd.Series([0.01, 0.02, 0.03], index=idx)
    turnover = pd.Series(0.0, index=idx)
    net = apply_transaction_costs(gross, turnover, bps=25.0)
    np.testing.assert_allclose(net.to_numpy(), gross.to_numpy())


def test_costs_reindex_misaligned_turnover() -> None:
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    gross = pd.Series([0.01, 0.02, 0.03], index=idx)
    turnover = pd.Series([0.5], index=[idx[0]])
    net = apply_transaction_costs(gross, turnover, bps=100.0)
    # Only the first observation should be reduced.
    assert net.iloc[0] == pytest.approx(0.01 - 0.01 * 0.5)
    assert net.iloc[1] == pytest.approx(0.02)
    assert net.iloc[2] == pytest.approx(0.03)
