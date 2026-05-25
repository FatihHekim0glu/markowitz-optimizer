"""Unit tests for :func:`markowitz.backtest.turnover.compute_turnover`."""

from __future__ import annotations

import numpy as np
import pytest

from markowitz.backtest.turnover import compute_turnover


def test_turnover_zero_when_new_equals_drift() -> None:
    w_prev = np.array([0.5, 0.5])
    r = np.array([0.10, -0.05])
    denom = 1.0 + float(w_prev @ r)
    w_drift = w_prev * (1.0 + r) / denom
    assert compute_turnover(w_drift, w_prev, r) == pytest.approx(0.0, abs=1e-15)


def test_turnover_is_one_sided_half_l1() -> None:
    w_prev = np.array([0.5, 0.5])
    r = np.array([0.0, 0.0])  # no drift
    w_new = np.array([0.7, 0.3])
    expected = 0.5 * np.sum(np.abs(w_new - w_prev))
    assert compute_turnover(w_new, w_prev, r) == pytest.approx(expected)


def test_turnover_full_rebalance_to_cash_equivalent() -> None:
    w_prev = np.array([1.0, 0.0])
    w_new = np.array([0.0, 1.0])
    r = np.zeros(2)
    assert compute_turnover(w_new, w_prev, r) == pytest.approx(1.0)


def test_turnover_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="align"):
        compute_turnover(np.zeros(3), np.zeros(2), np.zeros(2))


def test_turnover_catastrophic_denominator() -> None:
    # 1 + w.r == 0: full wipeout, should still return a finite number.
    w_prev = np.array([1.0])
    w_new = np.array([1.0])
    r = np.array([-1.0])
    out = compute_turnover(w_new, w_prev, r)
    assert np.isfinite(out)
