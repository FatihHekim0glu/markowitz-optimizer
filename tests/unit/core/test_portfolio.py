"""Tests for the immutable Portfolio value object."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from markowitz.core.exceptions import NumericalError
from markowitz.core.portfolio import Portfolio, portfolio_performance


class TestPortfolioDataclass:
    def test_frozen(self) -> None:
        p = Portfolio(weights=np.array([0.5, 0.5]), expected_return=0.1, volatility=0.2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.expected_return = 0.0  # type: ignore[misc]

    def test_weights_are_copied(self) -> None:
        w = np.array([0.5, 0.5])
        p = Portfolio(weights=w, expected_return=0.1, volatility=0.2)
        w[0] = 999.0
        assert p.weights[0] == 0.5  # unaffected by mutation of the source

    def test_weights_are_read_only(self) -> None:
        p = Portfolio(weights=np.array([0.5, 0.5]), expected_return=0.1, volatility=0.2)
        with pytest.raises(ValueError):
            p.weights[0] = 1.0  # read-only flag should reject writes

    def test_variance_is_volatility_squared(self) -> None:
        p = Portfolio(weights=np.array([1.0]), expected_return=0.1, volatility=0.25)
        assert p.variance() == pytest.approx(0.0625)

    def test_default_sharpe_is_none(self) -> None:
        p = Portfolio(weights=np.array([1.0]), expected_return=0.1, volatility=0.2)
        assert p.sharpe is None

    def test_explicit_sharpe(self) -> None:
        p = Portfolio(
            weights=np.array([1.0]),
            expected_return=0.1,
            volatility=0.2,
            sharpe=0.4,
        )
        assert p.sharpe == 0.4


class TestPortfolioPerformance:
    def test_basic_two_asset(self) -> None:
        w = np.array([0.5, 0.5])
        mu = np.array([0.10, 0.20])
        Sigma = np.array([[0.04, 0.0], [0.0, 0.09]])
        p = portfolio_performance(w, mu, Sigma)
        assert p.expected_return == pytest.approx(0.15)
        assert p.variance() == pytest.approx(0.25 * 0.04 + 0.25 * 0.09)
        assert p.volatility == pytest.approx(np.sqrt(0.25 * 0.04 + 0.25 * 0.09))
        assert p.sharpe is None

    def test_sharpe_with_rf(self) -> None:
        w = np.array([1.0, 0.0])
        mu = np.array([0.10, 0.20])
        Sigma = np.diag([0.04, 0.09])
        p = portfolio_performance(w, mu, Sigma, rf=0.02)
        assert p.sharpe == pytest.approx((0.10 - 0.02) / 0.2)

    def test_sharpe_undefined_when_vol_zero(self) -> None:
        # An all-zero portfolio has zero variance and Sharpe is undefined.
        w = np.zeros(2)
        mu = np.array([0.1, 0.2])
        Sigma = np.eye(2)
        p = portfolio_performance(w, mu, Sigma, rf=0.01)
        assert p.sharpe is None

    def test_raises_on_dim_mismatch(self) -> None:
        with pytest.raises(NumericalError):
            portfolio_performance(np.zeros(2), np.zeros(3), np.eye(2))

    def test_raises_on_2d_weights(self) -> None:
        with pytest.raises(NumericalError):
            portfolio_performance(np.zeros((2, 1)), np.zeros(2), np.eye(2))

    def test_raises_on_bad_sigma_shape(self) -> None:
        with pytest.raises(NumericalError):
            portfolio_performance(np.zeros(2), np.zeros(2), np.eye(3))

    def test_raises_on_nan(self) -> None:
        with pytest.raises(NumericalError):
            portfolio_performance(np.array([np.nan, 0.0]), np.zeros(2), np.eye(2))

    def test_returns_portfolio_instance(self) -> None:
        p = portfolio_performance(np.array([1.0]), np.array([0.1]), np.array([[0.04]]))
        assert isinstance(p, Portfolio)
