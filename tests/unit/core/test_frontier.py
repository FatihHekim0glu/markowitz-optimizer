"""Unit tests for ``markowitz.core.frontier.AnalyticFrontier``."""

from __future__ import annotations

import numpy as np
import pytest

from markowitz.core.exceptions import (
    InfeasibleFrontierError,
    NumericalError,
    SingularCovarianceError,
)
from markowitz.core.frontier import AnalyticFrontier
from markowitz.core.portfolio import Portfolio


@pytest.fixture
def two_asset() -> tuple[np.ndarray, np.ndarray]:
    mu = np.array([0.10, 0.15])
    Sigma = np.array([[0.0225, 0.01125], [0.01125, 0.0625]])
    return mu, Sigma


class TestConstruction:
    def test_basic_construction(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        mu, S = two_asset
        f = AnalyticFrontier(mu, S)
        assert f.n_assets == 2
        abcd = f.abcd
        assert abcd.A > 0 and abcd.C > 0 and abcd.D > 0

    def test_rejects_non_pd(self) -> None:
        with pytest.raises(SingularCovarianceError):
            AnalyticFrontier(np.array([0.1, 0.2]), np.zeros((2, 2)))

    def test_rejects_dim_mismatch(self) -> None:
        with pytest.raises(NumericalError):
            AnalyticFrontier(np.array([0.1, 0.2, 0.3]), np.eye(2))

    def test_rejects_2d_mu(self) -> None:
        with pytest.raises(NumericalError):
            AnalyticFrontier(np.zeros((2, 1)), np.eye(2))

    def test_rejects_non_square_sigma(self) -> None:
        with pytest.raises(NumericalError):
            AnalyticFrontier(np.zeros(2), np.zeros((2, 3)))

    def test_rejects_non_finite_mu(self) -> None:
        with pytest.raises(NumericalError):
            AnalyticFrontier(np.array([0.1, np.nan]), np.eye(2))


class TestGMV:
    def test_weights_sum_to_one(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        gmv = f.gmv()
        assert isinstance(gmv, Portfolio)
        assert sum(gmv.weights) == pytest.approx(1.0, abs=1e-12)

    def test_variance_equals_one_over_A(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        gmv = f.gmv()
        assert gmv.variance() == pytest.approx(1.0 / f.abcd.A, abs=1e-12)

    def test_expected_return_is_b_over_a(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        gmv = f.gmv()
        assert gmv.expected_return == pytest.approx(f.abcd.B / f.abcd.A, abs=1e-12)


class TestTangency:
    def test_weights_sum_to_one(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        tan = f.tangency(rf=0.02)
        assert sum(tan.weights) == pytest.approx(1.0, abs=1e-12)

    def test_sharpe_is_set(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        tan = f.tangency(rf=0.02)
        assert tan.sharpe is not None and tan.sharpe > 0

    def test_infeasible_at_gmv_rf(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        rf_bad = f.abcd.B / f.abcd.A  # exactly mu_GMV
        with pytest.raises(InfeasibleFrontierError):
            f.tangency(rf=rf_bad)


class TestEfficientReturn:
    def test_at_target_return(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        p = f.efficient_return(mu_p=0.12)
        assert p.expected_return == pytest.approx(0.12)
        assert sum(p.weights) == pytest.approx(1.0, abs=1e-12)

    def test_variance_matches_hyperbola(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        mu_p = 0.13
        p = f.efficient_return(mu_p)
        A, B, C, D = f.abcd
        expected_var = (A * mu_p * mu_p - 2.0 * B * mu_p + C) / D
        assert p.variance() == pytest.approx(expected_var, abs=1e-12)

    def test_degenerate_raises(self) -> None:
        # mu collinear with 1 -> D = 0.
        mu = 0.05 * np.ones(3)
        Sigma = np.diag([0.04, 0.09, 0.16])
        f = AnalyticFrontier(mu, Sigma)
        assert f.is_degenerate
        with pytest.raises(InfeasibleFrontierError):
            f.efficient_return(0.10)


class TestFrontier:
    def test_returns_requested_count(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        pts = f.frontier(n_points=25)
        assert len(pts) == 25
        assert all(isinstance(p, Portfolio) for p in pts)

    def test_default_bounds_run_from_gmv_upward(
        self, two_asset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        f = AnalyticFrontier(*two_asset)
        pts = f.frontier(n_points=11)
        mu_gmv = f.abcd.B / f.abcd.A
        assert pts[0].expected_return == pytest.approx(mu_gmv)
        assert pts[-1].expected_return > mu_gmv

    def test_explicit_bounds(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        pts = f.frontier(n_points=5, mu_min=0.10, mu_max=0.20)
        assert pts[0].expected_return == pytest.approx(0.10)
        assert pts[-1].expected_return == pytest.approx(0.20)

    def test_rejects_bad_count(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        with pytest.raises(ValueError):
            f.frontier(n_points=1)

    def test_rejects_inverted_bounds(self, two_asset: tuple[np.ndarray, np.ndarray]) -> None:
        f = AnalyticFrontier(*two_asset)
        with pytest.raises(ValueError):
            f.frontier(n_points=5, mu_min=0.2, mu_max=0.1)


class TestEdgeCases:
    def test_n_equals_1_construct(self) -> None:
        mu = np.array([0.10])
        Sigma = np.array([[0.04]])
        f = AnalyticFrontier(mu, Sigma)
        assert f.n_assets == 1
        # D = A*C - B^2 = (1/0.04)*(0.01/0.04) - (0.1/0.04)^2 = 6.25 - 6.25 = 0.
        assert f.is_degenerate
        gmv = f.gmv()
        np.testing.assert_allclose(gmv.weights, [1.0], atol=1e-12)
        assert gmv.expected_return == pytest.approx(0.10, abs=1e-12)
        assert gmv.volatility == pytest.approx(0.2, abs=1e-12)
        # Degenerate frontier -> efficient_return raises for any target.
        with pytest.raises(InfeasibleFrontierError):
            f.efficient_return(0.10)

    def test_near_singular_perfectly_correlated_raises(self) -> None:
        Sigma = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]])
        with pytest.raises(SingularCovarianceError):
            AnalyticFrontier(np.array([0.10, 0.15]), Sigma)

    def test_high_condition_number_warns(self) -> None:
        # Well-PD diagonal with condition number ~1e10; must construct cleanly.
        Sigma = np.diag([1.0, 1e-10])
        mu = np.array([0.10, 0.15])
        f = AnalyticFrontier(mu, Sigma)
        assert f.n_assets == 2
        assert f.abcd.A > 0
        assert f.abcd.C > 0
