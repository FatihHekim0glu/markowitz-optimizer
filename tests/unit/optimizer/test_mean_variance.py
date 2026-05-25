"""Unit tests for the MeanVariance optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cvxpy")

from markowitz.core import AnalyticFrontier
from markowitz.optimizer import (
    CustomConstraint,
    InfeasibleError,
    LeverageCap,
    LongOnly,
    MeanVariance,
    OptimizationError,
    SectorCap,
    WeightBounds,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_problem() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(2024)
    n = 5
    tickers = [f"S{i}" for i in range(n)]
    A = rng.standard_normal((n, n))
    Sigma_raw = A @ A.T / n + 0.05 * np.eye(n)
    sigma = pd.DataFrame(Sigma_raw, index=tickers, columns=tickers)
    mu_arr = np.array([0.10, 0.08, 0.12, 0.06, 0.15])
    mu = pd.Series(mu_arr, index=tickers)
    return mu, sigma


@pytest.fixture
def small_problem_arrays(
    small_problem: tuple[pd.Series, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    mu, sigma = small_problem
    return mu.to_numpy(), sigma.to_numpy()


# ---------------------------------------------------------------------------
# Basic plumbing
# ---------------------------------------------------------------------------


class TestBasics:
    def test_pandas_inputs_preserve_tickers(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma)
        assert opt.tickers == list(mu.index)
        assert opt.n_assets == len(mu)
        assert opt.last_weights is None

    def test_numpy_inputs_get_default_names(
        self, small_problem_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        mu, sigma = small_problem_arrays
        opt = MeanVariance(mu, sigma)
        assert opt.tickers == [f"A{i}" for i in range(len(mu))]

    def test_negative_regularize_rejected(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        with pytest.raises(ValueError, match="non-negative"):
            MeanVariance(mu, sigma, regularize=-0.1)

    def test_shape_mismatch(self) -> None:
        mu = np.array([0.1, 0.2])
        sigma = np.eye(3)
        with pytest.raises(ValueError, match="incompatible"):
            MeanVariance(mu, sigma)

    def test_empty_mu_rejected(self) -> None:
        with pytest.raises(ValueError):
            MeanVariance(np.array([]), np.zeros((0, 0)))

    def test_add_clear_constraints(self, small_problem: tuple[pd.Series, pd.DataFrame]) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma)
        opt.add_constraint(LongOnly())
        assert len(opt.constraints) == 1
        opt.add_constraint(WeightBounds(0.0, 0.5))
        assert len(opt.constraints) == 2
        opt.clear_constraints()
        assert opt.constraints == []


# ---------------------------------------------------------------------------
# Analytical-parity tests
# ---------------------------------------------------------------------------


class TestAnalyticalParity:
    def test_min_volatility_matches_analytical(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        # Disable long-only / box constraints so the numerical problem matches
        # the analytical GMV (which is fully unconstrained except sum(w)=1).
        opt = MeanVariance(mu, sigma, weight_bounds=(None, None))
        w_num = opt.min_volatility()
        analytical = AnalyticFrontier(mu.to_numpy(), sigma.to_numpy()).gmv().weights
        np.testing.assert_allclose(w_num.to_numpy(), analytical, atol=1e-6, rtol=1e-6)

    def test_max_sharpe_matches_analytical(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        rf = 0.02
        opt = MeanVariance(mu, sigma, weight_bounds=(None, None))
        w_num = opt.max_sharpe(risk_free_rate=rf)
        analytical = AnalyticFrontier(mu.to_numpy(), sigma.to_numpy()).tangency(rf).weights
        np.testing.assert_allclose(w_num.to_numpy(), analytical, atol=1e-6, rtol=1e-6)

    def test_max_quadratic_utility_matches_analytical(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        risk_aversion = 4.0
        opt = MeanVariance(mu, sigma, weight_bounds=(None, None))
        w_num = opt.max_quadratic_utility(risk_aversion=risk_aversion)
        # Analytical: with budget constraint sum(w) = 1 the Lagrangian gives
        #   w = Sigma^{-1} (mu + nu * 1) / lambda
        # where nu is chosen to satisfy sum(w) = 1.  Solve in closed form.
        sigma_arr = sigma.to_numpy()
        mu_arr = mu.to_numpy()
        ones = np.ones_like(mu_arr)
        s_inv_mu = np.linalg.solve(sigma_arr, mu_arr)
        s_inv_one = np.linalg.solve(sigma_arr, ones)
        # Solve risk_aversion = ones @ (s_inv_mu + nu * s_inv_one)
        nu = (risk_aversion - ones @ s_inv_mu) / (ones @ s_inv_one)
        w_analytical = (s_inv_mu + nu * s_inv_one) / risk_aversion
        np.testing.assert_allclose(w_num.to_numpy(), w_analytical, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Constraint enforcement
# ---------------------------------------------------------------------------


class TestConstraintEnforcement:
    def test_long_only_enforced(self, small_problem: tuple[pd.Series, pd.DataFrame]) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0))
        w = opt.min_volatility()
        assert (w.to_numpy() >= -1e-8).all()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weight_bounds_enforced(self, small_problem: tuple[pd.Series, pd.DataFrame]) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma, weight_bounds=(0.05, 0.4))
        w = opt.min_volatility()
        assert (w.to_numpy() >= 0.05 - 1e-7).all()
        assert (w.to_numpy() <= 0.4 + 1e-7).all()

    def test_sector_cap_enforced(self, small_problem: tuple[pd.Series, pd.DataFrame]) -> None:
        mu, sigma = small_problem
        sector_map = {"S0": "tech", "S1": "tech", "S2": "fin", "S3": "fin", "S4": "energy"}
        opt = MeanVariance(mu, sigma)
        opt.add_constraint(SectorCap(sector_map, caps={"tech": 0.3}))
        w = opt.max_quadratic_utility(risk_aversion=1.0)
        tech_total = float(w["S0"] + w["S1"])
        assert tech_total <= 0.3 + 1e-6

    def test_leverage_cap_enforced(self, small_problem: tuple[pd.Series, pd.DataFrame]) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma, weight_bounds=(-1.0, 1.0))
        opt.add_constraint(LeverageCap(1.5))
        w = opt.max_quadratic_utility(risk_aversion=0.5)
        assert float(np.abs(w.to_numpy()).sum()) <= 1.5 + 1e-6


# ---------------------------------------------------------------------------
# Efficient frontier targets
# ---------------------------------------------------------------------------


class TestEfficientFrontier:
    def test_efficient_return_at_targets(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma, weight_bounds=(None, None))
        gmv = opt.min_volatility()
        gmv_ret = float(mu @ gmv)
        max_ret = float(mu.max())
        targets = np.linspace(gmv_ret + 1e-4, max_ret - 1e-4, 5)
        variances = []
        for t in targets:
            w = opt.efficient_return(target_return=float(t))
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
            # Realised return meets target.
            assert float(mu @ w) >= float(t) - 1e-6
            variances.append(float(w @ sigma @ w))
        # Variance should be monotone non-decreasing in target return.
        for a, b in zip(variances, variances[1:]):
            assert b >= a - 1e-8

    def test_efficient_risk_caps_volatility(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0))
        # First find the feasible GMV vol; target slightly above guarantees feasibility.
        gmv = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0)).min_volatility()
        gmv_vol = float(np.sqrt(gmv.to_numpy() @ sigma.to_numpy() @ gmv.to_numpy()))
        target_vol = gmv_vol * 1.2
        w = opt.efficient_risk(target_volatility=target_vol)
        realised = float(np.sqrt(w.to_numpy() @ sigma.to_numpy() @ w.to_numpy()))
        assert realised <= target_vol + 1e-5

    def test_efficient_risk_rejects_nonpositive(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma)
        with pytest.raises(ValueError):
            opt.efficient_risk(target_volatility=0.0)


# ---------------------------------------------------------------------------
# Reporting / failure modes
# ---------------------------------------------------------------------------


class TestReportingAndFailures:
    def test_portfolio_performance_tuple(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma)
        w = opt.min_volatility()
        ret, vol, sharpe = opt.portfolio_performance(risk_free_rate=0.02)
        assert isinstance(ret, float)
        assert isinstance(vol, float)
        assert isinstance(sharpe, float)
        assert vol > 0
        # Manual calculation cross-check.
        assert abs(ret - float(mu @ w)) < 1e-10

    def test_portfolio_performance_without_solve_raises(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma)
        with pytest.raises(ValueError, match="No weights"):
            opt.portfolio_performance()

    def test_portfolio_performance_accepts_explicit_weights(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma)
        w = pd.Series(1 / len(mu), index=mu.index)
        ret, vol, sharpe = opt.portfolio_performance(w, risk_free_rate=0.0)
        assert vol > 0

    def test_max_quadratic_utility_negative_lambda_rejected(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma)
        with pytest.raises(ValueError, match="positive"):
            opt.max_quadratic_utility(risk_aversion=-1.0)

    def test_infeasible_target_return_raises_typed_exception(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0))
        # Target above the best long-only achievable expected return.
        absurd_target = float(mu.max()) + 1.0
        with pytest.raises(InfeasibleError):
            opt.efficient_return(target_return=absurd_target)

    def test_max_sharpe_degenerate_raises(self) -> None:
        # All assets earn below the risk-free rate.
        mu = pd.Series([0.01, 0.02, 0.015], index=["A", "B", "C"])
        sigma = pd.DataFrame(np.eye(3) * 0.04, index=mu.index, columns=mu.index)
        opt = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0))
        with pytest.raises(InfeasibleError):
            opt.max_sharpe(risk_free_rate=0.05)

    def test_chaining_returns_self(self, small_problem: tuple[pd.Series, pd.DataFrame]) -> None:
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma)
        result = opt.add_constraint(LongOnly()).add_constraint(LeverageCap(1.5))
        assert result is opt


# ---------------------------------------------------------------------------
# Numerical knobs and degenerate inputs
# ---------------------------------------------------------------------------


class TestRegularisationAndDegeneracies:
    def test_regularize_adds_ridge(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        """Passing ``regularize=eps`` must add ``eps*I`` to Sigma in place."""
        mu, sigma = small_problem
        eps = 1e-3
        opt = MeanVariance(mu, sigma, regularize=eps)
        # The stored sigma is the symmetrised input plus eps*I; the diagonal
        # should therefore exceed the original by exactly eps.
        original_diag = np.diag(sigma.to_numpy())
        # Reach into _sigma deliberately to verify the ridge was applied
        # in-place rather than only at solve time.
        new_diag = np.diag(opt._sigma)
        np.testing.assert_allclose(new_diag - original_diag, eps, atol=1e-12)

    def test_portfolio_performance_zero_vol_returns_nan_sharpe(self) -> None:
        """If volatility is zero, Sharpe is undefined and reported as NaN."""
        mu = pd.Series([0.1, 0.1], index=["A", "B"])
        sigma = pd.DataFrame(np.zeros((2, 2)), index=mu.index, columns=mu.index)
        opt = MeanVariance(mu, sigma, weight_bounds=(None, None))
        zero_weights = pd.Series([0.0, 0.0], index=mu.index)
        ret, vol, sharpe = opt.portfolio_performance(zero_weights, risk_free_rate=0.01)
        assert vol == 0.0
        assert np.isnan(sharpe)
        assert ret == 0.0


class TestMaxSharpeWithConstraints:
    def test_max_sharpe_with_weight_bounds_constraint(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        """A user-added WeightBounds constraint must survive y-space reform."""
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0))
        opt.add_constraint(WeightBounds(0.1, 0.4))
        w = opt.max_sharpe(risk_free_rate=0.02)
        assert (w.to_numpy() >= 0.1 - 1e-6).all()
        assert (w.to_numpy() <= 0.4 + 1e-6).all()
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_max_sharpe_with_unsupported_custom_constraint_raises(
        self, small_problem: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        """CustomConstraint has no y-space translation; must error explicitly."""
        mu, sigma = small_problem
        opt = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0))

        def builder(w, ctx):  # type: ignore[no-untyped-def]
            del ctx
            return [w[0] == 0.5]

        opt.add_constraint(CustomConstraint(builder, description="pin S0"))
        with pytest.raises(OptimizationError, match="not supported"):
            opt.max_sharpe(risk_free_rate=0.02)
