"""Parity tests against PyPortfolioOpt's EfficientFrontier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cvxpy")
pypfopt = pytest.importorskip("pypfopt")

from pypfopt.efficient_frontier import EfficientFrontier  # noqa: E402

from markowitz.optimizer import MeanVariance  # noqa: E402


@pytest.fixture
def random_problem() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(20250523)
    n = 10
    tickers = [f"T{i:02d}" for i in range(n)]
    A = rng.standard_normal((n, n))
    sigma = pd.DataFrame(A @ A.T / n + 0.05 * np.eye(n), index=tickers, columns=tickers)
    mu = pd.Series(rng.uniform(0.03, 0.20, size=n), index=tickers)
    return mu, sigma


def _pypfopt_weights(ef: EfficientFrontier) -> np.ndarray:
    raw = ef.clean_weights(rounding=None) if False else ef.weights
    arr = np.asarray(raw, dtype=float).reshape(-1)
    return arr


class TestPyPortfolioOptParity:
    def test_min_volatility_parity(self, random_problem: tuple[pd.Series, pd.DataFrame]) -> None:
        mu, sigma = random_problem
        ours = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0)).min_volatility()
        ef = EfficientFrontier(mu, sigma, weight_bounds=(0.0, 1.0), solver="CLARABEL")
        ef.min_volatility()
        theirs = _pypfopt_weights(ef)
        # Observed deviation against PyPortfolioOpt CLARABEL is ~5e-17, so 1e-7
        # is a comfortably tight upper bound.
        np.testing.assert_allclose(ours.to_numpy(), theirs, atol=1e-7)

    def test_max_sharpe_parity(self, random_problem: tuple[pd.Series, pd.DataFrame]) -> None:
        mu, sigma = random_problem
        rf = 0.02
        ours = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0)).max_sharpe(risk_free_rate=rf)
        ef = EfficientFrontier(mu, sigma, weight_bounds=(0.0, 1.0), solver="CLARABEL")
        ef.max_sharpe(risk_free_rate=rf)
        theirs = _pypfopt_weights(ef)
        # Observed max-Sharpe deviation is ~1.16e-9 (different reformulation
        # path on PyPortfolioOpt's side); 1e-7 still leaves two orders of
        # margin.
        np.testing.assert_allclose(ours.to_numpy(), theirs, atol=1e-7, rtol=1e-7)

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_efficient_return_parity(
        self, random_problem: tuple[pd.Series, pd.DataFrame], idx: int
    ) -> None:
        mu, sigma = random_problem
        targets = np.linspace(float(mu.min()) + 0.01, float(mu.max()) - 0.01, 3)
        target = float(targets[idx])

        ours = MeanVariance(mu, sigma, weight_bounds=(0.0, 1.0)).efficient_return(
            target_return=target
        )
        ef = EfficientFrontier(mu, sigma, weight_bounds=(0.0, 1.0), solver="CLARABEL")
        ef.efficient_return(target_return=target)
        theirs = _pypfopt_weights(ef)
        # Same parity tolerance as min_volatility (observed ~5e-17).
        np.testing.assert_allclose(ours.to_numpy(), theirs, atol=1e-7)
