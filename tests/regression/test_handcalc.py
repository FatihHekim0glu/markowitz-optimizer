"""Regression test pinning hand-calculated frontier values.

Reference inputs
----------------
    mu     = (0.10, 0.15)
    sigma  = (0.15, 0.25)
    rho    = 0.3
    Sigma  = [[0.0225,  0.01125],
              [0.01125, 0.0625]]

The reference values below were computed in exact arithmetic with
``fractions.Fraction`` and then converted to ``float`` for the
comparison.  Reproduction (paste into a Python REPL)::

    from fractions import Fraction as F
    s11, s22, s12 = F(225, 10_000), F(625, 10_000), F(1125, 100_000)
    det = s11 * s22 - s12 * s12
    inv11, inv22, inv12 = s22 / det, s11 / det, -s12 / det
    mu1, mu2 = F(1, 10), F(3, 20)

    A = inv11 + 2 * inv12 + inv22
    B = inv11 * mu1 + inv12 * (mu1 + mu2) + inv22 * mu2
    C = inv11 * mu1**2 + 2 * inv12 * mu1 * mu2 + inv22 * mu2**2
    D = A * C - B * B
    # w_gmv = (Sigma^-1 1) / A, var_gmv = 1 / A, mu_gmv = B / A
    # w_tan = (Sigma^-1 (mu - rf 1)) / (B - A rf),
    #   max Sharpe = sqrt(C - 2 B rf + A rf^2)
    # w(mu_p) = lam1 Sigma^-1 mu + lam2 Sigma^-1 1, lam1 = (A mu_p - B)/D,
    #   lam2 = (C - B mu_p)/D, var(mu_p) = (A mu_p^2 - 2 B mu_p + C)/D

The closed-form ``var(mu_p) = (A mu_p^2 - 2 B mu_p + C) / D`` together
with the algebraic identity ``mu_p = mu1 w1 + mu2 (1 - w1)`` for the
two-asset case yields the integer-valued weights tabulated below.
"""

from __future__ import annotations

import numpy as np
import pytest

from markowitz.core.frontier import AnalyticFrontier

ABS_TOL = 1e-10


@pytest.fixture(scope="module")
def frontier() -> AnalyticFrontier:
    mu = np.array([0.10, 0.15])
    Sigma = np.array([[0.0225, 0.01125], [0.01125, 0.0625]])
    return AnalyticFrontier(mu, Sigma)


class TestMertonScalars:
    """Pin (A, B, C, D) to their exact-arithmetic values."""

    def test_abcd(self, frontier: AnalyticFrontier) -> None:
        A, B, C, D = frontier.abcd
        assert pytest.approx(48.84004884004884, abs=ABS_TOL) == A
        assert pytest.approx(5.323565323565323, abs=ABS_TOL) == B
        assert pytest.approx(0.6202686202686203, abs=ABS_TOL) == C
        assert pytest.approx(1.9536019536019535, abs=ABS_TOL) == D


class TestGMV:
    def test_weights(self, frontier: AnalyticFrontier) -> None:
        gmv = frontier.gmv()
        np.testing.assert_allclose(gmv.weights, [0.82, 0.18], atol=ABS_TOL)

    def test_variance(self, frontier: AnalyticFrontier) -> None:
        gmv = frontier.gmv()
        assert gmv.variance() == pytest.approx(0.020475, abs=ABS_TOL)

    def test_expected_return(self, frontier: AnalyticFrontier) -> None:
        gmv = frontier.gmv()
        assert gmv.expected_return == pytest.approx(0.109, abs=ABS_TOL)


class TestTangency:
    def test_weights(self, frontier: AnalyticFrontier) -> None:
        tan = frontier.tangency(rf=0.02)
        # w_tan = (0.6359550561797753, 0.36404494382022473)
        np.testing.assert_allclose(
            tan.weights,
            [0.6359550561797753, 0.3640449438202247],
            atol=ABS_TOL,
        )

    def test_max_sharpe(self, frontier: AnalyticFrontier) -> None:
        tan = frontier.tangency(rf=0.02)
        assert tan.sharpe == pytest.approx(0.6533467891265915, abs=ABS_TOL)


@pytest.mark.parametrize(
    ("mu_p", "w_expected", "var_expected"),
    [
        (0.11, (0.8, 0.2), 0.0205),
        (0.12, (0.6, 0.4), 0.0235),
        (0.13, (0.4, 0.6), 0.0315),
    ],
)
class TestEfficientReturns:
    def test_weights(
        self,
        frontier: AnalyticFrontier,
        mu_p: float,
        w_expected: tuple[float, float],
        var_expected: float,
    ) -> None:
        p = frontier.efficient_return(mu_p)
        np.testing.assert_allclose(p.weights, w_expected, atol=ABS_TOL)

    def test_variance(
        self,
        frontier: AnalyticFrontier,
        mu_p: float,
        w_expected: tuple[float, float],
        var_expected: float,
    ) -> None:
        p = frontier.efficient_return(mu_p)
        assert p.variance() == pytest.approx(var_expected, abs=ABS_TOL)

    def test_expected_return_exact(
        self,
        frontier: AnalyticFrontier,
        mu_p: float,
        w_expected: tuple[float, float],
        var_expected: float,
    ) -> None:
        p = frontier.efficient_return(mu_p)
        assert p.expected_return == pytest.approx(mu_p, abs=ABS_TOL)
