"""Tests for Merton's ABCD scalars."""

from __future__ import annotations

import numpy as np
import pytest

from markowitz.core.exceptions import NumericalError, SingularCovarianceError
from markowitz.core.merton_scalars import MertonABCD, compute_abcd


def _abcd_via_inverse(mu: np.ndarray, Sigma: np.ndarray) -> MertonABCD:
    """Reference implementation that explicitly forms Sigma^-1.

    This is the slow, numerically inferior approach we are avoiding in
    production code; it is fine as a test oracle for small problems.
    """

    Sinv = np.linalg.solve(Sigma, np.eye(Sigma.shape[0]))
    ones = np.ones(mu.shape[0])
    A = float(ones @ Sinv @ ones)
    B = float(ones @ Sinv @ mu)
    C = float(mu @ Sinv @ mu)
    D = A * C - B * B
    return MertonABCD(A=A, B=B, C=C, D=D)


class TestComputeABCD:
    def test_two_asset_matches_oracle(self) -> None:
        mu = np.array([0.10, 0.15])
        Sigma = np.array([[0.0225, 0.01125], [0.01125, 0.0625]])
        abcd = compute_abcd(mu, Sigma)
        oracle = _abcd_via_inverse(mu, Sigma)
        np.testing.assert_allclose(tuple(abcd), tuple(oracle), atol=1e-12)

    def test_diagonal_three_asset(self) -> None:
        mu = np.array([0.05, 0.07, 0.12])
        Sigma = np.diag([0.04, 0.09, 0.25])
        abcd = compute_abcd(mu, Sigma)
        oracle = _abcd_via_inverse(mu, Sigma)
        np.testing.assert_allclose(tuple(abcd), tuple(oracle), atol=1e-12)

    def test_d_is_non_negative(self) -> None:
        rng = np.random.default_rng(seed=7)
        L = rng.standard_normal((4, 4))
        Sigma = L @ L.T + 1e-3 * np.eye(4)
        mu = rng.standard_normal(4)
        abcd = compute_abcd(mu, Sigma)
        assert abcd.D >= -1e-10  # tiny FP slack
        # And matches oracle within tight tolerance.
        np.testing.assert_allclose(tuple(abcd), tuple(_abcd_via_inverse(mu, Sigma)), atol=1e-9)

    def test_collinear_mu_gives_zero_d(self) -> None:
        # If mu is proportional to 1, B^2 = A * C, so D = 0.
        mu = 0.07 * np.ones(3)
        Sigma = np.diag([0.1, 0.2, 0.3])
        abcd = compute_abcd(mu, Sigma)
        assert abs(abcd.D) < 1e-12

    def test_namedtuple_unpacking(self) -> None:
        mu = np.array([0.10, 0.15])
        Sigma = np.array([[0.0225, 0.0], [0.0, 0.0625]])
        A, B, C, D = compute_abcd(mu, Sigma)
        assert A > 0 and C > 0 and D >= 0
        assert pytest.approx(0.10 / 0.0225 + 0.15 / 0.0625, abs=1e-12) == B

    def test_raises_on_shape_mismatch(self) -> None:
        with pytest.raises(NumericalError):
            compute_abcd(np.array([0.1, 0.2, 0.3]), np.eye(2))

    def test_raises_on_non_square(self) -> None:
        with pytest.raises(NumericalError):
            compute_abcd(np.array([0.1, 0.2]), np.zeros((2, 3)))

    def test_raises_on_2d_mu(self) -> None:
        with pytest.raises(NumericalError):
            compute_abcd(np.zeros((2, 1)), np.eye(2))

    def test_raises_on_non_finite_mu(self) -> None:
        with pytest.raises(NumericalError):
            compute_abcd(np.array([0.1, np.nan]), np.eye(2))

    def test_raises_on_singular_sigma(self) -> None:
        with pytest.raises(SingularCovarianceError):
            compute_abcd(np.array([0.1, 0.2]), np.zeros((2, 2)))

    def test_compute_abcd_n_equals_1(self) -> None:
        # For N=1: A = 1/sigma^2, B = mu/sigma^2, C = mu^2/sigma^2, D = 0.
        mu_val = 0.10
        sigma2 = 0.04
        expected = (
            1.0 / sigma2,
            mu_val / sigma2,
            mu_val * mu_val / sigma2,
        )
        abcd = compute_abcd(np.array([mu_val]), np.array([[sigma2]]))
        np.testing.assert_allclose((abcd.A, abcd.B, abcd.C), expected, atol=1e-14)
        # D = A*C - B^2 is algebraically zero for N=1; allow FP slack.
        assert abs(abcd.D) <= 1e-14 * abcd.A * abcd.C
