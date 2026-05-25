"""Tests for ``markowitz.core.linalg``."""

from __future__ import annotations

import numpy as np
import pytest

from markowitz.core.exceptions import NumericalError, SingularCovarianceError
from markowitz.core.linalg import (
    cholesky_solve,
    condition_number,
    psd_check,
    regularize,
)


@pytest.fixture
def spd_2x2() -> np.ndarray:
    return np.array([[2.0, 0.3], [0.3, 1.0]])


@pytest.fixture
def spd_3x3() -> np.ndarray:
    return np.array(
        [
            [4.0, 1.0, 0.5],
            [1.0, 3.0, 0.2],
            [0.5, 0.2, 2.0],
        ]
    )


class TestCholeskySolve:
    def test_matches_explicit_inverse_2x2(self, spd_2x2: np.ndarray) -> None:
        # We deliberately compare against np.linalg.solve (not inv) since
        # forming inv is the practice we are avoiding.
        b = np.array([1.0, 2.0])
        x = cholesky_solve(spd_2x2, b)
        expected = np.linalg.solve(spd_2x2, b)
        np.testing.assert_allclose(x, expected, atol=1e-12)

    def test_matches_for_3x3(self, spd_3x3: np.ndarray) -> None:
        b = np.array([1.0, 0.0, -1.0])
        x = cholesky_solve(spd_3x3, b)
        np.testing.assert_allclose(spd_3x3 @ x, b, atol=1e-12)

    def test_multi_rhs(self, spd_3x3: np.ndarray) -> None:
        b = np.eye(3)
        x = cholesky_solve(spd_3x3, b)
        np.testing.assert_allclose(spd_3x3 @ x, b, atol=1e-12)

    def test_raises_on_non_pd(self) -> None:
        bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite
        with pytest.raises(SingularCovarianceError):
            cholesky_solve(bad, np.array([1.0, 0.0]))

    def test_raises_on_non_finite(self) -> None:
        bad = np.array([[1.0, np.nan], [np.nan, 1.0]])
        with pytest.raises(NumericalError):
            cholesky_solve(bad, np.array([1.0, 0.0]))

    def test_raises_on_non_finite_b(self, spd_2x2: np.ndarray) -> None:
        with pytest.raises(NumericalError):
            cholesky_solve(spd_2x2, np.array([1.0, np.inf]))

    def test_raises_on_shape_mismatch(self, spd_2x2: np.ndarray) -> None:
        with pytest.raises(ValueError):
            cholesky_solve(spd_2x2, np.array([1.0, 2.0, 3.0]))

    def test_raises_on_non_square(self) -> None:
        with pytest.raises(ValueError):
            cholesky_solve(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))

    def test_check_finite_false_skips_validation(self, spd_2x2: np.ndarray) -> None:
        # When check_finite is False we should not raise on finite inputs;
        # this just covers the branch.
        x = cholesky_solve(spd_2x2, np.array([1.0, 1.0]), check_finite=False)
        assert x.shape == (2,)


class TestPsdCheck:
    def test_accepts_spd(self, spd_2x2: np.ndarray) -> None:
        psd_check(spd_2x2)

    def test_accepts_3x3_spd(self, spd_3x3: np.ndarray) -> None:
        psd_check(spd_3x3)

    def test_rejects_non_symmetric(self) -> None:
        m = np.array([[1.0, 0.3], [0.5, 1.0]])
        with pytest.raises(SingularCovarianceError):
            psd_check(m)

    def test_allows_non_symmetric_when_disabled(self) -> None:
        m = np.array([[2.0, 0.3], [0.5, 2.0]])
        # Even with require_symmetric=False the eigenvalue test should
        # operate on the symmetrized matrix and accept this case.
        psd_check(m, require_symmetric=False)

    def test_rejects_indefinite(self) -> None:
        with pytest.raises(SingularCovarianceError):
            psd_check(np.array([[1.0, 2.0], [2.0, 1.0]]))

    def test_rejects_zero_matrix(self) -> None:
        with pytest.raises(SingularCovarianceError):
            psd_check(np.zeros((3, 3)))

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(SingularCovarianceError):
            psd_check(np.array([[1.0, np.nan], [np.nan, 1.0]]))

    def test_rejects_non_square(self) -> None:
        with pytest.raises(ValueError):
            psd_check(np.zeros((2, 3)))

    def test_diagnostics_attached(self) -> None:
        try:
            psd_check(np.zeros((2, 2)))
        except SingularCovarianceError as exc:
            assert exc.condition_number == float("inf") or exc.condition_number > 0
            assert exc.min_eigenvalue <= 1e-10


class TestConditionNumber:
    def test_identity_is_one(self) -> None:
        assert condition_number(np.eye(5)) == pytest.approx(1.0)

    def test_diagonal_ratio(self) -> None:
        m = np.diag([1.0, 10.0, 100.0])
        assert condition_number(m) == pytest.approx(100.0)

    def test_non_pd_returns_inf(self) -> None:
        assert condition_number(np.zeros((3, 3))) == float("inf")

    def test_non_finite_returns_inf(self) -> None:
        m = np.array([[1.0, np.nan], [np.nan, 1.0]])
        assert condition_number(m) == float("inf")

    def test_empty_returns_inf(self) -> None:
        assert condition_number(np.empty((0, 0))) == float("inf")

    def test_non_square_returns_inf(self) -> None:
        assert condition_number(np.ones((2, 3))) == float("inf")


class TestRegularize:
    def test_auto_eps_preserves_symmetry(self, spd_2x2: np.ndarray) -> None:
        out = regularize(spd_2x2, eps="auto")
        np.testing.assert_allclose(out, out.T)

    def test_explicit_eps_adds_diagonal(self) -> None:
        m = np.zeros((3, 3))
        out = regularize(m, eps=1e-3)
        np.testing.assert_allclose(out, 1e-3 * np.eye(3))

    def test_auto_eps_makes_zero_matrix_pd(self) -> None:
        out = regularize(np.zeros((4, 4)))
        psd_check(out)

    def test_negative_eps_rejected(self) -> None:
        with pytest.raises(ValueError):
            regularize(np.eye(2), eps=-1.0)

    def test_unsupported_method(self) -> None:
        with pytest.raises(ValueError):
            regularize(np.eye(2), method="bogus")  # type: ignore[arg-type]

    def test_non_square_rejected(self) -> None:
        with pytest.raises(ValueError):
            regularize(np.zeros((2, 3)))

    def test_symmetrizes_input(self) -> None:
        m = np.array([[2.0, 0.0], [1.0, 2.0]])
        out = regularize(m, eps=0.0)
        np.testing.assert_allclose(out, 0.5 * (m + m.T))
