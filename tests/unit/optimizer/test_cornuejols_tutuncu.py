"""Unit tests for the Cornuejols-Tutuncu max-Sharpe reformulation."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from markowitz.optimizer.cornuejols_tutuncu import (  # noqa: E402
    back_transform,
    detect_degeneracy,
    reformulate_max_sharpe,
)
from markowitz.optimizer.exceptions import InfeasibleError, NumericalError  # noqa: E402


class TestBackTransform:
    def test_round_trip(self) -> None:
        y = np.array([0.2, 0.5, 0.3, 1.0])
        w = back_transform(y)
        np.testing.assert_allclose(w.sum(), 1.0)
        np.testing.assert_allclose(w, y / y.sum())

    def test_negative_components_preserved_sign(self) -> None:
        y = np.array([1.0, -0.2, 0.5])
        w = back_transform(y)
        np.testing.assert_allclose(w, y / y.sum())

    def test_zero_sum_raises_numerical(self) -> None:
        y = np.array([1e-15, -1e-15])
        with pytest.raises(NumericalError):
            back_transform(y)

    def test_exactly_zero_raises(self) -> None:
        with pytest.raises(NumericalError):
            back_transform(np.zeros(3))

    def test_nan_sum_raises(self) -> None:
        with pytest.raises(NumericalError):
            back_transform(np.array([1.0, np.nan, 1.0]))


class TestDetectDegeneracy:
    def test_long_only_no_positive_excess_raises(self) -> None:
        mu = np.array([0.01, 0.005, -0.002])
        with pytest.raises(InfeasibleError, match="degenerate"):
            detect_degeneracy(mu, risk_free_rate=0.05, weight_bounds=(0.0, 1.0), long_only=True)

    def test_long_only_with_positive_excess_passes(self) -> None:
        mu = np.array([0.10, 0.05, -0.01])
        # Should not raise.
        detect_degeneracy(mu, risk_free_rate=0.02, weight_bounds=(0.0, 1.0), long_only=True)

    def test_long_short_all_zero_excess_raises(self) -> None:
        mu = np.array([0.02, 0.02, 0.02])
        with pytest.raises(InfeasibleError):
            detect_degeneracy(mu, risk_free_rate=0.02, weight_bounds=(-1.0, 1.0), long_only=False)


class TestReformulation:
    def test_problem_is_convex_and_solves(self) -> None:
        mu = np.array([0.10, 0.12, 0.08])
        Sigma = np.array(
            [
                [0.04, 0.005, 0.0],
                [0.005, 0.06, 0.002],
                [0.0, 0.002, 0.05],
            ]
        )
        reform = reformulate_max_sharpe(
            mu,
            Sigma,
            risk_free_rate=0.02,
            weight_bounds=(0.0, 1.0),
            long_only=True,
        )
        assert reform.problem.is_dcp()
        reform.problem.solve()
        assert reform.problem.status == "optimal"
        w = back_transform(np.asarray(reform.y.value))
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-7)
        assert (w >= -1e-8).all()

    def test_bound_translation_respects_box_in_w_space(self) -> None:
        mu = np.array([0.20, 0.10])
        Sigma = np.eye(2) * 0.05
        reform = reformulate_max_sharpe(
            mu,
            Sigma,
            risk_free_rate=0.0,
            weight_bounds=(0.1, 0.7),
            long_only=True,
        )
        reform.problem.solve()
        w = back_transform(np.asarray(reform.y.value))
        assert w[0] <= 0.7 + 1e-6
        assert w[1] >= 0.1 - 1e-6

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            reformulate_max_sharpe(
                np.array([0.1, 0.1]),
                np.eye(3),
                risk_free_rate=0.0,
                weight_bounds=(0.0, 1.0),
            )

    def test_extra_builder_invoked(self) -> None:
        mu = np.array([0.10, 0.15])
        Sigma = np.eye(2) * 0.04
        seen: list[bool] = []

        def builder(y: cp.Variable, kappa: cp.Variable) -> list[cp.Constraint]:
            seen.append(True)
            return [y[0] <= 0.5 * kappa]

        reform = reformulate_max_sharpe(
            mu,
            Sigma,
            risk_free_rate=0.01,
            weight_bounds=(0.0, 1.0),
            long_only=True,
            extra_constraint_builders=[builder],
        )
        reform.problem.solve()
        assert seen == [True]
        w = back_transform(np.asarray(reform.y.value))
        assert w[0] <= 0.5 + 1e-6

    def test_weight_bounds_both_none(self) -> None:
        """``weight_bounds=(None, None)`` must skip both box constraints.

        Exercises the inner ``lower is None`` / ``upper is None`` branches of
        :func:`reformulate_max_sharpe`; the resulting problem must still
        build cleanly and solve under the long-only non-negativity constraint
        alone.
        """
        mu = np.array([0.10, 0.12, 0.08])
        Sigma = np.array(
            [
                [0.04, 0.005, 0.0],
                [0.005, 0.06, 0.002],
                [0.0, 0.002, 0.05],
            ]
        )
        reform = reformulate_max_sharpe(
            mu,
            Sigma,
            risk_free_rate=0.02,
            weight_bounds=(None, None),
            long_only=True,
        )
        assert reform.problem.is_dcp()
        reform.problem.solve()
        assert reform.problem.status == "optimal"
        w = back_transform(np.asarray(reform.y.value))
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-7)
        assert (w >= -1e-8).all()
