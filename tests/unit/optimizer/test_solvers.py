"""Unit tests for the typed ``solve_problem`` wrapper.

These cover the failure-mode mapping in
:func:`markowitz.optimizer.solvers.solve_problem`: solver exceptions get
wrapped as ``SolverError``, status strings outside the recognised buckets
also raise ``SolverError``, and unbounded problems surface as
``UnboundedError``.
"""

from __future__ import annotations

from typing import Any

import pytest

cp = pytest.importorskip("cvxpy")

from markowitz.optimizer.exceptions import (  # noqa: E402
    SolverError,
    UnboundedError,
)
from markowitz.optimizer.solvers import solve_problem  # noqa: E402


class TestSolverErrorWrapping:
    def test_solver_exception_wrapped(self) -> None:
        # Construct a trivial CVXPY problem and demand a non-existent solver.
        # CVXPY raises before any backend is invoked; ``solve_problem`` should
        # translate that into our typed SolverError.
        x = cp.Variable()
        problem = cp.Problem(cp.Minimize(x), [x >= 0])
        with pytest.raises(SolverError) as info:
            solve_problem(problem, solver="NONEXISTENT_SOLVER")
        # Status carrier should be populated for downstream introspection.
        assert info.value.solver_status == "solver_error"
        assert info.value.solver_message is not None


class TestUnboundedDetection:
    def test_unbounded_status_raises_unbounded_error(self) -> None:
        # A free maximisation of a scalar variable with no constraints is
        # unbounded above; CLARABEL reports ``unbounded`` (possibly with the
        # ``_inaccurate`` suffix), both of which map to UnboundedError.
        x = cp.Variable()
        problem = cp.Problem(cp.Maximize(x))
        with pytest.raises(UnboundedError):
            solve_problem(problem, solver="CLARABEL")


class _FakeObjective:
    """Stand-in for ``cp.Problem.objective`` with a ``value`` attribute."""

    value: Any = None


class _FakeProblem:
    """Minimal duck-typed stand-in for ``cp.Problem``.

    Lets us drive ``solve_problem`` past the ``problem.solve`` call without
    actually invoking CVXPY, so we can exercise the
    "status-not-in-any-bucket" branch deterministically.
    """

    def __init__(self, status: str) -> None:
        self.status = status
        self.value: Any = None
        self.objective = _FakeObjective()

    def solve(self, **_: Any) -> None:  # mimic cvxpy's signature loosely
        return None


class TestUnknownStatus:
    def test_unknown_status_raises_solver_error(self) -> None:
        fake = _FakeProblem(status="weird")
        with pytest.raises(SolverError) as info:
            solve_problem(fake, solver="CLARABEL")  # type: ignore[arg-type]
        assert info.value.solver_status == "weird"
