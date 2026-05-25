"""Unit tests for the optimizer exception hierarchy."""

from __future__ import annotations

import pytest

from markowitz.optimizer.exceptions import (
    InfeasibleError,
    NumericalError,
    OptimizationError,
    SolverError,
    UnboundedError,
)


class TestOptimizationError:
    def test_message_is_preserved(self) -> None:
        err = OptimizationError("boom")
        assert str(err) == "boom"
        assert err.solver_status is None
        assert err.solver_message is None
        assert err.constraints_violated is None

    def test_carries_diagnostic_payload(self) -> None:
        err = OptimizationError(
            "details",
            solver_status="infeasible",
            solver_message="cones overlapping",
            constraints_violated=["LongOnly", "WeightBounds"],
        )
        assert err.solver_status == "infeasible"
        assert err.solver_message == "cones overlapping"
        assert err.constraints_violated == ["LongOnly", "WeightBounds"]

    @pytest.mark.parametrize(
        "subclass",
        [InfeasibleError, UnboundedError, SolverError, NumericalError],
    )
    def test_subclasses_inherit_base(self, subclass: type[OptimizationError]) -> None:
        err = subclass("x", solver_status="status")
        assert isinstance(err, OptimizationError)
        assert err.solver_status == "status"

    def test_catchable_as_base(self) -> None:
        with pytest.raises(OptimizationError) as info:
            raise InfeasibleError("nope")
        assert isinstance(info.value, InfeasibleError)
