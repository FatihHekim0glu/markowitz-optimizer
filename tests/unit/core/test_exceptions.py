"""Tests for the core exception hierarchy."""

from __future__ import annotations

import pytest

from markowitz.core.exceptions import (
    CoreError,
    InfeasibleFrontierError,
    NumericalError,
    SingularCovarianceError,
)


class TestHierarchy:
    def test_all_subclasses_inherit_from_core_error(self) -> None:
        for cls in (
            SingularCovarianceError,
            InfeasibleFrontierError,
            NumericalError,
        ):
            assert issubclass(cls, CoreError)

    def test_core_error_inherits_from_exception(self) -> None:
        assert issubclass(CoreError, Exception)

    def test_subclasses_are_distinct(self) -> None:
        assert not issubclass(InfeasibleFrontierError, SingularCovarianceError)
        assert not issubclass(NumericalError, InfeasibleFrontierError)


class TestSingularCovarianceError:
    def test_default_attributes(self) -> None:
        err = SingularCovarianceError("oops")
        assert str(err) == "oops"
        assert err.condition_number == float("inf")
        assert err.min_eigenvalue == 0.0

    def test_custom_diagnostics(self) -> None:
        err = SingularCovarianceError(
            "near singular",
            condition_number=1.0e12,
            min_eigenvalue=-1e-9,
        )
        assert err.condition_number == 1.0e12
        assert err.min_eigenvalue == -1e-9

    def test_keyword_only_diagnostics(self) -> None:
        with pytest.raises(TypeError):
            SingularCovarianceError("m", 1.0, 0.0)  # type: ignore[misc]

    def test_can_be_raised_and_caught_as_core_error(self) -> None:
        with pytest.raises(CoreError):
            raise SingularCovarianceError("bad")


class TestOtherErrors:
    def test_infeasible_frontier_message(self) -> None:
        err = InfeasibleFrontierError("no tangency")
        assert str(err) == "no tangency"

    def test_numerical_error_message(self) -> None:
        err = NumericalError("nan in input")
        assert str(err) == "nan in input"

    def test_catch_specific_then_general(self) -> None:
        try:
            raise NumericalError("x")
        except SingularCovarianceError:  # pragma: no cover - should not match
            raise AssertionError("wrong branch")
        except CoreError as exc:
            assert isinstance(exc, NumericalError)
