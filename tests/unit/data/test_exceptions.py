"""Tests for the typed exception hierarchy."""

from __future__ import annotations

import pytest

from markowitz.data import exceptions as exc


@pytest.mark.parametrize(
    "subclass",
    [
        exc.EmptyDataError,
        exc.InsufficientDataError,
        exc.DataIntegrityError,
        exc.RateLimitError,
        exc.ProviderUnavailableError,
        exc.CacheCorruptError,
        exc.CalendarMismatchError,
    ],
)
def test_all_errors_subclass_base(subclass: type[Exception]) -> None:
    assert issubclass(subclass, exc.DataLayerError)
    assert issubclass(subclass, Exception)


def test_base_class_distinct_from_builtins() -> None:
    assert exc.DataLayerError is not Exception
    assert not issubclass(exc.DataLayerError, ValueError)


def test_errors_can_be_raised_and_caught_as_base() -> None:
    with pytest.raises(exc.DataLayerError):
        raise exc.EmptyDataError("nope")
    with pytest.raises(exc.DataLayerError):
        raise exc.InsufficientDataError("nope")


def test_messages_round_trip() -> None:
    e = exc.RateLimitError("too many requests")
    assert str(e) == "too many requests"
