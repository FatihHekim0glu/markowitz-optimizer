"""Unit tests for :class:`markowitz.data_providers.SP500UniverseBuilder`."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from markowitz.data_providers.sp500_universe import (
    CURRENT_SP500,
    SP500UniverseBuilder,
)

pytestmark = pytest.mark.unit


class _StubProvider:
    """Provider stub whose grouped-daily is driven by a per-date dict."""

    def __init__(self, frames: dict[date, pd.DataFrame]) -> None:
        self._frames = frames
        self.call_count = 0

    def get_grouped_daily(self, date_: date) -> pd.DataFrame:
        self.call_count += 1
        return self._frames.get(date_, pd.DataFrame(columns=["Close"]))


def _frame_for(symbols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"Close": [1.0] * len(symbols)}, index=symbols)
    df.index.name = "ticker"
    return df


def test_intersection_with_current_sp500() -> None:
    as_of = date(2024, 1, 2)
    active = ["AAPL", "MSFT", "GOOG", "ZZZ_NOT_IN_INDEX"]
    builder = SP500UniverseBuilder(_StubProvider({as_of: _frame_for(active)}))

    members = builder.get_membership_as_of(as_of)
    assert set(members) <= set(CURRENT_SP500)
    assert "AAPL" in members
    assert "MSFT" in members
    assert "ZZZ_NOT_IN_INDEX" not in members
    assert members == sorted(members)


def test_membership_window_uses_pandas_freq() -> None:
    # 3 month-ends fall inside this window.
    frames = {
        date(2024, 1, 31): _frame_for(["AAPL", "MSFT"]),
        date(2024, 2, 29): _frame_for(["AAPL", "NVDA"]),
        date(2024, 3, 31): _frame_for(["AAPL"]),
    }
    builder = SP500UniverseBuilder(_StubProvider(frames))
    out = builder.get_membership_window(date(2024, 1, 1), date(2024, 3, 31), freq="ME")
    assert set(out.keys()) == set(frames.keys())
    assert out[date(2024, 1, 31)] == sorted({"AAPL", "MSFT"})
    assert out[date(2024, 2, 29)] == sorted({"AAPL", "NVDA"})


def test_membership_window_rejects_inverted_window() -> None:
    builder = SP500UniverseBuilder(_StubProvider({}))
    with pytest.raises(ValueError, match=r"start.*<=.*end"):
        builder.get_membership_window(date(2024, 2, 1), date(2024, 1, 1))


def test_empty_grouped_daily_returns_empty_list() -> None:
    builder = SP500UniverseBuilder(_StubProvider({}))
    assert builder.get_membership_as_of(date(2024, 1, 2)) == []


def test_fallback_emits_survivorship_warning_when_no_provider() -> None:
    builder = SP500UniverseBuilder(provider=None)
    with pytest.warns(UserWarning, match="survivorship-biased"):
        members = builder.get_membership_as_of(date(2024, 1, 2))
    assert members == sorted(CURRENT_SP500)


def test_results_are_cached_per_date() -> None:
    as_of = date(2024, 1, 2)
    provider = _StubProvider({as_of: _frame_for(["AAPL", "MSFT"])})
    builder = SP500UniverseBuilder(provider)

    first = builder.get_membership_as_of(as_of)
    second = builder.get_membership_as_of(as_of)
    assert first == second
    assert provider.call_count == 1, "second call should hit the in-memory cache"


def test_provider_exception_falls_back_to_static_list() -> None:
    class _Boom:
        def get_grouped_daily(self, date_: date) -> pd.DataFrame:
            raise RuntimeError("polygon offline")

    builder = SP500UniverseBuilder(_Boom())
    with pytest.warns(UserWarning, match="survivorship-biased"):
        members = builder.get_membership_as_of(date(2024, 1, 2))
    assert members == sorted(CURRENT_SP500)
