"""Tests for trading-calendar utilities."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from markowitz.data.calendars import reindex_to_sessions, trading_sessions
from markowitz.data.exceptions import CalendarMismatchError


def test_trading_sessions_returns_tznaive_index() -> None:
    idx = trading_sessions("2024-01-02", "2024-01-31")
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.tz is None
    assert idx.is_monotonic_increasing


def test_calendars_holiday_exclusion_july_4(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force fallback path to test our hand-rolled holiday list independent of
    # pandas_market_calendars (which may or may not be installed).
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    idx = trading_sessions("2024-07-01", "2024-07-08")
    july4 = pd.Timestamp("2024-07-04")
    assert july4 not in idx


def test_calendars_no_weekends_in_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    idx = trading_sessions("2024-01-06", "2024-01-08")  # Sat..Mon
    sat = pd.Timestamp("2024-01-06")
    sun = pd.Timestamp("2024-01-07")
    mon = pd.Timestamp("2024-01-08")
    assert sat not in idx
    assert sun not in idx
    assert mon in idx


def test_reindex_to_sessions_empty_passthrough() -> None:
    empty = pd.DataFrame(columns=["close"], dtype="float64")
    out = reindex_to_sessions(empty)
    assert out.empty


def test_reindex_to_sessions_rejects_non_dti() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=[0, 1])
    with pytest.raises(CalendarMismatchError, match="DatetimeIndex"):
        reindex_to_sessions(df)


def test_reindex_to_sessions_rejects_tz_aware() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(CalendarMismatchError, match="tz-naive"):
        reindex_to_sessions(df)


def test_reindex_to_sessions_no_missing_returns_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    idx = trading_sessions("2024-01-02", "2024-01-12")
    df = pd.DataFrame({"close": range(len(idx))}, index=idx, dtype="float64")
    out = reindex_to_sessions(df)
    pd.testing.assert_frame_equal(out, df)


def test_reindex_to_sessions_raises_on_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    full = trading_sessions("2024-01-02", "2024-01-12")
    # Drop one session
    partial = full.drop([full[3]])
    df = pd.DataFrame({"close": range(len(partial))}, index=partial, dtype="float64")
    with pytest.raises(CalendarMismatchError, match="missing"):
        reindex_to_sessions(df, on_missing="raise")


def test_reindex_to_sessions_ffill_fills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    full = trading_sessions("2024-01-02", "2024-01-12")
    partial = full.drop([full[3]])
    df = pd.DataFrame(
        {"close": [10.0] * len(partial)},
        index=partial,
    )
    out = reindex_to_sessions(df, on_missing="ffill", max_gap_sessions=1)
    assert len(out) == len(full)
    assert not out["close"].isna().any()


def test_reindex_to_sessions_gap_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    full = trading_sessions("2024-01-02", "2024-01-31")
    # Drop a 3-day run
    drop_idx = [full[5], full[6], full[7]]
    partial = full.drop(drop_idx)
    df = pd.DataFrame({"close": [1.0] * len(partial)}, index=partial)
    with pytest.raises(CalendarMismatchError, match="Gap"):
        reindex_to_sessions(df, on_missing="ffill", max_gap_sessions=1)


def test_reindex_to_sessions_non_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-04"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(CalendarMismatchError, match="monotonically"):
        reindex_to_sessions(df)


def test_reindex_to_sessions_unknown_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    full = trading_sessions("2024-01-02", "2024-01-12")
    partial = full.drop([full[3]])
    df = pd.DataFrame({"close": [1.0] * len(partial)}, index=partial)
    with pytest.raises(CalendarMismatchError, match="unknown on_missing"):
        reindex_to_sessions(df, on_missing="bogus", max_gap_sessions=10)


def test_reindex_to_sessions_nan_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    full = trading_sessions("2024-01-02", "2024-01-12")
    partial = full.drop([full[3]])
    df = pd.DataFrame({"close": [1.0] * len(partial)}, index=partial)
    out = reindex_to_sessions(df, on_missing="nan", max_gap_sessions=2)
    assert out["close"].isna().sum() == 1


def test_trading_sessions_with_mcal_if_available() -> None:
    # Exercises the non-fallback path when pandas_market_calendars is installed.
    pytest.importorskip("pandas_market_calendars")
    idx = trading_sessions("2024-07-01", "2024-07-08")
    assert pd.Timestamp("2024-07-04") not in idx
    assert pd.Timestamp("2024-07-01") in idx


def test_trading_sessions_mcal_path_with_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject a stub pandas_market_calendars to exercise the tz_localize branch."""
    sessions_tz = pd.DatetimeIndex(
        ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05"], tz="America/New_York"
    )
    schedule = pd.DataFrame(
        {"market_open": sessions_tz, "market_close": sessions_tz}, index=sessions_tz
    )

    def get_calendar(_name: str) -> SimpleNamespace:
        return SimpleNamespace(schedule=lambda start_date, end_date: schedule)

    stub = ModuleType("pandas_market_calendars")
    stub.get_calendar = get_calendar  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", stub)

    idx = trading_sessions("2024-07-01", "2024-07-08")
    assert idx.tz is None
    assert pd.Timestamp("2024-07-01") in idx
    assert pd.Timestamp("2024-07-05") in idx
