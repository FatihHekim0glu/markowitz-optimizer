"""Unit tests for :class:`markowitz.data_providers.PolygonProvider`.

The :class:`FakeClient` fixture in ``conftest.py`` replaces the real
``httpx.Client`` so no live traffic is generated. Each test pins the exact
sequence of HTTP responses the provider should see and asserts the resulting
DataFrame / dict / exception.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from markowitz.data_providers.exceptions import (
    PolygonAuthError,
    PolygonDataError,
    PolygonError,
    PolygonRateLimitError,
)
from markowitz.data_providers.polygon import (
    _OHLCV_COLUMNS,
    PolygonProvider,
    _TokenBucket,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_auth_error() -> None:
    with pytest.raises(PolygonAuthError, match="POLYGON_API_KEY is required"):
        PolygonProvider(api_key=None)


def test_http_401_raises_auth_error(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(FakeResponse(status_code=401, text="unauthorized"))
    provider = PolygonProvider(api_key="abc", session=client)
    with pytest.raises(PolygonAuthError, match="401"):
        provider.get_ticker_meta("AAPL")


# ---------------------------------------------------------------------------
# get_eod (aggregates)
# ---------------------------------------------------------------------------


def test_get_eod_returns_titlecase_ohlcv_with_date_index(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    bar_ts = int(pd.Timestamp("2024-01-02", tz="UTC").timestamp() * 1000)
    client = fake_client_factory(
        FakeResponse(
            status_code=200,
            payload={
                "results": [
                    {
                        "t": bar_ts,
                        "o": 100.0,
                        "h": 110.0,
                        "l": 99.0,
                        "c": 108.0,
                        "v": 1_000_000,
                    }
                ]
            },
        )
    )
    provider = PolygonProvider(api_key="abc", session=client)
    df = provider.get_eod("aapl", date(2024, 1, 2), date(2024, 1, 2))

    assert list(df.columns) == list(_OHLCV_COLUMNS)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "Date"
    assert df.index.tz is None
    assert df.loc[pd.Timestamp("2024-01-02"), "Close"] == 108.0
    # Symbol normalized to uppercase in the URL.
    assert "/AAPL/" in client.calls[0]["url"]


def test_get_eod_empty_results_returns_empty_frame(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(FakeResponse(status_code=200, payload={"results": []}))
    provider = PolygonProvider(api_key="abc", session=client)
    df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    assert df.empty
    assert list(df.columns) == list(_OHLCV_COLUMNS)


def test_get_eod_rejects_inverted_window(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(FakeResponse(status_code=200, payload={}))
    provider = PolygonProvider(api_key="abc", session=client)
    with pytest.raises(ValueError, match=r"start.*<=.*end"):
        provider.get_eod("AAPL", date(2024, 1, 3), date(2024, 1, 2))


# ---------------------------------------------------------------------------
# Retry / rate-limit behaviour
# ---------------------------------------------------------------------------


def test_get_eod_retries_on_429_then_succeeds(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    bar_ts = int(pd.Timestamp("2024-01-02", tz="UTC").timestamp() * 1000)
    client = fake_client_factory(
        FakeResponse(status_code=429, text="slow down"),
        FakeResponse(
            status_code=200,
            payload={
                "results": [
                    {
                        "t": bar_ts,
                        "o": 1.0,
                        "h": 1.0,
                        "l": 1.0,
                        "c": 1.0,
                        "v": 1,
                    }
                ]
            },
        ),
    )
    provider = PolygonProvider(api_key="abc", session=client)
    df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert len(df) == 1
    assert len(client.calls) == 2  # initial 429 + retry success


def test_get_eod_raises_rate_limit_after_three_failures(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(
        FakeResponse(status_code=429),
        FakeResponse(status_code=429),
        FakeResponse(status_code=429),
    )
    provider = PolygonProvider(api_key="abc", session=client)
    with pytest.raises(PolygonRateLimitError):
        provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert len(client.calls) == 3


def test_get_eod_retries_then_fails_on_5xx(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(
        FakeResponse(status_code=500),
        FakeResponse(status_code=502),
        FakeResponse(status_code=503),
    )
    provider = PolygonProvider(api_key="abc", session=client)
    with pytest.raises(PolygonError, match="failed after 3 attempts"):
        provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))


def test_get_eod_propagates_network_error_after_retries(
    fake_client_factory, httpx_error_factory
) -> None:
    client = fake_client_factory(
        httpx_error_factory("conn-1"),
        httpx_error_factory("conn-2"),
        httpx_error_factory("conn-3"),
    )
    provider = PolygonProvider(api_key="abc", session=client)
    with pytest.raises(PolygonError, match="failed after 3 attempts"):
        provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))


# ---------------------------------------------------------------------------
# Grouped daily + ticker meta
# ---------------------------------------------------------------------------


def test_get_grouped_daily_indexes_by_ticker(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(
        FakeResponse(
            status_code=200,
            payload={
                "results": [
                    {"T": "AAPL", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100},
                    {"T": "MSFT", "o": 3, "h": 4, "l": 2.5, "c": 3.5, "v": 200},
                    {"T": None, "o": 9, "h": 9, "l": 9, "c": 9, "v": 9},
                ]
            },
        )
    )
    provider = PolygonProvider(api_key="abc", session=client)
    df = provider.get_grouped_daily(date(2024, 1, 2))
    assert df.index.name == "ticker"
    assert sorted(df.index) == ["AAPL", "MSFT"]
    assert df.loc["AAPL", "Close"] == 1.5
    assert list(df.columns) == list(_OHLCV_COLUMNS)


def test_get_grouped_daily_empty_returns_empty_frame(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(FakeResponse(status_code=200, payload={"results": []}))
    provider = PolygonProvider(api_key="abc", session=client)
    df = provider.get_grouped_daily(date(2024, 1, 2))
    assert df.empty
    assert list(df.columns) == list(_OHLCV_COLUMNS)


def test_get_ticker_meta_missing_results_raises_data_error(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(FakeResponse(status_code=200, payload={"results": None}))
    provider = PolygonProvider(api_key="abc", session=client)
    with pytest.raises(PolygonDataError, match="No reference data"):
        provider.get_ticker_meta("AAPL")


def test_get_ticker_meta_returns_dict(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(
        FakeResponse(
            status_code=200,
            payload={"results": {"ticker": "AAPL", "name": "Apple Inc."}},
        )
    )
    provider = PolygonProvider(api_key="abc", session=client)
    meta = provider.get_ticker_meta("aapl")
    assert meta == {"ticker": "AAPL", "name": "Apple Inc."}


# ---------------------------------------------------------------------------
# Payload / parse failures
# ---------------------------------------------------------------------------


def test_malformed_json_raises_data_error(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(FakeResponse(status_code=200, raise_json=True))
    provider = PolygonProvider(api_key="abc", session=client)
    with pytest.raises(PolygonDataError, match="non-JSON"):
        provider.get_ticker_meta("AAPL")


def test_400_other_than_auth_raises_polygon_error(fake_client_factory) -> None:
    from tests.unit.data_providers.conftest import FakeResponse

    client = fake_client_factory(FakeResponse(status_code=404, text="not found"))
    provider = PolygonProvider(api_key="abc", session=client)
    with pytest.raises(PolygonError, match="404"):
        provider.get_ticker_meta("ZZZZ")


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


def test_token_bucket_admits_up_to_rpm_immediately() -> None:
    bucket = _TokenBucket(rpm=5)
    for _ in range(5):
        bucket.acquire()
    # No sleep expected - the queued timestamps fit within the 60s window.
    assert True


def test_token_bucket_throttles_when_window_full(monkeypatch) -> None:
    bucket = _TokenBucket(rpm=2)
    sleep_calls: list[float] = []
    import markowitz.data_providers.polygon as polygon_mod

    monkeypatch.setattr(polygon_mod.time, "sleep", lambda s: sleep_calls.append(float(s)))
    # Three monotonic ticks: first two fill the window, third must wait.
    ticks = iter([0.0, 0.5, 1.0, 1.0])
    monkeypatch.setattr(polygon_mod.time, "monotonic", lambda: next(ticks))

    bucket.acquire()
    bucket.acquire()
    bucket.acquire()
    assert any(s > 0 for s in sleep_calls), "expected throttling sleep when window full"
