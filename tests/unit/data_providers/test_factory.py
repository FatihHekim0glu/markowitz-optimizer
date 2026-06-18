"""Unit tests for :func:`markowitz.data_providers.make_provider`."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from markowitz.data_providers.factory import make_provider
from markowitz.data_providers.polygon import PolygonProvider
from markowitz.data_providers.yfinance import YFinanceProvider

pytestmark = pytest.mark.unit


class _StubYF:
    """yfinance stand-in that returns a tiny single-row close frame."""

    def fetch(self, ticker: str, start: Any, end: Any, **_: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {"close": [123.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="Date"),
        )


def test_returns_polygon_when_explicit_key_passed() -> None:
    provider = make_provider(api_key="sk_test_123")
    assert isinstance(provider, PolygonProvider)


def test_returns_yfinance_when_no_key_anywhere() -> None:
    provider = make_provider(api_key=None, inner_yfinance=_StubYF())
    assert isinstance(provider, YFinanceProvider)
    # The fallback must actually be wired up - exercise the round-trip.
    df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert df.loc[pd.Timestamp("2024-01-02"), "Close"] == 123.0


def test_env_var_picked_up_when_explicit_key_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "sk_env_456")
    provider = make_provider()
    assert isinstance(provider, PolygonProvider)
