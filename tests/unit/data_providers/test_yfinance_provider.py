"""Unit tests for :class:`markowitz.data_providers.YFinanceProvider`.

The legacy ``markowitz.data.providers.YFinanceProvider`` adapter returns a
single-``close``-column frame; the new wrapper here reshapes it into the
canonical TitleCase OHLCV contract. These tests verify the reshape and the
two ``PolygonError`` raises for features yfinance can't supply.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from markowitz.data_providers.exceptions import PolygonError
from markowitz.data_providers.yfinance import YFinanceProvider

pytestmark = pytest.mark.unit


class _CloseOnlyStub:
    def fetch(self, ticker: str, start: Any, end: Any, **_: Any) -> pd.DataFrame:
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            name="Date",
        )
        return pd.DataFrame({"close": [100.0, 101.0]}, index=idx)


def test_get_eod_reshapes_close_only_to_titlecase_ohlcv() -> None:
    provider = YFinanceProvider(inner=_CloseOnlyStub())
    df = provider.get_eod("aapl", date(2024, 1, 2), date(2024, 1, 3))
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.loc[pd.Timestamp("2024-01-03"), "Close"] == 101.0
    # Open/High/Low are NaN because the inner provider didn't supply them.
    assert df["Open"].isna().all()
    assert (df["Volume"] == 0.0).all()
    assert df.index.name == "Date"


def test_get_ticker_meta_raises_polygon_error() -> None:
    provider = YFinanceProvider(inner=_CloseOnlyStub())
    with pytest.raises(PolygonError, match="get_ticker_meta requires POLYGON_API_KEY"):
        provider.get_ticker_meta("AAPL")


def test_get_grouped_daily_raises_polygon_error() -> None:
    provider = YFinanceProvider(inner=_CloseOnlyStub())
    with pytest.raises(PolygonError, match="get_grouped_daily requires POLYGON_API_KEY"):
        provider.get_grouped_daily(date(2024, 1, 2))
