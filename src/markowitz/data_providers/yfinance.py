"""yfinance adapter that conforms to the same surface as :class:`PolygonProvider`.

The legacy :mod:`markowitz.data.providers` module already wraps yfinance for the
single-``close`` column workflow used by the optimizer's Parquet cache. This
adapter delegates to it and reshapes the output into the TitleCase OHLCV frame
that the rest of :mod:`markowitz.data_providers` consumes, so a caller can
swap Polygon out for yfinance without changing downstream code.

Only :meth:`get_eod` is supported. :meth:`get_ticker_meta` and
:meth:`get_grouped_daily` have no equivalent in yfinance for our purposes
(no point-in-time index membership) and raise :class:`PolygonError` to make
that explicit at the call site.
"""

from __future__ import annotations

from datetime import date
from typing import Any, cast

import pandas as pd

from .exceptions import PolygonError

_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class YFinanceProvider:
    """yfinance-backed provider matching :class:`PolygonProvider`'s surface.

    Parameters
    ----------
    inner:
        Optional pre-built provider exposing ``fetch(ticker, start, end)`` and
        returning a frame with a ``close`` column. Defaults to
        :class:`markowitz.data.providers.YFinanceProvider`. Exposed as a hook
        for tests so we never touch the network.
    """

    name = "yfinance"

    def __init__(self, inner: Any = None) -> None:
        if inner is None:
            from markowitz.data.providers import (  # noqa: PLC0415
                YFinanceProvider as _LegacyYF,
            )

            inner = _LegacyYF()
        self._inner = inner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:
        raise PolygonError(
            "get_ticker_meta requires POLYGON_API_KEY; yfinance has no equivalent"
        )

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return daily OHLCV via yfinance, normalized to TitleCase columns.

        yfinance returns ``Open/High/Low/Close/Volume`` natively when called
        through ``yf.download``, but the legacy adapter in this repo collapses
        the frame down to a single ``close`` column. We reconstruct the full
        OHLCV view by re-querying yfinance directly when available, falling
        back to a close-only frame (Open/High/Low filled with NaN, Volume with
        0) when the inner provider doesn't expose it.
        """
        ticker = ticker.strip().upper()
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        frame = self._inner.fetch(ticker, start, end)
        if "Close" in frame.columns and {"Open", "High", "Low", "Volume"}.issubset(
            frame.columns
        ):
            ohlcv = frame[list(_OHLCV_COLUMNS)].copy()
        else:
            close = frame["close"] if "close" in frame.columns else frame["Close"]
            ohlcv = pd.DataFrame(
                {
                    "Open": pd.Series(index=close.index, dtype="float64"),
                    "High": pd.Series(index=close.index, dtype="float64"),
                    "Low": pd.Series(index=close.index, dtype="float64"),
                    "Close": close.astype("float64"),
                    "Volume": pd.Series(0.0, index=close.index, dtype="float64"),
                }
            )
        ohlcv.index = pd.DatetimeIndex(ohlcv.index, name="Date")
        return cast(pd.DataFrame, ohlcv)

    def get_grouped_daily(self, date_: date) -> pd.DataFrame:
        raise PolygonError(
            "get_grouped_daily requires POLYGON_API_KEY; yfinance has no equivalent"
        )

    def close(self) -> None:  # parity with PolygonProvider
        return None
