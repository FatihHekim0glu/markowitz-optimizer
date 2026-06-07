"""External market-data providers and the S&P 500 point-in-time universe.

This package layers a Polygon.io REST client and a thin yfinance adapter
behind a uniform interface, exposing a :func:`make_provider` factory and a
:class:`SP500UniverseBuilder` that produces survivorship-bias-aware membership
snapshots. It is independent of the legacy ``markowitz.data`` package — that
module still owns the on-disk Parquet cache and Fama-French utilities; this
one adds remote OHLCV ingestion and universe construction needed for
walk-forward research on a realistic equity universe.
"""

from __future__ import annotations

from .exceptions import (
    PolygonAuthError,
    PolygonDataError,
    PolygonError,
    PolygonRateLimitError,
)
from .factory import make_provider
from .polygon import PolygonProvider
from .sp500_universe import CURRENT_SP500, SP500UniverseBuilder
from .yfinance import YFinanceProvider

__all__ = [
    "CURRENT_SP500",
    "PolygonAuthError",
    "PolygonDataError",
    "PolygonError",
    "PolygonProvider",
    "PolygonRateLimitError",
    "SP500UniverseBuilder",
    "YFinanceProvider",
    "make_provider",
]
