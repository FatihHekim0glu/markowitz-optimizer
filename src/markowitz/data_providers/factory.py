"""Factory that picks the right provider based on environment configuration.

:func:`make_provider` returns :class:`PolygonProvider` when a Polygon API key
is available (either passed in or read from ``POLYGON_API_KEY``), and the
:class:`YFinanceProvider` adapter otherwise. The factory is the single
recommended entry point for callers that want to be agnostic about whether a
key is configured. The Streamlit demo and the universe builder both go
through it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .exceptions import PolygonAuthError
from .polygon import PolygonProvider
from .yfinance import YFinanceProvider

logger = logging.getLogger(__name__)


def make_provider(
    api_key: str | None = None,
    *,
    inner_yfinance: Any = None,
) -> PolygonProvider | YFinanceProvider:
    """Return a Polygon provider if a key is configured, otherwise yfinance.

    Parameters
    ----------
    api_key:
        Explicit Polygon API key. When ``None`` the factory falls back to the
        ``POLYGON_API_KEY`` environment variable.
    inner_yfinance:
        Optional pre-built provider passed straight through to
        :class:`YFinanceProvider` when the fallback path is taken, useful
        for tests that need to inject a stub instead of touching the network.
    """
    key = (api_key or os.environ.get("POLYGON_API_KEY", "")).strip()
    if key:
        try:
            return PolygonProvider(api_key=key)
        except PolygonAuthError:
            # Construction failed despite a non-empty key, so fall back rather
            # than propagate and let the demo keep running with yfinance data.
            logger.warning(
                "POLYGON_API_KEY present but PolygonProvider rejected it; falling back to yfinance"
            )
    else:
        logger.info("POLYGON_API_KEY not set; using yfinance provider (no PIT universe)")
    return YFinanceProvider(inner=inner_yfinance)
