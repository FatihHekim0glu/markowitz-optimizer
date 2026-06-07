"""Typed Polygon error hierarchy.

All errors raised by :class:`PolygonProvider` inherit from
:class:`PolygonError` so a single ``except PolygonError`` clause catches the
whole family. Subclasses let call sites pattern-match on auth (401/403),
rate-limit (429), and payload-shape failures separately.
"""

from __future__ import annotations


class PolygonError(Exception):
    """Base class for every Polygon-originated failure."""


class PolygonAuthError(PolygonError):
    """Raised on HTTP 401/403 — missing or invalid API key."""


class PolygonRateLimitError(PolygonError):
    """Raised after exhausting retries on HTTP 429."""


class PolygonDataError(PolygonError):
    """Raised when the response payload is missing expected fields or is empty."""
