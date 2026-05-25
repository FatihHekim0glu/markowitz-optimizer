"""Tests for the low-level provider plumbing in markowitz.data.providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from markowitz.data.exceptions import DataIntegrityError, EmptyDataError
from markowitz.data.providers import CachedProvider, _normalize_frame


def test_normalize_frame_empty_raises() -> None:
    with pytest.raises(EmptyDataError, match="no rows"):
        _normalize_frame(pd.DataFrame(), ticker="X")


def test_normalize_frame_coerces_non_dti_index() -> None:
    df = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.Index(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    out = _normalize_frame(df, ticker="AAA")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None


def test_normalize_frame_strips_tz() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    out = _normalize_frame(df, ticker="AAA")
    assert out.index.tz is None


@pytest.mark.parametrize("alias", ["Close", "Adj Close", "adj_close"])
def test_normalize_frame_renames_close_aliases(alias: str) -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame({alias: [1.0, 2.0, 3.0]}, index=idx)
    out = _normalize_frame(df, ticker="AAA")
    assert list(out.columns) == ["close"]


def test_normalize_frame_no_close_raises() -> None:
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    df = pd.DataFrame({"open": [1.0, 2.0]}, index=idx)
    with pytest.raises(DataIntegrityError, match="no 'close'"):
        _normalize_frame(df, ticker="AAA")


def test_normalize_frame_all_nan_close_raises() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame({"close": [np.nan, np.nan, np.nan]}, index=idx)
    with pytest.raises(EmptyDataError, match="all-NaN"):
        _normalize_frame(df, ticker="AAA")


class _PartialThenFullProvider:
    """Inner provider: first call returns a half-coverage frame, then a full one."""

    name = "partial"

    def __init__(self, half: pd.DataFrame, full: pd.DataFrame) -> None:
        self._half = half
        self._full = full
        self.calls = 0

    def fetch(self, ticker: str, start: Any, end: Any, *, frequency: str = "1d") -> pd.DataFrame:
        self.calls += 1
        return (self._half if self.calls == 1 else self._full).copy()


def test_cached_provider_refetches_on_partial_coverage(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    full_idx = pd.date_range("2024-01-02", periods=20, freq="B")
    full = pd.DataFrame({"close": np.linspace(100.0, 120.0, 20)}, index=full_idx)
    half = full.iloc[:8].copy()  # only the first half of the window
    inner = _PartialThenFullProvider(half=half, full=full)

    cached = CachedProvider(inner, tmp_path)
    cached.fetch("AAA", "2024-01-02", "2024-01-30")
    cached.fetch("AAA", "2024-01-02", "2024-01-30")
    assert inner.calls == 2
