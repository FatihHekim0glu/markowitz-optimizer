"""Tests for load_prices and the provider stack."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from markowitz.data.exceptions import (
    DataIntegrityError,
    EmptyDataError,
    InsufficientDataError,
    ProviderUnavailableError,
    RateLimitError,
)
from markowitz.data.loaders import align_universe, compute_returns, load_prices
from markowitz.data.providers import (
    CachedProvider,
    FallbackProvider,
    PriceProvider,
    YFinanceProvider,
)

# ---------------------------------------------------------------------------
# Synthetic providers
# ---------------------------------------------------------------------------


class StaticProvider:
    """In-memory provider returning a pre-built frame per ticker."""

    name = "static"

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames
        self.call_count = 0

    def fetch(
        self,
        ticker: str,
        start: Any,
        end: Any,
        *,
        frequency: str = "1d",
    ) -> pd.DataFrame:
        self.call_count += 1
        if ticker not in self._frames:
            raise EmptyDataError(f"no synthetic data for {ticker}")
        return self._frames[ticker].copy()


class AlwaysRateLimited:
    name = "ratelimited"

    def fetch(self, *args: object, **kwargs: object) -> pd.DataFrame:
        raise RateLimitError("nope")


class AlwaysUnavailable:
    name = "unavailable"

    def fetch(self, *args: object, **kwargs: object) -> pd.DataFrame:
        raise ProviderUnavailableError("offline")


def _make_price_frame(n: int = 300, start: str = "2023-01-02") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({"close": np.linspace(100.0, 150.0, n)}, index=idx)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_provider_protocol_runtime_checkable() -> None:
    # Static provider satisfies the structural protocol
    prov = StaticProvider({})
    assert isinstance(prov, PriceProvider)


def test_yfinance_provider_missing_dep_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ProviderUnavailableError, match="yfinance"):
        YFinanceProvider()


# ---------------------------------------------------------------------------
# load_prices
# ---------------------------------------------------------------------------


def test_load_prices_single_ticker() -> None:
    prov = StaticProvider({"AAA": _make_price_frame()})
    out = load_prices("AAA", "2023-01-02", "2024-01-01", provider=prov, min_history_sessions=100)
    assert list(out.columns) == ["AAA"]
    assert out.shape[0] == 300


def test_load_prices_multiple_tickers() -> None:
    prov = StaticProvider({"AAA": _make_price_frame(), "BBB": _make_price_frame()})
    out = load_prices(
        ["AAA", "BBB"],
        "2023-01-02",
        "2024-01-01",
        provider=prov,
        min_history_sessions=100,
    )
    assert list(out.columns) == ["AAA", "BBB"]


def test_load_prices_insufficient_history_raises() -> None:
    prov = StaticProvider({"AAA": _make_price_frame(n=50)})
    with pytest.raises(InsufficientDataError, match="observations"):
        load_prices("AAA", "2023-01-02", "2024-01-01", provider=prov)


def test_load_prices_rejects_duplicate_tickers() -> None:
    prov = StaticProvider({"AAA": _make_price_frame()})
    with pytest.raises(ValueError, match="duplicate"):
        load_prices(
            ["AAA", "AAA"],
            "2023-01-02",
            "2024-01-01",
            provider=prov,
            min_history_sessions=10,
        )


def test_load_prices_rejects_empty_iterable() -> None:
    prov = StaticProvider({})
    with pytest.raises(ValueError):
        load_prices([], "2023-01-02", "2024-01-01", provider=prov)


def test_load_prices_bad_provider_response_raises() -> None:
    bad = _make_price_frame().rename(columns={"close": "wrong"})
    prov = StaticProvider({"AAA": bad})
    # _normalize_frame is bypassed (StaticProvider returns raw); load_prices
    # should detect the missing 'close' column.
    with pytest.raises(DataIntegrityError):
        load_prices("AAA", "2023-01-02", "2024-01-01", provider=prov)


# ---------------------------------------------------------------------------
# CachedProvider
# ---------------------------------------------------------------------------


def test_cached_provider_warm_cache_avoids_inner_call(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    inner = StaticProvider({"AAA": _make_price_frame()})
    cached = CachedProvider(inner, tmp_path)
    cached.fetch("AAA", "2023-01-02", "2024-01-01")
    assert inner.call_count == 1
    # Second fetch in the same range should be served from cache
    cached.fetch("AAA", "2023-01-02", "2024-01-01")
    assert inner.call_count == 1


def test_cached_provider_cold_cache_calls_inner(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    inner = StaticProvider({"AAA": _make_price_frame()})
    cached = CachedProvider(inner, tmp_path)
    cached.fetch("AAA", "2023-01-02", "2024-01-01")
    assert inner.call_count == 1


# ---------------------------------------------------------------------------
# FallbackProvider
# ---------------------------------------------------------------------------


def test_fallback_provider_advances_on_rate_limit() -> None:
    primary = AlwaysRateLimited()
    secondary = StaticProvider({"AAA": _make_price_frame()})
    fp = FallbackProvider([primary, secondary])
    out = fp.fetch("AAA", "2023-01-02", "2024-01-01")
    assert not out.empty
    assert secondary.call_count == 1


def test_fallback_provider_advances_on_unavailable() -> None:
    primary = AlwaysUnavailable()
    secondary = StaticProvider({"AAA": _make_price_frame()})
    fp = FallbackProvider([primary, secondary])
    out = fp.fetch("AAA", "2023-01-02", "2024-01-01")
    assert not out.empty


def test_fallback_provider_all_fail_raises() -> None:
    fp = FallbackProvider([AlwaysRateLimited(), AlwaysUnavailable()])
    with pytest.raises(ProviderUnavailableError, match="all providers failed"):
        fp.fetch("AAA", "2023-01-02", "2024-01-01")


def test_fallback_provider_requires_at_least_one() -> None:
    with pytest.raises(ValueError):
        FallbackProvider([])


def test_fallback_provider_does_not_swallow_other_errors() -> None:
    class BoomProvider:
        name = "boom"

        def fetch(self, *args: object, **kwargs: object) -> pd.DataFrame:
            raise EmptyDataError("nothing")

    fp = FallbackProvider([BoomProvider(), StaticProvider({"AAA": _make_price_frame()})])
    # EmptyDataError is *not* in the fallback list, so it propagates.
    with pytest.raises(EmptyDataError):
        fp.fetch("AAA", "2023-01-02", "2024-01-01")


# ---------------------------------------------------------------------------
# YFinanceProvider (with mocked yfinance module)
# ---------------------------------------------------------------------------


def _install_fake_yfinance(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame: pd.DataFrame | None = None,
    raise_exc: Exception | None = None,
) -> None:
    module = ModuleType("yfinance")

    def download(
        tickers: str,
        start: str,
        end: str,
        interval: str,
        auto_adjust: bool,
        progress: bool,
        threads: bool,
    ) -> pd.DataFrame | None:
        if raise_exc is not None:
            raise raise_exc
        return frame

    module.download = download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", module)


def test_yfinance_provider_fetch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    raw = pd.DataFrame({"Close": np.linspace(100.0, 110.0, 10)}, index=idx)
    _install_fake_yfinance(monkeypatch, frame=raw)

    prov = YFinanceProvider()
    out = prov.fetch("AAPL", "2024-01-02", "2024-01-16")
    assert list(out.columns) == ["close"]
    assert out["close"].dtype == np.float64
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing


def test_yfinance_provider_multiindex_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    multi = pd.MultiIndex.from_product([["Close", "Open"], ["AAPL"]])
    raw = pd.DataFrame(
        np.tile(np.arange(5.0).reshape(-1, 1), (1, 2)),
        index=idx,
        columns=multi,
    )
    _install_fake_yfinance(monkeypatch, frame=raw)
    prov = YFinanceProvider()
    out = prov.fetch("AAPL", "2024-01-02", "2024-01-09")
    assert list(out.columns) == ["close"]


def test_yfinance_provider_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_yfinance(monkeypatch, frame=pd.DataFrame())
    prov = YFinanceProvider()
    with pytest.raises(EmptyDataError):
        prov.fetch("XYZ", "2024-01-02", "2024-01-09")


def test_yfinance_provider_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_yfinance(monkeypatch, raise_exc=RuntimeError("HTTP 429 rate limit exceeded"))
    prov = YFinanceProvider()
    with pytest.raises(RateLimitError):
        prov.fetch("AAPL", "2024-01-02", "2024-01-09")


def test_yfinance_provider_generic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_yfinance(monkeypatch, raise_exc=RuntimeError("connection reset"))
    prov = YFinanceProvider()
    with pytest.raises(ProviderUnavailableError):
        prov.fetch("AAPL", "2024-01-02", "2024-01-09")


def test_yfinance_provider_missing_close(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    bad = pd.DataFrame({"Open": [1.0, 2.0, 3.0]}, index=idx)
    _install_fake_yfinance(monkeypatch, frame=bad)
    prov = YFinanceProvider()
    with pytest.raises(DataIntegrityError):
        prov.fetch("AAPL", "2024-01-02", "2024-01-09")


def test_load_prices_default_provider_uses_yfinance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-02", periods=300, freq="B")
    raw = pd.DataFrame({"Close": np.linspace(100.0, 150.0, 300)}, index=idx)
    _install_fake_yfinance(monkeypatch, frame=raw)
    out = load_prices("AAPL", "2024-01-02", "2025-04-01", min_history_sessions=100)
    assert list(out.columns) == ["AAPL"]


# ---------------------------------------------------------------------------
# Edge cases: duplicate index, empty returns, common-window too short
# ---------------------------------------------------------------------------


def test_load_prices_dedups_duplicate_index() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"])
    # Provider returns a frame whose index has a duplicate timestamp.
    frame = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=idx)
    prov = StaticProvider({"AAA": frame})
    out = load_prices("AAA", "2024-01-02", "2024-01-05", provider=prov, min_history_sessions=3)
    assert out.index.is_unique
    assert len(out) == 3
    # last duplicate kept
    assert out.loc[pd.Timestamp("2024-01-02"), "AAA"] == 101.0


def test_compute_returns_empty_after_dropna_raises() -> None:
    idx = pd.DatetimeIndex(["2024-01-02"])
    prices = pd.DataFrame({"AAA": [100.0]}, index=idx)
    with pytest.raises(InsufficientDataError, match="empty"):
        compute_returns(prices, method="simple", dropna=True)


def test_align_universe_common_window_too_short_raises() -> None:
    idx = pd.date_range("2024-01-02", periods=300, freq="B")
    rets = pd.DataFrame(
        {"AAA": np.linspace(0.01, 0.02, 300), "BBB": np.linspace(0.01, 0.02, 300)},
        index=idx,
    )
    # Both tickers individually have >= 60 obs (AAA: first 200, BBB: last 200),
    # but the inner-join window (rows where BOTH are non-NaN) is only ~100.
    rets.loc[rets.index[200:], "AAA"] = np.nan
    rets.loc[rets.index[:100], "BBB"] = np.nan
    with pytest.raises(InsufficientDataError, match="common window"):
        align_universe(rets, min_obs_per_ticker=150, common_window=True)
