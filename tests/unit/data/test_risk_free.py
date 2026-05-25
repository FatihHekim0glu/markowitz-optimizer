"""Tests for the FRED risk-free wrapper and (de)annualization helpers."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from markowitz.data.exceptions import EmptyDataError, ProviderUnavailableError
from markowitz.data.risk_free import (
    annualize_rate,
    constant_rate,
    de_annualize,
    risk_free_rate,
)


def test_de_annualize_basic() -> None:
    annual = pd.Series([0.05, 0.10])
    out = de_annualize(annual, "daily")
    expected_first = (1.05) ** (1.0 / 252) - 1.0
    assert out.iloc[0] == pytest.approx(expected_first)


def test_de_annualize_roundtrip() -> None:
    annual = pd.Series([0.01, 0.05, 0.10, 0.20])
    per_period = de_annualize(annual, "monthly")
    back = annualize_rate(per_period, "monthly")
    np.testing.assert_allclose(back.to_numpy(), annual.to_numpy(), rtol=1e-12)


def test_de_annualize_unknown_frequency() -> None:
    with pytest.raises(ValueError, match="unknown frequency"):
        de_annualize(pd.Series([0.05]), "hourly")


def test_constant_rate_construction() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    s = constant_rate(0.04, idx)
    assert (s == 0.04).all()
    assert s.dtype == np.float64


def test_constant_rate_rejects_nonfinite() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    with pytest.raises(ValueError):
        constant_rate(float("nan"), idx)


def test_risk_free_rate_missing_fredapi_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fredapi", None)
    with pytest.raises(ProviderUnavailableError, match="fredapi"):
        risk_free_rate("2024-01-01", "2024-02-01")


def _install_fake_fred(
    monkeypatch: pytest.MonkeyPatch,
    *,
    series: pd.Series | None,
    raise_exc: Exception | None = None,
) -> None:
    """Inject a fake ``fredapi`` module with a controllable ``Fred`` class."""
    module = ModuleType("fredapi")

    class _FakeFred:
        def __init__(self, api_key: Any = None) -> None:
            self.api_key = api_key

        def get_series(self, series_id: str, **_: object) -> pd.Series | None:
            if raise_exc is not None:
                raise raise_exc
            return series

    module.Fred = _FakeFred  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fredapi", module)


def test_risk_free_rate_daily_decimal_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    # FRED returns annualized percent; constant 5.04 means 5.04% per year
    annual_pct = pd.Series([5.04] * 5, index=idx, name="DGS1MO")
    _install_fake_fred(monkeypatch, series=annual_pct)

    out = risk_free_rate("2024-01-02", "2024-01-08", frequency="daily")
    expected = (1.0504) ** (1.0 / 252) - 1.0
    assert out.iloc[0] == pytest.approx(expected, rel=1e-9)
    assert out.index.tz is None


def test_risk_free_rate_monthly_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-02", periods=70, freq="B")
    annual_pct = pd.Series(np.linspace(4.0, 5.0, 70), index=idx, name="DGS1MO")
    _install_fake_fred(monkeypatch, series=annual_pct)

    out = risk_free_rate("2024-01-02", "2024-04-01", frequency="monthly")
    # Should produce one observation per resampled month-end
    assert len(out) <= 4
    assert out.notna().all()


def test_fred_invalid_series_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fred(monkeypatch, series=None, raise_exc=RuntimeError("Bad Request"))
    with pytest.raises(ProviderUnavailableError, match="FRED fetch failed"):
        risk_free_rate("2024-01-01", "2024-02-01", series_id="NOPE")


def test_fred_empty_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fred(monkeypatch, series=pd.Series([], dtype="float64"))
    with pytest.raises(EmptyDataError):
        risk_free_rate("2024-01-01", "2024-02-01")


def test_risk_free_rate_resampled_empty_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # All-NaN series: passes the initial empty check (length > 0) but the
    # monthly resample-then-dropna leaves an empty series.
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    nan_series = pd.Series([np.nan] * 5, index=idx, name="DGS1MO", dtype="float64")
    _install_fake_fred(monkeypatch, series=nan_series)
    with pytest.raises(EmptyDataError, match="resampling"):
        risk_free_rate("2024-01-02", "2024-01-08", frequency="monthly")
