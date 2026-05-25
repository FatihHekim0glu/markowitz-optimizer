"""Unit tests for the shared estimator helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.estimators._common import (
    cholesky_solve,
    coerce_returns,
    ewma_weights,
    halflife_to_lambda,
    infer_periods_per_year,
    resolve_decay,
    symmetrize,
)
from markowitz.estimators.exceptions import (
    AmbiguousFrequencyError,
    DimensionMismatchError,
    EstimatorConfigError,
)


def test_coerce_dataframe_round_trip() -> None:
    idx = pd.bdate_range("2021-01-04", periods=10)
    df = pd.DataFrame(np.arange(30, dtype=float).reshape(10, 3), index=idx, columns=list("XYZ"))
    values, names, index = coerce_returns(df)
    assert values.shape == (10, 3)
    assert values.dtype == np.float64
    assert list(names) == ["X", "Y", "Z"]
    assert index is idx


def test_coerce_ndarray_synthetic_names() -> None:
    arr = np.random.default_rng(0).normal(size=(20, 4))
    values, names, index = coerce_returns(arr)
    assert values.shape == (20, 4)
    assert list(names) == ["x0", "x1", "x2", "x3"]
    assert index is None


def test_coerce_rejects_1d() -> None:
    with pytest.raises(DimensionMismatchError):
        coerce_returns(np.zeros(10))


def test_coerce_rejects_single_row_df() -> None:
    df = pd.DataFrame([[1.0, 2.0]], columns=["a", "b"])
    with pytest.raises(DimensionMismatchError):
        coerce_returns(df)


def test_coerce_rejects_single_row_ndarray() -> None:
    with pytest.raises(DimensionMismatchError):
        coerce_returns(np.array([[1.0, 2.0]]))


def test_infer_ppy_user_override() -> None:
    assert infer_periods_per_year(None, user_ppy=52) == 52


def test_infer_ppy_rejects_negative() -> None:
    with pytest.raises(EstimatorConfigError):
        infer_periods_per_year(None, user_ppy=0)


def test_infer_ppy_business_daily() -> None:
    idx = pd.bdate_range("2020-01-02", periods=60)
    assert infer_periods_per_year(idx, user_ppy=None) == 252


def test_infer_ppy_monthly_index() -> None:
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    assert infer_periods_per_year(idx, user_ppy=None) == 12


def test_infer_ppy_ambiguous_raises() -> None:
    with pytest.raises(AmbiguousFrequencyError):
        infer_periods_per_year(None, user_ppy=None)


def test_infer_ppy_median_fallback_weekly() -> None:
    base = pd.bdate_range("2020-01-02", periods=20)
    perturbed = base + pd.to_timedelta(np.arange(20) % 2, unit="D")
    idx = pd.DatetimeIndex([perturbed[0] + pd.Timedelta(days=7 * i) for i in range(20)])
    assert infer_periods_per_year(idx, user_ppy=None) == 52


def test_ewma_weights_sum_to_one() -> None:
    w = ewma_weights(50, 0.95)
    assert w.shape == (50,)
    assert pytest.approx(w.sum()) == 1.0
    assert np.all(w >= 0.0)
    assert w[-1] > w[0]


def test_ewma_weights_rejects_invalid_lambda() -> None:
    with pytest.raises(EstimatorConfigError):
        ewma_weights(10, 0.0)
    with pytest.raises(EstimatorConfigError):
        ewma_weights(10, 1.0)


def test_halflife_to_lambda_roundtrip() -> None:
    lam = halflife_to_lambda(63)
    assert 0.0 < lam < 1.0
    assert pytest.approx(lam, abs=1e-12) == 0.5 ** (1.0 / 63)


def test_halflife_to_lambda_rejects_nonpositive() -> None:
    with pytest.raises(EstimatorConfigError):
        halflife_to_lambda(0)


def test_resolve_decay_xor_requires_one() -> None:
    with pytest.raises(EstimatorConfigError):
        resolve_decay(halflife_years=None, lam=None, periods_per_year=252, require_xor=True)
    with pytest.raises(EstimatorConfigError):
        resolve_decay(halflife_years=0.5, lam=0.94, periods_per_year=252, require_xor=True)


def test_resolve_decay_loose_rejects_both() -> None:
    with pytest.raises(EstimatorConfigError):
        resolve_decay(halflife_years=0.5, lam=0.94, periods_per_year=252, require_xor=False)


def test_resolve_decay_loose_requires_something() -> None:
    with pytest.raises(EstimatorConfigError):
        resolve_decay(halflife_years=None, lam=None, periods_per_year=252, require_xor=False)


def test_resolve_decay_from_halflife_years() -> None:
    lam = resolve_decay(halflife_years=0.25, lam=None, periods_per_year=252, require_xor=True)
    assert pytest.approx(lam, abs=1e-12) == 0.5 ** (1.0 / (0.25 * 252))


def test_symmetrize_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    m = rng.normal(size=(6, 6))
    s = symmetrize(m)
    assert np.allclose(s, s.T)


def test_cholesky_solve_matches_direct() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 5))
    spd = a @ a.T + np.eye(5)
    b = rng.normal(size=5)
    x = cholesky_solve(spd, b)
    assert np.allclose(spd @ x, b, atol=1e-10)


def test_infer_ppy_handles_infer_freq_exception() -> None:
    # pd.infer_freq raises ValueError for indices with fewer than 3 entries.
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")])
    # Two business-day-spaced timestamps -> median fallback should hit daily branch.
    assert infer_periods_per_year(idx, user_ppy=None) == 252


def test_infer_ppy_median_fallback_weekly_irregular() -> None:
    # Force the median-spacing path (not the freqstr/infer_freq paths) at ~7d.
    base = pd.Timestamp("2020-01-02")
    offsets = [0, 6, 14, 20, 27, 34, 41, 48, 56, 63]
    idx = pd.DatetimeIndex([base + pd.Timedelta(days=d) for d in offsets])
    assert infer_periods_per_year(idx, user_ppy=None) == 52


def test_infer_ppy_median_fallback_daily() -> None:
    # Irregular daily-spaced index that defeats freq inference.
    base = pd.Timestamp("2020-01-02")
    offsets = [0, 1, 2, 4, 5, 6, 8, 9, 11, 12]  # ~1-day median, non-uniform
    idx = pd.DatetimeIndex([base + pd.Timedelta(days=d) for d in offsets])
    assert infer_periods_per_year(idx, user_ppy=None) == 252


def test_infer_ppy_median_fallback_monthly() -> None:
    base = pd.Timestamp("2020-01-31")
    offsets = [0, 29, 60, 91, 121, 152, 180, 211, 242, 272]
    idx = pd.DatetimeIndex([base + pd.Timedelta(days=d) for d in offsets])
    assert infer_periods_per_year(idx, user_ppy=None) == 12


def test_infer_ppy_median_fallback_quarterly() -> None:
    base = pd.Timestamp("2020-01-31")
    offsets = [0, 88, 180, 271, 360, 452, 540, 632]
    idx = pd.DatetimeIndex([base + pd.Timedelta(days=d) for d in offsets])
    assert infer_periods_per_year(idx, user_ppy=None) == 4


def test_ewma_weights_underflow_returns_uniform(monkeypatch: pytest.MonkeyPatch) -> None:
    # The total<=0 branch is defensive: with any lam in (0,1) the final age=0
    # term contributes ~ (1-lam) so total is never exactly zero. To exercise
    # the fallback we force np.power to return all zeros (extreme underflow).
    import markowitz.estimators._common as common_mod

    real_power = np.power

    def fake_power(base: float, exponent: np.ndarray) -> np.ndarray:
        out = real_power(base, exponent)
        out[:] = 0.0
        return out

    monkeypatch.setattr(common_mod.np, "power", fake_power)
    w = ewma_weights(10000, 1e-300)
    assert w.shape == (10000,)
    assert np.allclose(w, 1.0 / 10000)
    assert pytest.approx(w.sum()) == 1.0


def test_resolve_decay_default_lam_when_both_none() -> None:
    lam = resolve_decay(
        halflife_years=None,
        lam=None,
        periods_per_year=252,
        require_xor=False,
        default_lam=0.94,
    )
    assert lam == pytest.approx(0.94)
