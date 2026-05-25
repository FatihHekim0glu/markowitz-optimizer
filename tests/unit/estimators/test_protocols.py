"""Protocol conformance tests for estimator interfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.estimators import (
    CAPMReturns,
    CovarianceEstimator,
    EWMACovariance,
    EWMAMean,
    ImpliedReturns,
    JorionBayesStein,
    LedoitWolfShrinkage,
    MeanEstimator,
    SampleCovariance,
    SampleMean,
)


@pytest.fixture
def daily_returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=300)
    data = rng.normal(loc=0.0005, scale=0.01, size=(300, 4))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C", "D"])


def test_mean_protocol_runtime_checkable(daily_returns: pd.DataFrame) -> None:
    estimators = [
        SampleMean(periods_per_year=252).fit(daily_returns),
        EWMAMean(lam=0.94, periods_per_year=252).fit(daily_returns),
        JorionBayesStein(periods_per_year=252).fit(daily_returns),
    ]
    for est in estimators:
        assert isinstance(est, MeanEstimator)


def test_capm_implied_satisfy_mean_protocol(daily_returns: pd.DataFrame) -> None:
    market = daily_returns.mean(axis=1).to_numpy()
    capm = CAPMReturns(periods_per_year=252).fit(daily_returns, market=market)
    weights = np.full(daily_returns.shape[1], 1.0 / daily_returns.shape[1])
    implied = ImpliedReturns().fit(daily_returns, weights=weights)
    assert isinstance(capm, MeanEstimator)
    assert isinstance(implied, MeanEstimator)


def test_cov_protocol_runtime_checkable(daily_returns: pd.DataFrame) -> None:
    estimators = [
        SampleCovariance(periods_per_year=252).fit(daily_returns),
        LedoitWolfShrinkage(periods_per_year=252).fit(daily_returns),
        EWMACovariance(lam=0.94, periods_per_year=252).fit(daily_returns),
    ]
    for est in estimators:
        assert isinstance(est, CovarianceEstimator)


def test_unfitted_object_does_not_satisfy_protocol() -> None:
    est = SampleMean()
    assert not isinstance(est, MeanEstimator)
    cov = SampleCovariance()
    assert not isinstance(cov, CovarianceEstimator)


def test_fit_returns_self(daily_returns: pd.DataFrame) -> None:
    sm = SampleMean(periods_per_year=252)
    assert sm.fit(daily_returns) is sm
    sc = SampleCovariance(periods_per_year=252)
    assert sc.fit(daily_returns) is sc
