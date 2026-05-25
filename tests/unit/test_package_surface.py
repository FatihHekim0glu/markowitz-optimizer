"""Smoke tests for the top-level public API surface."""

from __future__ import annotations


def test_version_present() -> None:
    import markowitz

    assert isinstance(markowitz.__version__, str)
    assert len(markowitz.__version__) > 0


def test_headline_imports() -> None:
    """Headline classes must be importable in one hop."""
    from markowitz import (
        AbsoluteView,
        AnalyticFrontier,
        BlackLitterman,
        LedoitWolfShrinkage,
        LeverageCap,
        LongOnly,
        MeanVariance,
        OneOverN,
        OptimizationError,
        Portfolio,
        RelativeView,
        SampleCovariance,
        SampleMean,
        SectorCap,
        Views,
        WalkForward,
        WeightBounds,
    )

    # Just verify they're not None
    for cls in [
        AnalyticFrontier,
        Portfolio,
        SampleMean,
        SampleCovariance,
        LedoitWolfShrinkage,
        MeanVariance,
        BlackLitterman,
        Views,
        WalkForward,
        OneOverN,
    ]:
        assert cls is not None

    # Touch the constraint / view / error classes so linters see them used.
    assert OptimizationError is not None
    assert LongOnly is not None
    assert WeightBounds is not None
    assert SectorCap is not None
    assert LeverageCap is not None
    assert AbsoluteView is not None
    assert RelativeView is not None


def test_all_consistent() -> None:
    """Every name in __all__ must be importable."""
    import markowitz

    for name in markowitz.__all__:
        assert hasattr(markowitz, name), f"__all__ lists {name!r} but it's not present"
