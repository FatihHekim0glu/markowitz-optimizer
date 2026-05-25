"""Shared pytest fixtures and hypothesis profile registration."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, settings

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile("dev", max_examples=25, deadline=2000)
settings.register_profile("nightly", max_examples=1000, deadline=None)
settings.load_profile("ci")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy Generator (seed=42) for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_returns_small(rng: np.random.Generator) -> pd.DataFrame:
    """1y of daily returns for 3 synthetic assets (sigma=1%)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    cols = ["A", "B", "C"]
    return pd.DataFrame(
        rng.standard_normal((252, 3)) * 0.01,
        index=dates,
        columns=cols,
    )


@pytest.fixture
def synthetic_returns_medium(rng: np.random.Generator) -> pd.DataFrame:
    """5y of daily returns for 10 synthetic assets (sigma=1%)."""
    dates = pd.date_range("2018-01-01", periods=1260, freq="B")
    cols = [f"T{i}" for i in range(10)]
    return pd.DataFrame(
        rng.standard_normal((1260, 10)) * 0.01,
        index=dates,
        columns=cols,
    )


@pytest.fixture
def psd_cov_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    """Factory producing random ``n x n`` positive-semi-definite covariances.

    Builds ``L L^T`` from a lower-triangular ``L`` whose diagonal is shifted
    by ``eps`` to guarantee strict positive-definiteness.
    """

    def _make(n: int, eps: float = 1e-4) -> np.ndarray:
        L = np.tril(rng.standard_normal((n, n)))
        np.fill_diagonal(L, np.abs(np.diag(L)) + eps)
        return L @ L.T

    return _make
