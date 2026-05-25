"""Property-based tests for the analytic mean-variance frontier.

We sample SPD covariance matrices via a Cholesky-factor strategy:
a strictly positive diagonal plus a bounded strictly lower-triangular
block yields ``L L^T`` which is always positive definite, with a
controllable condition number.  This gives Hypothesis a rich
parameter space without ever producing degenerate inputs.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from markowitz.core.frontier import AnalyticFrontier

# Keep dimensions modest: anything > 6 makes Hypothesis shrinking slow
# without exercising any additional code paths.
_DIMS = st.integers(min_value=2, max_value=6)


def _build_spd(n: int, diag: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Return ``L L^T`` for a Cholesky factor with strictly positive diagonal."""

    L = np.zeros((n, n), dtype=np.float64)
    idx = np.tril_indices(n, k=-1)
    L[idx] = lower
    L[np.diag_indices(n)] = diag
    return L @ L.T


@st.composite
def _mu_sigma(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    n = draw(_DIMS)
    diag = draw(
        hnp.arrays(
            np.float64,
            (n,),
            elements=st.floats(min_value=0.2, max_value=2.0),
        )
    )
    n_lower = n * (n - 1) // 2
    lower = draw(
        hnp.arrays(
            np.float64,
            (n_lower,),
            elements=st.floats(min_value=-0.5, max_value=0.5),
        )
    )
    mu = draw(
        hnp.arrays(
            np.float64,
            (n,),
            elements=st.floats(min_value=-0.3, max_value=0.3),
            unique=True,
        )
    )
    Sigma = _build_spd(n, diag, lower)
    return mu, Sigma


_PROP_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@_PROP_SETTINGS
@given(_mu_sigma())
def test_weights_sum_to_one(data: tuple[np.ndarray, np.ndarray]) -> None:
    mu, Sigma = data
    f = AnalyticFrontier(mu, Sigma)
    assume(not f.is_degenerate)

    gmv = f.gmv()
    assert sum(gmv.weights) == pytest.approx(1.0, abs=1e-8)

    er = f.efficient_return(float(np.mean(mu)))
    assert sum(er.weights) == pytest.approx(1.0, abs=1e-8)

    for p in f.frontier(n_points=10):
        assert sum(p.weights) == pytest.approx(1.0, abs=1e-8)


@_PROP_SETTINGS
@given(_mu_sigma())
def test_gmv_has_minimum_variance(data: tuple[np.ndarray, np.ndarray]) -> None:
    mu, Sigma = data
    f = AnalyticFrontier(mu, Sigma)
    assume(not f.is_degenerate)

    gmv = f.gmv()
    pts = f.frontier(n_points=15)
    for p in pts:
        # Tiny rounding slack at the GMV vertex itself.
        assert gmv.variance() <= p.variance() + 1e-10


@_PROP_SETTINGS
@given(_mu_sigma())
def test_variance_matches_hyperbola(data: tuple[np.ndarray, np.ndarray]) -> None:
    mu, Sigma = data
    f = AnalyticFrontier(mu, Sigma)
    assume(not f.is_degenerate)

    A, B, C, D = f.abcd
    mu_gmv = B / A
    for offset in (-0.05, -0.02, 0.0, 0.03, 0.07):
        mu_p = float(mu_gmv + offset)
        closed_form = (A * mu_p * mu_p - 2.0 * B * mu_p + C) / D
        closed_form = max(closed_form, 0.0)
        computed = f.efficient_return(mu_p).variance()
        assert computed == pytest.approx(closed_form, abs=1e-8, rel=1e-8)


@_PROP_SETTINGS
@given(_mu_sigma())
def test_frontier_monotone_above_gmv(data: tuple[np.ndarray, np.ndarray]) -> None:
    mu, Sigma = data
    f = AnalyticFrontier(mu, Sigma)
    assume(not f.is_degenerate)

    A, B, _, _ = f.abcd
    mu_gmv = B / A
    grid = np.linspace(mu_gmv, mu_gmv + 0.2, 12)
    variances = [f.efficient_return(float(m)).variance() for m in grid]
    # Variance is a strictly convex quadratic in mu_p centered at mu_gmv,
    # so above the vertex it must be non-decreasing.
    for v_prev, v_next in itertools.pairwise(variances):
        assert v_next + 1e-10 >= v_prev

    grid_below = np.linspace(mu_gmv - 0.2, mu_gmv, 12)
    variances_below = [f.efficient_return(float(m)).variance() for m in grid_below]
    # Mirror image: below the vertex, variance is non-increasing.
    for v_prev, v_next in itertools.pairwise(variances_below):
        assert v_next <= v_prev + 1e-10
