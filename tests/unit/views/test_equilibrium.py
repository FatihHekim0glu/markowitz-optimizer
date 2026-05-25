"""Unit tests for :mod:`markowitz.views.equilibrium`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.views.equilibrium import implied_returns, infer_delta


def _make_cov(assets: list[str], values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=assets, columns=assets)


def test_implied_returns_diagonal_universe() -> None:
    assets = ["A", "B"]
    cov = _make_cov(assets, np.diag([0.04, 0.09]))
    w = pd.Series([0.6, 0.4], index=assets)
    pi = implied_returns(cov, w, delta=2.5)
    assert pi.index.tolist() == assets
    np.testing.assert_allclose(pi.to_numpy(), [2.5 * 0.04 * 0.6, 2.5 * 0.09 * 0.4])


def test_implied_returns_matches_matrix_form() -> None:
    assets = ["X", "Y", "Z"]
    cov_vals = np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.05, 0.02],
            [0.005, 0.02, 0.07],
        ]
    )
    cov = _make_cov(assets, cov_vals)
    w = pd.Series([0.3, 0.4, 0.3], index=assets)
    pi = implied_returns(cov, w, delta=3.0)
    np.testing.assert_allclose(pi.to_numpy(), 3.0 * cov_vals @ w.to_numpy())


def test_implied_returns_reindexes_weights() -> None:
    assets = ["A", "B"]
    cov = _make_cov(assets, np.diag([0.04, 0.09]))
    w = pd.Series([0.4, 0.6], index=["B", "A"])  # reversed order
    pi = implied_returns(cov, w, delta=2.5)
    np.testing.assert_allclose(pi.to_numpy(), [2.5 * 0.04 * 0.6, 2.5 * 0.09 * 0.4])


def test_implied_returns_rejects_non_square() -> None:
    cov = pd.DataFrame(np.ones((2, 3)))
    w = pd.Series([0.5, 0.5])
    with pytest.raises(ValueError, match="square"):
        implied_returns(cov, w)


def test_implied_returns_rejects_mismatched_labels() -> None:
    cov = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "C"])
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    with pytest.raises(ValueError, match="index and columns"):
        implied_returns(cov, w)


def test_implied_returns_rejects_missing_weights() -> None:
    cov = _make_cov(["A", "B"], np.eye(2))
    w = pd.Series([0.5], index=["A"])
    with pytest.raises(ValueError, match="missing"):
        implied_returns(cov, w)


def test_implied_returns_rejects_non_positive_delta() -> None:
    cov = _make_cov(["A"], np.array([[0.04]]))
    w = pd.Series([1.0], index=["A"])
    with pytest.raises(ValueError, match="delta"):
        implied_returns(cov, w, delta=0.0)


def test_infer_delta_simple_case() -> None:
    r = pd.Series([0.10, 0.06, 0.14, 0.02, 0.08])
    rf = 0.03
    delta = infer_delta(r, rf)
    expected = (r.mean() - rf) / r.var(ddof=1)
    assert delta == pytest.approx(expected)


def test_infer_delta_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="At least two"):
        infer_delta(pd.Series([0.1]), 0.0)


def test_infer_delta_rejects_zero_variance() -> None:
    # Use a series whose variance is *exactly* zero in IEEE-754 (constant
    # integers promoted to float yield bit-identical entries).
    with pytest.raises(ValueError, match="variance"):
        infer_delta(pd.Series([1.0, 1.0, 1.0, 1.0]), 0.0)
