"""No-views special-case tests for :class:`markowitz.views.BlackLitterman`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markowitz.views import BlackLitterman, Views

ASSETS = ["A", "B", "C"]


@pytest.fixture
def cov_df() -> pd.DataFrame:
    arr = np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.06, 0.02],
            [0.005, 0.02, 0.08],
        ]
    )
    return pd.DataFrame(arr, index=ASSETS, columns=ASSETS)


@pytest.fixture
def market_weights() -> pd.Series:
    return pd.Series([0.4, 0.35, 0.25], index=ASSETS)


def test_no_views_posterior_equals_prior(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    bl = BlackLitterman(cov_df, market_weights, views=None)
    pi = bl.implied_returns()
    mu_bl = bl.posterior_returns()
    pd.testing.assert_series_equal(pi.rename("mu_bl"), mu_bl, check_exact=False)


def test_no_views_posterior_cov_equals_prior(
    cov_df: pd.DataFrame, market_weights: pd.Series
) -> None:
    bl = BlackLitterman(cov_df, market_weights, views=None)
    sigma_bl = bl.posterior_covariance()
    pd.testing.assert_frame_equal(sigma_bl, cov_df)


def test_no_views_weights_recover_market(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    bl = BlackLitterman(cov_df, market_weights, views=None, delta=2.5)
    w_bl = bl.posterior_weights()
    np.testing.assert_allclose(w_bl.to_numpy(), market_weights.to_numpy(), atol=1e-10)


def test_empty_views_container_equivalent_to_none(
    cov_df: pd.DataFrame, market_weights: pd.Series
) -> None:
    bl_empty = BlackLitterman(cov_df, market_weights, views=Views([], ASSETS))
    bl_none = BlackLitterman(cov_df, market_weights, views=None)
    pd.testing.assert_series_equal(bl_empty.posterior_returns(), bl_none.posterior_returns())


def test_summary_shape(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    bl = BlackLitterman(cov_df, market_weights, views=None)
    df = bl.summary()
    assert list(df.columns) == ["pi", "mu_bl", "delta_mu", "w_mkt", "w_bl", "delta_w"]
    assert list(df.index) == ASSETS
    np.testing.assert_allclose(df["delta_mu"].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(df["delta_w"].to_numpy(), 0.0, atol=1e-10)


def test_constructor_rejects_non_square() -> None:
    bad = pd.DataFrame(np.ones((2, 3)))
    with pytest.raises(Exception, match="square"):
        BlackLitterman(bad, pd.Series([0.5, 0.5, 0.5]))


def test_constructor_rejects_non_symmetric() -> None:
    arr = np.array([[1.0, 0.2], [0.3, 1.0]])
    bad = pd.DataFrame(arr, index=["A", "B"], columns=["A", "B"])
    with pytest.raises(Exception, match="symmetric"):
        BlackLitterman(bad, pd.Series([0.5, 0.5], index=["A", "B"]))


def test_constructor_rejects_bad_tau(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    with pytest.raises(Exception, match="tau"):
        BlackLitterman(cov_df, market_weights, tau=0.0)


def test_constructor_rejects_bad_delta(cov_df: pd.DataFrame, market_weights: pd.Series) -> None:
    with pytest.raises(Exception, match="delta"):
        BlackLitterman(cov_df, market_weights, delta=-1.0)


def test_constructor_rejects_unknown_omega_method(
    cov_df: pd.DataFrame, market_weights: pd.Series
) -> None:
    with pytest.raises(Exception, match="omega_method"):
        BlackLitterman(cov_df, market_weights, omega_method="bogus")


def test_constructor_rejects_unknown_cov_mode(
    cov_df: pd.DataFrame, market_weights: pd.Series
) -> None:
    with pytest.raises(Exception, match="posterior_covariance_mode"):
        BlackLitterman(cov_df, market_weights, posterior_covariance_mode="weird")


def test_constructor_rejects_missing_weights(cov_df: pd.DataFrame) -> None:
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    with pytest.raises(Exception, match="missing"):
        BlackLitterman(cov_df, w)


def test_cov_label_mismatch_raises() -> None:
    """Covariance must have matching row and column labels."""
    arr = np.array([[1.0, 0.2], [0.2, 1.0]])
    bad = pd.DataFrame(arr, index=["A", "B"], columns=["X", "Y"])
    with pytest.raises(Exception, match="row and column labels"):
        BlackLitterman(bad, pd.Series([0.5, 0.5], index=["A", "B"]))


def test_empty_market_weights_raises(cov_df: pd.DataFrame) -> None:
    """Empty market_weights Series must raise."""
    empty = pd.Series([], dtype=float)
    with pytest.raises(Exception, match="non-empty"):
        BlackLitterman(cov_df, empty)
