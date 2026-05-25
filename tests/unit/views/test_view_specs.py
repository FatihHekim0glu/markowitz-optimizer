"""Unit tests for :mod:`markowitz.views.view_specs`."""

from __future__ import annotations

import numpy as np
import pytest

from markowitz.views.exceptions import ViewValidationError
from markowitz.views.view_specs import AbsoluteView, RelativeView, Views

ASSETS = ["AUL", "CAN", "FRA", "GER", "JPN", "UKG", "USA"]


def test_absolute_view_pq_row() -> None:
    v = AbsoluteView(asset="GER", expected_return=0.0525)
    container = Views([v], ASSETS)
    p, q = container.build_pq()
    assert p.shape == (1, 7)
    assert q.shape == (1,)
    expected_row = np.zeros(7)
    expected_row[ASSETS.index("GER")] = 1.0
    np.testing.assert_array_equal(p[0], expected_row)
    assert q[0] == pytest.approx(0.0525)


def test_relative_view_pq_row_sums_to_zero() -> None:
    v = RelativeView(long_leg={"GER": 1.0}, short_leg={"FRA": 1.0}, spread=0.05)
    container = Views([v], ASSETS)
    p, q = container.build_pq()
    assert p[0].sum() == pytest.approx(0.0)
    assert p[0, ASSETS.index("GER")] == pytest.approx(1.0)
    assert p[0, ASSETS.index("FRA")] == pytest.approx(-1.0)
    assert q[0] == pytest.approx(0.05)


def test_relative_view_basket_weights() -> None:
    v = RelativeView(
        long_leg={"GER": 0.5, "FRA": 0.5},
        short_leg={"UKG": 0.3, "USA": 0.7},
        spread=0.02,
    )
    container = Views([v], ASSETS)
    p, _ = container.build_pq()
    assert p[0].sum() == pytest.approx(0.0, abs=1e-12)


def test_unknown_asset_raises() -> None:
    v = AbsoluteView(asset="BRA", expected_return=0.05)
    with pytest.raises(ViewValidationError, match="not in universe"):
        Views([v], ASSETS)


def test_relative_unknown_long_leg_asset_raises() -> None:
    v = RelativeView(long_leg={"ITA": 1.0}, short_leg={"USA": 1.0}, spread=0.01)
    with pytest.raises(ViewValidationError, match="long-leg"):
        Views([v], ASSETS)


def test_relative_unknown_short_leg_asset_raises() -> None:
    v = RelativeView(long_leg={"USA": 1.0}, short_leg={"ITA": 1.0}, spread=0.01)
    with pytest.raises(ViewValidationError, match="short-leg"):
        Views([v], ASSETS)


def test_relative_view_nonzero_row_sum_raises() -> None:
    v = RelativeView(long_leg={"GER": 1.0}, short_leg={"FRA": 0.5}, spread=0.01)
    with pytest.raises(ViewValidationError, match="sum to 0"):
        Views([v], ASSETS)


def test_duplicate_assets_raise() -> None:
    with pytest.raises(ViewValidationError, match="duplicate"):
        Views([AbsoluteView("A", 0.05)], ["A", "B", "A"])


def test_confidence_out_of_range_raises() -> None:
    with pytest.raises(ViewValidationError, match=r"\[0, 1\]"):
        Views([AbsoluteView("USA", 0.05, confidence=1.5)], ASSETS)


def test_partial_confidences_raise() -> None:
    v1 = AbsoluteView("USA", 0.05, confidence=0.5)
    v2 = AbsoluteView("GER", 0.04)  # no confidence
    with pytest.raises(ViewValidationError, match="all views or for none"):
        Views([v1, v2], ASSETS)


def test_he_litterman_omega_diagonal() -> None:
    v = AbsoluteView("GER", 0.0525)
    container = Views([v], ASSETS)
    sigma = np.eye(7) * 0.04
    omega = container.build_omega_he_litterman(tau=0.05, cov=sigma)
    assert omega.shape == (1, 1)
    assert omega[0, 0] == pytest.approx(0.05 * 0.04)


def test_he_litterman_omega_rejects_bad_tau() -> None:
    v = AbsoluteView("USA", 0.05)
    container = Views([v], ASSETS)
    sigma = np.eye(7)
    with pytest.raises(ViewValidationError, match="tau"):
        container.build_omega_he_litterman(tau=0.0, cov=sigma)


def test_he_litterman_omega_rejects_shape_mismatch() -> None:
    v = AbsoluteView("USA", 0.05)
    container = Views([v], ASSETS)
    sigma = np.eye(3)
    with pytest.raises(ViewValidationError, match="Covariance shape"):
        container.build_omega_he_litterman(tau=0.05, cov=sigma)


def test_len_and_confidence_flags() -> None:
    v1 = AbsoluteView("USA", 0.05)
    container_no_conf = Views([v1], ASSETS)
    assert len(container_no_conf) == 1
    assert container_no_conf.has_confidences() is False
    assert container_no_conf.confidences() is None

    v2 = AbsoluteView("USA", 0.05, confidence=0.8)
    container_conf = Views([v2], ASSETS)
    assert container_conf.has_confidences() is True
    np.testing.assert_allclose(container_conf.confidences(), [0.8])


def test_empty_views_container() -> None:
    container = Views([], ASSETS)
    assert len(container) == 0
    p, q = container.build_pq()
    assert p.shape == (0, len(ASSETS))
    assert q.shape == (0,)


def test_empty_relative_legs_raise() -> None:
    bad = RelativeView(long_leg={}, short_leg={"USA": 1.0}, spread=0.0)
    with pytest.raises(ViewValidationError, match="non-empty"):
        Views([bad], ASSETS)
