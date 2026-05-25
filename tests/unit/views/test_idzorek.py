"""Unit tests for :mod:`markowitz.views.idzorek`."""

from __future__ import annotations

import numpy as np
import pytest

from markowitz.views.exceptions import OmegaSpecificationError
from markowitz.views.idzorek import idzorek_omega, idzorek_omega_approx


def _toy_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(0)
    n = 4
    A = rng.standard_normal((n, n))
    sigma = A @ A.T + np.eye(n) * 0.01
    tau = 0.05
    tau_sigma = tau * sigma
    p = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
        ]
    )
    w_mkt = np.array([0.25, 0.25, 0.25, 0.25])
    pi = 2.5 * sigma @ w_mkt
    return p, tau_sigma, pi, w_mkt, sigma, 2.5


def test_approx_full_confidence_yields_zero_variance() -> None:
    p, tau_sigma, _, _, _, _ = _toy_inputs()
    omega = idzorek_omega_approx(p, tau_sigma, np.array([1.0, 1.0]))
    np.testing.assert_array_equal(np.diag(omega), [0.0, 0.0])


def test_approx_zero_confidence_yields_large_variance() -> None:
    p, tau_sigma, _, _, _, _ = _toy_inputs()
    omega = idzorek_omega_approx(p, tau_sigma, np.array([0.0, 0.0]))
    assert np.diag(omega)[0] > 1e6
    assert np.diag(omega)[1] > 1e6


def test_approx_half_confidence_matches_formula() -> None:
    p, tau_sigma, _, _, _, _ = _toy_inputs()
    c = np.array([0.5, 0.5])
    omega = idzorek_omega_approx(p, tau_sigma, c)
    expected_diag = np.einsum("ki,ij,kj->k", p, tau_sigma, p)  # since (1-c)/c = 1
    np.testing.assert_allclose(np.diag(omega), expected_diag)


def test_approx_returns_diagonal_matrix() -> None:
    p, tau_sigma, _, _, _, _ = _toy_inputs()
    omega = idzorek_omega_approx(p, tau_sigma, np.array([0.5, 0.8]))
    off_diag = omega - np.diag(np.diag(omega))
    np.testing.assert_array_equal(off_diag, np.zeros_like(off_diag))


def test_approx_rejects_bad_confidences() -> None:
    p, tau_sigma, _, _, _, _ = _toy_inputs()
    with pytest.raises(OmegaSpecificationError, match=r"\[0, 1\]"):
        idzorek_omega_approx(p, tau_sigma, np.array([0.5, 1.5]))


def test_approx_rejects_shape_mismatch() -> None:
    p, tau_sigma, _, _, _, _ = _toy_inputs()
    with pytest.raises(OmegaSpecificationError, match="rows but"):
        idzorek_omega_approx(p, tau_sigma, np.array([0.5]))


def test_approx_rejects_two_dim_confidences() -> None:
    p, tau_sigma, _, _, _, _ = _toy_inputs()
    with pytest.raises(OmegaSpecificationError, match="one-dimensional"):
        idzorek_omega_approx(p, tau_sigma, np.array([[0.5, 0.5]]))


def test_exact_full_confidence_zero_variance() -> None:
    p, tau_sigma, pi, w_mkt, _, delta = _toy_inputs()
    omega = idzorek_omega(p, tau_sigma, np.array([1.0, 1.0]), pi, delta, w_mkt)
    np.testing.assert_array_equal(np.diag(omega), [0.0, 0.0])


def test_exact_zero_confidence_large_variance() -> None:
    p, tau_sigma, pi, w_mkt, _, delta = _toy_inputs()
    omega = idzorek_omega(p, tau_sigma, np.array([0.0, 0.0]), pi, delta, w_mkt)
    assert np.diag(omega)[0] > 1e6


def test_exact_returns_diagonal() -> None:
    p, tau_sigma, pi, w_mkt, _, delta = _toy_inputs()
    omega = idzorek_omega(p, tau_sigma, np.array([0.5, 0.7]), pi, delta, w_mkt)
    off_diag = omega - np.diag(np.diag(omega))
    np.testing.assert_array_equal(off_diag, np.zeros_like(off_diag))


def test_exact_monotonic_in_confidence() -> None:
    """Higher confidence should produce smaller omega entries."""
    p, tau_sigma, pi, w_mkt, _, delta = _toy_inputs()
    o_low = np.diag(idzorek_omega(p, tau_sigma, np.array([0.2, 0.2]), pi, delta, w_mkt))
    o_high = np.diag(idzorek_omega(p, tau_sigma, np.array([0.8, 0.8]), pi, delta, w_mkt))
    assert o_low[0] > o_high[0]
    assert o_low[1] > o_high[1]


def test_exact_rejects_bad_confidences() -> None:
    p, tau_sigma, pi, w_mkt, _, delta = _toy_inputs()
    with pytest.raises(OmegaSpecificationError):
        idzorek_omega(p, tau_sigma, np.array([1.5, 0.5]), pi, delta, w_mkt)


def test_exact_rejects_shape_mismatch() -> None:
    p, tau_sigma, pi, w_mkt, _, delta = _toy_inputs()
    with pytest.raises(OmegaSpecificationError, match="rows but"):
        idzorek_omega(p, tau_sigma, np.array([0.5]), pi, delta, w_mkt)


def test_brentq_fallback_on_bracket_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When bracket endpoints have the same sign, fall back to the approx.

    We force the same-sign condition by monkeypatching the internal
    `_tilt_residual` helper to return a constant value regardless of the
    log-omega argument, so f(lo) * f(hi) > 0 always holds.
    """
    from markowitz.views import idzorek as idz_mod

    monkeypatch.setattr(idz_mod, "_tilt_residual", lambda *_args, **_kw: 1.0)

    p, tau_sigma, pi, w_mkt, _, delta = _toy_inputs()
    c = np.array([0.5, 0.7])
    omega = idzorek_omega(p, tau_sigma, c, pi, delta, w_mkt)

    # Fall-back path returns the closed-form approx along the diagonal.
    expected = np.diag(idzorek_omega_approx(p, tau_sigma, c))
    np.testing.assert_allclose(np.diag(omega), expected)
    # Off-diagonals stay zero (diagonal matrix invariant preserved).
    off_diag = omega - np.diag(np.diag(omega))
    np.testing.assert_array_equal(off_diag, np.zeros_like(off_diag))


def test_brentq_raises_runtime_error_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatch brentq to raise; the try/except must fall back to approx."""
    from markowitz.views import idzorek as idz_mod

    def _raising_brentq(*_args: object, **_kw: object) -> float:
        raise RuntimeError("forced brentq failure for test")

    monkeypatch.setattr(idz_mod, "brentq", _raising_brentq)

    p, tau_sigma, pi, w_mkt, _, delta = _toy_inputs()
    c = np.array([0.5, 0.7])
    omega = idzorek_omega(p, tau_sigma, c, pi, delta, w_mkt)

    expected = np.diag(idzorek_omega_approx(p, tau_sigma, c))
    np.testing.assert_allclose(np.diag(omega), expected)
