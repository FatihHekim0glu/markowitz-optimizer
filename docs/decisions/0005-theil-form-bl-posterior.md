# ADR-0005: Theil mixed-estimation form for Black-Litterman posterior

* Status: Accepted
* Date: 2026-05-23

## Context

The Black-Litterman posterior mean `μ_BL` has two algebraically equivalent forms: the precision-weighted form (inverts N×N matrices) and the Theil mixed-estimation form (inverts only K×K, where K = number of views, typically K ≪ N).

## Decision

Implement the **Theil mixed-estimation form**:
```
mu_BL = pi + tau·Sigma·Pᵀ · (P·tau·Sigma·Pᵀ + Omega)⁻¹ · (Q − P·pi)
```
The only inverse is K×K. Posterior covariance `Sigma_BL = Sigma + M` derived via Woodbury identity to avoid forming `(τΣ)⁻¹`.

## Decision drivers

- Numerical stability: avoids inverting N×N matrices that may be ill-conditioned.
- Performance: K is typically 1 to 5; the K×K solve is trivial.
- Cholesky-only linear algebra throughout.

## Considered options

- Option A: Theil form. **Chosen.**
- Option B: Precision-weighted form. Rejected: forms `(τΣ)⁻¹` which is unstable.
- Option C: Both, user-selectable. Rejected: API bloat with no user-visible benefit.

## Consequences

- The K=0 (no views) case is a fast path returning `μ_BL = π` and `M = 0`.
- All linear solves use `scipy.linalg.cho_factor` / `cho_solve`.
- The He-Litterman 1999 reproduction test validates against published numbers regardless of which form was used.

## Links

- Theil, H. (1971). *Principles of Econometrics*.
- He, G. & Litterman, R. (1999). "The Intuition Behind Black-Litterman Model Portfolios."
