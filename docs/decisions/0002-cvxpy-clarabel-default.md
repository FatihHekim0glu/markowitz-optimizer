# ADR-0002: cvxpy + CLARABEL as default solver

* Status: Accepted
* Date: 2026-05-23

## Context

The numerical optimizer must handle long-only constraints, weight bounds, sector caps, leverage caps, and turnover caps. Multiple solver backends could fill this role.

## Decision

Default to **cvxpy + CLARABEL**. CLARABEL is open-source (Apache-2.0), an interior-point solver, robust to ill-conditioned covariance matrices, and is the default conic solver in cvxpy 1.5+.

## Decision drivers

- License must permit commercial use (rules out commercial-only solvers).
- Robustness on ill-conditioned Σ (sample covariance with N ≈ T).
- Active maintenance and Windows-installable wheels.

## Considered options

- Option A: cvxpy + CLARABEL. **Chosen.**
- Option B: cvxpy + OSQP — chosen for parametric sweeps (frontier) where warm-starts matter.
- Option C: scipy SLSQP only — kept as a no-extra-deps fallback.
- Option D: MOSEK — commercial license excludes it.

## Consequences

- `cvxpy` is in the `[robust]` optional extra; users who only want closed-form math don't need it.
- Max-Sharpe goes through Cornuejols-Tütüncü (ADR-0003) since CLARABEL solves convex QPs only.
- OSQP is dispatched explicitly for parametric frontier sweeps (DPP caching).

## Links

- Goulart, P. J. & Chen, Y. (2024). "Clarabel: An interior-point solver for conic programs with quadratic objectives." arXiv:2405.12762.
