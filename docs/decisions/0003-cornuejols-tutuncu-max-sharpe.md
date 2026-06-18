# ADR-0003: Max-Sharpe via Cornuejols-Tütüncü reformulation

* Status: Accepted
* Date: 2026-05-23

## Context

Maximizing the Sharpe ratio `(μᵀw − rf) / √(wᵀΣw)` is a nonconvex problem. Direct minimization of negative Sharpe via a generic NLP solver (e.g., SLSQP) is unreliable: the objective has a removable singularity at `μᵀw = rf`, multiple stationary points when shorting is allowed, and only quasi-convex structure.

## Decision

Implement the Cornuejols-Tütüncü change of variables: introduce `y = w / κ` with `κ > 0`, fix `μ̃ᵀy = 1` (where `μ̃ = μ − rf·1`), and **minimize `yᵀΣy`**, a convex QP. Recover `w* = y* / 1ᵀy*`.

## Decision drivers

- Numerical reliability: convex QP guarantees unique global optimum.
- Constraint compatibility: long-only, weight bounds, sector caps, leverage all translate cleanly to `(y, κ)` space.
- Solver compatibility: CLARABEL solves the resulting QP.

## Considered options

- Option A: Cornuejols-Tütüncü reformulation. **Chosen.**
- Option B: Direct nonconvex SLSQP on negative Sharpe. Rejected: PyPortfolioOpt does this and inherits the local-optimum class of bugs.
- Option C: Grid search over the frontier and pick max Sharpe. Rejected: O(n_points) extra solves.

## Consequences

- Pre-check degeneracy: `max(μ) ≤ rf` ⇒ raise `InfeasibleError`.
- Each linear/box constraint requires a `(y, κ)`-space translation; documented in `cornuejols_tutuncu.py`.
- Tests verify the reformulation algebra against the closed-form tangency from ADR-0001.

## Links

- Cornuejols, G. & Tütüncü, R. (2007). *Optimization Methods in Finance*, Ch. 8 §8.2.
