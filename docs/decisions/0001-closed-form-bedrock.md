# ADR-0001: Closed-form A/B/C/D scalars as test bedrock

* Status: Accepted
* Date: 2026-05-23

## Context

Numerical optimizers (cvxpy + CLARABEL, scipy SLSQP) need a reference implementation to be validated against. Floating-point parity with another solver only shows agreement, not correctness.

## Decision

Implement Merton (1972) closed-form A/B/C/D scalars (`A = 1ᵀΣ⁻¹1`, `B = 1ᵀΣ⁻¹μ`, `C = μᵀΣ⁻¹μ`, `D = AC − B²`) as a separate `core/` module. Every numerical solver test compares its output against this closed-form oracle.

## Decision drivers

- Test oracle without solver-in-the-loop drift.
- Two-asset hand calculation must work for didactic docs and the bedrock regression test.
- Recruiter-visible signal: closed-form mastery distinguishes this from cvxpy wrappers.

## Considered options

- Option A: Closed-form A/B/C/D oracle. **Chosen.**
- Option B: PyPortfolioOpt as oracle. Rejected: oracle drift, circular reasoning.
- Option C: Per-problem oracles. Rejected: inconsistent and confusing.

## Consequences

- Test failures point unambiguously at our implementation.
- The two-asset hand-calculation in the docs is exact, not illustrative.
- Constrained problems still need PyPortfolioOpt parity tests as a secondary layer.
- All linear solves go through Cholesky factorization; no `np.linalg.inv`.

## Links

- Merton, R. C. (1972). "An Analytic Derivation of the Efficient Portfolio Frontier." *JFQA*.
- ADR-0002, ADR-0003, ADR-0008.
