# ADR-0004: Ledoit-Wolf identity-target shrinkage as default Σ

* Status: Accepted
* Date: 2026-05-23

## Context

Sample covariance is unstable when N approaches T. Eigenvalue spread under Marčenko-Pastur explodes; the smallest eigenvalues invert to dominate `Σ⁻¹μ` and produce extreme weights. Shrinkage is the standard fix.

## Decision

Default covariance estimator is **Ledoit-Wolf identity-target shrinkage** (Ledoit & Wolf 2004): `Σ̂_LW = δ̂·F + (1−δ̂)·S` with `F = (tr(S)/N)·I` and `δ̂` estimated in closed form.

## Decision drivers

- Safe-by-default for users with N ≈ T.
- Identity target is the most defensible "no opinion" prior.
- Closed-form δ̂ requires no hyperparameter tuning.
- sklearn parity test enforces 1e-10 numerical agreement.

## Considered options

- Option A: Sample covariance as default. Rejected: unstable for N ≈ T.
- Option B: LW identity target. **Chosen.**
- Option C: LW constant-correlation target. Better for equities but breaks "no opinion" framing; kept as opt-in.
- Option D: Nonlinear shrinkage (Ledoit-Wolf 2017, 2020). Deferred to v2.

## Consequences

- `SampleCovariance` ships as a deliberate teaching baseline to demonstrate the failure mode.
- `EstimatorConfigError` raised if user requests a target that isn't `identity` or `constant_corr`.
- All Σ estimators output PSD matrices verified by property test.

## Links

- Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *J. Multivariate Anal.* 88(2).
