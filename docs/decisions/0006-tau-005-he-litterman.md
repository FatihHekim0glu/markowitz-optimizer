# ADR-0006: τ = 0.05 (He-Litterman convention)

* Status: Accepted
* Date: 2026-05-23

## Context

The Black-Litterman τ scalar reflects uncertainty in the prior. The literature is genuinely inconsistent: He-Litterman (1999) and Idzorek (2005) use small τ (0.025–0.05); Meucci (2010) uses τ = 1 ("tau-less convention") and absorbs the scaling into Ω.

## Decision

Default to **τ = 0.05** (He-Litterman convention). This is the value used in the reference reproduction test (He-Litterman 1999 Table 7).

## Decision drivers

- Reproducibility: the He-Litterman 1999 numbers depend on τ = 0.05.
- Reference compatibility: PyPortfolioOpt uses the same convention.
- Idzorek's confidence-implied Ω procedure assumes the He-Litterman convention.

## Considered options

- Option A: τ = 0.05 He-Litterman default. **Chosen.**
- Option B: τ = 1 Meucci tau-less default. Rejected: would break reproduction test.
- Option C: τ = 1/T derived from data. Rejected: introduces an additional hyperparameter dependency.

## Consequences

- Documentation states the convention loudly.
- The `tau` argument is exposed for user override.
- `TauScaleError` raised if τ ≤ 0 or τ > 1.

## Links

- He, G. & Litterman, R. (1999). "The Intuition Behind Black-Litterman Model Portfolios."
- Walters, J. (2014). "The Black-Litterman Model in Detail." SSRN #1314585.
- Meucci, A. (2010). "The Black-Litterman Approach: Original Model and Extensions."
