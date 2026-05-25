# ADR-0008: Simple (not log) returns

* Status: Accepted
* Date: 2026-05-23

## Context

The Markowitz mean-variance objective `wᵀΣw − λ·μᵀw` and the budget constraint `1ᵀw = 1` assume linear-in-r aggregation: portfolio return = `w·r`. This only holds for simple returns `r_t = P_t/P_{t-1} − 1`. Log returns are convenient for time-series modeling but do not aggregate linearly across assets.

## Decision

Library-wide convention: **simple returns**. `compute_returns(prices, method="simple")` is the default; `method="log"` is exposed for diagnostics only.

## Decision drivers

- Mathematical consistency: matches what the optimizer's algebra assumes.
- Reproducibility: matches the convention of DGU 2009 and most empirical portfolio research.
- Pedagogical clarity: avoids the Jensen-correction discussion in the basic tutorials.

## Considered options

- Option A: Simple returns as default. **Chosen.**
- Option B: Log returns as default. Rejected: would require Jensen correction every time the optimizer sees them.
- Option C: User must specify. Rejected: silent footguns when users default to "obvious" log returns.

## Consequences

- Annualization: mean × 252 (daily) or × 12 (monthly); std × √n.
- `compute_returns` exposes both with the simple default.
- Tests verify that for small returns, `log(1+r_simple) ≈ r_log` to 1e-8.

## Links

- Campbell, J. Y., Lo, A. W. & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*, Ch. 1.
