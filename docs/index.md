# markowitz-optimizer

A research-grade Python library for **mean-variance**, **shrinkage-based**, **Black-Litterman**, and **robust / CVaR** portfolio optimization.

## Why another optimizer?

Most off-the-shelf portfolio libraries hide the parts that matter for research:
the choice of covariance estimator, the constraint geometry, the solver
tolerances, and the out-of-sample protocol. `markowitz-optimizer` exposes those
levers and ships with:

- A numerically stable QP backend for long-only, box, and turnover constraints.
- Ledoit-Wolf, OAS, and graphical-lasso covariance estimators with documented
  shrinkage targets.
- The Black-Litterman posterior with both Idzorek and He-Litterman conventions.
- Robust mean-variance and CVaR (Rockafellar-Uryasev) formulations.
- Reproductions of well-known empirical results (He-Litterman 1999,
  DeMiguel-Garlappi-Uppal 2009) bundled as validation tests.

## Quick taste

```python
import numpy as np
from markowitz import MeanVariance, LedoitWolfCovariance

rng = np.random.default_rng(0)
returns = rng.standard_normal((252, 10)) * 0.01

cov = LedoitWolfCovariance().fit(returns).covariance_
mu  = returns.mean(axis=0)

opt = MeanVariance(risk_aversion=5.0, long_only=True)
weights = opt.fit(mu, cov).weights_
```

## What to read first

- **[Getting Started](getting-started.md)** - install and run your first
  optimization in five minutes.
- **[Theory](theory/mean-variance.md)** - the math, with primary citations.
- **[Tutorials](tutorials.md)** - six Jupyter notebooks from two-asset
  intuition to a full out-of-sample horse race.
- **[Validation](validation/he-litterman-1999.md)** - reproductions of
  benchmark results from the literature.

## Design principles

1. **Estimators and optimizers are separate.** A covariance estimator is a
   `fit / transform`-style object; an optimizer takes an estimated `mu` and
   `Sigma` and returns weights.
2. **Constraints are first-class.** Box, group, turnover, and tracking-error
   constraints compose; the QP layer rejects infeasible problems with a
   diagnostic, not a silent fallback.
3. **No silent regularization.** Any ridge added to `Sigma` to make a problem
   solvable is logged and returned in the result object.
4. **Reproducibility.** Every example, notebook, and validation page is seeded
   and runnable from a fresh clone.
