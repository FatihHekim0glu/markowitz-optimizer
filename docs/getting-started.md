# Getting Started

## Install

`markowitz-optimizer` targets Python 3.10+.

=== "uv (recommended)"

    ```bash
    uv add markowitz-optimizer
    ```

=== "pip"

    ```bash
    pip install markowitz-optimizer
    ```

=== "from source"

    ```bash
    git clone https://github.com/FatihHekim0glu/markowitz-optimizer
    cd markowitz-optimizer
    uv sync
    ```

## A 5-minute optimization

The library splits the problem into three steps:

1. **Estimate** `mu` and `Sigma` from data (or supply them directly).
2. **Configure** an optimizer with constraints and a risk preference.
3. **Solve** to obtain weights and a diagnostics object.

```python
import numpy as np
from markowitz import LedoitWolfCovariance, MeanVariance

rng = np.random.default_rng(42)
n_assets = 8
n_obs = 504
returns = rng.standard_normal((n_obs, n_assets)) * 0.012

# 1. Estimate
cov_est = LedoitWolfCovariance().fit(returns)
Sigma = cov_est.covariance_
mu = returns.mean(axis=0)

# 2. Configure
opt = MeanVariance(
    risk_aversion=4.0,
    long_only=True,
    sum_to_one=True,
)

# 3. Solve
result = opt.fit(mu, Sigma)
print(result.weights_)
print("solver status:", result.status_)
print("realized shrinkage intensity:", cov_est.shrinkage_)
```

## What you get back

Every optimizer returns a result object exposing at minimum:

| attribute        | meaning                                                     |
|------------------|-------------------------------------------------------------|
| `weights_`       | optimal portfolio weights, shape `(n_assets,)`              |
| `objective_`     | optimal objective value at the solution                     |
| `status_`        | solver status (`"optimal"`, `"infeasible"`, ...)            |
| `dual_`          | dual variables for active constraints                       |
| `ridge_applied_` | any regularization added to `Sigma` for numerical stability |

## Where to go next

- The **[Theory](theory/mean-variance.md)** pages explain the math behind each
  estimator and optimizer with primary citations.
- The **[Tutorials](tutorials.md)** notebooks build intuition step-by-step.
- The **[API Reference](api-reference/markowitz.md)** lists every public symbol.

## Reproducibility notes

- All examples and notebooks set NumPy seeds explicitly.
- No example fetches network data; synthetic returns are generated in-process.
- Solver tolerances are exposed as constructor arguments; defaults match the
  values used in the validation suite.
