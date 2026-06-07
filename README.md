# markowitz-optimizer

> **Now live as an interactive web tool at https://fatihhekimoglu-platform.vercel.app/tools/markowitz-optimizer** — part of the fatihhekimoglu.com quantitative-tools platform. The Streamlit app here remains usable for local development; the hosted version uses the same compute library wrapped in a FastAPI backend.

> From-scratch mean-variance portfolio optimization toolkit reproducing canonical literature results.

[![CI](https://github.com/FatihHekim0glu/markowitz-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/FatihHekim0glu/markowitz-optimizer/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/FatihHekim0glu/markowitz-optimizer/branch/main/graph/badge.svg)](https://codecov.io/gh/FatihHekim0glu/markowitz-optimizer)
[![PyPI](https://img.shields.io/pypi/v/markowitz-optimizer.svg)](https://pypi.org/project/markowitz-optimizer/)
[![Python versions](https://img.shields.io/pypi/pyversions/markowitz-optimizer.svg)](https://pypi.org/project/markowitz-optimizer/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-informational)](https://FatihHekim0glu.github.io/markowitz-optimizer)

`markowitz-optimizer` is a pedagogically transparent, research-grade implementation of mean-variance
portfolio theory. Every numerical routine is derived from first principles in the documentation and
cross-checked against an independent reference: closed-form Lagrangian scalars, `PyPortfolioOpt`,
and `sklearn.covariance` where applicable. The aim is a library that a graduate student can read
end-to-end while also being numerically defensible enough to back small-scale research papers.

## Key results

- Reproduces He & Litterman (1999) Table 2 implied equilibrium returns to within `1e-4` on the
  canonical seven-country equity example (test: `tests/regression/test_he_litterman_1999.py`).
- `markowitz.estimators.covariance.LedoitWolfShrinkage` agrees with
  `sklearn.covariance.LedoitWolf` to `1e-10` Frobenius distance across 200 randomized fixtures
  (test: `tests/parity/test_ledoit_wolf_parity.py`).
- Naive sample-based mean-variance optimization frequently underperforms the equal-weight `1/N`
  benchmark out of sample on the 10-industry FF dataset, reproducing the qualitative finding of
  DeMiguel, Garlappi, and Uppal (2009) (test: `tests/regression/test_demiguel_2009.py`).

## Quick start

```python
import numpy as np
import pandas as pd
from markowitz.optimizer.mean_variance import MeanVariance

rng = np.random.default_rng(0)
tickers = ["AAPL", "MSFT", "JNJ", "XOM"]
mu = pd.Series([0.12, 0.10, 0.08, 0.06], index=tickers)
sigma = pd.DataFrame(np.cov(rng.standard_normal((4, 252))), index=tickers, columns=tickers)

opt = MeanVariance(mu, sigma)
w_tangency = opt.max_sharpe(risk_free_rate=0.02)
print(w_tangency.round(4))
```

## Live demos

- Hosted documentation: <https://FatihHekim0glu.github.io/markowitz-optimizer>
- Interactive Streamlit app: <https://markowitz-optimizer.streamlit.app>
- Reproducible notebooks: see `notebooks/`

## Features

- Closed-form efficient frontier via the Merton A/B/C/D scalars.
- Numerical optimizer with linear and box constraints (`cvxpy` + CLARABEL).
- Cornuejols-Tutuncu reformulation for the tangency portfolio without a risk-free asset.
- Ledoit-Wolf and Oracle Approximating Shrinkage covariance estimators.
- Black-Litterman posterior with Theil mixed-estimation form and the Idzorek confidence mapping.
- Walk-forward backtesting with turnover, transaction costs, and rolling performance statistics.
- Streamlit application and rendered `mkdocs-material` site for interactive exploration.

## Methodology

<details>
<summary>Closed-form mean-variance frontier</summary>

The efficient frontier in `(sigma, mu)`-space is parameterized by the four Merton scalars
`A = 1' Sigma^{-1} 1`, `B = 1' Sigma^{-1} mu`, `C = mu' Sigma^{-1} mu`, and `D = A*C - B^2`.
Variance as a function of target return is `sigma^2(r) = (A r^2 - 2 B r + C) / D`. These
scalars also yield the global minimum-variance and tangency portfolios in closed form, and serve
as the test oracle for the numerical optimizer.

</details>

<details>
<summary>Covariance shrinkage</summary>

Sample covariance is unstable when the number of assets approaches the sample size. The
Ledoit-Wolf shrinkage target is the scaled identity `mu_id * I` where `mu_id` is the average
sample variance; the shrinkage intensity is estimated in closed form from the sample fourth
moments. Parity with `sklearn.covariance.LedoitWolf` is enforced to `1e-10` per fixture.

</details>

<details>
<summary>Black-Litterman posterior</summary>

The posterior expected return vector is computed via Theil's mixed estimation,
`E[r] = ((tau Sigma)^{-1} + P' Omega^{-1} P)^{-1} ((tau Sigma)^{-1} pi + P' Omega^{-1} Q)`,
with `tau = 0.05` following He & Litterman (1999). View confidence can either be supplied
directly through `Omega` or mapped from per-view confidences using the Idzorek (2005) method.

</details>

<details>
<summary>Walk-forward backtest</summary>

The default protocol uses a rolling `M = 120` monthly window: estimate `mu` and `Sigma`,
solve the optimizer, rebalance, apply linear transaction costs, advance one month. Returns are
simple (not log) so that portfolio aggregation is exact: `r_p = w' r`.

</details>

## Validation

| Reference | What is reproduced | Tolerance | Test file |
|-----------|-------------------|-----------|-----------|
| Markowitz (1952) | Closed-form A/B/C/D frontier vs numerical optimum | `1e-8` | `tests/unit/test_frontier_closed_form.py` |
| Merton (1972) | Two-fund separation, GMV / tangency identities | `1e-9` | `tests/unit/test_merton_scalars.py` |
| Ledoit & Wolf (2004) | Identity-target shrinkage vs `sklearn` | `1e-10` | `tests/parity/test_ledoit_wolf_parity.py` |
| He & Litterman (1999) | Implied equilibrium and posterior on Table 2 | `1e-4` | `tests/regression/test_he_litterman_1999.py` |
| DeMiguel et al. (2009) | `1/N` vs sample MV out-of-sample Sharpe ordering | qualitative | `tests/regression/test_demiguel_2009.py` |
| PyPortfolioOpt | `max_sharpe`, `min_volatility` weights | `1e-6` | `tests/parity/test_pypfopt_parity.py` |

## Limitations

The library targets a single-period, friction-light setting: turnover and proportional transaction
costs are modeled, but margin, borrow fees, short-selling constraints beyond simple bounds, and
taxes are not. Covariance estimation assumes returns are iid; regime-switching, conditional
heteroskedasticity, and factor structure are out of scope for the v0.1 release. Data ingestion via
`yfinance` is best-effort and should not be relied on for production use.

## Citation

If you use `markowitz-optimizer` in academic work, please cite the underlying literature in
addition to the software:

```bibtex
@article{Markowitz1952,
  author  = {Markowitz, Harry},
  title   = {Portfolio Selection},
  journal = {The Journal of Finance},
  volume  = {7},
  number  = {1},
  pages   = {77--91},
  year    = {1952},
  doi     = {10.1111/j.1540-6261.1952.tb01525.x}
}

@techreport{HeLitterman1999,
  author      = {He, Guangliang and Litterman, Robert},
  title       = {The Intuition Behind {Black--Litterman} Model Portfolios},
  institution = {Goldman Sachs Investment Management Research},
  year        = {1999}
}

@article{DeMiguel2009,
  author  = {DeMiguel, Victor and Garlappi, Lorenzo and Uppal, Raman},
  title   = {Optimal Versus Naive Diversification: How Inefficient is the {$1/N$} Portfolio Strategy?},
  journal = {The Review of Financial Studies},
  volume  = {22},
  number  = {5},
  pages   = {1915--1953},
  year    = {2009},
  doi     = {10.1093/rfs/hhm075}
}

@article{LedoitWolf2004,
  author  = {Ledoit, Olivier and Wolf, Michael},
  title   = {A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices},
  journal = {Journal of Multivariate Analysis},
  volume  = {88},
  number  = {2},
  pages   = {365--411},
  year    = {2004},
  doi     = {10.1016/S0047-259X(03)00096-4}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
