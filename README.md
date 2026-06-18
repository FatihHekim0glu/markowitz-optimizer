# markowitz-optimizer

> **Now live as an interactive web tool at https://fatihhekimoglu-platform.vercel.app/tools/markowitz-optimizer**, part of the fatihhekimoglu.com quantitative-tools platform. The Streamlit app here remains usable for local development; the hosted version uses the same compute library wrapped in a FastAPI backend.

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
On the data side, the library ships a Polygon.io REST client and a survivorship-bias-aware S&P 500
point-in-time universe builder so walk-forward backtests do not silently leak look-ahead through a
modern constituent list.

## Key results

- Reproduces He & Litterman (1999) Table 2 implied equilibrium returns to within `1e-4` on the
  canonical seven-country equity example (test: `tests/regression/test_he_litterman_1999.py`).
- `markowitz.estimators.covariance.LedoitWolfShrinkage` agrees with
  `sklearn.covariance.LedoitWolf` to `1e-10` Frobenius distance across randomized fixtures
  (test: `tests/parity/test_sklearn_lw_parity.py`).
- The walk-forward harness reproduces He & Litterman (1999) and checks for look-ahead leakage
  (`tests/regression/test_no_lookahead.py`). The full DeMiguel, Garlappi, and Uppal (2009) `1/N`
  horse race over the Fama-French datasets is on the v0.2 roadmap, not the current release; see
  `docs/validation/demiguel-2009.md` for the planned protocol.

## Data sources & universe

`markowitz.data_providers` is the remote-data layer. Two providers conform to the same
`get_eod` / `get_ticker_meta` / `get_grouped_daily` surface:

- **`PolygonProvider`**: Polygon.io REST client. Adjusted daily OHLCV, sliding-window token
  bucket (~100 rpm Starter tier), exponential-backoff retries on 429 and 5xx, typed exception
  hierarchy (`PolygonError` / `PolygonAuthError` / `PolygonRateLimitError` / `PolygonDataError`).
- **`YFinanceProvider`**: thin adapter over the existing yfinance pipeline used by
  `markowitz.data`. Only `get_eod` is supported; `get_ticker_meta` and `get_grouped_daily`
  raise `PolygonError` because yfinance has no equivalent.

`make_provider()` picks the right one: it returns `PolygonProvider` when `POLYGON_API_KEY` is
set in the environment (or passed explicitly), and the yfinance adapter otherwise. This lets the
Streamlit demo and the universe builder be written against one surface.

### Survivorship-bias-aware S&P 500 universe

`SP500UniverseBuilder.get_membership_as_of(date_)` intersects a recent snapshot of the index
(`CURRENT_SP500`) with the Polygon grouped-daily snapshot for that date. A ticker counts as a
member iff it appears in the static list *and* has a real trading bar on the as-of date. This is
materially better than the naive "today's-list on yesterday's-date" approach because:

- Symbols that had not yet IPO'd drop out (no grouped-daily row), preventing look-ahead leakage
  from the modern constituent list into early-window backtests.
- Every returned ticker is guaranteed to have same-day OHLCV available, which is the dominant
  correctness concern in walk-forward research.

`get_membership_window(start, end, freq='ME')` builds membership at each rebalance date in the
window (month-end by default), giving the backtest harness a date-keyed dict of universes.

Known limitations (read before using in published research):

- Tickers that were once in the index but have since been delisted or acquired (Lehman, EMC,
  Sprint, ...) are absent. That is the pure "survivor" blind spot and biases backtests upward
  on average. A truly bias-free history requires a paid index-rebalance feed.
- Without `POLYGON_API_KEY` the builder warns once and returns the static today-list. That
  path *is* survivorship-biased and is provided only so the offline demo runs.

```python
import os
from datetime import date

from markowitz.data_providers import SP500UniverseBuilder, make_provider

os.environ["POLYGON_API_KEY"] = "..."  # required for the PIT path
provider = make_provider()
builder = SP500UniverseBuilder(provider)
members = builder.get_membership_as_of(date(2015, 6, 30))
print(len(members), members[:5])
```

The Streamlit sidebar exposes the same toggle under **Universe: Custom tickers | S&P 500
point-in-time** and renders `data:` and `universe:` badges on the landing page so the active
data path is always visible.

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
| Markowitz (1952) | Closed-form A/B/C/D frontier vs numerical optimum | `1e-8` | `tests/unit/core/test_frontier.py` |
| Merton (1972) | Two-fund separation, GMV / tangency identities | `1e-9` | `tests/unit/core/test_merton_scalars.py` |
| Ledoit & Wolf (2004) | Identity-target shrinkage vs `sklearn` | `1e-10` | `tests/parity/test_sklearn_lw_parity.py` |
| He & Litterman (1999) | Implied equilibrium and posterior on Table 2 | `1e-4` | `tests/regression/test_he_litterman_1999.py` |
| PyPortfolioOpt | `max_sharpe`, `min_volatility` weights | `1e-6` | `tests/parity/test_pypfopt_parity.py` |

The DeMiguel, Garlappi, and Uppal (2009) `1/N` horse race is planned for v0.2 and is not part of
this release; the protocol is sketched in `docs/validation/demiguel-2009.md`.

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
