# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- CVaR optimization (Rockafellar-Uryasev linear-programming formulation)
- DeMiguel-Garlappi-Uppal (2009) reproduction on Fama-French datasets
- Interactive view-builder in the Streamlit Black-Litterman tab

## [0.1.0] - 2026-05-23

### Added

- Closed-form efficient frontier via the Merton A/B/C/D scalars, including the global
  minimum-variance and tangency portfolios in `markowitz.core.frontier`.
- Numerical mean-variance optimizer (`markowitz.optimizer.mean_variance.MeanVariance`)
  with linear and box constraints, solved via `cvxpy` + CLARABEL.
- Cornuejols-Tutuncu change-of-variables for the maximum-Sharpe portfolio without a risk-free
  asset (`markowitz.optimizer.cornuejols_tutuncu`).
- Covariance shrinkage estimators: Ledoit-Wolf identity-target and Oracle Approximating
  Shrinkage (`markowitz.estimators.covariance`).
- Black-Litterman posterior in Theil mixed-estimation form, plus the Idzorek (2005)
  confidence-to-`Omega` mapping (`markowitz.views.black_litterman`,
  `markowitz.views.idzorek`).
- Walk-forward backtest engine with proportional transaction costs, turnover tracking, and
  rolling performance statistics (`markowitz.backtest`).
- Streamlit application under `app/` for interactive frontier and Black-Litterman exploration.
- `mkdocs-material` documentation site under `docs/` with rendered notebooks and API reference.
- He-Litterman (1999) Table 2 reproduction regression test
  (`tests/regression/test_he_litterman_1999.py`).

[Unreleased]: https://github.com/FatihHekim0glu/markowitz-optimizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/markowitz-optimizer/releases/tag/v0.1.0
