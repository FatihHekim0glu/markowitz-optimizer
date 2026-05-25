# Future work

Captured items, no order, no commitments. The presence of an item here is a substitute for adding it now, not a promise to add it later. Every feature listed below was explicitly considered and excluded from the v1.0 scope.

## Optimization

- Hierarchical Risk Parity (López de Prado 2016)
- Risk-parity / equal-risk-contribution as a first-class objective
- Conditional Value-at-Risk (CVaR) minimization (Rockafellar-Uryasev) as primary objective
- Robust optimization with ellipsoidal uncertainty sets (Tütüncü-Koenig 2004)
- Resampled efficient frontier (Michaud 1998)
- Multi-period dynamic programming optimization

## Estimators

- Nonlinear shrinkage (Ledoit-Wolf 2017, 2020)
- Rotation-invariant estimators (Bouchaud, Bun, Knowles, Potters)
- Factor-based covariance via `factorlab` integration
- Bayesian hierarchical models
- GARCH-family multivariate volatility

## Data

- Real-time / streaming data feeds
- Intraday / tick data
- Options chains and Greeks
- Fundamentals (earnings, balance sheets)
- Alternative data
- Multi-currency FX conversion
- CRSP / Compustat adapters

## Backtesting and execution

- Custom transaction-cost models beyond flat bps
- Tax-aware rebalancing (wash sales, lot accounting)
- Live broker integration
- Order management
- Regime-switching strategies
- Machine-learning return forecasts as estimator inputs

## Visualization and tooling

- Interactive Pyodide notebooks in docs
- PDF report export
- Custom Plotly themes per user
- R bindings
- GPU acceleration

## Process notes

If a future maintainer wants to move any of these into scope, the gate is:
1. Open a GitHub issue tagged `proposal:v2`.
2. Reference the published literature or institutional precedent.
3. Demonstrate that the feature does not duplicate `factorlab`, `riskmetrics`, or `ma-crossover-backtester`.
4. Demonstrate that the feature does not change the project's narrative ("Markowitz, validated, with honest reporting") into something else.
