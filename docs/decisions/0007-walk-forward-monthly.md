# ADR-0007: Walk-forward M = 120 monthly rolling window

* Status: Accepted
* Date: 2026-05-23

## Context

The walk-forward backtest needs a fixed protocol: rolling vs expanding window, lookback length, rebalance frequency. The protocol must match published benchmarks (DeMiguel-Garlappi-Uppal 2009) for the reproduction test to be meaningful.

## Decision

Default: **rolling window, M = 120 monthly observations, monthly rebalance**. These are exactly the DGU (2009) settings.

## Decision drivers

- Reproduction: DGU 2009 uses M=120, monthly. Defaults must match for the reproduction test.
- Stationarity: rolling discards old data, adapting to regime changes; expanding accumulates indefinitely.
- Statistical power: M=120 gives enough observations for stable covariance estimation while remaining computationally cheap per rebalance.

## Considered options

- Option A: Rolling M=120, monthly. **Chosen.**
- Option B: Expanding window from t=0. Rejected: estimator variance shrinks over time, making early/late comparisons unfair.
- Option C: Multiple defaults (parameterized). Rejected: defaults should match the reproduction; users override per-call.

## Consequences

- Robustness check supports M ∈ {60, 120, 240} via parameter.
- DGU reproduction test runs with M=120 monthly on three Fama-French datasets.
- Turnover formula is one-sided drift-adjusted (DGU Eq. 11): `TO = 0.5 · Σ|w_new − w_drift|`.

## Links

- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal Versus Naive Diversification." *RFS* 22(5).
