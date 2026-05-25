# Validation: DeMiguel, Garlappi & Uppal (2009)

!!! warning "Status: planned for v0.2"
    This reproduction is on the roadmap. The walk-forward backtest infrastructure
    is implemented (`markowitz.backtest.WalkForward`), but full reproduction of
    DeMiguel-Garlappi-Uppal (2009) on Fama-French datasets is deferred to v0.2.

DeMiguel, Garlappi, and Uppal [@demiguel2009] ran a large horse race of fourteen portfolio strategies against the naive $1/N$ rule on seven empirical datasets. Their controversial conclusion: out of sample, $1/N$ was hard to beat. This page sketches the planned reproduction protocol; see the warning above for current status.

## Protocol (planned)

For each strategy and dataset:

1. Use a rolling estimation window of $M = 120$ months.
2. At each rebalance, fit moments on the trailing window.
3. Form portfolio weights and hold for one month.
4. Record realized return.
5. After the full backtest, compute:
   - out-of-sample Sharpe ratio,
   - certainty-equivalent return at $\gamma = 1$,
   - portfolio turnover.

The relevant out-of-sample metrics in the paper are Tables 3-5.

## Strategies to cover

The reproduction will target the strategies most relevant to mean-variance
research:

| strategy                                    | building block                            |
|---------------------------------------------|-------------------------------------------|
| Equally-weighted $1/N$                      | `markowitz.backtest.OneOverN`             |
| Plug-in mean-variance                       | `MeanVariance` + `SampleMean` / `SampleCovariance` |
| Mean-variance with Ledoit-Wolf $\hat\Sigma$ | `MeanVariance` + `LedoitWolfShrinkage`    |
| Minimum-variance, sample covariance         | `MeanVariance(risk_aversion=large)` + `SampleCovariance` |
| Minimum-variance, Ledoit-Wolf               | `MeanVariance(risk_aversion=large)` + `LedoitWolfShrinkage` |
| Black-Litterman with equilibrium prior      | `BlackLitterman` + `MeanVariance`         |

## Planned reproduction sketch

Once shipped, the reproduction will look approximately like:

```python
# Sketch only - not yet implemented in v0.1
from markowitz.backtest import WalkForward
# Plus a Fama-French data loader (deferred to v0.2)

wf = WalkForward(
    window=120,
    rebalance="monthly",
    strategy=...,  # any callable returning weights from a returns DataFrame
)
result = wf.run(returns_df)
print(result.sharpe(), result.turnover())
```

The walk-forward engine already exists; what is deferred to v0.2 is (a) the
bundled Fama-French dataset loaders and (b) the comparison harness that runs
all six strategies over all seven datasets and produces the paper's tables.

## Expected qualitative findings

The reproduction is expected to confirm the paper's headline results:

- The plug-in mean-variance strategy underperforms $1/N$ on Sharpe and has
  extreme turnover.
- Ledoit-Wolf shrinkage closes most (not all) of the Sharpe gap and reduces
  turnover by an order of magnitude.
- Minimum-variance with Ledoit-Wolf is the most consistent positive-edge
  strategy across datasets.

## References

[@demiguel2009]; [@ledoit2004]. See [Citations](../citations.md).
