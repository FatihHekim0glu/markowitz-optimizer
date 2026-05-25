# Tutorials

Six Jupyter notebooks that build from two-asset intuition up to a full out-of-sample horse race. All notebooks use **synthetic data only** (no network), set explicit random seeds, and depend only on the public API.

The notebooks live under [`notebooks/`](https://github.com/FatihHekim0glu/markowitz-optimizer/tree/main/notebooks) in the repository. Open them in JupyterLab from a `uv sync`'d clone:

```bash
uv run jupyter lab notebooks/
```

| # | Notebook | What you learn |
|---|----------|----------------|
| 01 | [Two-Asset Intuition](https://github.com/FatihHekim0glu/markowitz-optimizer/blob/main/notebooks/01_two_asset_intuition.ipynb) | The mean-variance trade-off, geometrically, with just two assets. |
| 02 | [Efficient Frontier](https://github.com/FatihHekim0glu/markowitz-optimizer/blob/main/notebooks/02_efficient_frontier.ipynb) | Sweep the risk-aversion parameter to trace the unconstrained and long-only frontiers. |
| 03 | [Markowitz Breaks](https://github.com/FatihHekim0glu/markowitz-optimizer/blob/main/notebooks/03_markowitz_breaks.ipynb) | Reproduce the error-maximization pathology with finite samples. |
| 04 | [Shrinkage & Fixes](https://github.com/FatihHekim0glu/markowitz-optimizer/blob/main/notebooks/04_shrinkage_and_fixes.ipynb) | Apply Ledoit-Wolf and OAS shrinkage and watch the weights stabilize. |
| 05 | [Black-Litterman](https://github.com/FatihHekim0glu/markowitz-optimizer/blob/main/notebooks/05_black_litterman.ipynb) | Build a prior from market caps, layer in views, and inspect the posterior. |
| 06 | [Out-of-Sample Showdown](https://github.com/FatihHekim0glu/markowitz-optimizer/blob/main/notebooks/06_out_of_sample_showdown.ipynb) | A rolling-window horse race across plug-in, shrinkage, Black-Litterman, and $1/N$. |

## Conventions used in every notebook

- `numpy` is imported as `np`; `pandas` as `pd`; `matplotlib.pyplot` as `plt`.
- All random draws use `np.random.default_rng(seed)` with an explicit integer seed.
- "Returns" always means **per-period excess returns** unless stated otherwise.
- Volatilities and Sharpe ratios are **annualized assuming 252 trading days** per year, except where the notebook is illustrating monthly data.

## Suggested reading order

If you are new to portfolio optimization, follow the notebooks in order; each builds on terminology introduced in the previous one. If you already know the basics, notebooks 3, 4, and 6 are the empirically interesting ones.
