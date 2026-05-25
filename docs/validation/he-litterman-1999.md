# Validation: He & Litterman (1999)

He and Litterman [@he1999] illustrate the Black-Litterman model on a seven-country equity universe. This page reproduces their Table 2 equilibrium-implied excess returns using the library.

## Setup

The original paper uses USD-denominated equity returns for Australia, Canada, France, Germany, Japan, the UK, and the US. We follow the paper's published inputs:

- Market-capitalization weights $w_{\mathrm{mkt}}$ from Table 2.
- Annualized covariance $\Sigma$ derived from Table 1 (correlations and volatilities).
- Risk aversion $\delta = 2.5$.
- $\tau = 0.05$.

Equilibrium excess returns are computed by reverse optimization:

$$
\Pi \;=\; \delta\, \Sigma\, w_{\mathrm{mkt}}.
$$

## Reproducing Table 2

```python
import numpy as np
import pandas as pd
from markowitz.views import BlackLitterman, Views, AbsoluteView

ASSETS = ["AUL", "CAN", "FRA", "GER", "JPN", "UKG", "USA"]
# Σ (annualized), market caps, δ=2.5, τ=0.05
# (see tests/regression/test_he_litterman_1999.py for the full numerical inputs)
bl = BlackLitterman(cov=COV_DF, market_weights=W_MKT_SERIES, delta=2.5, tau=0.05)
pi = bl.implied_returns()
# Reference test: tests/regression/test_he_litterman_1999.py
# Tolerance: 1e-3 per-asset (six countries); Canada is a documented paper anomaly (1e-2 bound)
```

`BlackLitterman.implied_returns()` returns $\Pi = \delta\, \Sigma\, w_{\mathrm{mkt}}$ as a pandas
`Series` labelled by the covariance index.

## Measured deviation

- **Max non-Canada deviation:** ~9e-4 (well inside the 1e-3 per-asset tolerance).
- **Canada deviation:** ~8e-3 (documented paper anomaly; we bound it at 1e-2).

The Canada residual is a known rounding artefact of the inputs printed in
He-Litterman (1999); it is not a defect of the library.

## Adding views

The paper's Table 4 layers two views on top of the equilibrium prior:

- **View 1.** German equity will outperform other European equity by 5%.
- **View 2.** Canadian equity will outperform US equity by 3%.

These can be encoded with `RelativeView` objects in a `Views` container and
passed to `BlackLitterman(views=...)`. See the regression test for the encoded
pick matrix.

## Notes on conventions

- The paper uses **excess** returns; `BlackLitterman` operates on excess returns
  throughout.
- He and Litterman use a diagonal $\Omega$ proportional to $P \tau \Sigma P^\top$
  (the `"he_litterman"` strategy, which is the default).
- The full numerical inputs (covariance matrix, market weights) and assertions
  live in `tests/regression/test_he_litterman_1999.py`.

## References

[@he1999]; [@black1992]. See [Citations](../citations.md).
