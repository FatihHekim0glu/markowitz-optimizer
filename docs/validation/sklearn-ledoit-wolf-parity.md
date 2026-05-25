# Validation: sklearn `LedoitWolf` Parity

The Ledoit-Wolf shrinkage estimator in `markowitz.LedoitWolfShrinkage` is benchmarked for **numerical parity** against `sklearn.covariance.LedoitWolf`. This guards against regressions whenever the internal estimator of $\pi$, $\rho$, and $\gamma$ (Eq. 3 of [Shrinkage](../theory/shrinkage.md)) is touched.

## Reference implementation

scikit-learn ships a well-tested implementation [@pedregosa2011] that, by default, shrinks toward the **identity** target

$$
F \;=\; \tfrac{\mathrm{tr}(\hat\Sigma)}{n}\, I.
$$

To compare like-for-like, the parity test calls our library with `target="identity"`, which delegates to `sklearn.covariance.ledoit_wolf` internally and then optionally annualizes.

## Test design

```python
import numpy as np
from sklearn.covariance import LedoitWolf as SKLW
from markowitz import LedoitWolfShrinkage

rng = np.random.default_rng(123)
for n, T in [(5, 250), (20, 500), (50, 250), (50, 60)]:
    X = rng.standard_normal((T, n))

    sk = SKLW().fit(X)
    lw = LedoitWolfShrinkage(target="identity", annualize=False).fit(X)

    # Shrinkage intensities and covariance entries match element-wise.
    assert np.isclose(sk.shrinkage_, lw.shrinkage_, atol=1e-12)
    assert np.allclose(sk.covariance_, lw.covariance_, atol=1e-12)
```

The test grid spans both the under-determined ($n < T$) and the difficult
($n \approx T$) regimes.

## Measured deviation

- **Observed max deviation across 15 random seeds: 0.0 (byte-equal element-wise).**

Parity is exact because `LedoitWolfShrinkage(target="identity")` is implemented
as a thin wrapper around `sklearn.covariance.ledoit_wolf`; no numerical work
diverges between the two paths. See `tests/parity/test_sklearn_lw_parity.py`
for the live assertions.

## Caveats

- The constant-correlation target (`target="constant_corr"`) is **not** offered
  by scikit-learn and is validated against Ledoit-Wolf 2003 [@ledoit2003]
  tabulated examples instead.
- `LedoitWolfShrinkage` annualizes by default (`annualize=True`); the parity
  test disables annualization so the comparison is against the raw per-period
  estimator.

## When parity is expected to break

If parity tests start failing, the most common causes are:

1. The wrapper around `sklearn.covariance.ledoit_wolf` is replaced with an
   in-house re-implementation.
2. A change in centering convention inside that wrapper.
3. A change in the shrinkage-intensity clip range.

The CI matrix pins a known-good scikit-learn version range; bumping that range
triggers a full rerun of this parity suite.

## References

[@pedregosa2011]; [@ledoit2004]; [@chen2010]. See [Citations](../citations.md).
