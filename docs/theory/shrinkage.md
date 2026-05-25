# Shrinkage Covariance Estimators

The sample covariance $\hat\Sigma$ is unbiased but high-variance, especially
when $n$ is comparable to $T$. Shrinkage estimators replace it with a convex
combination

$$
\hat\Sigma^{\mathrm{sh}}(\delta) \;=\; (1-\delta)\,\hat\Sigma \;+\; \delta\, F,
\qquad \delta \in [0, 1],
\tag{1}
$$

where $F$ is a structured **target** (well-conditioned but biased) and
$\delta$ is the **shrinkage intensity**.

## Choice of target

Three targets are standard:

| target           | $F_{ij}$                                                  | best when                       |
|------------------|-----------------------------------------------------------|---------------------------------|
| Identity-scaled  | $\bar\sigma^2 \cdot \mathbb{1}\{i=j\}$                    | assets are roughly homogeneous  |
| Constant correl. | $\bar\rho \cdot s_i s_j$ off-diagonal, $s_i^2$ on diag.   | equity universes [@ledoit2004]  |
| Single-index     | $\beta_i \beta_j \sigma_m^2 + \mathrm{diag}$ of residuals | factor-driven returns           |

Here $s_i^2 = \hat\Sigma_{ii}$, $\bar\sigma^2$ is the mean diagonal, and
$\bar\rho$ is the mean off-diagonal correlation.

## Optimal intensity: Ledoit-Wolf

Ledoit and Wolf [@ledoit2004] derive the $\delta^\star$ that minimizes the
Frobenius-norm risk

$$
R(\delta) \;=\; \mathbb{E}\,\bigl\|\hat\Sigma^{\mathrm{sh}}(\delta) - \Sigma\bigr\|_F^2.
\tag{2}
$$

The minimizer has the closed form

$$
\delta^\star \;=\; \frac{\pi - \rho}{\gamma},
\qquad
\begin{aligned}
\pi   &= \textstyle\sum_{i,j} \mathrm{AsyVar}(\sqrt{T}\,\hat\Sigma_{ij}),\\
\rho  &= \textstyle\sum_{i,j} \mathrm{AsyCov}(\sqrt{T}\,\hat\Sigma_{ij},\,\sqrt{T}\,F_{ij}),\\
\gamma &= \|F - \Sigma\|_F^2.
\end{aligned}
\tag{3}
$$

In practice each term is estimated from the sample; we clip $\delta^\star$ to
$[0,1]$.

## Oracle Approximating Shrinkage (OAS)

Chen et al. [@chen2010] propose an estimator that, under Gaussian
assumptions, has lower MSE than Ledoit-Wolf for small $T$:

$$
\delta_{\mathrm{OAS}} \;=\;
\frac{(1 - 2/n)\,\mathrm{tr}(\hat\Sigma^2) + \mathrm{tr}(\hat\Sigma)^2}
     {(T + 1 - 2/n)\bigl(\mathrm{tr}(\hat\Sigma^2) - \mathrm{tr}(\hat\Sigma)^2/n\bigr)}.
\tag{4}
$$

OAS shrinks toward the identity-scaled target.

## Why shrinkage helps optimization

Inverting $\hat\Sigma^{\mathrm{sh}}$ instead of $\hat\Sigma$ has two effects:

1. The smallest eigenvalues are inflated, which **caps** $\|w^\star\|_2$ and
   suppresses the error-maximization pathology described in
   [Estimation Error](estimation-error.md).
2. The condition number is dramatically reduced, improving QP solver
   reliability.

The Ledoit-Wolf and OAS estimators are exposed in this library via the
`LedoitWolfCovariance` and `OASCovariance` classes; parity with
`sklearn.covariance` is verified in the
[validation suite](../validation/sklearn-ledoit-wolf-parity.md).

## References

[@ledoit2004]; [@chen2010]; [@ledoit2003]. See [Citations](../citations.md).
