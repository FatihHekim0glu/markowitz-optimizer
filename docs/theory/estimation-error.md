# Estimation Error

The mean-variance solution (Eq. 2 in [Mean-Variance](mean-variance.md)) treats
$\mu$ and $\Sigma$ as known. In practice they are replaced by sample
estimates $\hat\mu$ and $\hat\Sigma$ from $T$ observations of $n$ assets.
This page quantifies how that substitution wrecks the optimization.

## Sample moments and their error

Let $r_t \in \mathbb{R}^n$ be excess returns. The unbiased sample estimators
are

$$
\hat\mu \;=\; \frac{1}{T}\sum_{t=1}^{T} r_t,
\qquad
\hat\Sigma \;=\; \frac{1}{T-1}\sum_{t=1}^{T} (r_t - \hat\mu)(r_t - \hat\mu)^\top.
\tag{1}
$$

Under iid Gaussian returns, $\hat\mu \sim \mathcal{N}(\mu, \Sigma/T)$ and
$(T-1)\hat\Sigma \sim \mathcal{W}_n(\Sigma, T-1)$. The marginal standard
error on each component of $\hat\mu_i$ is $\sigma_i / \sqrt{T}$ - for
monthly equity data with $\sigma_i \approx 5\%$ and $T = 60$, the
standard error on the mean is roughly $0.65\%$ per month, larger than
typical equity risk premia.

## The plug-in fallacy

Substituting $\hat\mu, \hat\Sigma$ into (Eq. 2) yields the *plug-in* portfolio
$\hat w$. Michaud [@michaud1989] called this an "error-maximization machine":
the optimizer overweights assets whose sample mean is high *by chance* and
whose sample variance is low *by chance*.

For an unconstrained tangency portfolio, Kan and Zhou [@kan2007]
derive the expected out-of-sample loss

$$
\mathbb{E}\bigl[U(w^\star) - U(\hat w)\bigr]
\;\approx\;
\frac{n}{2T}\, \theta^2 \;+\; \frac{n}{T},
\tag{2}
$$

where $\theta^2 = \mu^\top \Sigma^{-1} \mu$ is the squared maximum Sharpe.
The loss grows **linearly in $n/T$**, so doubling the universe at fixed
sample size doubles the expected utility shortfall.

## Sensitivity to $\hat\Sigma$

The unconstrained optimum scales with $\hat\Sigma^{-1}$, and small
eigenvalues of $\hat\Sigma$ become large eigenvalues of $\hat\Sigma^{-1}$.
The condition number of a sample covariance matrix from $T$ observations of
$n$ iid standard normals behaves like

$$
\kappa(\hat\Sigma) \;\sim\; \left(\frac{1+\sqrt{n/T}}{1-\sqrt{n/T}}\right)^2,
\tag{3}
$$

by Marchenko-Pastur [@marchenko1967]. For $n/T = 0.5$, the condition
number already exceeds 30, and for $n \ge T$ the sample covariance is
singular.

## Implications

Three families of fixes recur in the literature, each addressed on its own
page:

- **Shrinkage** of $\hat\Sigma$ towards a structured target
  ([Shrinkage](shrinkage.md)).
- **Bayesian priors** on returns, of which Black-Litterman is the most
  popular ([Black-Litterman](black-litterman.md)).
- **Robust** and distributionally-robust formulations
  ([Robust & CVaR](robust-cvar.md)).

## References

[@michaud1989]; [@kan2007]; [@marchenko1967]. See [Citations](../citations.md).
