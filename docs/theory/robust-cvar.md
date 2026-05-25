# Robust Mean-Variance and CVaR

Beyond shrinkage and Bayesian priors, two further families of estimators
hedge against the limitations of the plug-in approach: **robust**
optimization, which protects against worst-case parameter realizations, and
**CVaR** optimization, which targets the tail of the loss distribution
directly.

## Robust mean-variance

Suppose $\mu$ lies in an ellipsoidal uncertainty set centered at the
estimate $\hat\mu$:

$$
\mathcal{U}_\mu(\kappa)
\;=\;
\bigl\{\, \mu \;:\; (\mu - \hat\mu)^\top \Sigma_\mu^{-1} (\mu - \hat\mu) \le \kappa^2 \bigr\}.
\tag{1}
$$

The worst-case mean-variance problem is

$$
\max_w\; \min_{\mu \in \mathcal{U}_\mu(\kappa)}
\;\mu^\top w - \tfrac{\gamma}{2} w^\top \Sigma w
\quad \text{s.t.}\quad \mathbf{1}^\top w = 1.
\tag{2}
$$

The inner minimum has the closed form
$\hat\mu^\top w - \kappa \sqrt{w^\top \Sigma_\mu w}$, giving the **SOCP**

$$
\max_w\; \hat\mu^\top w - \kappa\,\sqrt{w^\top \Sigma_\mu w}
\;-\; \tfrac{\gamma}{2} w^\top \Sigma w,
\tag{3}
$$

solved by any SOCP backend [@goldfarb2003]. The parameter $\kappa$ controls
the conservatism of the solution; for a $(1-\alpha)$ confidence ellipsoid
under Gaussianity, $\kappa = \sqrt{\chi^2_{n, 1-\alpha} / T}$.

## CVaR (Rockafellar-Uryasev)

For loss $L(w) = -w^\top r$ and confidence level $\alpha \in (0, 1)$, the
**Value-at-Risk** is

$$
\mathrm{VaR}_\alpha(w) \;=\; \inf\{\, \zeta \;:\; \mathbb{P}(L(w) \le \zeta) \ge \alpha \,\}.
\tag{4}
$$

The **Conditional VaR** is the conditional mean of losses beyond VaR:

$$
\mathrm{CVaR}_\alpha(w) \;=\; \mathbb{E}\bigl[\,L(w)\;\big|\;L(w) \ge \mathrm{VaR}_\alpha(w)\,\bigr].
\tag{5}
$$

Rockafellar and Uryasev [@rockafellar2000] showed that CVaR has the
variational representation

$$
\mathrm{CVaR}_\alpha(w)
\;=\;
\min_\zeta\;
\zeta \;+\; \frac{1}{1-\alpha}\,\mathbb{E}\bigl[(L(w) - \zeta)_+\bigr].
\tag{6}
$$

Given a sample $\{r_1, \dots, r_T\}$, the empirical CVaR optimization
becomes a **linear program**:

$$
\begin{aligned}
\min_{w, \zeta, u}\;& \zeta \;+\; \frac{1}{(1-\alpha) T} \sum_{t=1}^T u_t \\
\text{s.t.}\;& u_t \ge -w^\top r_t - \zeta,\quad u_t \ge 0,\quad t=1,\dots,T,\\
& \mathbf{1}^\top w = 1, \quad w \in \mathcal{W}.
\tag{7}
\end{aligned}
$$

This is solved directly by `CVaROptimizer` with any LP-capable backend.

## When to use which

| situation                                          | recommended formulation |
|----------------------------------------------------|-------------------------|
| Returns approximately Gaussian, moderate $n/T$     | Mean-variance + LW shrinkage |
| Strong priors or absolute views                    | Black-Litterman         |
| Concern about mean estimation error specifically   | Robust mean (SOCP)      |
| Heavy-tailed returns, regulatory tail constraint   | CVaR                    |

## References

[@goldfarb2003]; [@rockafellar2000]; [@ben-tal2009]. See
[Citations](../citations.md).
