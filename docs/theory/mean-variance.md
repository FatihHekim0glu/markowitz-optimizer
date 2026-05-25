# Mean-Variance Optimization

Markowitz [@markowitz1952] cast portfolio choice as a quadratic program in the
mean and covariance of asset returns. Given an asset universe of $n$ risky
assets with expected return vector $\mu \in \mathbb{R}^n$ and positive
semi-definite covariance matrix $\Sigma \in \mathbb{R}^{n \times n}$, the
investor chooses portfolio weights $w \in \mathbb{R}^n$.

## The canonical problem

The classical mean-variance objective trades expected return against variance
through a risk-aversion parameter $\gamma > 0$:

$$
\max_{w}\; \mu^\top w \;-\; \tfrac{\gamma}{2}\, w^\top \Sigma w
\quad \text{s.t.} \quad \mathbf{1}^\top w = 1.
\tag{1}
$$

With no further constraints, the closed-form solution is

$$
w^\star \;=\; \frac{1}{\gamma}\,\Sigma^{-1}\!\bigl(\mu - \lambda \mathbf{1}\bigr),
\qquad
\lambda \;=\; \frac{\mathbf{1}^\top \Sigma^{-1} \mu - \gamma}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}.
\tag{2}
$$

## The efficient frontier

Sweeping $\gamma$ traces the **efficient frontier** in $(\sigma, \mu)$ space.
For the fully invested, unconstrained case the frontier is the hyperbola

$$
\sigma_p^2 \;=\; \frac{1}{C}\bigl(1 + (\mu_p - A/C)^2 \cdot C / (BC - A^2)\bigr),
\tag{3}
$$

where $A = \mathbf{1}^\top \Sigma^{-1} \mu$, $B = \mu^\top \Sigma^{-1} \mu$,
and $C = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}$ are the standard Merton
constants [@merton1972].

## With long-only and box constraints

In practice the problem is solved as a QP:

$$
\min_w\; \tfrac{1}{2} w^\top \Sigma w - \tfrac{1}{\gamma} \mu^\top w
\quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; \ell \le w \le u.
\tag{4}
$$

Box constraints $\ell \le w \le u$ subsume the long-only case ($\ell = 0$).
Adding turnover constraints

$$
\|w - w_0\|_1 \;\le\; \tau,
\tag{5}
$$

where $w_0$ is the current portfolio, requires auxiliary variables but remains
a convex QP.

## Why this is only a starting point

Problem (4) is well-posed *given* $\mu$ and $\Sigma$. The catch is that both
must be **estimated** from finite samples. The next pages describe how
estimation error propagates into wildly unstable weights, and what to do
about it.

## References

[@markowitz1952]; [@merton1972]. See [Citations](../citations.md) for full
bibliographic details.
