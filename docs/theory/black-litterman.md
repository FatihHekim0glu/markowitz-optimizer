# The Black-Litterman Model

Black and Litterman [@black1992] addressed the brittleness of plug-in
mean-variance by combining a **market-implied prior** on returns with
investor **views**.

## Reverse optimization: the prior

Assume the market portfolio $w_{\mathrm{mkt}}$ is the optimum of an
unconstrained mean-variance problem with risk aversion $\delta$. Inverting
the first-order condition gives the **equilibrium implied returns**:

$$
\Pi \;=\; \delta\, \Sigma\, w_{\mathrm{mkt}}.
\tag{1}
$$

The prior on the true (unobserved) mean $\mu$ is

$$
\mu \;\sim\; \mathcal{N}\bigl(\Pi,\; \tau \Sigma\bigr),
\tag{2}
$$

where $\tau$ is a small scalar reflecting confidence in the equilibrium
(typical values $0.01 \le \tau \le 0.05$).

## Encoding views

A view is a linear statement about $\mu$:

$$
P \mu \;=\; q \;+\; \varepsilon, \qquad
\varepsilon \sim \mathcal{N}(0,\, \Omega),
\tag{3}
$$

where $P \in \mathbb{R}^{k \times n}$ picks linear combinations of assets,
$q \in \mathbb{R}^k$ is the view magnitude, and $\Omega \in \mathbb{R}^{k \times k}$
is the view uncertainty (diagonal in practice).

## Posterior

Combining (2) and (3) by standard Gaussian conjugacy gives the **posterior
mean** of $\mu$:

$$
\bar\mu \;=\;
\Bigl[(\tau \Sigma)^{-1} + P^\top \Omega^{-1} P\Bigr]^{-1}
\Bigl[(\tau \Sigma)^{-1} \Pi + P^\top \Omega^{-1} q\Bigr],
\tag{4}
$$

and posterior covariance

$$
M \;=\; \bigl[(\tau \Sigma)^{-1} + P^\top \Omega^{-1} P\bigr]^{-1}.
\tag{5}
$$

The Black-Litterman portfolio plugs $(\bar\mu,\; \Sigma + M)$ into the
mean-variance optimizer of [Mean-Variance](mean-variance.md).

## Setting $\Omega$: the Idzorek convention

Idzorek [@idzorek2005] specifies $\Omega$ by **confidence levels**
$c_k \in [0, 1]$ rather than variances. For view $k$ with row $p_k$, set

$$
\Omega_{kk} \;=\; \frac{1 - c_k}{c_k} \cdot p_k\, (\tau \Sigma)\, p_k^\top.
\tag{6}
$$

This recovers the intuitive limits: $c_k \to 0$ ignores the view,
$c_k \to 1$ enforces it exactly.

## He-Litterman convention

He and Litterman [@he1999] use a simpler diagonal form:

$$
\Omega \;=\; \mathrm{diag}\bigl(P\, (\tau \Sigma)\, P^\top\bigr),
\tag{7}
$$

which implicitly assigns each view a confidence consistent with its
projection of the prior covariance. The
[He-Litterman 1999 validation page](../validation/he-litterman-1999.md)
reproduces the eight-country example using this convention.

## Practical notes

- Always shrink $\Sigma$ before reverse-optimization; the equilibrium
  weights are extremely sensitive to small eigenvalues.
- For a fully invested optimization, use $\delta$ implied by the historical
  market Sharpe rather than the textbook $\delta = 2.5$.
- The library exposes `BlackLitterman(prior=..., views=..., omega="idzorek")`
  with both conventions selectable.

## References

[@black1992]; [@he1999]; [@idzorek2005]. See [Citations](../citations.md).
