"""Page 4 — Black-Litterman view blending."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from components.plots import weights_bar
from components.sidebar import render_sidebar
from components.theme import inject_css, register_plotly_template

st.set_page_config(page_title="Black-Litterman", layout="wide")
register_plotly_template()
inject_css()

try:
    from markowitz.views import BlackLitterman  # type: ignore

    HAS_BL = True
    _import_error: str | None = None
except Exception as exc:
    HAS_BL = False
    _import_error = str(exc)


@st.cache_data(show_spinner=False)
def _synthetic_returns(tickers: tuple[str, ...], seed: int, n_days: int = 1260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(tickers)
    drifts = rng.uniform(0.05, 0.14, size=n) / 252.0
    vols = rng.uniform(0.15, 0.35, size=n) / np.sqrt(252.0)
    a = rng.normal(size=(n, n))
    base = a @ a.T / n
    d = np.sqrt(np.diag(base))
    corr = base / np.outer(d, d)
    cov = np.outer(vols, vols) * corr
    chol = np.linalg.cholesky(cov + 1e-8 * np.eye(n))
    z = rng.standard_normal(size=(n_days, n))
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame(drifts + z @ chol.T, index=idx, columns=list(tickers))


def _implied_returns(market_weights: np.ndarray, cov: np.ndarray, delta: float) -> np.ndarray:
    return delta * cov @ market_weights


def _bl_posterior(
    pi: np.ndarray,
    cov: np.ndarray,
    tau: float,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tau_cov = tau * cov
    inv_tau = np.linalg.pinv(tau_cov)
    inv_om = np.linalg.pinv(Omega)
    middle = inv_tau + P.T @ inv_om @ P
    mu_post = np.linalg.pinv(middle) @ (inv_tau @ pi + P.T @ inv_om @ Q)
    cov_post = cov + np.linalg.pinv(middle)
    return mu_post, cov_post


st.title("Black-Litterman")
cfg = render_sidebar(key_prefix="bl")

if not HAS_BL:
    st.info(
        f"Library BL module unavailable ({_import_error}); using closed-form fallback.",
        icon=":material/info:",
    )

returns = _synthetic_returns(cfg.tickers, cfg.seed)
cov_annual = returns.cov().values * 252.0
n = len(cfg.tickers)

st.markdown("### Market prior")
col1, col2 = st.columns(2)
with col1:
    delta = st.slider("Risk aversion δ", 1.0, 6.0, 2.5, step=0.1)
with col2:
    tau = st.slider("Scalar τ on prior covariance", 0.01, 0.20, 0.05, step=0.01)

eq_weights = np.full(n, 1.0 / n)
pi = _implied_returns(eq_weights, cov_annual, delta)
prior_series = pd.Series(pi, index=cfg.tickers, name="implied_return")

st.dataframe(
    prior_series.to_frame().style.format("{:.2%}"),
    use_container_width=True,
    height=min(360, 38 * (n + 1)),
)

st.markdown("### Views")
st.caption(
    "Each view is an absolute annual return for one asset. Leave blank to skip. "
    "Confidence is the standard deviation of the view (lower = stronger)."
)

view_data: list[tuple[str, float, float]] = []
with st.form("bl_views"):
    grid = st.columns([2, 1, 1])
    grid[0].markdown("**Asset**")
    grid[1].markdown("**View (annual return)**")
    grid[2].markdown("**Confidence σ**")
    st.caption("Views are limited to the first 6 assets in the universe.")
    for tkr in cfg.tickers[: min(n, 6)]:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.text_input(
                "asset", value=tkr, key=f"bl_a_{tkr}", disabled=True, label_visibility="collapsed"
            )
        with c2:
            q = st.number_input(
                "q",
                value=0.0,
                step=0.01,
                format="%.3f",
                key=f"bl_q_{tkr}",
                label_visibility="collapsed",
            )
        with c3:
            s = st.number_input(
                "s",
                value=0.05,
                min_value=0.005,
                step=0.005,
                format="%.3f",
                key=f"bl_s_{tkr}",
                label_visibility="collapsed",
            )
        if abs(q) > 1e-9:
            view_data.append((tkr, float(q), float(s)))
    submitted = st.form_submit_button("Blend views", type="primary")

if submitted or "bl_posterior" not in st.session_state:
    try:
        if view_data:
            P = np.zeros((len(view_data), n))
            Q = np.zeros(len(view_data))
            omega_diag = np.zeros(len(view_data))
            for i, (tkr, q, s) in enumerate(view_data):
                P[i, list(cfg.tickers).index(tkr)] = 1.0
                Q[i] = q
                omega_diag[i] = s * s
            Omega = np.diag(omega_diag)
            if HAS_BL:
                try:
                    bl = BlackLitterman(pi=pi, sigma=cov_annual, tau=tau)
                    bl.add_views(P=P, Q=Q, Omega=Omega)
                    post = bl.posterior()
                    mu_post = np.asarray(post["mu"])
                    cov_post = np.asarray(post["cov"])
                except Exception:
                    mu_post, cov_post = _bl_posterior(pi, cov_annual, tau, P, Q, Omega)
            else:
                mu_post, cov_post = _bl_posterior(pi, cov_annual, tau, P, Q, Omega)
        else:
            mu_post, cov_post = pi.copy(), cov_annual.copy()

        raw = np.linalg.pinv(delta * cov_post) @ mu_post
        if cfg.long_only:
            raw = np.clip(raw, 0.0, None)
        s = raw.sum()
        w_post = raw / s if s > 0 else eq_weights
        w_post = np.minimum(w_post, cfg.max_weight)
        s2 = w_post.sum()
        w_post = w_post / s2 if s2 > 0 else eq_weights

        st.session_state["bl_posterior"] = {
            "mu": pd.Series(mu_post, index=cfg.tickers),
            "weights": pd.Series(w_post, index=cfg.tickers),
        }
    except Exception as exc:
        st.error(f"Blend failed: {exc.__class__.__name__}: {exc}")
        st.stop()

post = st.session_state["bl_posterior"]
left, right = st.columns(2, gap="large")
with left:
    st.markdown("#### Posterior returns")
    df_compare = pd.DataFrame({"prior": prior_series, "posterior": post["mu"]})
    df_compare["delta"] = df_compare["posterior"] - df_compare["prior"]
    st.dataframe(
        df_compare.style.format("{:.2%}"),
        use_container_width=True,
        height=min(420, 38 * (n + 1)),
    )
with right:
    st.plotly_chart(weights_bar(post["weights"]), use_container_width=True)
