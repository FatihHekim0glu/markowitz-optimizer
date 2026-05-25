"""Page 5 — Data diagnostics: distributions, correlations, conditioning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from components.plots import corr_heatmap
from components.sidebar import render_sidebar
from components.theme import PALETTE, inject_css, register_plotly_template

st.set_page_config(page_title="Diagnostics", layout="wide")
register_plotly_template()
inject_css()


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


def _moments_table(returns: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats  # type: ignore

    table = pd.DataFrame(index=returns.columns)
    table["mean (ann.)"] = returns.mean() * 252
    table["vol (ann.)"] = returns.std() * np.sqrt(252)
    table["skew"] = returns.apply(stats.skew)
    table["ex. kurtosis"] = returns.apply(stats.kurtosis)
    table["min"] = returns.min()
    table["max"] = returns.max()
    return table


def _return_distribution(returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    palette_cycle = [
        PALETTE["navy"],
        PALETTE["amber"],
        PALETTE["teal"],
        PALETTE["slate_blue"],
        PALETTE["bronze"],
        PALETTE["crimson"],
    ]
    for i, col in enumerate(returns.columns):
        fig.add_trace(
            go.Histogram(
                x=returns[col],
                name=str(col),
                opacity=0.55,
                nbinsx=60,
                marker={"color": palette_cycle[i % len(palette_cycle)]},
            )
        )
    fig.update_layout(
        barmode="overlay",
        title="Daily return distribution",
        xaxis={"title": "Daily return", "tickformat": ".1%"},
        yaxis={"title": "Count"},
        height=420,
    )
    return fig


def _condition_metrics(cov: np.ndarray) -> dict:
    eigvals = np.linalg.eigvalsh(cov + 1e-12 * np.eye(cov.shape[0]))
    eigvals = np.clip(eigvals, 1e-18, None)
    cond = float(eigvals.max() / eigvals.min())
    return {
        "min_eig": float(eigvals.min()),
        "max_eig": float(eigvals.max()),
        "condition_number": cond,
        "rank": int(np.linalg.matrix_rank(cov)),
    }


st.title("Diagnostics")
cfg = render_sidebar(key_prefix="diag")

returns = _synthetic_returns(cfg.tickers, cfg.seed)
corr = returns.corr()
moments = _moments_table(returns)
cond = _condition_metrics(returns.cov().values)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(returns):,}")
c2.metric("Assets", f"{returns.shape[1]}")
c3.metric("Avg pairwise ρ", f"{corr.values[np.triu_indices_from(corr.values, k=1)].mean():.2f}")
c4.metric("Cov condition #", f"{cond['condition_number']:.1e}")

tab_dist, tab_corr, tab_moments, tab_cov = st.tabs(
    ["Distributions", "Correlation", "Moments", "Covariance health"]
)

with tab_dist:
    st.plotly_chart(_return_distribution(returns), use_container_width=True)

with tab_corr:
    st.plotly_chart(corr_heatmap(corr), use_container_width=True)

with tab_moments:
    st.dataframe(
        moments.style.format(
            {
                "mean (ann.)": "{:.2%}",
                "vol (ann.)": "{:.2%}",
                "skew": "{:.2f}",
                "ex. kurtosis": "{:.2f}",
                "min": "{:.2%}",
                "max": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

with tab_cov:
    st.write(
        "Eigenvalue spread of the sample covariance — large condition numbers "
        "indicate near-singular Σ and unstable mean-variance solutions."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Min eigenvalue", f"{cond['min_eig']:.2e}")
    m2.metric("Max eigenvalue", f"{cond['max_eig']:.2e}")
    m3.metric("Rank", f"{cond['rank']} / {returns.shape[1]}")

    eigvals = np.linalg.eigvalsh(returns.cov().values)
    eig_fig = go.Figure(
        go.Bar(
            x=[f"λ{i + 1}" for i in range(len(eigvals))],
            y=np.sort(eigvals)[::-1],
            marker={"color": PALETTE["navy"]},
        )
    )
    eig_fig.update_layout(
        title="Sorted eigenvalues of Σ",
        yaxis={"type": "log", "title": "Eigenvalue (log)"},
        height=360,
    )
    st.plotly_chart(eig_fig, use_container_width=True)
