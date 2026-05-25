"""Page 2 — Compare covariance and mean estimators side-by-side."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from components.plots import frontier_figure, weights_bar
from components.sidebar import render_sidebar
from components.theme import inject_css, register_plotly_template

st.set_page_config(page_title="Method comparison", layout="wide")
register_plotly_template()
inject_css()

try:
    from markowitz.estimators import (  # type: ignore
        JorionBayesStein,
        SampleMean,
    )
    from markowitz.estimators import LedoitWolfShrinkage as LedoitWolfCov  # type: ignore
    from markowitz.estimators import SampleCovariance as SampleCov  # type: ignore

    HAS_ESTIMATORS = True
    _import_error: str | None = None
except Exception as exc:
    HAS_ESTIMATORS = False
    _import_error = str(exc)


@st.cache_data(show_spinner=False)
def _synthetic_returns(
    tickers: tuple[str, ...], seed: int, n_days: int = 1260
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    drifts = rng.uniform(0.05, 0.14, size=n_assets) / 252.0
    vols = rng.uniform(0.15, 0.35, size=n_assets) / np.sqrt(252.0)
    a = rng.normal(size=(n_assets, n_assets))
    base = a @ a.T / n_assets
    d = np.sqrt(np.diag(base))
    corr = base / np.outer(d, d)
    cov = np.outer(vols, vols) * corr
    chol = np.linalg.cholesky(cov + 1e-8 * np.eye(n_assets))
    z = rng.standard_normal(size=(n_days, n_assets))
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame(drifts + z @ chol.T, index=idx, columns=list(tickers))


def _estimate(returns: pd.DataFrame, mean_method: str, cov_method: str):
    """Return (mu, cov) annualized. Uses library if available, else fallback."""
    if HAS_ESTIMATORS:
        try:
            if cov_method == "LedoitWolf":
                cov = LedoitWolfCov().fit(returns.values).covariance_
            else:
                cov = SampleCov().fit(returns.values).covariance_

            if mean_method == "JorionBayesStein":
                mu = JorionBayesStein().fit(returns.values, cov).mean_
            else:
                mu = SampleMean().fit(returns.values).mean_
            return np.asarray(mu) * 252.0, np.asarray(cov) * 252.0
        except Exception:
            pass

    sample_cov = returns.cov().values
    if cov_method == "LedoitWolf":
        try:
            from sklearn.covariance import LedoitWolf  # type: ignore

            cov = LedoitWolf().fit(returns.values).covariance_
        except Exception:
            cov = sample_cov
    else:
        cov = sample_cov

    mu = returns.mean().values
    if mean_method == "JorionBayesStein":
        grand = float(np.mean(mu))
        cov_inv = np.linalg.pinv(cov)
        ones = np.ones_like(mu)
        scale = float(ones @ cov_inv @ (mu - grand))
        denom = float(ones @ cov_inv @ ones)
        shrink = max(0.0, min(1.0, scale / (denom + 1e-12)))
        mu = (1 - shrink) * mu + shrink * grand

    return mu * 252.0, cov * 252.0


def _frontier_from(mu: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
    inv = np.linalg.pinv(cov)
    ones = np.ones_like(mu)
    a = float(ones @ inv @ ones)
    b = float(ones @ inv @ mu)
    c = float(mu @ inv @ mu)
    d = max(a * c - b * b, 1e-9)
    targets = np.linspace(mu.min(), mu.max(), 40)
    vols = np.sqrt((a * targets * targets - 2 * b * targets + c) / d)
    return pd.DataFrame({"volatility": vols, "expected_return": targets})


def _tangency(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    raw = np.linalg.pinv(cov) @ (mu - rf)
    if raw.sum() == 0:
        return np.full_like(mu, 1.0 / len(mu))
    w = np.clip(raw / raw.sum(), 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(mu, 1.0 / len(mu))


st.title("Method comparison")
cfg = render_sidebar(key_prefix="compare")

if not HAS_ESTIMATORS:
    st.info(
        f"Library estimators not available ({_import_error}); "
        "using sklearn / closed-form fallbacks.",
        icon=":material/info:",
    )

returns = _synthetic_returns(cfg.tickers, cfg.seed)

methods = [
    ("Sample", "Sample"),
    ("Sample", "LedoitWolf"),
    ("JorionBayesStein", "LedoitWolf"),
]

rows = []
frontiers: dict[str, pd.DataFrame] = {}
weights: dict[str, pd.Series] = {}

with st.spinner("Computing estimators..."):
    for mean_m, cov_m in methods:
        label = f"{mean_m} + {cov_m}"
        try:
            mu, cov = _estimate(returns, mean_m, cov_m)
            front = _frontier_from(mu, cov)
            w = _tangency(mu, cov, cfg.risk_free_rate)
            frontiers[label] = front
            weights[label] = pd.Series(w, index=returns.columns, name=label)
            rows.append(
                {
                    "method": label,
                    "min_vol": float(front["volatility"].min()),
                    "max_return": float(front["expected_return"].max()),
                    "tangency_return": float(np.dot(w, mu)),
                    "tangency_vol": float(np.sqrt(w @ cov @ w)),
                }
            )
        except Exception as exc:
            st.warning(f"{label} failed: {exc.__class__.__name__}")

if not rows:
    st.error("All methods failed to compute.")
    st.stop()

summary_df = pd.DataFrame(rows)
summary_df["tangency_sharpe"] = (
    summary_df["tangency_return"] - cfg.risk_free_rate
) / summary_df["tangency_vol"].replace(0, np.nan)
st.dataframe(
    summary_df.style.format(
        {
            "min_vol": "{:.2%}",
            "max_return": "{:.2%}",
            "tangency_return": "{:.2%}",
            "tangency_vol": "{:.2%}",
            "tangency_sharpe": "{:.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

tabs = st.tabs([row["method"] for row in rows])
for tab, row in zip(tabs, rows, strict=False):
    with tab:
        label = row["method"]
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.plotly_chart(frontier_figure(frontiers[label]), use_container_width=True)
        with right:
            st.plotly_chart(weights_bar(weights[label]), use_container_width=True)
