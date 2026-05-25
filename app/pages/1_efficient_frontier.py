"""Page 1 — Efficient frontier."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from components.plots import frontier_figure, weights_bar
from components.sidebar import SidebarConfig, render_sidebar
from components.state import (
    SESSION_LAST_HASH,
    SESSION_RESULTS,
    config_hash,
)
from components.theme import inject_css, register_plotly_template

st.set_page_config(page_title="Efficient frontier", layout="wide")
register_plotly_template()
inject_css()

try:
    from markowitz.core import AnalyticFrontier  # type: ignore
    from markowitz.optimizer import MeanVariance  # type: ignore

    HAS_OPTIMIZER = True
    _import_error: str | None = None
except Exception as exc:  # ImportError or partial install
    HAS_OPTIMIZER = False
    _import_error = str(exc)


@st.cache_data(show_spinner=False)
def _synthetic_returns(tickers: tuple[str, ...], start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)
    if n < 30:
        idx = pd.bdate_range(end=end, periods=252 * 3)
        n = len(idx)
    drifts = rng.uniform(0.05, 0.14, size=n_assets) / 252.0
    vols = rng.uniform(0.15, 0.35, size=n_assets) / np.sqrt(252.0)
    a = rng.normal(size=(n_assets, n_assets))
    base = a @ a.T / n_assets
    d = np.sqrt(np.diag(base))
    corr = base / np.outer(d, d)
    cov = np.outer(vols, vols) * corr
    chol = np.linalg.cholesky(cov + 1e-8 * np.eye(n_assets))
    z = rng.standard_normal(size=(n, n_assets))
    eps = z @ chol.T
    rets = drifts + eps
    return pd.DataFrame(rets, index=idx, columns=list(tickers))


def _load_returns(cfg: SidebarConfig) -> pd.DataFrame:
    if cfg.use_real_data:
        try:
            import yfinance as yf  # type: ignore

            data = yf.download(
                list(cfg.tickers),
                start=cfg.start.isoformat(),
                end=cfg.end.isoformat(),
                progress=False,
                auto_adjust=True,
            )
            prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
            prices = prices.dropna(how="all").ffill().dropna()
            rets = prices.pct_change().dropna()
            if rets.empty:
                raise RuntimeError("No returns after cleaning.")
            return rets
        except Exception as exc:
            st.warning(
                f"Real data unavailable ({exc.__class__.__name__}); "
                "falling back to synthetic returns."
            )
    return _synthetic_returns(cfg.tickers, cfg.start.isoformat(), cfg.end.isoformat(), cfg.seed)


@st.cache_data(show_spinner=False)
def _compute_frontier(returns: pd.DataFrame, cfg_payload: dict) -> dict:
    """Compute analytic + constrained frontier; falls back to fully Python.

    Returns dict with keys: frontier (DataFrame), weights (Series),
    summary (dict).
    """
    rf = float(cfg_payload["risk_free_rate"])
    mu_daily = returns.mean()
    cov_daily = returns.cov()
    mu = mu_daily * 252.0
    cov = cov_daily * 252.0

    if HAS_OPTIMIZER:
        try:
            af = AnalyticFrontier(mu.values, cov.values)
            grid = af.frontier(n_points=40)
            frontier_df = pd.DataFrame(grid, columns=["volatility", "expected_return"])
        except Exception:
            frontier_df = _fallback_frontier(mu.values, cov.values)

        try:
            opt = MeanVariance(
                mu.values,
                cov.values,
                long_only=bool(cfg_payload["long_only"]),
                max_weight=float(cfg_payload["max_weight"]),
            )
            tan_w = np.asarray(opt.max_sharpe(risk_free_rate=rf))
        except Exception:
            tan_w = _fallback_tangency(mu.values, cov.values, rf)
    else:
        frontier_df = _fallback_frontier(mu.values, cov.values)
        tan_w = _fallback_tangency(mu.values, cov.values, rf)

    weights = pd.Series(tan_w, index=mu.index, name="weight")

    min_vol_idx = int(frontier_df["volatility"].idxmin())
    sharpe = (frontier_df["expected_return"] - rf) / frontier_df["volatility"].replace(0, np.nan)
    max_sharpe_idx = int(sharpe.idxmax()) if sharpe.notna().any() else min_vol_idx
    frontier_df["label"] = None
    frontier_df.loc[min_vol_idx, "label"] = "min_vol"
    frontier_df.loc[max_sharpe_idx, "label"] = "max_sharpe"

    port_ret = float(weights @ mu.values)
    port_vol = float(np.sqrt(weights.values @ cov.values @ weights.values))
    port_sharpe = (port_ret - rf) / port_vol if port_vol > 0 else float("nan")
    summary = {
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe": port_sharpe,
        "n_assets": int((weights.abs() > 1e-4).sum()),
    }
    return {"frontier": frontier_df, "weights": weights, "summary": summary}


def _fallback_frontier(mu: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
    inv = np.linalg.pinv(cov)
    ones = np.ones_like(mu)
    a = float(ones @ inv @ ones)
    b = float(ones @ inv @ mu)
    c = float(mu @ inv @ mu)
    d = a * c - b * b
    if d <= 0:
        d = 1e-9
    targets = np.linspace(mu.min(), mu.max(), 40)
    vols = np.sqrt((a * targets * targets - 2 * b * targets + c) / d)
    return pd.DataFrame({"volatility": vols, "expected_return": targets})


def _fallback_tangency(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    excess = mu - rf
    raw = np.linalg.pinv(cov) @ excess
    if raw.sum() == 0:
        return np.full_like(mu, 1.0 / len(mu))
    w = raw / raw.sum()
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(mu, 1.0 / len(mu))


# ---------------------------------------------------------------------------

st.title("Efficient frontier")
cfg = render_sidebar(key_prefix="frontier")

if not HAS_OPTIMIZER:
    st.warning(
        f"Library not fully installed ({_import_error}); showing analytic fallback only.",
        icon=":material/warning:",
    )

cfg_dict = asdict(cfg)
cfg_dict.pop("run_token", None)
current_hash = config_hash(cfg_dict)
should_run = cfg.run_token > 0 and st.session_state.get(SESSION_LAST_HASH) != (
    current_hash,
    cfg.run_token,
)

if cfg.run_token == 0 and SESSION_RESULTS not in st.session_state:
    st.info("Configure the universe and click **Run optimization** in the sidebar.")
elif should_run or SESSION_RESULTS not in st.session_state:
    with st.spinner("Optimizing portfolio..."):
        try:
            returns = _load_returns(cfg)
            result = _compute_frontier(returns, cfg_dict)
            st.session_state[SESSION_RESULTS] = result
            st.session_state[SESSION_LAST_HASH] = (current_hash, cfg.run_token)
        except Exception as exc:
            st.error(f"Optimization failed: {exc.__class__.__name__}: {exc}")
            st.stop()

result = st.session_state.get(SESSION_RESULTS)
if result is None:
    st.stop()

summary = result["summary"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected return", f"{summary['expected_return']:.2%}")
c2.metric("Volatility", f"{summary['volatility']:.2%}")
c3.metric("Sharpe", f"{summary['sharpe']:.2f}")
c4.metric("Active assets", f"{summary['n_assets']}")

left, right = st.columns([3, 2], gap="large")
with left:
    st.plotly_chart(frontier_figure(result["frontier"]), use_container_width=True)
with right:
    st.plotly_chart(weights_bar(result["weights"]), use_container_width=True)

with st.expander("Frontier data"):
    st.dataframe(result["frontier"], use_container_width=True, hide_index=True)
