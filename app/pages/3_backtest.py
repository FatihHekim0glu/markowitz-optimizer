"""Page 3 — Walk-forward backtest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from components.plots import drawdown, equity_curves  # noqa: E402
from components.sidebar import render_sidebar  # noqa: E402
from components.theme import inject_css, register_plotly_template  # noqa: E402

st.set_page_config(page_title="Backtest", layout="wide")
register_plotly_template()
inject_css()

try:
    from markowitz.backtest import WalkForwardBacktest  # type: ignore

    HAS_BACKTEST = True
    _import_error: str | None = None
except Exception as exc:
    HAS_BACKTEST = False
    _import_error = str(exc)


@st.cache_data(show_spinner=False)
def _synthetic_returns(
    tickers: tuple[str, ...], seed: int, n_days: int = 2000
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


@st.cache_data(show_spinner=False)
def _fallback_backtest(
    returns: pd.DataFrame,
    rebalance: str,
    lookback: int,
    long_only: bool,
    max_weight: float,
    cost_bps: float,
) -> dict:
    """Closed-form walk-forward MVO backtest used when the library is partial."""
    rebal_dates = returns.resample(rebalance).first().index
    # Snap to nearest available trading day in returns.index:
    idx_pos = returns.index.get_indexer(rebal_dates, method='nearest')
    schedule = list(returns.index[idx_pos])

    weights_hist: list[pd.Series] = []
    port_returns = pd.Series(0.0, index=returns.index)
    benchmark = returns.mean(axis=1)
    prev_w = np.full(returns.shape[1], 1.0 / returns.shape[1])
    turnover_total = 0.0

    last_w = prev_w.copy()
    for i in range(1, len(returns)):
        date = returns.index[i]
        if date in schedule:
            window = returns.iloc[max(0, i - lookback) : i]
            if len(window) >= 30:
                mu = window.mean().values
                cov = window.cov().values + 1e-6 * np.eye(returns.shape[1])
                raw = np.linalg.pinv(cov) @ mu
                if raw.sum() == 0:
                    w = prev_w
                else:
                    w = raw / raw.sum()
                if long_only:
                    w = np.clip(w, 0.0, None)
                w = np.minimum(w, max_weight)
                s = w.sum()
                w = w / s if s > 0 else prev_w
                turnover = float(np.sum(np.abs(w - last_w)))
                turnover_total += turnover
                port_returns.iloc[i] = float(last_w @ returns.iloc[i].values) - (
                    turnover * cost_bps * 1e-4
                )
                weights_hist.append(pd.Series(w, index=returns.columns, name=date))
                last_w = w
                continue
        port_returns.iloc[i] = float(last_w @ returns.iloc[i].values)

    equity = pd.DataFrame(
        {
            "Strategy": (1 + port_returns).cumprod(),
            "Equal-weight": (1 + benchmark).cumprod(),
        }
    )
    weights_df = pd.DataFrame(weights_hist) if weights_hist else pd.DataFrame()
    return {
        "equity": equity,
        "returns": port_returns,
        "weights": weights_df,
        "turnover_total": turnover_total,
    }


def _stats(returns: pd.Series, rf: float) -> dict:
    if returns.empty:
        return {}
    ann_ret = float((1 + returns.mean()) ** 252 - 1)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")
    equity = (1 + returns).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd),
    }


st.title("Backtest")
cfg = render_sidebar(key_prefix="backtest")

if not HAS_BACKTEST:
    st.info(
        f"Library backtest unavailable ({_import_error}); using built-in walk-forward.",
        icon=":material/info:",
    )

colA, colB, colC = st.columns(3)
with colA:
    lookback = st.slider("Lookback (days)", 60, 756, 252, step=21)
with colB:
    cost_bps = st.slider("Per-turn cost (bps)", 0, 50, 5)
with colC:
    st.metric("Rebalance", {"ME": "Monthly", "QE": "Quarterly", "YE": "Annual"}[cfg.rebalance])

run = st.button("Run backtest", type="primary", key="bt_run")

if not run and "bt_result" not in st.session_state:
    st.info("Press **Run backtest** to simulate.")
    st.stop()

if run:
    with st.spinner("Running walk-forward simulation..."):
        try:
            returns = _synthetic_returns(cfg.tickers, cfg.seed)
            result = _fallback_backtest(
                returns,
                rebalance=cfg.rebalance,
                lookback=lookback,
                long_only=cfg.long_only,
                max_weight=cfg.max_weight,
                cost_bps=float(cost_bps),
            )
            st.session_state["bt_result"] = result
            st.session_state["bt_cfg"] = {
                "rf": cfg.risk_free_rate,
                "lookback": lookback,
                "cost_bps": cost_bps,
            }
        except Exception as exc:
            st.error(f"Backtest failed: {exc.__class__.__name__}: {exc}")
            st.stop()

result = st.session_state["bt_result"]
stats = _stats(result["returns"], st.session_state["bt_cfg"]["rf"])

if not stats:
    st.warning("No statistics available — try a longer date range or different universe.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Annualized return", f"{stats['ann_return']:.2%}")
m2.metric("Annualized vol", f"{stats['ann_vol']:.2%}")
m3.metric("Sharpe", f"{stats['sharpe']:.2f}")
m4.metric("Max drawdown", f"{stats['max_drawdown']:.2%}")

st.plotly_chart(equity_curves(result["equity"]), use_container_width=True)
st.plotly_chart(drawdown(result["equity"]), use_container_width=True)

with st.expander("Weight history"):
    if not result["weights"].empty:
        st.dataframe(result["weights"].tail(20).style.format("{:.2%}"))
    else:
        st.caption("No rebalances recorded.")
