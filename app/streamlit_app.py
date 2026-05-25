"""Multipage Streamlit entrypoint for the Markowitz optimizer dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `components` importable when Streamlit runs this script directly.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from components.theme import inject_css, register_plotly_template  # noqa: E402

st.set_page_config(
    page_title="Markowitz Optimizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Mean-variance portfolio optimization dashboard."},
)

register_plotly_template()
inject_css()

st.title("Markowitz Optimizer")
st.caption(
    "Mean-variance portfolio construction, shrinkage estimators, "
    "Black-Litterman views, walk-forward backtests."
)

col_l, col_r = st.columns([3, 2], gap="large")
with col_l:
    st.markdown(
        """
This dashboard is organized as five pages, each accessible from the sidebar:

1. **Efficient frontier** — analytic + constrained frontier with key portfolios.
2. **Method comparison** — sample vs. shrinkage covariance and mean estimators.
3. **Backtest** — walk-forward portfolio simulation with turnover and costs.
4. **Black-Litterman** — combine equilibrium priors with user views.
5. **Diagnostics** — return distributions, correlations, condition numbers.

Configure the universe and estimators in the left sidebar, then press
**Run optimization** to populate the active page.
"""
    )

with col_r:
    st.info(
        "Synthetic returns are used by default so the demo works offline. "
        "Toggle **Load real data** in the sidebar to fetch prices via yfinance.",
        icon=":material/info:",
    )

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Default universe", "10 tickers")
m2.metric("Frequency", "Daily")
m3.metric("Estimators", "3 mean × 2 cov")
m4.metric("Backtest", "Walk-forward")

st.caption(
    "Library is loaded as `markowitz`. Pages degrade gracefully when parts of "
    "the library are not yet installed."
)
