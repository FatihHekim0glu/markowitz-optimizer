"""Sidebar controls returning an immutable SidebarConfig."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import streamlit as st

DEFAULT_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "KO",
    "WMT",
)

Rebalance = Literal["ME", "QE", "YE"]
CovMethod = Literal["Sample", "LedoitWolf"]
MeanMethod = Literal["Sample", "JorionBayesStein", "CAPM"]


@dataclass(frozen=True)
class SidebarConfig:
    tickers: tuple[str, ...]
    start: date
    end: date
    rebalance: Rebalance
    cov_method: CovMethod
    mean_method: MeanMethod
    long_only: bool
    max_weight: float
    risk_free_rate: float
    use_real_data: bool = False
    seed: int = 7
    run_token: int = field(default=0, compare=False)


def _parse_tickers(text: str) -> tuple[str, ...]:
    raw = [t.strip().upper() for t in text.replace("\n", ",").split(",")]
    return tuple(t for t in raw if t)


def render_sidebar(*, key_prefix: str = "main") -> SidebarConfig:
    """Render sidebar controls and return a SidebarConfig.

    The Run button writes an incremented token into session state; pages should
    re-compute when `(config_hash, run_token)` changes.
    """
    with st.sidebar:
        st.markdown("### Universe")
        tickers_text = st.text_area(
            "Tickers",
            value=", ".join(DEFAULT_TICKERS),
            height=80,
            key=f"{key_prefix}_tickers",
            help="Comma- or newline-separated symbols.",
        )
        tickers = _parse_tickers(tickers_text) or DEFAULT_TICKERS

        st.markdown("### Sample window")
        today = date.today()
        default_start = today - timedelta(days=365 * 5)
        col_a, col_b = st.columns(2)
        with col_a:
            start = st.date_input("Start", value=default_start, key=f"{key_prefix}_start")
        with col_b:
            end = st.date_input("End", value=today, key=f"{key_prefix}_end")

        st.markdown("### Estimation")
        cov_method = st.selectbox(
            "Covariance estimator",
            options=("Sample", "LedoitWolf"),
            index=1,
            key=f"{key_prefix}_cov",
        )
        mean_method = st.selectbox(
            "Mean estimator",
            options=("Sample", "JorionBayesStein", "CAPM"),
            index=1,
            key=f"{key_prefix}_mean",
        )

        st.markdown("### Constraints")
        long_only = st.toggle("Long-only", value=True, key=f"{key_prefix}_long_only")
        max_weight = st.slider(
            "Max single weight",
            min_value=0.05,
            max_value=1.0,
            value=0.40,
            step=0.05,
            key=f"{key_prefix}_max_w",
        )
        risk_free_rate = st.number_input(
            "Risk-free rate (annual)",
            min_value=-0.05,
            max_value=0.25,
            value=0.04,
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_rf",
        )

        st.markdown("### Backtest")
        rebalance = st.selectbox(
            "Rebalance",
            options=("ME", "QE", "YE"),
            index=1,
            format_func=lambda x: {"ME": "Monthly", "QE": "Quarterly", "YE": "Annual"}[x],
            key=f"{key_prefix}_rebal",
        )

        st.markdown("### Data source")
        use_real_data = st.toggle(
            "Load real data (yfinance)",
            value=False,
            key=f"{key_prefix}_real",
            help="Off by default — synthetic returns keep the demo offline.",
        )

        st.divider()
        ready = len(tickers) >= 2 and start < end
        run_clicked = st.button(
            "Run optimization",
            type="primary",
            disabled=not ready,
            use_container_width=True,
            key=f"{key_prefix}_run",
        )
        if not ready:
            st.caption("Add at least two tickers and a valid date range.")

    token_key = f"{key_prefix}_run_token"
    if token_key not in st.session_state:
        st.session_state[token_key] = 0
    if run_clicked:
        st.session_state[token_key] = int(st.session_state[token_key]) + 1

    return SidebarConfig(
        tickers=tickers,
        start=start,
        end=end,
        rebalance=rebalance,
        cov_method=cov_method,
        mean_method=mean_method,
        long_only=bool(long_only),
        max_weight=float(max_weight),
        risk_free_rate=float(risk_free_rate),
        use_real_data=bool(use_real_data),
        run_token=int(st.session_state[token_key]),
    )
