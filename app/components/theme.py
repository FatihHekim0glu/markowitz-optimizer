"""Plotly template registration and CSS injection for the dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# House palette
PALETTE = {
    "navy": "#1F3A5F",
    "amber": "#D4A017",
    "teal": "#3E8E7E",
    "slate_blue": "#5B7B9A",
    "grey": "#8A8F98",
    "bronze": "#8B6F47",
    "crimson": "#A8324C",
}

PALETTE_ORDER = [
    PALETTE["navy"],
    PALETTE["amber"],
    PALETTE["teal"],
    PALETTE["slate_blue"],
    PALETTE["bronze"],
    PALETTE["crimson"],
    PALETTE["grey"],
]


def register_plotly_template(name: str = "fund") -> None:
    """Register the 'fund' Plotly template and make it the default."""
    template = go.layout.Template(
        layout=go.Layout(
            font={"family": "Inter, system-ui, sans-serif", "size": 13, "color": "#1A1A1A"},
            title={
                "font": {
                    "family": "Inter, system-ui, sans-serif",
                    "size": 18,
                    "color": PALETTE["navy"],
                },
                "x": 0.0,
                "xanchor": "left",
            },
            colorway=PALETTE_ORDER,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin={"l": 56, "r": 24, "t": 56, "b": 48},
            xaxis={
                "showgrid": True,
                "gridcolor": "#ECEEF1",
                "zeroline": False,
                "ticks": "outside",
                "tickcolor": "#C8CCD2",
                "linecolor": "#C8CCD2",
                "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
            },
            yaxis={
                "showgrid": True,
                "gridcolor": "#ECEEF1",
                "zeroline": False,
                "ticks": "outside",
                "tickcolor": "#C8CCD2",
                "linecolor": "#C8CCD2",
                "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
            },
            legend={
                "bgcolor": "rgba(255,255,255,0.85)",
                "bordercolor": "#ECEEF1",
                "borderwidth": 1,
                "font": {"size": 11},
            },
            hoverlabel={
                "bgcolor": "#FFFFFF",
                "bordercolor": PALETTE["navy"],
                "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
            },
            colorscale={
                "sequential": [
                    [0.0, "#F5F6F8"],
                    [0.5, PALETTE["slate_blue"]],
                    [1.0, PALETTE["navy"]],
                ],
                "diverging": [
                    [0.0, PALETTE["crimson"]],
                    [0.5, "#F5F6F8"],
                    [1.0, PALETTE["navy"]],
                ],
            },
        )
    )
    pio.templates[name] = template
    pio.templates.default = name


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.stApp {
    font-feature-settings: "tnum" 1, "lnum" 1;
}

code, pre, .stCode, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums;
}

[data-testid="stMetricValue"] {
    color: #1F3A5F;
    font-weight: 600;
}

[data-testid="stHeader"], #MainMenu, footer, .stDeployButton {
    visibility: hidden;
    height: 0;
}

div[data-testid="stToolbar"] {
    display: none;
}

h1, h2, h3 {
    color: #1F3A5F;
    letter-spacing: -0.01em;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

div[data-testid="stSidebar"] {
    background-color: #F5F6F8;
    border-right: 1px solid #ECEEF1;
}
</style>
"""


def inject_css() -> None:
    """Inject the dashboard CSS once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)
