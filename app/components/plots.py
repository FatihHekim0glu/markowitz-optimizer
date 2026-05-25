"""Plotly figure builders. Each takes a DataFrame and returns go.Figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import PALETTE


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color=PALETTE["grey"], size=13),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def frontier_figure(frontier_df: pd.DataFrame) -> go.Figure:
    """Risk-return scatter of the efficient frontier.

    Expects columns: 'volatility', 'expected_return'. Optional: 'sharpe',
    'label' (e.g. 'min_vol', 'max_sharpe', 'tangency').
    """
    if frontier_df is None or frontier_df.empty:
        return _empty("Run the optimizer to plot the frontier.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["expected_return"],
            mode="lines+markers",
            line=dict(color=PALETTE["navy"], width=2.5),
            marker=dict(size=5, color=PALETTE["navy"]),
            name="Efficient frontier",
            hovertemplate="vol=%{x:.3%}<br>ret=%{y:.3%}<extra></extra>",
        )
    )

    if "label" in frontier_df.columns:
        marks = frontier_df.dropna(subset=["label"])
        for _, row in marks.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["volatility"]],
                    y=[row["expected_return"]],
                    mode="markers+text",
                    text=[str(row["label"])],
                    textposition="top center",
                    marker=dict(size=11, color=PALETTE["amber"], symbol="diamond"),
                    showlegend=False,
                    hovertemplate=f"{row['label']}<br>vol=%{{x:.3%}}<br>ret=%{{y:.3%}}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Efficient frontier",
        xaxis=dict(title="Volatility (annualized)", tickformat=".1%"),
        yaxis=dict(title="Expected return (annualized)", tickformat=".1%"),
        height=460,
    )
    return fig


def weights_bar(weights: pd.Series | pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of portfolio weights."""
    if weights is None or len(weights) == 0:
        return _empty("No weights to display.")

    if isinstance(weights, pd.DataFrame):
        series = weights.iloc[:, 0]
    else:
        series = weights

    series = series.dropna().sort_values(ascending=True)
    if top_n is not None and len(series) > top_n:
        series = series.tail(top_n)

    colors = [
        PALETTE["navy"] if v >= 0 else PALETTE["crimson"] for v in series.values
    ]
    fig = go.Figure(
        go.Bar(
            x=series.values,
            y=series.index.astype(str),
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Portfolio weights",
        xaxis=dict(title="Weight", tickformat=".0%"),
        yaxis=dict(title=""),
        height=max(280, 22 * len(series) + 120),
    )
    return fig


def equity_curves(equity_df: pd.DataFrame) -> go.Figure:
    """Cumulative growth-of-1 lines for one or more strategies."""
    if equity_df is None or equity_df.empty:
        return _empty("No equity series available.")

    fig = go.Figure()
    for col in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df[col],
                mode="lines",
                name=str(col),
                line=dict(width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra>" + str(col) + "</extra>",
            )
        )
    fig.update_layout(
        title="Equity curves (growth of 1)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Value", tickformat=".2f"),
        height=420,
        hovermode="x unified",
    )
    return fig


def drawdown(equity_df: pd.DataFrame) -> go.Figure:
    """Drawdown chart from equity curves."""
    if equity_df is None or equity_df.empty:
        return _empty("No equity series available.")

    rolling_max = equity_df.cummax()
    dd = equity_df / rolling_max - 1.0

    fig = go.Figure()
    for col in dd.columns:
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd[col],
                mode="lines",
                name=str(col),
                fill="tozeroy",
                line=dict(width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra>" + str(col) + "</extra>",
            )
        )
    fig.update_layout(
        title="Drawdown",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Drawdown", tickformat=".0%"),
        height=320,
        hovermode="x unified",
    )
    return fig


def corr_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap."""
    if corr_df is None or corr_df.empty:
        return _empty("No correlation matrix available.")

    z = np.asarray(corr_df.values, dtype=float)
    labels = [str(c) for c in corr_df.columns]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, PALETTE["crimson"]],
                [0.5, "#F5F6F8"],
                [1.0, PALETTE["navy"]],
            ],
            colorbar=dict(title="ρ", tickformat=".1f"),
            hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Correlation",
        height=max(360, 26 * len(labels) + 120),
        xaxis=dict(tickangle=-45),
    )
    return fig
