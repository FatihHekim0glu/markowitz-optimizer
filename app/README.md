---
title: Markowitz Optimizer
emoji: ""
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.40.0
app_file: streamlit_app.py
pinned: false
license: mit
short_description: Mean-variance portfolio optimization dashboard.
---

# Markowitz Optimizer Dashboard

Mean-variance portfolio optimization with a multi-page Streamlit interface:

1. Efficient Frontier
2. Method Comparison
3. Backtest
4. Black-Litterman
5. Diagnostics

## Local development

From the repository root:

```bash
uv sync
uv run streamlit run app/streamlit_app.py
```

The library is imported as `from markowitz import ...` and is expected to be
installed in editable mode (handled by `uv sync`).

## Data

By default the app generates a deterministic synthetic return panel so the
dashboard works fully offline. Toggle "Load real data" in the sidebar to fetch
prices via `yfinance` (network required).

## Hugging Face Spaces

This folder is structured to deploy directly as a Streamlit Space. The
frontmatter above is consumed by HF Spaces; `requirements.txt` pins the runtime
dependencies.
