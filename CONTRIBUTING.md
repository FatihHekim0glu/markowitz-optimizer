# Contributing to markowitz-optimizer

Thank you for considering a contribution. This project values reproducibility, numerical
correctness, and pedagogical clarity above raw feature count. The notes below describe how to
set up a development environment, where the test layers live, and the conventions we expect for
commits and pull requests.

## Development environment

We use [`uv`](https://docs.astral.sh/uv/) for dependency resolution and virtualenv management.

```bash
git clone https://github.com/FatihHekim0glu/markowitz-optimizer
cd markowitz-optimizer
uv sync --all-extras
uv run pre-commit install
uv run pytest -q
```

`uv sync --all-extras` installs the runtime dependencies plus every optional extra
(`data`, `viz`, `robust`, `app`, `test`, `docs`, `dev`). After this you should be able to run
the full test suite, build the docs, and launch the Streamlit app without further setup.

## Test layers

Tests are grouped under `tests/` and tagged with `pytest` markers. Run a single layer with
`uv run pytest -m <marker>`.

| Marker | Purpose | Typical runtime |
|--------|---------|-----------------|
| `unit` | Pure-function and small-class tests with no external dependencies. | < 5 s |
| `parity` | Cross-check against reference implementations such as PyPortfolioOpt and `sklearn`. | 30-60 s |
| `regression` | Pinned numerical snapshots, including the He-Litterman (1999) tables. | 1-2 min |
| `property` | Hypothesis-driven property tests on shape, sign, and identity invariants. | 30-60 s |
| `integration` | End-to-end backtest and pipeline tests crossing module boundaries. | 1-3 min |

Tests marked `network` or `slow` are excluded from the default CI run and must be opted into
explicitly via `uv run pytest -m "network or slow"`.

## Code style

- Formatting and linting are enforced by `ruff` (configured in `pyproject.toml`).
- Type checking is enforced by `mypy` in strict mode against `src/markowitz`.
- A spell check (`codespell`) and the no-AI-attribution guard
  (`tools/check_no_ai_attribution.py`) run as pre-commit hooks.

## Commit convention

We follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) so that
the changelog can be assembled automatically. Typical prefixes:

- `feat(optimizer): add CVaR objective`
- `fix(estimators): clamp Ledoit-Wolf shrinkage intensity to [0, 1]`
- `docs(theory): expand Merton scalars derivation`
- `refactor(backtest): extract turnover accounting`
- `test(parity): add fixtures for non-square panels`
- `chore(deps): bump cvxpy to 1.6`

Breaking changes append `!` to the type, e.g. `feat(api)!: drop Python 3.10 support`, and include
a `BREAKING CHANGE:` footer.

## Branch naming

Branches are short, kebab-case, and prefixed with a type that mirrors the commit prefix:

- `feat/cvar-objective`
- `fix/lw-shrinkage-bounds`
- `docs/black-litterman-derivation`

Hotfix branches target the latest release tag rather than `main`.

## Pull request workflow

1. Open a draft pull request as early as possible. The template under
   `.github/PULL_REQUEST_TEMPLATE.md` is required reading; fill in the summary, validation
   evidence, and citation sections.
2. Reference an issue with `Closes #N` whenever one exists.
3. CI must be green and coverage must remain at or above the 90% project floor before review.
4. Squash-merge is the default; the squash commit message must itself be Conventional.

## Reporting bugs and proposing features

Use the bug report and feature request templates under `.github/ISSUE_TEMPLATE/`. For
security-sensitive issues, follow `SECURITY.md` instead of opening a public issue.
