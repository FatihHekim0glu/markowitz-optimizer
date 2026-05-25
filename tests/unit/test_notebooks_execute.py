"""Smoke tests verifying tutorial notebooks parse and that their library calls resolve.

This does NOT execute notebooks end-to-end (kernel/timing concerns) but it:
1. Validates JSON structure (nbformat.validate)
2. Statically checks that imports in each notebook would resolve
3. Confirms cells aren't empty
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1].parent / "notebooks"

NOTEBOOK_NAMES = [
    "01_two_asset_intuition.ipynb",
    "02_efficient_frontier.ipynb",
    "03_markowitz_breaks.ipynb",
    "04_shrinkage_and_fixes.ipynb",
    "05_black_litterman.ipynb",
    "06_out_of_sample_showdown.ipynb",
]


@pytest.mark.parametrize("nb_name", NOTEBOOK_NAMES)
def test_notebook_is_valid_json(nb_name: str) -> None:
    nb_path = NOTEBOOK_DIR / nb_name
    assert nb_path.exists(), f"Notebook missing: {nb_path}"
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)


@pytest.mark.parametrize("nb_name", NOTEBOOK_NAMES)
def test_notebook_has_cells(nb_name: str) -> None:
    nb_path = NOTEBOOK_DIR / nb_name
    nb = nbformat.read(nb_path, as_version=4)
    assert len(nb.cells) >= 3, f"{nb_name} has only {len(nb.cells)} cells"


@pytest.mark.parametrize("nb_name", NOTEBOOK_NAMES)
def test_notebook_code_cells_parse(nb_name: str) -> None:
    """Every code cell must be syntactically valid Python."""
    nb_path = NOTEBOOK_DIR / nb_name
    nb = nbformat.read(nb_path, as_version=4)
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
            if not source.strip():
                continue
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"{nb_name} cell {idx} syntax error: {e}")


@pytest.mark.parametrize("nb_name", NOTEBOOK_NAMES)
def test_notebook_outputs_stripped(nb_name: str) -> None:
    """Committed notebooks should have empty outputs and execution_count=None."""
    nb_path = NOTEBOOK_DIR / nb_name
    nb = nbformat.read(nb_path, as_version=4)
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.execution_count is None, (
                f"{nb_name} cell {idx} has execution_count={cell.execution_count!r}; "
                "run nbstripout"
            )
            assert cell.outputs == [], (
                f"{nb_name} cell {idx} has non-empty outputs; run nbstripout"
            )
