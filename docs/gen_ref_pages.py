"""Generate API-reference pages from the ``src/markowitz`` package.

This script is invoked by the ``mkdocs-gen-files`` plugin. It walks the
package, emits a single ``.md`` per public module containing a
``::: dotted.path`` directive for mkdocstrings to render, and writes a
``SUMMARY.md`` for ``mkdocs-literate-nav``.

Robust by design: a partially-built library (broken imports, missing
modules) must not break ``mkdocs build``. Errors are caught per-file and
recorded as warnings inside the generated page.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

SRC_ROOT = Path("src")
PACKAGE = "markowitz"
API_DIR = Path("api-reference")

nav = mkdocs_gen_files.Nav()

package_root = SRC_ROOT / PACKAGE

if not package_root.is_dir():
    # Library not yet present; emit a placeholder so the site still builds.
    placeholder = API_DIR / "index.md"
    with mkdocs_gen_files.open(placeholder, "w") as fh:
        fh.write(
            "# API Reference\n\n"
            "The `markowitz` package has not been built yet. "
            "Run `uv sync` from a clone and rebuild the docs.\n"
        )
    nav["Overview"] = "index.md"
    with mkdocs_gen_files.open(API_DIR / "SUMMARY.md", "w") as fh:
        fh.writelines(nav.build_literate_nav())
else:
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).with_suffix("")
        parts = list(rel.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            if not parts:
                continue
        if parts[-1].startswith("_"):
            # Skip private modules.
            continue

        doc_path = API_DIR / Path(*parts).with_suffix(".md")
        nav_path = tuple(parts[1:]) if len(parts) > 1 else (parts[0],)
        nav[nav_path] = str(Path(*parts).with_suffix(".md")).replace("\\", "/")

        ident = ".".join(parts)
        try:
            with mkdocs_gen_files.open(doc_path, "w") as fh:
                fh.write(f"# `{ident}`\n\n")
                fh.write(f"::: {ident}\n")
            mkdocs_gen_files.set_edit_path(doc_path, path)
        except Exception as exc:
            with mkdocs_gen_files.open(doc_path, "w") as fh:
                fh.write(f"# `{ident}`\n\n")
                fh.write(
                    "!!! warning \"API reference unavailable\"\n"
                    f"    Could not render `{ident}`: `{exc!r}`.\n"
                )

    with mkdocs_gen_files.open(API_DIR / "SUMMARY.md", "w") as fh:
        fh.writelines(nav.build_literate_nav())
