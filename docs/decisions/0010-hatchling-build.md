# ADR-0010: Hatchling + hatch-vcs as build backend

* Status: Accepted
* Date: 2026-05-23

## Context

A modern Python project needs a PEP 517/518 build backend. Options include setuptools, hatchling, flit, poetry, pdm.

## Decision

**Hatchling + hatch-vcs.** Hatchling is the build backend; hatch-vcs derives the package version from git tags automatically.

## Decision drivers

- Modern: hatchling is the PyPA-recommended default for new projects in 2025+.
- Git-tag versioning: hatch-vcs writes `_version.py` at build time from the latest `vX.Y.Z` tag, removing manual version bumping.
- Clean configuration: pure-TOML, no `setup.py` shim.
- Compatible with PyPI OIDC trusted publishing.

## Considered options

- Option A: Hatchling + hatch-vcs. **Chosen.**
- Option B: Setuptools + setuptools_scm. Equivalent functionality but more historical baggage.
- Option C: Poetry. Rejected: opinionated lockfile and dependency resolver that conflicts with uv.
- Option D: PDM. Rejected: smaller community than hatchling.

## Consequences

- Tagging `v0.1.0` produces wheel/sdist with version `0.1.0`.
- Untagged commits get `0.1.0.dev<N>+g<sha>` (PEP 440 dev release).
- `src/markowitz/_version.py` is gitignored (auto-generated).
- Users read the version via `importlib.metadata.version("markowitz-optimizer")` exposed as `markowitz.__version__`.

## Links

- hatch: https://hatch.pypa.io/
- hatch-vcs: https://github.com/ofek/hatch-vcs
