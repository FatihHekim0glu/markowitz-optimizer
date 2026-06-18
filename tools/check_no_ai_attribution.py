"""Repository guard: forbid any trace of AI-tool attribution in tracked files.

Usage:
    python tools/check_no_ai_attribution.py FILE [FILE ...]

Intended as both a pre-commit hook and a CI gate. Scans the provided files for
a fixed regex set of forbidden tokens (case-insensitive, plus the robot
emoji). Exits with status 1 on any match, printing ``path:line:token`` lines.

The script allowlists itself and ``codespell-ignore.txt`` to avoid false
positives caused by the token list living inside the repository.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Files that legitimately contain the forbidden tokens (they ARE the
# enforcement mechanism) and must therefore be skipped.
ALLOWLIST_NAMES: frozenset[str] = frozenset(
    {
        "check_no_ai_attribution.py",
        "codespell-ignore.txt",
        # The commit-message guard workflow spells out the forbidden tokens in
        # its own grep pattern, so it is the enforcement mechanism too.
        "no-ai-attribution.yml",
    }
)

# Extensions we are willing to scan. Anything else is silently skipped.
SCANNABLE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".md",
        ".rst",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".cfg",
        ".ini",
        ".json",
        ".html",
        ".css",
        ".js",
        ".sh",
        ".ps1",
        ".cff",
        ".bib",
    }
)

# Patterns are joined with ``|`` into a single case-insensitive regex.
_RAW_PATTERNS: tuple[str, ...] = (
    r"\bclaude\b",
    r"\banthropic\b",
    r"\bcopilot\b",
    r"\bchatgpt\b",
    r"\bopenai\b",
    r"\bgpt[-\s]?[0-9]",
    r"\bcursor\.(?:so|sh|com)\b",
    r"\bcodeium\b",
    r"\btabnine\b",
    r"\bsonnet\b",
    r"\bhaiku\b",
    r"\bopus\b",
    r"claude\.ai",
    r"anthropic\.com",
    r"co-authored-by:\s*claude",
    r"generated\s+(?:by|with)\s+(?:claude|anthropic|gpt|chatgpt|copilot)",
    r"\U0001F916",  # robot face emoji
)

_PATTERN: re.Pattern[str] = re.compile("|".join(_RAW_PATTERNS), re.IGNORECASE)


def _should_skip(path: Path) -> bool:
    """Return True if the file is allowlisted or has an unscannable suffix."""
    if path.name in ALLOWLIST_NAMES:
        return True
    return path.suffix.lower() not in SCANNABLE_SUFFIXES


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Scan one file. Return a list of ``(line_number, matched_token)``."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    findings: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _PATTERN.finditer(line):
            findings.append((lineno, match.group(0)))
    return findings


def main(argv: list[str]) -> int:
    """Entry point. Print findings, return 1 if anything was found else 0."""
    if not argv:
        return 0

    total_findings = 0
    for raw in argv:
        path = Path(raw)
        if not path.is_file() or _should_skip(path):
            continue
        for lineno, token in _scan_file(path):
            sys.stdout.write(f"{path.as_posix()}:{lineno}:{token}\n")
            total_findings += 1

    if total_findings:
        sys.stderr.write(
            f"\nno-ai-attribution: {total_findings} forbidden token(s) found.\n"
            "Remove all references to AI tools/vendors from the listed lines.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
