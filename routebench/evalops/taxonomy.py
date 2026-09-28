"""One canonical task taxonomy, shared by datasets, classifier, router and quality storage.

Before this module the eval suites tagged cases `coding`/`extraction`/`summarization` while
`gateway/router.py` looked up `code`/`extract`/`summarize`. Every lookup missed, so the router
silently substituted its `0.5` default for *measured* quality. Routing was therefore not
data-driven in the way README claimed. Canonical names live here; legacy spellings are accepted
only through `canonical()`, which fails loudly on anything unknown.
"""
from __future__ import annotations

from typing import Final

# Canonical names == the TaskClass literals consumed by gateway/router.py.
CANONICAL: Final[tuple[str, ...]] = (
    "code", "extract", "sql", "summarize", "reason", "tool_use", "classify", "chat",
)

#: Task classes that carry a measured quality score in the routing matrix.
MEASURED: Final[tuple[str, ...]] = ("code", "extract", "sql", "summarize", "reason", "tool_use")

_ALIASES: Final[dict[str, str]] = {
    "coding": "code", "code_generation": "code", "codegen": "code",
    "extraction": "extract", "json": "extract", "structured": "extract",
    "text2sql": "sql", "text_to_sql": "sql", "nl2sql": "sql",
    "summarization": "summarize", "summary": "summarize", "groundedness": "summarize",
    "reasoning": "reason", "math": "reason", "arithmetic": "reason",
    "tool-use": "tool_use", "tooluse": "tool_use", "function_call": "tool_use",
    "function-calling": "tool_use", "tools": "tool_use",
    "classification": "classify",
}


class UnknownTaskClass(ValueError):
    """Raised instead of silently mapping an unknown tag onto a default quality."""


def canonical(name: str) -> str:
    """Normalise a task label. Unknown labels raise rather than defaulting to `chat`.

    A silent default is exactly how a stale tag becomes an invented `0.5` quality score.
    """
    key = str(name).strip().lower().replace(" ", "_")
    if key in CANONICAL:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise UnknownTaskClass(
        f"unknown task class {name!r}; canonical={list(CANONICAL)} aliases={sorted(_ALIASES)}"
    )


def is_measured(name: str) -> bool:
    return canonical(name) in MEASURED
