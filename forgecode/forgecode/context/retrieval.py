"""Deterministic BM25-style lexical ranking without a remote embedding service."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())


def rank(repo: str | Path, query: str) -> list[tuple[Path, float]]:
    root = Path(repo)
    documents: list[tuple[Path, list[str]]] = []
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".ts", ".tsx", ".js", ".md", ".toml", ".json"}:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        try:
            documents.append((path, _tokens(path.name + " " + path.read_text(encoding="utf-8"))))
        except UnicodeDecodeError:
            continue
    query_terms = set(_tokens(query))
    document_frequency = Counter(term for term in query_terms for _, words in documents if term in words)
    scored: list[tuple[Path, float]] = []
    for path, words in documents:
        counts = Counter(words)
        score = sum(counts[t] * math.log((len(documents) + 1) / (document_frequency[t] + 0.5)) for t in query_terms)
        scored.append((path, score))
    return sorted(scored, key=lambda item: (-item[1], str(item[0])))

