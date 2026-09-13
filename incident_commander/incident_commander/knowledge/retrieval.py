"""Runbook retrieval with authorization enforced before ranking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True)
class Runbook:
    title: str
    roles: list[str]
    content: str
    path: str


def load(directory: str | Path) -> list[Runbook]:
    books: list[Runbook] = []
    for path in Path(directory).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        match = re.match(r"---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            continue
        meta = yaml.safe_load(match.group(1))
        books.append(Runbook(meta["title"], meta["roles"], match.group(2).strip(), str(path)))
    return books


def retrieve(query: str, role: str, directory: str | Path, limit: int = 5) -> list[Runbook]:
    terms = set(re.findall(r"\w+", query.lower()))
    authorized = [book for book in load(directory) if role in book.roles]
    return sorted(
        authorized,
        key=lambda book: -len(terms & set(re.findall(r"\w+", (book.title + " " + book.content).lower()))),
    )[:limit]

