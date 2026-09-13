"""Budgeted repository context pack."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import repo_map, retrieval
from .evict import estimate_tokens, fit


@dataclass(frozen=True)
class ContextPack:
    text: str
    files: list[str]
    tokens: int


def build(repo: str | Path, task: str, budget_tokens: int) -> ContextPack:
    root = Path(repo).resolve()
    mapping = repo_map.build(root)
    recent = subprocess.run(
        ["git", "log", "-3", "--oneline"], cwd=root, text=True, capture_output=True, check=False,
    ).stdout
    chunks = [f"REPO MAP\n{mapping}", f"RECENT GIT\n{recent}"]
    files: list[str] = []
    for path, _score in retrieval.rank(root, task):
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        chunks.append(f"FILE {relative}\n{content}")
        files.append(relative)
    selected = fit(chunks, budget_tokens)
    text = "\n\n".join(selected)
    return ContextPack(text=text, files=[f for f in files if f in text], tokens=estimate_tokens(text))


def build_context_pack(repo: str | Path, task: str, budget_tokens: int) -> str:
    return build(repo, task, budget_tokens).text
