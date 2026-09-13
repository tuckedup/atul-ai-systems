"""Lightweight import graph used as a context-ranking signal."""
from __future__ import annotations

import ast
from pathlib import Path


def imports(path: Path) -> set[str]:
    if path.suffix != ".py":
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def build(repo: str | Path) -> dict[str, list[str]]:
    root = Path(repo)
    graph: dict[str, list[str]] = {}
    for path in root.rglob("*.py"):
        if ".git" not in path.parts:
            try:
                graph[str(path.relative_to(root))] = sorted(imports(path))
            except (SyntaxError, UnicodeDecodeError):
                graph[str(path.relative_to(root))] = []
    return graph

