"""Language-neutral symbol map for compact repository orientation."""
from __future__ import annotations

import ast
import re
from pathlib import Path


def symbols(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        tree = ast.parse(text)
        return [node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    return re.findall(r"(?:function|class|interface|const)\s+([A-Za-z_$][\w$]*)", text)


def build(repo: str | Path) -> str:
    root = Path(repo)
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".py", ".ts", ".tsx", ".js"} or any(p.startswith(".") for p in path.parts):
            continue
        try:
            names = symbols(path)
        except (SyntaxError, UnicodeDecodeError):
            names = []
        lines.append(f"{path.relative_to(root)}: {', '.join(names)}")
    return "\n".join(lines)

