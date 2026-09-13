"""ForgeCode tools, all scoped to an explicit task copy."""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

from aisys.tools import registry, sensitive_path, tool

from ..sandbox import SubprocessSandbox


def _root(repo: str) -> Path:
    root = Path(repo).resolve()
    marker = Path.cwd().resolve() / ".local" / "forgecode" / "tasks"
    if marker not in root.parents:
        raise ValueError("ForgeCode tools accept only a task fixture copy")
    return root


def _path(repo: str, path: str) -> Path:
    root = _root(repo)
    candidate = (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes the task copy")
    return candidate


@tool(risk="low")
def read_file(repo: str, path: str) -> str:
    """Read a UTF-8 file from the task copy."""
    return _path(repo, path).read_text(encoding="utf-8")


@tool(risk="medium", risk_predicate=sensitive_path)
def write_file(repo: str, path: str, content: str) -> str:
    """Write a UTF-8 file within the task copy."""
    target = _path(repo, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target.relative_to(_root(repo)))


@tool(risk="low")
def search_code(repo: str, query: str) -> list[str]:
    """Return matching source lines from the task copy."""
    matches: list[str] = []
    for path in _root(repo).rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(_root(repo)).parts):
            continue
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if query.lower() in line.lower():
                    matches.append(f"{path.relative_to(_root(repo))}:{number}:{line.strip()}")
        except UnicodeDecodeError:
            continue
    return matches[:200]


@tool(risk="low")
def git_diff(repo: str) -> str:
    """Return the task copy's current diff."""
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff"], cwd=_root(repo), text=True, capture_output=True, check=False,
    )
    return result.stdout


@tool(risk="low")
def git_history(repo: str, limit: int = 10) -> str:
    """Return recent commits from the task copy."""
    result = subprocess.run(
        ["git", "log", f"-{max(1, min(limit, 100))}", "--oneline"],
        cwd=_root(repo), text=True, capture_output=True, check=False,
    )
    return result.stdout


@tool(risk="low")
def list_symbols(repo: str, path: str) -> list[str]:
    """List top-level Python symbols in a source file."""
    tree = ast.parse(_path(repo, path).read_text(encoding="utf-8"))
    return [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]


@tool(risk="medium")
def run_terminal(repo: str, command: str, timeout_s: float = 60.0, memory_mb: int = 512) -> dict[str, Any]:
    """Run a shell command with limits and cwd pinned to the task copy."""
    task_root = _root(repo)
    sandbox = object.__new__(SubprocessSandbox)
    sandbox.fixture = task_root
    sandbox.workspace_root = Path.cwd().resolve()
    sandbox.timeout_s = timeout_s
    sandbox.memory_bytes = memory_mb * 1024 * 1024
    sandbox.workdir = task_root
    result = sandbox.run(command)
    return result.__dict__


@tool(risk="medium")
def run_tests(repo: str, command: str = "python -m pytest -q") -> dict[str, Any]:
    """Run the fixture's test command under the subprocess sandbox."""
    return run_terminal(repo, command)


@tool(risk="medium")
def run_linter(repo: str, command: str = "python -m ruff check .") -> dict[str, Any]:
    """Run the fixture's linter under the subprocess sandbox."""
    return run_terminal(repo, command)


@tool(risk="high")
def create_patch(repo: str, message: str) -> str:
    """Materialize the reviewed git diff as a patch file in the task copy."""
    del message
    patch = git_diff(repo)
    target = _root(repo) / "forgecode.patch"
    target.write_text(patch, encoding="utf-8")
    return str(target)


__all__ = ["registry"]
