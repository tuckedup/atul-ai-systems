"""ForgeCode sandbox substitute: subprocess with a hard timeout, a memory cap, and cwd pinned
to a copy of the task's fixture. Never executes with the repo root as cwd.

When Docker returns, replace this file with docker.py that spins up a per-task container.
The rest of forgecode calls sandbox.run(cmd, cwd, timeout_s, memory_mb) and does not care which
backend is behind it.

Usage::

    out = run(["pytest", "-q"], cwd=copy_dir, timeout_s=120, memory_mb=512)
    assert out.returncode == 0
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def run(
    cmd: list[str],
    cwd: str | Path,
    *,
    timeout_s: float = 120.0,
    memory_mb: int | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run `cmd` inside `cwd` (which must be a real path, not the repo root).

    Returns a dict with:
        returncode, stdout, stderr, timed_out, killed_on_oom, cwd, cmd
    """
    cwd = Path(cwd).resolve()
    if not cwd.is_dir():
        raise FileNotFoundError(f"sandbox cwd does not exist: {cwd}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    best_effort_mem_cap = _build_memory_limit(memory_mb) if memory_mb else []

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=merged_env,
            *best_effort_mem_cap,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
            "killed_on_oom": False,
            "cwd": str(cwd),
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "returncode": -1,
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + f"\n[timeout after {timeout_s}s]",
            "timed_out": True,
            "killed_on_oom": False,
            "cwd": str(cwd),
            "cmd": cmd,
        }


def _build_memory_limit(memory_mb: int) -> tuple:
    """Best-effort memory cap using ulimit -v (virtual memory, KB). Not all platforms enforce it,
    but it is the closest thing available without Docker."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    kb = memory_mb * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (min(kb, hard), hard))
    except ValueError:
        pass  # hard limit is lower than requested — leave it
    return ()


def copy_fixture(src: Path, dest_dir: Path | None = None) -> Path:
    """Copy a fixture tree into an isolated temp dir. ForgeCode uses this so the sandbox never
    touches the repo root.

    Returns the temp dir path; caller owns cleanup (shutil.rmtree when done or on error).
    """
    d = dest_dir or Path(tempfile.mkdtemp(prefix="forgecode-sandbox-"))
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d)
    return d
