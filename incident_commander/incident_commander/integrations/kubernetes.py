"""Approved Kubernetes actions executed only with an explicit kube context and cwd."""
import subprocess
from pathlib import Path


def execute(action: str, resource: str, context: str, cwd: str | Path) -> subprocess.CompletedProcess[str]:
    if action not in {"rollout", "restart", "scale"}:
        raise ValueError("unsupported Kubernetes action")
    return subprocess.run(
        ["kubectl", "--context", context, action, resource], cwd=Path(cwd).resolve(),
        text=True, capture_output=True, timeout=120, check=False,
    )

