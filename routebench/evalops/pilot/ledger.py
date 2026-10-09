"""Cumulative local-GPU compute ledger with a hard ceiling.

Every GPU-using step appends one JSON line (wall-clock seconds, purpose, verdict). `execute` refuses to start when the ledger total plus
the run's own total cap would exceed the authorised ceiling, so the allowance cannot be exceeded by accumulation. The ledger records API
spend too (always 0 here) so a non-zero value is visible. Wall-clock seconds are an upper bound on GPU-seconds.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CEILING_SECONDS = 40 * 3600  # the authorised total, all steps included
LEDGER_NAME = "compute_ledger.jsonl"


class LedgerCeilingExceeded(RuntimeError):
    pass


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def total_seconds(path: Path) -> float:
    return float(sum(e["wall_seconds"] for e in read(path)))


def total_api_usd(path: Path) -> float:
    return float(sum(e.get("api_usd", 0.0) for e in read(path)))


def check_budget(path: Path, planned_seconds: float, ceiling: float = CEILING_SECONDS) -> None:
    """Refuse when spent + planned > ceiling, or when any API spend has been recorded."""
    if total_api_usd(path) > 0:
        raise LedgerCeilingExceeded("API spend has been recorded; the authorisation is $0")
    spent = total_seconds(path)
    if spent + planned_seconds > ceiling:
        raise LedgerCeilingExceeded(
            f"ledger has {spent / 3600:.2f} h; a further {planned_seconds / 3600:.2f} h would exceed the {ceiling / 3600:.0f} h ceiling")


def append(path: Path, *, kind: str, wall_seconds: float, verdict: str = "", note: str = "",
           extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "wall_seconds": round(float(wall_seconds), 2),
             "api_usd": 0.0, "verdict": verdict, "note": note, **dict(extra or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry
