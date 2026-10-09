"""External watchdog: enforces the wall-clock caps on a child process from OUTSIDE, so a backend that never calls `tick` cannot hang a run.

The child announces phase starts and ends through a small progress file (`ProgressFile`, written atomically). The parent measures
every deadline on ITS OWN clock from the moment it first observes the announcement, so a stuck or lying child cannot extend its budget.
Enforced limits:
  * startup    - from launch until the first phase starts: `startup_seconds` (+ grace)
  * per phase  - `phase_seconds[phase]` (+ grace) while a phase is announced as running
  * total      - `total_seconds` from launch (+ grace), always
The grace is deliberately small and later than the cooperative in-process caps, so a well-behaved backend is stopped cleanly first and
the kill is the backstop. On Windows the whole process tree is killed (`taskkill /T`), because a GPU worker may have children.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .config import PilotConfig


class ProgressFile:
    """Atomic phase announcements: {"seq", "phase", "event"} with a strictly increasing sequence number.

    Best-effort by design: on Windows the watchdog may hold the file open at the instant of the replace, which raises
    PermissionError. Reporting retries briefly and, if it still cannot write, counts the failure and carries on; a progress-file
    problem must never crash the run. (If reports are lost, the watchdog still enforces the startup and total caps.)
    """

    RETRIES = 40
    RETRY_SLEEP = 0.025

    def __init__(self, path: Path) -> None:
        self.path, self.seq, self.failures = path, 0, 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, phase: str, event: str) -> None:
        self.seq += 1
        tmp = self.path.with_suffix(".tmp")
        payload = json.dumps({"seq": self.seq, "phase": phase, "event": event})
        for attempt in range(self.RETRIES):
            try:
                tmp.write_text(payload, encoding="utf8")
                os.replace(tmp, self.path)
                return
            except OSError:
                time.sleep(self.RETRY_SLEEP * (1 if attempt < 10 else 2))
        self.failures += 1


def read_progress(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None


def kill_tree(proc: subprocess.Popen[Any]) -> None:
    """Kill the process and its children."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:  # pragma: no cover - exercised on POSIX only
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.kill()


def supervise(argv: list[str], *, cfg: PilotConfig, progress_path: Path, env: Mapping[str, str] | None = None,
              log_path: Path | None = None, poll: float = 0.25, clock: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep,
              killer: Callable[[subprocess.Popen[Any]], None] = kill_tree, cwd: Path | None = None) -> dict[str, Any]:
    """Run `argv` under the caps. Returns {'exit_code', 'killed', 'reason', 'phase', 'elapsed_seconds'}."""
    grace = cfg.watchdog_grace_seconds
    log = log_path.open("wb") if log_path else subprocess.DEVNULL
    started = clock()
    proc = subprocess.Popen(argv, env=dict(env) if env is not None else None, stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(cwd) if cwd else None)
    phase: str | None = None
    phase_started = 0.0
    saw_start = False
    last_seq = -1
    try:
        while True:
            rc = proc.poll()
            now = clock()
            prog = read_progress(progress_path)
            if prog and prog.get("seq", -1) != last_seq:
                last_seq = prog["seq"]
                if prog.get("event") == "start":
                    phase, phase_started, saw_start = prog.get("phase"), now, True
                elif prog.get("event") == "end":
                    phase = None
            if rc is not None:
                return {"exit_code": rc, "killed": False, "reason": None, "phase": phase, "elapsed_seconds": now - started}
            reason = None
            if now - started > cfg.total_seconds + grace:
                reason = f"total cap of {cfg.total_seconds} s exceeded"
            elif not saw_start and now - started > cfg.startup_seconds + grace:
                reason = f"no phase started within the startup cap of {cfg.startup_seconds} s"
            elif phase is not None and now - phase_started > cfg.phase_seconds.get(phase, 0) + grace:
                reason = f"phase {phase!r} exceeded its {cfg.phase_seconds.get(phase)} s cap"
            if reason:
                killer(proc)
                return {"exit_code": proc.returncode, "killed": True, "reason": reason, "phase": phase,
                        "elapsed_seconds": clock() - started}
            sleep(poll)
    finally:
        if proc.poll() is None:  # pragma: no cover - only if supervise itself is interrupted
            killer(proc)
        if log_path and hasattr(log, "close"):
            log.close()
