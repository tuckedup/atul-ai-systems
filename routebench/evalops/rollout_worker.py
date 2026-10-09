"""Observation worker for the rollout controller: ingest, validate, tick, and enforce the canary deadline even when telemetry is dead.

`RolloutController` is caller-driven. This worker is the caller. Each `step()`:
  1. asks the injected `source()` for new observations (a callable returning mappings or `OnlineObservation`s; it is the operator's independently measured evidence, never invented here);
  2. feeds each to `controller.observe`, skipping already-recorded ids and QUARANTINING (recording, never crashing on) invalid ones;
  3. calls `controller.tick()`, which advances a passing window, rolls back a regression, and enforces the canary deadline.
A source that raises is counted and the step still proceeds to `tick()`, so a broken or silent telemetry feed cannot keep a canary alive: with no observations the deadline rolls it back.
There is no provider call here; the 90-second rollback guarantee is only as good as the interval at which `step()` is called and is not a measured production figure.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .rollout_controller import OnlineObservation, RolloutController

TERMINAL = frozenset({"completed", "rolled-back"})


class RolloutWorker:
    def __init__(
        self,
        controller: RolloutController,
        source: Callable[[], Iterable[OnlineObservation | Mapping[str, Any]]],
        *,
        interval_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not interval_s > 0:
            raise ValueError("interval_s must be positive")
        self.controller, self.source, self.interval_s, self._sleep = controller, source, interval_s, sleep
        self.ingested = 0
        self.duplicates = 0
        self.rejected: list[dict[str, str]] = []
        self.source_errors = 0
        self.last_source_error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def step(self) -> dict[str, Any]:
        try:
            items = list(self.source())
        except Exception as exc:  # noqa: BLE001 - dead telemetry must not stop the deadline from being enforced
            items = []
            self.source_errors += 1
            self.last_source_error = f"{type(exc).__name__}: {exc}"
        for item in items:
            try:
                self.controller.observe(item)
                self.ingested += 1
            except ValueError as exc:
                if "duplicate observation_id" in str(exc):
                    self.duplicates += 1
                else:
                    self.rejected.append({"reason": str(exc), "item": repr(item)[:200]})
            except RuntimeError:
                break  # the rollout is no longer accepting observations (terminal)
        return self.controller.tick()

    def run(self, *, max_steps: int | None = None) -> dict[str, Any]:
        """Step until the rollout reaches a terminal state, `stop()` is called, or `max_steps` is reached."""
        steps = 0
        status = self.controller.status()
        while not self._stop.is_set():
            status = self.step()
            steps += 1
            if status["status"] in TERMINAL or (max_steps is not None and steps >= max_steps):
                break
            self._sleep(self.interval_s)
        return status | {"worker": self.summary(steps)}

    def summary(self, steps: int | None = None) -> dict[str, Any]:
        return {"steps": steps, "ingested": self.ingested, "duplicates_skipped": self.duplicates, "rejected": list(self.rejected),
                "source_errors": self.source_errors, "last_source_error": self.last_source_error}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("worker already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="rollout-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
