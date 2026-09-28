"""Cached, resumable, spend-capped execution of judge variants over a case set.

Three properties this module exists to guarantee:

*   **Resumability.** Judging a few thousand cases across half a dozen variants is a long,
    paid, network-bound job that will be interrupted. Every judgment is written to a
    content-addressed cache the moment it returns, so a rerun costs nothing for work already
    done. The cache key covers the case content *and* the judge configuration *and* the rubric
    hash -- change a rubric and the old judgments correctly stop being reused, rather than
    being silently attributed to the new rubric.

*   **A hard spend cap.** Budget is *reserved* before each call and only converted to committed
    spend (or given back) once the call finishes, atomically under one lock -- so concurrent
    workers cannot all pass a "spent so far" check before any of them has recorded anything and
    collectively overshoot the cap. A model with no configured price is refused rather than
    treated as free: `aisys.llm.estimate_cost` returns `None` and warns for an unpriced model,
    and `None` accumulating as zero is how a budget silently becomes unbounded.

*   **No label access.** The runner takes `CaseRecord`s only. It never opens
    `annotations.jsonl`, so no code path exists by which a human label could reach a judge
    prompt. That is a structural guarantee, not a convention.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aisys.llm import _PRICING

from .dataset import CaseRecord, Judgment, content_hash
from .judge import JudgeConfig, judge_case
from .rubrics import for_task


class BudgetExceeded(RuntimeError):
    pass


class UnpricedModel(RuntimeError):
    """Raised rather than logging an unknown price as a measured zero."""


def _cache_key(case: CaseRecord, config: JudgeConfig) -> str:
    try:
        rubric_hash = for_task(case.task_class, config.rubric_dir).hash
    except Exception:  # noqa: BLE001 - a missing rubric is handled by the judge, not the cache
        rubric_hash = "no-rubric"
    return content_hash(case.fingerprint, config.config_hash, rubric_hash, case.task_class)


class _Reservation:
    """A handle for one in-flight reservation, returned by `SpendMeter.reserve()`.

    Use it as a context manager: `with meter.reserve(...) as slot: ... slot.record(cost, ...)`.
    That guarantees the reservation is released even if the protected call raises before
    `record()`/`release()` runs -- a failed call must give its budget back, not leak it.
    Calling `record()` or `release()` explicitly settles the reservation early; `__exit__` is
    then a no-op.
    """

    __slots__ = ("_meter", "_amount_usd", "_settled")

    def __init__(self, meter: "SpendMeter", amount_usd: float) -> None:
        self._meter = meter
        self._amount_usd = amount_usd
        self._settled = False

    def record(self, cost: float | None, *, model: str) -> None:
        """Commit the real cost and release whatever part of the reservation was unused.

        A `None` cost (unpriced model) is never recorded as zero: the reservation is released
        and `UnpricedModel` is raised, exactly as the old `SpendMeter.record()` did.
        """
        if self._settled:
            raise RuntimeError("reservation already recorded/released")
        if cost is None:
            self._settled = True
            self._meter._release(self._amount_usd)
            raise UnpricedModel(
                f"no price configured for {model!r} in aisys/pricing.yaml; refusing to record "
                "an unknown cost as $0.00. Add the model's per-1M rates before running."
            )
        self._settled = True
        self._meter._commit(self._amount_usd, cost)

    def release(self) -> None:
        """Give the reservation back unused -- for a failed/`provider_error`/undecided call."""
        if self._settled:
            return
        self._settled = True
        self._meter._release(self._amount_usd)

    def __enter__(self) -> "_Reservation":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._settled:
            self.release()


@dataclass
class SpendMeter:
    """Thread-safe budget accounting with atomic in-flight reservation.

    A pre-call check alone ("spent + reserve <= cap, so proceed") is not a hard cap under
    concurrency: with N worker threads, every one of them can pass that check before any of
    them has recorded a real cost, because `record()` used to run only *after* the call
    returned. Total spend could then overshoot the cap by up to N x reserve_usd. `reserve()`
    (aliased below by `admit()` for existing callers) closes that gap: the admission check and
    the increment of an in-flight `_reserved_usd` total happen under the SAME lock, so
    `spent_usd` (committed) plus `_reserved_usd` (in flight) never exceeds `cap_usd` at any
    instant a caller can observe.

    Reservation sizing assumption: `judge_case` (see `judge.py`) can make up to
    `JudgeConfig.max_attempts` provider calls for one judgment, retrying on a provider/parse
    error, and accumulates ALL of their cost onto that single `Judgment`. Reserving only one
    call's worth of budget would therefore under-reserve a judgment that retries, letting
    concurrent workers collectively punch through the cap on a retry-heavy run. `reserve()`
    takes `max_attempts` and scales the reservation by it -- callers pass the judge config's
    `max_attempts` -- so one judgment's reservation always covers its worst-case total cost,
    not just its first call.
    """

    cap_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    #: Worst-case cost of a SINGLE provider call. `reserve()` multiplies this by the caller's
    #: `max_attempts` to size the reservation for one whole (possibly retried) judgment.
    reserve_usd: float = 0.01
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    #: Total currently reserved for in-flight calls -- committed once real cost is known.
    _reserved_usd: float = field(default=0.0, repr=False)
    #: Count of outstanding (unsettled) reservations, for reporting only.
    _in_flight: int = field(default=0, repr=False)
    #: Per-thread stash so the old admit()-then-record() two-step still works (see `admit()`).
    _local: threading.local = field(default_factory=threading.local, repr=False)

    def reserve(self, max_attempts: int = 1) -> _Reservation:
        """Atomically reserve one judgment's worst-case cost, or raise `BudgetExceeded`."""
        amount = self.reserve_usd * max(1, max_attempts)
        with self._lock:
            if self.spent_usd + self._reserved_usd + amount > self.cap_usd:
                raise BudgetExceeded(
                    f"spend cap reached: ${self.spent_usd:.4f} committed + "
                    f"${self._reserved_usd:.4f} reserved of ${self.cap_usd:.2f} cap; the next "
                    f"judgment could reserve up to ${amount:.4f}"
                )
            self._reserved_usd += amount
            self._in_flight += 1
        return _Reservation(self, amount)

    def admit(self, max_attempts: int = 1) -> None:
        """Back-compat with the old admit()-then-record() calling convention.

        Reserves now and stashes the reservation on this thread; the next `record()` call made
        on the SAME thread settles it. New code should prefer `reserve()` used as a context
        manager, which does not depend on admit() and record() running on the same thread.
        """
        self._local.reservation = self.reserve(max_attempts=max_attempts)

    def record(self, cost: float | None, *, model: str) -> None:
        """Settle the reservation this thread's `admit()` created, if any.

        If nothing was reserved on this thread (a caller that never adopted `admit()`), record
        directly against the cap with no reservation accounting -- the pre-fix behaviour --
        so this remains a safe no-op change for any such caller.
        """
        reservation: _Reservation | None = getattr(self._local, "reservation", None)
        self._local.reservation = None
        if reservation is not None:
            reservation.record(cost, model=model)
            return
        if cost is None:
            raise UnpricedModel(
                f"no price configured for {model!r} in aisys/pricing.yaml; refusing to record "
                "an unknown cost as $0.00. Add the model's per-1M rates before running."
            )
        with self._lock:
            self.spent_usd += cost
            self.calls += 1

    def _commit(self, reserved_amount: float, cost: float) -> None:
        with self._lock:
            self._reserved_usd -= reserved_amount
            self._in_flight -= 1
            self.spent_usd += cost
            self.calls += 1

    def _release(self, reserved_amount: float) -> None:
        with self._lock:
            self._reserved_usd -= reserved_amount
            self._in_flight -= 1

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return self._reserved_usd

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def committed_usd(self) -> float:
        """Alias for `spent_usd`: cost that has actually been recorded, not merely reserved."""
        return self.spent_usd

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            reserved, in_flight = self._reserved_usd, self._in_flight
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 6),
            "calls": self.calls,
            "committed_usd": round(self.spent_usd, 6),
            "reserved_usd": round(reserved, 6),
            "in_flight": in_flight,
        }


def require_priced(models: Iterable[str]) -> None:
    """Fail before spending anything if any model's price is unknown."""
    missing = [m for m in dict.fromkeys(models) if m not in _PRICING]
    if missing:
        raise UnpricedModel(
            f"models {missing} have no entry in aisys/pricing.yaml. A run with an unpriced "
            "model cannot honour a spend cap, so it is refused up front."
        )


class JudgmentCache:
    """Append-only JSONL cache, loaded into memory on open."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: dict[str, Judgment] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        self._entries[raw["_key"]] = Judgment.model_validate(raw["judgment"])
                    except Exception:  # noqa: BLE001, S112 - a truncated tail must not lose the cache
                        continue

    def get(self, key: str) -> Judgment | None:
        return self._entries.get(key)

    def put(self, key: str, judgment: Judgment) -> None:
        with self._lock:
            self._entries[key] = judgment
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"_key": key, "judgment": judgment.model_dump(mode="json")},
                               sort_keys=True) + "\n"
                )

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class RunSummary:
    variant_id: str
    config: dict[str, Any]
    n_cases: int
    n_ok: int
    n_cached: int
    errors: dict[str, int]
    spend: dict[str, Any]
    wall_s: float
    models_served: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id, "config": self.config, "n_cases": self.n_cases,
            "n_ok": self.n_ok, "n_cached": self.n_cached, "errors": self.errors,
            "spend": self.spend, "wall_s": round(self.wall_s, 1),
            "models_served": self.models_served,
        }


def run_variant(
    cases: Sequence[CaseRecord],
    config: JudgeConfig,
    *,
    cache: JudgmentCache,
    meter: SpendMeter,
    concurrency: int = 8,
    progress: bool = True,
) -> tuple[list[Judgment], RunSummary]:
    """Judge every case under one configuration. Returns judgments in `cases` order."""
    require_priced([config.model])
    started = time.perf_counter()
    out: list[Judgment | None] = [None] * len(cases)
    cached = 0
    stopped = threading.Event()
    done = threading.Lock()
    counter = {"n": 0}

    def one(index_case: tuple[int, CaseRecord]) -> None:
        nonlocal cached
        index, case = index_case
        key = _cache_key(case, config)
        hit = cache.get(key)
        if hit is not None and hit.status == "ok":
            out[index] = hit
            cached += 1
            return
        if stopped.is_set():
            return
        try:
            # Reserve this judgment's worst-case cost up front (one call x max_attempts),
            # atomically against the cap -- see `SpendMeter.reserve()`. This is what makes the
            # cap hard under concurrency: nothing else can spend past the cap while this
            # judgment is in flight, even before its real cost is known.
            slot = meter.reserve(max_attempts=config.max_attempts)
        except BudgetExceeded:
            stopped.set()
            return
        with slot:
            judgment = judge_case(case, config)
            if judgment.status == "provider_error":
                # No usable cost was accrued (or it's already unrecoverable): give the
                # reservation back rather than let it sit against the cap for the rest of run.
                slot.release()
            else:
                try:
                    slot.record(judgment.cost_usd, model=config.model)
                except UnpricedModel:
                    stopped.set()
                    raise
        out[index] = judgment
        cache.put(key, judgment)
        if progress:
            with done:
                counter["n"] += 1
                if counter["n"] % 25 == 0:
                    sys.stderr.write(
                        f"\r  {config.variant_id}: {counter['n']} judged, "
                        f"${meter.spent_usd:.3f} spent"
                    )
                    sys.stderr.flush()

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(one, enumerate(cases)))
    if progress:
        sys.stderr.write("\r" + " " * 70 + "\r")

    judgments = [j for j in out if j is not None]
    errors: dict[str, int] = {}
    served: dict[str, int] = {}
    for j in judgments:
        if j.status != "ok":
            errors[j.status] = errors.get(j.status, 0) + 1
        if j.model_served:
            served[j.model_served] = served.get(j.model_served, 0) + 1
    if stopped.is_set():
        errors["budget_stopped"] = len(cases) - len(judgments)

    summary = RunSummary(
        variant_id=config.variant_id, config=config.as_dict(), n_cases=len(cases),
        n_ok=sum(1 for j in judgments if j.usable), n_cached=cached, errors=errors,
        spend=meter.as_dict(), wall_s=time.perf_counter() - started, models_served=served,
    )
    return judgments, summary
