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
from typing import Any, Self

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


def _legacy_cache_key(case: CaseRecord, config: JudgeConfig) -> str:
    """The cache key those judgments were written under, for IDENTIFICATION only.

    See `JudgeConfig.legacy_config_hash`. The legacy `config_hash` did not cover the prompt
    scaffolding, so a judgment found under this key cannot be shown to have been produced by the
    prompts now in the tree. `legacy_lookup` therefore returns it tagged, and `run_variant` never
    treats it as a hit.
    """
    try:
        rubric_hash = for_task(case.task_class, config.rubric_dir).hash
    except Exception:  # noqa: BLE001
        rubric_hash = "no-rubric"
    return content_hash(
        case.fingerprint, config.legacy_config_hash, rubric_hash, case.task_class
    )


#: What is known about the prompts behind a cached judgment.
#:
#: `current` -- the judgment's `config_hash` covers `prompt_template_hash`, so the prompts are
#: pinned and verified.
#: `unverified_legacy` -- found under the pre-2026-09-28 key, whose hash omitted the prompt
#: scaffolding entirely. The rubric content IS verified (every `rubric_version` in the cache
#: matches this tree), but `_ROLE` and the output-format blocks are not covered by anything.
#:
#: The rule, which exists because relabelling is the cheap way to fake a result: an
#: `unverified_legacy` judgment may be used for a clearly-labelled development probe and may be
#: counted as evidence that a re-run is or is not needed. It may never back a frozen calibration
#: bundle, and nothing may rewrite its provenance to `current`.
LEGACY_PROMPT_PROVENANCE = "unverified_legacy"
CURRENT_PROMPT_PROVENANCE = "current"


class _Reservation:
    """A handle for one in-flight reservation, returned by `SpendMeter.reserve()`.

    Use it as a context manager: `with meter.reserve(...) as slot: ... slot.record(cost, ...)`.
    That guarantees the reservation is released even if the protected call raises before
    `record()`/`release()` runs -- a failed call must give its budget back, not leak it.
    Calling `record()` or `release()` explicitly settles the reservation early; `__exit__` is
    then a no-op.
    """

    __slots__ = ("_amount_usd", "_meter", "_settled")

    def __init__(self, meter: SpendMeter, amount_usd: float) -> None:
        self._meter = meter
        self._amount_usd = amount_usd
        self._settled = False

    def record(self, cost: float | None, *, model: str) -> None:
        """Commit the real cost and release whatever part of the reservation was unused.

        A `None` cost (unpriced model) is never recorded as zero: the reservation is released
        and `UnpricedModel` is raised, exactly as the old `SpendMeter.record()` did.

        A cost that EXCEEDS its reservation is committed in full and recorded as a breach. It is
        already spent at the provider, so hiding it would make `spent_usd` a comfortable fiction;
        the cap's job from that point is to stop the next call, which `_commit` does because the
        overshoot lands in `spent_usd`.
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
        self._meter._commit(self._amount_usd, cost, model=model)

    def release(self, *, accrued_usd: float | None = None, model: str = "") -> None:
        """Settle a failed/undecided call, committing any cost it already burned.

        `accrued_usd` is the cost the attempt really incurred before failing. This exists because
        releasing the whole reservation and recording nothing -- the previous behaviour -- loses
        real money from the accounting: `judge_case` accumulates tokens and cost across every
        attempt onto the judgment, so a judgment that burns three attempts and then returns
        `provider_error` has spent at the provider whatever those attempts cost. A cap that does
        not see that spend is not a cap on spending; it is a cap on *successful* spending, which is
        the wrong quantity and the more forgiving one.
        """
        if self._settled:
            return
        self._settled = True
        if accrued_usd:
            self._meter._commit(self._amount_usd, float(accrued_usd), model=model)
        else:
            self._meter._release(self._amount_usd)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._settled:
            self.release()


def worst_case_call_usd(model: str, config: JudgeConfig, *, prompt_tokens: int | None = None) -> float:
    """A real upper bound on one provider call's cost for `config`, from the configured price.

    This replaces the flat `reserve_usd = 0.01` estimate as the reservation basis. The output side
    is bounded exactly -- `max_tokens` is the most the provider will emit. The input side is not
    knowable before the prompt is built, so it is bounded generously: `prompt_tokens` when the
    caller has counted them, otherwise `_PROMPT_TOKEN_CEILING`, which is well above any judge
    prompt this repository builds (the longest, a decompose judgment over an AggreFact document
    with boundary examples, is a few thousand tokens).

    Over-reserving is the safe direction. It can refuse a judgment that would in fact have fitted,
    which stops a run early and is visible; under-reserving lets real spend past the cap, which is
    invisible until the bill.

    Raises `UnpricedModel` rather than returning 0.0 for an unknown model, since a zero bound would
    make the cap unbounded -- the same failure `require_priced` exists to prevent.
    """
    price = _PRICING.get(model)
    if not price:
        raise UnpricedModel(
            f"no price configured for {model!r} in aisys/pricing.yaml; a spend cap cannot be "
            "enforced for an unpriced model."
        )
    prompt = _PROMPT_TOKEN_CEILING if prompt_tokens is None else max(0, prompt_tokens)
    return (
        prompt * float(price["input_per_1m"])
        + max(1, config.max_tokens) * float(price["output_per_1m"])
    ) / 1e6


#: Generous input-token ceiling for reservation sizing only. Never used to truncate anything.
_PROMPT_TOKEN_CEILING = 32_000


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
    #: Calls whose real cost exceeded their reservation, i.e. every case where the reservation was
    #: not the bound it claimed to be. Reported in `as_dict()` so a run cannot quietly rely on a
    #: cap it overshot.
    breaches: list[dict[str, Any]] = field(default_factory=list)
    #: True once committed spend has passed the cap. A run that ends with this set spent more than
    #: it was authorised to, and the report must say so.
    cap_exceeded: bool = False
    _changed: threading.Condition = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._changed = threading.Condition(self._lock)

    def reserve(
        self,
        max_attempts: int = 1,
        *,
        samples: int = 1,
        worst_case_usd: float | None = None,
        wait: bool = False,
    ) -> _Reservation:
        """Atomically reserve one judgment's worst-case cost, or raise `BudgetExceeded`.

        `samples` multiplies the reservation because `judge_case` draws k independent samples and
        retries EACH of them up to `max_attempts`, accumulating every call's cost onto one
        judgment. Reserving `max_attempts` alone under-reserves a `samples=3` variant by 3x --
        confirmed against the declared v10/v11 configs, which reserved 3 calls' worth of budget
        for a 9-call worst case.

        `worst_case_usd` is a real per-call bound derived from the model's price and token budget
        (`worst_case_call_usd`). Pass it. Without it the reservation falls back to `reserve_usd`,
        a flat $0.01 estimate that is not a bound at all: one 2,000-token gpt-4.1 judgment costs
        several cents, and a single call whose true cost exceeds its reservation walks straight
        through the cap.
        """
        per_call = self.reserve_usd if worst_case_usd is None else max(0.0, worst_case_usd)
        amount = per_call * max(1, max_attempts) * max(1, samples)
        with self._lock:
            # An outstanding conservative reservation is not exhausted spend. Wait for it to
            # settle before rejecting queued work, unless even zero reservations cannot fit.
            while (wait and self._in_flight and self.spent_usd + amount <= self.cap_usd
                   and self.spent_usd + self._reserved_usd + amount > self.cap_usd):
                self._changed.wait()
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

    def _commit(self, reserved_amount: float, cost: float, *, model: str = "") -> None:
        with self._lock:
            self._reserved_usd -= reserved_amount
            self._in_flight -= 1
            self.spent_usd += cost
            self.calls += 1
            if cost > reserved_amount + 1e-12:
                # The reservation was not a bound. Recorded rather than raised: the money is
                # already gone, and a run that dies here loses the accounting too. The cap still
                # binds the NEXT reservation, because the overshoot is in `spent_usd`.
                self.breaches.append({
                    "model": model, "reserved_usd": reserved_amount, "actual_usd": cost,
                    "over_usd": cost - reserved_amount,
                })
            if self.spent_usd > self.cap_usd + 1e-12:
                self.cap_exceeded = True
            self._changed.notify_all()

    def _release(self, reserved_amount: float) -> None:
        with self._lock:
            self._reserved_usd -= reserved_amount
            self._in_flight -= 1
            self._changed.notify_all()

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
            # Surfaced, not buried: a run that overshot its cap has to say so in the artifact the
            # experiment record keeps, or the cap is only a cap in the docstring.
            "cap_exceeded": self.cap_exceeded,
            "reservation_breaches": len(self.breaches),
            "worst_breach_usd": (round(max(b["over_usd"] for b in self.breaches), 6)
                                 if self.breaches else 0.0),
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

    def legacy_lookup(
        self, case: CaseRecord, config: JudgeConfig
    ) -> tuple[Judgment | None, str]:
        """Find a judgment for `case` under either key, and say which provenance it carries.

        Returns `(judgment, provenance)`. Checking the current key first matters: once a case has
        been re-judged under pinned prompts, the verified judgment must win over the legacy one
        rather than the lookup order deciding.

        This is the whole of "preserve reusable evidence where defensible". It makes the 2,968
        cached judgments findable and auditable without asserting they came from today's prompts —
        the claim the recovered legacy formula shows cannot be made either way.
        """
        current = self.get(_cache_key(case, config))
        if current is not None:
            return current, CURRENT_PROMPT_PROVENANCE
        return self.get(_legacy_cache_key(case, config)), LEGACY_PROMPT_PROVENANCE

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
    # Computed once: the reservation basis for every judgment in this run. `require_priced` above
    # has already refused an unpriced model, so this cannot fall back to a meaningless bound.
    per_call_bound = worst_case_call_usd(config.model, config)
    started = time.perf_counter()
    out: list[Judgment | None] = [None] * len(cases)
    cached = 0
    stopped = threading.Event()
    quota_stopped = threading.Event()
    done = threading.Lock()
    counter = {"n": 0}

    def one(index_case: tuple[int, CaseRecord]) -> None:
        nonlocal cached
        index, case = index_case
        key = _cache_key(case, config)
        hit = cache.get(key)
        if hit is not None and hit.usable:
            out[index] = hit
            cached += 1
            return
        if stopped.is_set():
            return
        try:
            # Reserve this judgment's worst-case cost up front, atomically against the cap -- see
            # `SpendMeter.reserve()`. This is what makes the cap hard under concurrency: nothing
            # else can spend past the cap while this judgment is in flight, even before its real
            # cost is known. The bound covers every call the judgment can make: k samples, each
            # retried up to `max_attempts`, sized from the model's configured price rather than
            # from a flat per-call guess.
            slot = meter.reserve(
                max_attempts=config.max_attempts,
                samples=config.samples,
                worst_case_usd=per_call_bound,
                wait=True,
            )
        except BudgetExceeded:
            stopped.set()
            return
        with slot:
            # A worker may have waited for a reservation while another found exhausted quota.
            if quota_stopped.is_set():
                return
            judgment = judge_case(case, config)
            if judgment.status == "quota_exhausted":
                quota_stopped.set()
                stopped.set()
            if judgment.status in {"provider_error", "quota_exhausted"}:
                # The attempts that failed still cost money: `judge_case` accumulates token and
                # cost accounting onto the judgment across every attempt. Committing that accrued
                # cost rather than releasing the whole reservation is the difference between a cap
                # on spending and a cap on successful spending.
                slot.release(accrued_usd=judgment.cost_usd, model=config.model)
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
        errors["quota_stopped" if quota_stopped.is_set() else "budget_stopped"] = (
            len(cases) - len(judgments))

    summary = RunSummary(
        variant_id=config.variant_id, config=config.as_dict(), n_cases=len(cases),
        n_ok=sum(1 for j in judgments if j.usable), n_cached=cached, errors=errors,
        spend=meter.as_dict(), wall_s=time.perf_counter() - started, models_served=served,
    )
    return judgments, summary
