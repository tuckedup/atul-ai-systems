"""Eval-gated shadow -> canary -> full rollout with rollback, behind a calibration gate.

The promotion ladder itself was already sound: each stage measures online quality and rolls
back when it drops more than `max_drop` below baseline. What was missing is the question one
level up -- *are the quality numbers driving those decisions trustworthy at all?*

Before this change `promote()` compared quality scores without ever asking where they came
from. Those scores come from an LLM judge, and the judge's only evidence of validity was a
calibration report that reported kappa 1.0 from a fixture in which the human and judge columns
were written from the same variable. So the gate was comparing two numbers produced by an
unvalidated instrument, and would have approved a regression just as confidently.

`promote()` now takes a `calibration` artifact path and refuses to run the ladder unless the
artifact is valid: measured on held-out data, bound to the current rubric hashes, complete, and
at or above the kappa policy. A blocked promotion writes the reasons into the audit event, so
"why did this not ship" is answerable from the audit log alone.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from aisys.audit import AuditLog

from .calibrate import ROUTEBENCH_MIN_KAPPA, validate_bundle

DEFAULT_ARTIFACT = Path(__file__).parent / "data" / "calibration_bundle.json"


@dataclass
class PromotionResult:
    model: str
    stage: str
    rolled_back: bool
    history: list[dict[str, object]] = field(default_factory=list)
    #: True when the ladder never ran because the judge was not trusted.
    blocked_by_calibration: bool = False
    calibration_reasons: list[str] = field(default_factory=list)


def promote(
    model: str,
    offline_quality: float,
    baseline_quality: float,
    measure_online: Callable[[float], float],
    audit: AuditLog,
    max_drop: float = 0.02,
    *,
    calibration: str | Path | None = DEFAULT_ARTIFACT,
    min_kappa: float = ROUTEBENCH_MIN_KAPPA,
    require_headline: bool = True,
    expected_dataset_hash: str | None = None,
) -> PromotionResult:
    """Run the promotion ladder, gated on a valid judge calibration.

    `calibration=None` skips the gate. That exists for unit tests of the ladder itself and for
    deliberately un-gated local experiments; it is not the production path, and the audit event
    records `calibration_checked: False` so a skipped gate is visible after the fact.

    `expected_dataset_hash` binds the artifact to a corpus. Pass
    `calibrate.corpus_dataset_hash(...)` on any path where the corpus is available: the gate
    otherwise checks the rubric hash but not the data, so an artifact whose kappa was measured on
    a different (smaller, easier, or simply older) corpus would be accepted as evidence about
    this one.
    """
    result = PromotionResult(model, "offline", False)

    if calibration is not None:
        ok, reasons = validate_bundle(
            calibration, min_kappa=min_kappa, require_headline=require_headline,
            expected_dataset_hash=expected_dataset_hash,
        )
        if not ok:
            result.blocked_by_calibration = True
            result.calibration_reasons = reasons
            result.stage = "blocked"
            audit.append({
                "type": "model_promotion", "model": model, "stage": "blocked",
                "rollback": False, "calibration_checked": True, "calibration_valid": False,
                "reasons": reasons, "history": [],
            })
            return result

    if offline_quality < baseline_quality - max_drop:
        result.rolled_back = True
        result.stage = "rollback"
        result.history.append({"stage": "offline", "quality": offline_quality, "accepted": False})
    else:
        for stage, weight in [("shadow", 0.0), ("canary-5", 0.05), ("canary-25", 0.25), ("full", 1.0)]:
            quality = measure_online(weight)
            accepted = quality >= baseline_quality - max_drop
            result.history.append(
                {"stage": stage, "weight": weight, "quality": quality, "accepted": accepted}
            )
            result.stage = stage
            if not accepted:
                result.stage = "rollback"
                result.rolled_back = True
                break

    audit.append({
        "type": "model_promotion", "model": model, "stage": result.stage,
        "rollback": result.rolled_back, "calibration_checked": calibration is not None,
        "calibration_valid": calibration is None or not result.blocked_by_calibration,
        "history": result.history,
    })
    return result
