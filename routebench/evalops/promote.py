"""Eval-gated shadow → canary → full rollout with rollback."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from aisys.audit import AuditLog


@dataclass
class PromotionResult:
    model: str
    stage: str
    rolled_back: bool
    history: list[dict[str, object]] = field(default_factory=list)


def promote(
    model: str,
    offline_quality: float,
    baseline_quality: float,
    measure_online: Callable[[float], float],
    audit: AuditLog,
    max_drop: float = 0.02,
) -> PromotionResult:
    result = PromotionResult(model, "offline", False)
    if offline_quality < baseline_quality - max_drop:
        result.rolled_back = True
        result.stage = "rollback"
        result.history.append({"stage": "offline", "quality": offline_quality, "accepted": False})
    else:
        for stage, weight in [("shadow", 0.0), ("canary-5", 0.05), ("canary-25", 0.25), ("full", 1.0)]:
            quality = measure_online(weight)
            accepted = quality >= baseline_quality - max_drop
            result.history.append({"stage": stage, "weight": weight, "quality": quality, "accepted": accepted})
            result.stage = stage
            if not accepted:
                result.stage = "rollback"
                result.rolled_back = True
                break
    audit.append({"type": "model_promotion", "model": model, "stage": result.stage, "rollback": result.rolled_back, "history": result.history})
    return result

