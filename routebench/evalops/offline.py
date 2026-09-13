"""Offline quality-matrix computation."""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aisys.evals import EvalCase, EvalSuite, run


def evaluate(model: str, suite_path: str | Path, invoke: Callable[[EvalCase, str], dict[str, Any]]) -> dict[str, float]:
    suite = EvalSuite.load(suite_path)
    result = run(suite, lambda case: invoke(case, model), name=model)
    by_tag: dict[str, list[float]] = defaultdict(list)
    for item in result.results:
        for tag in item.tags:
            by_tag[tag].append(item.score)
    return {tag: sum(scores) / len(scores) for tag, scores in by_tag.items()}


def write_matrix(path: str | Path, matrix: dict[str, dict[str, float]]) -> None:
    Path(path).write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")

