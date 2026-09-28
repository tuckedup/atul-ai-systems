"""Versioned task rubrics and the deterministic score rule built on top of them.

The central design decision in this file: the judge model is never asked for a score.

Asking a model for "a correctness score between 0 and 1" produces a number with no stable
meaning. The same response scores 0.7 and 0.85 on different runs, the distribution piles up
on round numbers, and no threshold separates pass from fail cleanly — which is precisely how
judge-vs-human kappa ends up in the 0.4s. Instead the model answers a small set of concrete,
observable yes/no questions drawn from a versioned rubric, and `score()` here — ordinary
Python, not the model — turns those verdicts into a number.

Two consequences worth stating:

* **Critical criteria dominate.** If a criterion marked `critical` fails, the score is 0.0
  regardless of everything else. This mirrors how a human actually grades correctness: a
  summary that contradicts its source is not 60% acceptable, and code that returns the wrong
  answer does not earn partial credit for being well named. Encoding that makes the score
  distribution bimodal, which is what gives a threshold something to bite on.
* **The rubric is versioned and hashed.** A judgment records `rubric_version` and the rubric
  hash. Change a criterion and the old judgments stop matching the bundle, so a stale
  calibration cannot be presented as evidence for a new rubric.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .taxonomy import canonical

RUBRIC_DIR = Path(__file__).parent / "rubrics"


class RubricError(ValueError):
    pass


@dataclass(frozen=True)
class Criterion:
    id: str
    question: str
    critical: bool = False
    weight: int = 1
    pass_when: str = ""
    fail_when: str = ""

    def as_prompt_block(self) -> str:
        lines = [f"- id: {self.id}"]
        lines.append(f"  question: {self.question}")
        if self.pass_when:
            lines.append(f"  answer yes when: {self.pass_when}")
        if self.fail_when:
            lines.append(f"  answer no when: {self.fail_when}")
        if self.critical:
            lines.append("  CRITICAL: a 'no' here fails the response outright.")
        return "\n".join(lines)


@dataclass(frozen=True)
class Rubric:
    id: str
    version: str
    task_class: str
    pass_definition: str
    criteria: tuple[Criterion, ...]
    boundary_examples: tuple[dict[str, str], ...] = ()
    evidence_required: bool = True
    source_notes: str = ""

    @property
    def hash(self) -> str:
        payload = {
            "id": self.id,
            "version": self.version,
            "task_class": self.task_class,
            "pass_definition": self.pass_definition,
            "criteria": [
                {
                    "id": c.id,
                    "question": c.question,
                    "critical": c.critical,
                    "weight": c.weight,
                    "pass_when": c.pass_when,
                    "fail_when": c.fail_when,
                }
                for c in self.criteria
            ],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def criterion_ids(self) -> frozenset[str]:
        return frozenset(c.id for c in self.criteria)

    def criterion(self, cid: str) -> Criterion:
        for c in self.criteria:
            if c.id == cid:
                return c
        raise RubricError(f"rubric {self.id}@{self.version} has no criterion {cid!r}")

    def as_prompt_block(self) -> str:
        parts = [f"PASS MEANS: {self.pass_definition.strip()}", "", "CRITERIA:"]
        parts += [c.as_prompt_block() for c in self.criteria]
        if self.boundary_examples:
            parts += ["", "BOUNDARY CASES (these are the calls people get wrong):"]
            parts += [
                f"- {e.get('situation', '').strip()} -> {e.get('verdict', '').strip().upper()}"
                f" ({e.get('why', '').strip()})"
                for e in self.boundary_examples
            ]
        return "\n".join(parts)


def _parse(raw: dict[str, Any], origin: str) -> Rubric:
    missing = {"id", "version", "task_class", "pass_definition", "criteria"} - set(raw)
    if missing:
        raise RubricError(f"{origin}: rubric missing required keys {sorted(missing)}")
    criteria: list[Criterion] = []
    seen: set[str] = set()
    for entry in raw["criteria"]:
        if "id" not in entry or "question" not in entry:
            raise RubricError(f"{origin}: every criterion needs 'id' and 'question', got {entry}")
        if entry["id"] in seen:
            raise RubricError(f"{origin}: duplicate criterion id {entry['id']!r}")
        seen.add(entry["id"])
        weight = int(entry.get("weight", 1))
        if weight < 1:
            raise RubricError(f"{origin}: criterion {entry['id']!r} weight must be >= 1")
        criteria.append(
            Criterion(
                id=str(entry["id"]),
                question=str(entry["question"]),
                critical=bool(entry.get("critical", False)),
                weight=weight,
                pass_when=str(entry.get("pass_when", "")),
                fail_when=str(entry.get("fail_when", "")),
            )
        )
    if not criteria:
        raise RubricError(f"{origin}: rubric has zero criteria")
    if not any(c.critical for c in criteria):
        raise RubricError(
            f"{origin}: rubric has no critical criterion. Without one, a response that is "
            "flatly wrong can still score above a pass threshold on secondary criteria."
        )
    return Rubric(
        id=str(raw["id"]),
        version=str(raw["version"]),
        task_class=canonical(raw["task_class"]),
        pass_definition=str(raw["pass_definition"]),
        criteria=tuple(criteria),
        boundary_examples=tuple(raw.get("boundary_examples", []) or ()),
        evidence_required=bool(raw.get("evidence_required", True)),
        source_notes=str(raw.get("source_notes", "")),
    )


@cache
def load_all(directory: str | None = None) -> dict[str, Rubric]:
    """Load every rubric, keyed by canonical task class."""
    d = Path(directory) if directory else RUBRIC_DIR
    if not d.is_dir():
        raise RubricError(f"rubric directory not found: {d}")
    out: dict[str, Rubric] = {}
    for f in sorted(d.glob("*.y*ml")):
        r = _parse(yaml.safe_load(f.read_text(encoding="utf-8")), str(f))
        if r.task_class in out:
            raise RubricError(f"two rubrics claim task_class {r.task_class!r}: {f} and earlier")
        out[r.task_class] = r
    if not out:
        raise RubricError(f"no rubrics found in {d}")
    return out


def for_task(task_class: str, directory: str | None = None) -> Rubric:
    key = canonical(task_class)
    rubrics = load_all(directory)
    if key not in rubrics:
        raise RubricError(f"no rubric for task class {key!r}; have {sorted(rubrics)}")
    return rubrics[key]


def bundle_hash(directory: str | None = None) -> str:
    """One hash over every rubric, for binding a calibration artifact to the rubric set."""
    rubrics = load_all(directory)
    joined = json.dumps({k: v.hash for k, v in sorted(rubrics.items())}, sort_keys=True)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


# ---------------- the deterministic score rule ----------------
def score(rubric: Rubric, verdicts: dict[str, bool]) -> float:
    """Turn per-criterion yes/no verdicts into a score in [0, 1].

    Rule:
      * any `critical` criterion answered no  -> 0.0
      * otherwise                             -> satisfied weight / total weight

    Every criterion in the rubric must be present in `verdicts`. A judge that skipped a
    criterion produced an incomplete judgment, and an incomplete judgment is an error, not a
    lower score -- silently treating "did not answer" as "no" would bias the judge toward fail.
    """
    missing = rubric.criterion_ids - set(verdicts)
    if missing:
        raise RubricError(
            f"incomplete judgment for rubric {rubric.id}@{rubric.version}: "
            f"missing verdicts for {sorted(missing)}"
        )
    unknown = set(verdicts) - rubric.criterion_ids
    if unknown:
        raise RubricError(
            f"judgment invented criteria not in rubric {rubric.id}@{rubric.version}: {sorted(unknown)}"
        )
    for c in rubric.criteria:
        if c.critical and not verdicts[c.id]:
            return 0.0
    total = sum(c.weight for c in rubric.criteria)
    satisfied = sum(c.weight for c in rubric.criteria if verdicts[c.id])
    return satisfied / total
