"""Evaluation framework shared by ForgeBench, RouteBench and Incident Commander.

Cases are YAML files. Graders are deterministic or model-based. The judge must be calibrated against
human labels (Cohen's kappa) before its scores are trusted. `compare()` produces the regression report
that CI uses to block merges.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from sklearn.metrics import cohen_kappa_score

from . import llm
from .tracing import traced

Grader = Callable[["EvalCase", Any], float]  # returns score in [0, 1]


class EvalCase(BaseModel):
    id: str
    input: Any
    expected: Any = None
    tags: list[str] = []
    grader: str = "exact"
    grader_args: dict[str, Any] = {}


class CaseResult(BaseModel):
    case_id: str
    score: float
    tags: list[str]
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    steps: int = 0
    human_interventions: int = 0
    error: str | None = None


class RunResult(BaseModel):
    name: str
    results: list[CaseResult]

    def success_rate(self, tag: str | None = None) -> float:
        rs = [r for r in self.results if tag is None or tag in r.tags]
        return sum(r.score >= 1.0 for r in rs) / len(rs) if rs else 0.0

    def mean(self, attr: str) -> float:
        return sum(getattr(r, attr) for r in self.results) / max(1, len(self.results))

    def table(self) -> str:
        tags = sorted({t for r in self.results for t in r.tags})
        lines = [f"{'tag':<16}{'n':>5}{'success':>10}", f"{'ALL':<16}{len(self.results):>5}{self.success_rate():>10.1%}"]
        lines += [f"{t:<16}{sum(t in r.tags for r in self.results):>5}{self.success_rate(t):>10.1%}" for t in tags]
        lines.append(f"tokens/task {self.mean('tokens'):.0f} | cost/task ${self.mean('cost_usd'):.4f} | "
                     f"latency {self.mean('latency_ms'):.0f}ms | steps {self.mean('steps'):.1f} | "
                     f"human-int rate {self.mean('human_interventions'):.2f}")
        return "\n".join(lines)


# ---------------- graders ----------------
def g_exact(c: EvalCase, out: Any) -> float:
    return float(str(out).strip() == str(c.expected).strip())


def g_contains(c: EvalCase, out: Any) -> float:
    return float(all(s in str(out) for s in (c.expected if isinstance(c.expected, list) else [c.expected])))


def g_regex(c: EvalCase, out: Any) -> float:
    return float(re.search(c.expected, str(out)) is not None)


def g_json_schema(c: EvalCase, out: Any) -> float:
    import jsonschema  # optional dep; declare in project
    try:
        jsonschema.validate(json.loads(out) if isinstance(out, str) else out, c.expected)
        return 1.0
    except Exception:
        return 0.0


def g_unit_tests(c: EvalCase, out: Any) -> float:
    """`out` is a workdir path with the candidate patch applied; grader_args.cmd runs hidden tests."""
    r = subprocess.run(c.grader_args["cmd"], shell=True, cwd=str(out), capture_output=True, timeout=c.grader_args.get("timeout", 600))
    return float(r.returncode == 0)


JUDGE_PROMPT = """You are a strict grader. Rubric:
{rubric}

TASK INPUT:
{input}

EXPECTED (may be empty):
{expected}

CANDIDATE OUTPUT:
{output}

Return ONLY JSON: {{"score": <0..1 float>, "reason": "<one sentence>"}}"""


def g_llm_judge(c: EvalCase, out: Any) -> float:
    r = llm.chat(
        [{"role": "user", "content": JUDGE_PROMPT.format(
            rubric=c.grader_args.get("rubric", "correctness, completeness, groundedness, instruction following"),
            input=c.input, expected=c.expected, output=out)}],
        model=c.grader_args.get("judge_model"), temperature=0.0, max_tokens=200,
    )
    return float(json.loads(re.sub(r"```json|```", "", r.text).strip())["score"])


def pairwise(c: EvalCase, a: Any, b: Any, judge_model: str | None = None) -> str:
    """Return 'A', 'B' or 'tie'. Runs both orders to cancel position bias."""
    def ask(x: Any, y: Any) -> str:
        r = llm.chat([{"role": "user", "content":
            f"Task:\n{c.input}\n\nResponse A:\n{x}\n\nResponse B:\n{y}\n\nWhich is better? Answer only A, B, or tie."}],
            model=judge_model, temperature=0.0, max_tokens=5)
        return r.text.strip().upper()[:3]
    first, second = ask(a, b), ask(b, a)
    if first.startswith("A") and second.startswith("B"):
        return "A"
    if first.startswith("B") and second.startswith("A"):
        return "B"
    return "tie"


GRADERS: dict[str, Grader] = {
    "exact": g_exact, "contains": g_contains, "regex": g_regex,
    "json_schema": g_json_schema, "unit_tests": g_unit_tests, "llm_judge": g_llm_judge,
}


# ---------------- suite + runner ----------------
@dataclass
class EvalSuite:
    cases: list[EvalCase] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> EvalSuite:
        cases = []
        for f in sorted(Path(path).glob("**/*.y*ml")):
            data = yaml.safe_load(f.read_text())
            cases += [EvalCase(**d) for d in (data if isinstance(data, list) else [data])]
        return cls(cases)

    def filter(self, tag: str) -> EvalSuite:
        return EvalSuite([c for c in self.cases if tag in c.tags])


@traced(kind="eval")
def run(suite: EvalSuite, fn: Callable[[EvalCase], dict[str, Any]], name: str = "run", concurrency: int = 4) -> RunResult:
    """`fn` returns {"output": ..., "tokens": int, "cost_usd": float, "latency_ms": float, "steps": int, "human_interventions": int}."""
    def one(c: EvalCase) -> CaseResult:
        try:
            r = fn(c)
            score = GRADERS[c.grader](c, r["output"])
            return CaseResult(case_id=c.id, score=score, tags=c.tags, **{k: r.get(k, 0) for k in
                              ("tokens", "cost_usd", "latency_ms", "steps", "human_interventions")})
        except Exception as e:
            return CaseResult(case_id=c.id, score=0.0, tags=c.tags, error=f"{type(e).__name__}: {e}")
    with ThreadPoolExecutor(concurrency) as ex:
        return RunResult(name=name, results=list(ex.map(one, suite.cases)))


# ---------------- judge calibration ----------------
def calibrate(judge_scores: list[float], human_scores: list[float], threshold: float = 0.5) -> dict[str, Any]:
    """Binarize at threshold and report Cohen's kappa + raw agreement. Require kappa >= 0.6 before trusting the judge."""
    j = [int(s >= threshold) for s in judge_scores]
    h = [int(s >= threshold) for s in human_scores]
    kappa = float(cohen_kappa_score(h, j))
    agreement = sum(a == b for a, b in zip(h, j)) / len(h)
    confusion = {"tp": sum(a and b for a, b in zip(h, j)), "fp": sum((not a) and b for a, b in zip(h, j)),
                 "fn": sum(a and (not b) for a, b in zip(h, j)), "tn": sum((not a) and (not b) for a, b in zip(h, j))}
    return {"kappa": kappa, "agreement": agreement, "n": len(h), "confusion": confusion, "trusted": kappa >= 0.6}


# ---------------- regression gate ----------------
class RegressionReport(BaseModel):
    baseline: str
    candidate: str
    overall_delta: float
    per_tag_delta: dict[str, float]
    blocked: bool
    reasons: list[str]


def compare(baseline: RunResult, candidate: RunResult, max_drop: float = 0.02, max_tag_drop: float = 0.05) -> RegressionReport:
    tags = sorted({t for r in baseline.results for t in r.tags})
    per_tag = {t: candidate.success_rate(t) - baseline.success_rate(t) for t in tags}
    overall = candidate.success_rate() - baseline.success_rate()
    reasons = []
    if overall < -max_drop:
        reasons.append(f"overall success dropped {overall:+.1%} (limit -{max_drop:.0%})")
    reasons += [f"tag '{t}' dropped {d:+.1%}" for t, d in per_tag.items() if d < -max_tag_drop]
    return RegressionReport(baseline=baseline.name, candidate=candidate.name, overall_delta=overall,
                            per_tag_delta=per_tag, blocked=bool(reasons), reasons=reasons)
