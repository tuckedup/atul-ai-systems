"""Second-pass auditor for cases the frozen v18 judge PASSED (offline implementation; no run without explicit approval).

Design (docs/ROUTEBENCH_PROPOSAL_ERROR_TARGETED_AUDIT_20260930.md, revised after review):
- The auditor judges support independently. Its prompt contains the source and the candidate only: never the gold label, never v18's verdict.
- It must name the disputed assertion, an error type, what the claim says vs what the source says, source evidence and an explanation.
- Quote matching is an AUTHENTICITY check only (the quote exists in the source). It does not prove the quote is relevant, so a flip also needs
  lexical overlap between quote and assertion plus a non-trivial claim/source contrast. Residual irrelevant-quote risk is reported, not hidden.
- Unsupported additions have no contradicting quote: they flip only when the auditor lists specific unsupported details that appear in the
  candidate, are absent from the source, and are not mere stop words.
- Spend is hard-capped by a single meter shared by every call (worst-case reservation includes reasoning tokens); an exhausted cap ends the run INCOMPLETE.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL = "o4-mini"
INPUT_PER_1M = 1.10
OUTPUT_PER_1M = 4.40  # reasoning tokens bill as output
MAX_TOKENS = 4000
MAX_ATTEMPTS = 2
PROMPT_TOKEN_ESTIMATE = 2900  # source + candidate (~2,000 observed for v18) + instructions
ERROR_TYPES = ("reasoning_error", "nuanced_meaning_shift", "tense_or_modality", "mis_referencing",
               "opinion_stated_as_fact", "contradiction", "unsupported_addition")
ADDITION = "unsupported_addition"
MIN_EXPLANATION_WORDS = 8
STOP = frozenset(["the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "with", "by", "as", "from", "has", "have", "had", "will", "would", "can", "could", "their", "they", "he", "she", "his", "her", "them", "there", "which", "who", "whom", "not", "no", "than", "then", "so", "also", "about", "into", "over", "after", "before"])

PROMPT = """You are auditing whether a summary is fully supported by its source. Judge the summary text only against the source text.
Do not use outside knowledge. A faithful paraphrase, compression or reordering of what the source says is SUPPORTED. Read every sentence.

Report a problem only when some assertion in the summary is not established by the source. Problem types:
- reasoning_error: a conclusion or causal/comparative link the source does not state or that does not follow from it
- nuanced_meaning_shift: wording that changes scope, degree, certainty, quantity or emphasis compared with the source
- tense_or_modality: past/future, actual/possible, required/recommended, planned/completed changed
- mis_referencing: the wrong person, speaker, entity, number or time is attached to a statement
- opinion_stated_as_fact: an opinion, hope or claim by someone presented as established fact
- contradiction: the source states the opposite
- unsupported_addition: a detail that the source does not contain at all (no source sentence addresses it)

Return ONLY one JSON object:
{{"verdict": "supported" | "unsupported",
 "disputed_assertion": "<exact words copied from the summary, or null>",
 "error_type": "<one type above, or null>",
 "claim_says": "<what the summary asserts, or null>",
 "source_says": "<what the source actually says on this point, or 'nothing'>",
 "evidence_quote": "<exact words copied from the source that show the mismatch, or null for unsupported_addition>",
 "unsupported_details": ["<specific words from the summary the source lacks; only for unsupported_addition>"],
 "explanation": "<two or three sentences on why the assertion is not supported>"}}
If everything is supported, return verdict "supported" and null for the other fields.

<source>
{source}
</source>

<summary>
{candidate}
</summary>
"""


def build_prompt(source: str, candidate: str) -> str:
    """The ONLY inputs are the source and the candidate; no label, no prior verdict, no split information."""
    return PROMPT.format(source=source, candidate=candidate)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def content_tokens(text: str) -> set[str]:
    return {t for t in norm(text).split() if t not in STOP and len(t) > 1}


@dataclass(frozen=True)
class AuditResult:
    verdict: str
    disputed_assertion: str | None = None
    error_type: str | None = None
    claim_says: str | None = None
    source_says: str | None = None
    evidence_quote: str | None = None
    unsupported_details: tuple[str, ...] = ()
    explanation: str = ""


class AuditParseError(ValueError):
    pass


def parse(text: str) -> AuditResult:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise AuditParseError("no JSON object")
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise AuditParseError(f"invalid JSON: {exc}") from exc
    verdict = raw.get("verdict")
    if verdict not in ("supported", "unsupported"):
        raise AuditParseError("verdict must be supported|unsupported")
    kind = raw.get("error_type")
    if verdict == "unsupported" and kind not in ERROR_TYPES:
        raise AuditParseError("unsupported verdict needs a known error_type")
    details = raw.get("unsupported_details") or []
    if not isinstance(details, list):
        raise AuditParseError("unsupported_details must be a list")

    def text_or_none(key: str) -> str | None:
        value = raw.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else None

    return AuditResult(verdict, text_or_none("disputed_assertion"), kind, text_or_none("claim_says"), text_or_none("source_says"),
                       text_or_none("evidence_quote"), tuple(str(d) for d in details if str(d).strip()), (raw.get("explanation") or "").strip())


def decide(result: AuditResult, source: str, candidate: str) -> tuple[bool, str]:
    """(flip_to_unsupported, reason). Conservative: every requirement must hold; otherwise the v18 verdict stands."""
    if result.verdict != "unsupported":
        return False, "auditor_supported"
    if not result.disputed_assertion or norm(result.disputed_assertion) not in norm(candidate):
        return False, "assertion_not_in_candidate"
    if len(result.explanation.split()) < MIN_EXPLANATION_WORDS:
        return False, "explanation_too_short"
    if not result.claim_says or not result.source_says:
        return False, "missing_claim_or_source_statement"
    if norm(result.claim_says) == norm(result.source_says):
        return False, "claim_equals_source"
    if result.error_type == ADDITION:
        details = [d for d in result.unsupported_details if content_tokens(d)]
        if not details:
            return False, "addition_without_specific_details"
        genuine = [d for d in details if norm(d) in norm(candidate) and not (content_tokens(d) & content_tokens(source))]
        # A detail counts only if none of its content words occur anywhere in the source (a conservative absence test).
        if not genuine:
            return False, "addition_details_present_in_source_or_not_in_candidate"
        return True, "flip_unsupported_addition"
    if not result.evidence_quote:
        return False, "no_evidence_quote"
    if len(result.evidence_quote.split()) < 3:
        return False, "quote_too_short"
    if norm(result.evidence_quote) not in norm(source):
        return False, "quote_not_in_source"  # authenticity failure: fabricated or altered quote
    if not (content_tokens(result.evidence_quote) & content_tokens(result.disputed_assertion)):
        return False, "quote_unrelated_to_assertion"  # real quote, no lexical link to the disputed assertion
    return True, "flip_with_authentic_related_quote"


# ------------------------------------------------------------------------------------------------------- spend and calls
class CapReached(RuntimeError):
    pass


def worst_case_call_usd(prompt_tokens: int = PROMPT_TOKEN_ESTIMATE, max_tokens: int = MAX_TOKENS) -> float:
    return (prompt_tokens * INPUT_PER_1M + max_tokens * OUTPUT_PER_1M) / 1e6


def plan_cost(n_cases: int, *, mean_prompt: int = 2600, mean_completion: int = 1700, retry_rate: float = 0.15) -> dict[str, float]:
    """Expected and worst-case spend for n audits (completion includes hidden reasoning tokens)."""
    per_call = (mean_prompt * INPUT_PER_1M + mean_completion * OUTPUT_PER_1M) / 1e6
    return {"calls": n_cases, "expected_usd": round(n_cases * per_call * (1 + retry_rate), 3),
            "worst_case_usd": round(n_cases * MAX_ATTEMPTS * worst_case_call_usd(), 3), "per_call_worst_usd": round(worst_case_call_usd(), 4)}


@dataclass
class Meter:
    """Hard total cap. A call is admitted only if spent + in-flight reservations + its worst case fits under the cap."""

    cap_usd: float
    spent: float = 0.0
    reserved: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self, usd: float) -> float:
        with self.lock:
            if self.spent + self.reserved + usd > self.cap_usd:
                raise CapReached(f"cap ${self.cap_usd:.2f} would be exceeded (spent {self.spent:.4f}, reserved {self.reserved:.4f})")
            self.reserved += usd
            return usd

    def settle(self, reserved: float, actual: float | None) -> None:
        with self.lock:
            self.reserved -= reserved
            self.spent += reserved if actual is None else actual  # unknown cost is charged at its bound, never as zero


Call = Callable[[str], tuple[str, float | None]]  # prompt -> (text, cost_usd)


def audit_one(case_id: str, source: str, candidate: str, call: Call, meter: Meter) -> dict[str, Any]:
    """Audit one case. Never raises: a failure is recorded as status != 'ok' and the case is NOT flipped."""
    prompt, last = build_prompt(source, candidate), ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        held = meter.reserve(worst_case_call_usd(max(PROMPT_TOKEN_ESTIMATE, len(prompt) // 3)))
        try:
            text, cost = call(prompt)
        except Exception as exc:  # noqa: BLE001 - provider failure is data
            meter.settle(held, None)
            last = f"{type(exc).__name__}: {exc}"
            continue
        meter.settle(held, cost)
        try:
            result = parse(text)
        except AuditParseError as exc:
            last = f"parse: {exc}"
            continue
        flip, reason = decide(result, source, candidate)
        return {"id": case_id, "status": "ok", "attempts": attempt, "flip": flip, "reason": reason,
                "result": result.__dict__ | {"unsupported_details": list(result.unsupported_details)}}
    return {"id": case_id, "status": "failed", "attempts": MAX_ATTEMPTS, "flip": False, "reason": last}


def run_stage(items: Sequence[Mapping[str, str]], call: Call, cap_usd: float, done: Mapping[str, Any] | None = None,
              write: Callable[[dict[str, Any]], None] = lambda row: None) -> dict[str, Any]:
    """Audit every item sequentially. Complete only if every item has an ok audit; a reached cap ends INCOMPLETE (rows kept)."""
    meter = Meter(cap_usd)
    rows: dict[str, Any] = dict(done or {})
    status = "COMPLETE"
    for item in items:
        if item["id"] in rows:
            continue
        try:
            row = audit_one(item["id"], item["source"], item["candidate"], call, meter)
        except CapReached as exc:
            status = f"INCOMPLETE: {exc}"
            break
        rows[item["id"]] = row
        write(row)
    if status == "COMPLETE" and any(r["status"] != "ok" for r in rows.values()):
        status = "INCOMPLETE: failed audits"
    return {"status": status, "rows": rows, "spent_usd": round(meter.spent, 5), "cap_usd": cap_usd}


def kappa(labels: Sequence[int], preds: Sequence[int]) -> float:
    n = len(labels)
    po = sum(a == b for a, b in zip(labels, preds, strict=True)) / n
    p1, q1 = sum(labels) / n, sum(preds) / n
    pe = p1 * q1 + (1 - p1) * (1 - q1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score_flips(labels: Mapping[str, int], v18: Mapping[str, int], flips: Mapping[str, bool]) -> dict[str, Any]:
    """Counts vs gold and the actual kappa for the full population (v18 verdict, with audited passes flipped to unsupported)."""
    ids = sorted(labels)
    new = {i: 0 if flips.get(i) else v18[i] for i in ids}
    corrected = sum(1 for i in ids if v18[i] != labels[i] and new[i] == labels[i])
    introduced = sum(1 for i in ids if v18[i] == labels[i] and new[i] != labels[i])
    ys = [labels[i] for i in ids]
    return {"n": len(ids), "corrected": corrected, "introduced": introduced, "v18_errors": sum(v18[i] != labels[i] for i in ids),
            "new_errors": sum(new[i] != labels[i] for i in ids), "kappa_v18": kappa(ys, [v18[i] for i in ids]),
            "kappa_audited": kappa(ys, [new[i] for i in ids])}


# ------------------------------------------------------------------------------------------------------------ CLI
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
WORK = ROOT / ".local" / "audit2-v1"
CONFIRM = "RUN-AUDIT2-PAID"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def protocol_hashes() -> dict[str, str]:
    """Everything that can change a verdict: the prompt, the decision rule and parser, the model and its limits."""
    code = "".join(inspect.getsource(f) for f in (decide, parse, norm, content_tokens)) + repr((ERROR_TYPES, MIN_EXPLANATION_WORDS, sorted(STOP)))
    return {"prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "rule_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "settings": _sha([MODEL, MAX_TOKENS, MAX_ATTEMPTS, "reasoning_effort=medium", "temperature=default"])}


def _population() -> dict[str, list[dict[str, Any]]]:
    """v18 passes by split from the cached frozen judgments, with labels. Cases with no usable v18 judgment are listed, never dropped silently."""
    from .dataset import HEADLINE_PROVENANCE, Annotation, CaseRecord, Corpus
    from .experiments import JudgmentCache, _cache_key
    from .run_calibration import GRID

    assignment = json.loads((DATA / "splits.json").read_text(encoding="utf8"))["assignment"]
    cases = [CaseRecord.model_validate_json(line) for line in (DATA / "cases.jsonl").read_text(encoding="utf8").splitlines()]
    annotations = [Annotation.model_validate_json(line) for line in (DATA / "annotations.jsonl").read_text(encoding="utf8").splitlines()]
    labels = Corpus(cases, annotations).resolved_labels(allow=HEADLINE_PROVENANCE)
    config = next(c for c in GRID if c.variant_id == "v18-reasoning-o4mini")
    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    out: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for case in sorted(cases, key=lambda c: c.case_id):
        split = assignment[case.group_id]
        if split not in out or case.case_id not in labels:
            continue
        judgment = cache.get(_cache_key(case, config))
        if judgment is None or not judgment.usable:
            raise ValueError(f"missing frozen v18 judgment {case.case_id}")
        out[split].append({"id": case.case_id, "source": case.context, "candidate": case.candidate_output.strip(),
                           "label": int(labels[case.case_id].label), "v18": int(judgment.raw_score >= 1.0)})
    return out


def prepare(cap_usd: float) -> None:
    population = _population()
    audited = {k: [r for r in v if r["v18"] == 1] for k, v in population.items()}
    total = sum(len(v) for v in audited.values())
    body = {**protocol_hashes(), "model": MODEL, "cap_usd_total": cap_usd, "stages": {k: len(v) for k, v in audited.items()},
            "population_sha256": {k: _sha([r["id"] for r in v]) for k, v in population.items()}, "cost_plan": plan_cost(total),
            "stage_order": ["train", "dev"], "gate_note": "dev runs only after the train stage is COMPLETE; no prompt or rule edits after this freeze"}
    WORK.mkdir(parents=True, exist_ok=False)
    (WORK / "population.json").write_text(json.dumps(population), encoding="utf8")
    (WORK / "protocol.json").write_text(json.dumps(body, indent=2), encoding="utf8")
    print(json.dumps(body, indent=2))


def _live_call(prompt: str) -> tuple[str, float | None]:
    from aisys import llm

    result = llm.chat([{"role": "user", "content": prompt}], model=MODEL, max_tokens=MAX_TOKENS, fallback=getattr(llm, "NO_FALLBACK", []),
                      reasoning_effort="medium")
    if result.model and not result.model.startswith(MODEL):
        raise RuntimeError(f"served model {result.model!r} is not {MODEL!r}")
    return result.text, result.cost_usd


def run(stage: str, cap_usd: float, confirm: str) -> int:
    protocol = json.loads((WORK / "protocol.json").read_text(encoding="utf8"))
    if confirm != CONFIRM:
        raise SystemExit(f"paid run requires --confirm {CONFIRM}")
    if {k: v for k, v in protocol.items() if k in protocol_hashes()} != protocol_hashes():
        raise SystemExit("prompt, decision rule or settings changed since the freeze")
    spent_so_far = sum(json.loads(t.read_text(encoding="utf8"))["spent_usd"] for t in WORK.glob("terminal_*.json"))
    cap = min(cap_usd, protocol["cap_usd_total"] - spent_so_far)
    if cap <= 0 or (WORK / f"terminal_{stage}.json").exists():
        raise SystemExit("total cap exhausted or stage already run; refusing a second attempt")
    if stage == "dev" and json.loads((WORK / "terminal_train.json").read_text(encoding="utf8"))["status"] != "COMPLETE":
        raise SystemExit("dev runs only after a COMPLETE train stage")
    population = json.loads((WORK / "population.json").read_text(encoding="utf8"))[stage]
    items = [r for r in population if r["v18"] == 1]
    partial = WORK / f"rows_{stage}.jsonl"
    with partial.open("a", encoding="utf8") as stream:
        out = run_stage(items, _live_call, cap, write=lambda row: (stream.write(json.dumps(row) + chr(10)), stream.flush()))
    flips = {i: bool(r["flip"]) for i, r in out["rows"].items()}
    report = score_flips({r["id"]: r["label"] for r in population}, {r["id"]: r["v18"] for r in population}, flips)
    terminal = {"stage": stage, "status": out["status"], "spent_usd": out["spent_usd"], "cap_usd": cap, "audited": len(out["rows"]),
                "flips": sum(flips.values()), "report": report, "reasons": {k: sum(1 for r in out["rows"].values() if r["reason"] == k)
                                                                           for k in {r["reason"] for r in out["rows"].values()}}}
    (WORK / f"terminal_{stage}.json").write_text(json.dumps(terminal, indent=2), encoding="utf8")
    print(json.dumps(terminal, indent=2))
    return 0 if out["status"] == "COMPLETE" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--stage", choices=["train", "dev"])
    parser.add_argument("--cap", type=float, default=3.0, help="USD cap: total for prepare, per-stage ceiling for run")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.cap)
        return 0
    if args.stage is None:
        parser.error("--stage required")
    return run(args.stage, args.cap, args.confirm)


if __name__ == "__main__":
    sys.exit(main())
