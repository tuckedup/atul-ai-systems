"""ONE opt-in judge candidate: a source-linked claim-relation audit (variant `claim-relation-v1`). Offline implementation; v18 is untouched.

Targeted error (hypothesis, not an established improvement): earlier judges either accepted nearby facts as an unstated link or rejected faithful
paraphrases. Examples reviewed by the AI panel: intended purpose read as achieved result; authorised amount read as payment; a share read as an
increase; possible read as confirmed; one entity's attribute transferred to all listed entities; U/You orthography rejected. Plausible failure modes:
the model can fabricate relations, over-segment a simple claim, copy an irrelevant source span, or be as wrong as v18 on cases where annotators
used a looser inference policy. Nothing here tunes to labels; the candidate must earn its result in a separately approved evaluation.

One structured response per case (no extra provider round trip). For EVERY assertion in the candidate the response must give: exact candidate span,
source evidence spans, actor, time, modality, numeric comparison, and causal/purpose/evidential relation (each an explicit value or the literal
"not_applicable"), a verdict (supported / unsupported / uncertain), an issue type and a rationale. The code, not the model, aggregates:
    any valid unsupported assertion          -> unsupported
    else uncertain / unreviewed span / support without evidence / possibly-hidden second claim -> uncertain
    else                                      -> supported
Quote authenticity (offsets and text exist in the stored source) is recorded separately and never upgrades or downgrades semantic support. An
unsupported addition needs no contradicting quote. No lexical-overlap rule gates support. Uncertain is an abstention: it is kept in the denominator
and may not count as a pass.

    python -m evalops.claim_relation demo --out DIR            (mocked inference on synthetic fixtures; no network, no GPU)
    build-cases -> freeze -> run (needs separate approval; verifies the frozen protocol first, stops at the first anomaly) -> compare
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VARIANT_ID = "claim-relation-v1"
V18_CONFIG_HASH = "12fc8b550d954359"
V18_PROMPT_TEMPLATE_HASH = "ef895d791c0bd860"
NOT_APPLICABLE = "not_applicable"
VERDICTS = ("supported", "unsupported", "uncertain")
MODALITIES = ("actual", "planned", "possible", "alleged", "opinion", NOT_APPLICABLE)
ISSUE_TYPES = ("none", "paraphrase_ok", "entity_mismatch", "purpose_vs_result", "authorization_vs_payment", "share_vs_increase", "possible_vs_confirmed",
               "attribute_transfer", "unlinked_facts", "unsupported_addition", "other")
RELATION_TYPES = ("causal", "purpose", "evidential", "temporal", "other")
CONJUNCTIONS = re.compile(r"\b(and|but|while|which|because|so|;)\b|;", re.IGNORECASE)
STOP = frozenset(["the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "with", "by", "as", "from", "has", "have", "had", "will", "would", "can", "could", "their", "they", "he", "she", "his", "her", "them", "there", "which", "who", "not", "no", "than", "then", "so", "also", "about", "into", "over", "after", "before"])
MAX_TOKENS = 6000

PROMPT = """You audit whether a candidate sentence is fully supported by a source. Use ONLY the source text. Do not use outside knowledge.

Policy: a faithful paraphrase or spelling/orthography normalisation that the source's context establishes is SUPPORTED. Check every value together with its entity, unit,
date and relationship. Do not equate: intended purpose with achieved result; an authorised amount with a payment; a current share with an increase; a possible, planned or
alleged event with a confirmed one; one listed entity's attribute with all of them; two nearby facts with a causal or evidential link the source states. If the source does not
contain a detail, the assertion is UNSUPPORTED even if no sentence contradicts it. If you cannot decide, answer "uncertain" and say what is missing.

Step 1. Split the candidate into its separate assertions so that EVERY part of the candidate belongs to some assertion.
Step 2. For each assertion give exact text copied from the candidate (candidate_span with start, end offsets: zero-based, end-exclusive) and exact source quotes with offsets as evidence.
For each assertion give: actor, time, modality (actual|planned|possible|alleged|opinion|not_applicable), numeric (an object {{candidate_value, source_value, same_entity, comparison}}
or "not_applicable"), relation (an object {{type: causal|purpose|evidential|temporal|other, stated_in_source: true|false, source_span: <span or null>}} or "not_applicable"),
verdict, issue_type, rationale. Never invent a field value: write "not_applicable" when it does not apply.
Step 3. List any candidate text you did not cover in uncovered_spans with a reason.

Return ONLY this JSON object:
{{"assertions": [{{"candidate_span": {{"start": 0, "end": 0, "text": ""}}, "evidence": [{{"start": 0, "end": 0, "text": ""}}], "actor": "", "time": "", "modality": "",
"numeric": "not_applicable", "relation": "not_applicable", "verdict": "supported|unsupported|uncertain", "issue_type": "none|paraphrase_ok|entity_mismatch|purpose_vs_result|authorization_vs_payment|share_vs_increase|possible_vs_confirmed|attribute_transfer|unlinked_facts|unsupported_addition|other",
"rationale": ""}}], "uncovered_spans": [{{"start": 0, "end": 0, "text": "", "reason": ""}}], "overall": {{"verdict": "supported|unsupported|uncertain", "rationale": ""}}}}

<source>
{source}
</source>

<candidate>
{candidate}
</candidate>
"""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def build_prompt(source: str, candidate: str, context: str = "") -> str:
    """The only inputs are the source, the candidate and (optionally) neighbouring summary context. No label, no prior verdict."""
    prompt = PROMPT.format(source=source, candidate=candidate)
    if context.strip():
        prompt = prompt.replace("<candidate>", f"<summary_context_for_reference_resolution_only_not_evidence>\n{context}\n</summary_context_for_reference_resolution_only_not_evidence>\n\n<candidate>", 1)
    return prompt


def content_hash() -> str:
    """Identity of this candidate: prompt, vocabularies and the validating/aggregating code. Distinct from v18's config and template hashes."""
    code = "".join(inspect.getsource(f) for f in (parse_response, validate_assertions, aggregate, _word_spans))
    return sha256(PROMPT + json.dumps([VERDICTS, MODALITIES, ISSUE_TYPES, RELATION_TYPES, sorted(STOP), MAX_TOKENS]) + code)


class ClaimRelationError(ValueError):
    """The model response is not a usable structured audit. Recorded as a failure, never repaired by guessing."""


@dataclass
class Assertion:
    span: dict[str, Any]
    evidence: list[dict[str, Any]]
    fields: dict[str, Any]
    verdict: str
    issue_type: str
    rationale: str
    flags: list[str] = field(default_factory=list)
    span_valid: bool = True
    evidence_authentic: list[bool] = field(default_factory=list)


def _require(obj: Mapping[str, Any], key: str, what: str) -> Any:
    if key not in obj:
        raise ClaimRelationError(f"{what}: missing {key!r} (write 'not_applicable' rather than omitting)")
    return obj[key]


def parse_response(text: str) -> dict[str, Any]:
    """Type- and vocabulary-validated parse of one response. Raises ClaimRelationError; never fills in a fabricated value."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ClaimRelationError("no JSON object")
    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ClaimRelationError(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("assertions"), list) or not raw["assertions"]:
        raise ClaimRelationError("assertions must be a non-empty list")
    for i, a in enumerate(raw["assertions"]):
        what = f"assertion {i}"
        if not isinstance(a, dict):
            raise ClaimRelationError(f"{what}: not an object")
        span = _require(a, "candidate_span", what)
        if not isinstance(span, dict) or not isinstance(span.get("text"), str) or not span["text"]:
            raise ClaimRelationError(f"{what}: candidate_span needs text")
        if not isinstance(_require(a, "evidence", what), list):
            raise ClaimRelationError(f"{what}: evidence must be a list")
        for key in ("actor", "time", "modality", "numeric", "relation"):
            _require(a, key, what)
        if a["modality"] not in MODALITIES:
            raise ClaimRelationError(f"{what}: modality must be one of {MODALITIES}")
        for key in ("numeric", "relation"):
            if a[key] != NOT_APPLICABLE and not isinstance(a[key], dict):
                raise ClaimRelationError(f"{what}: {key} must be an object or 'not_applicable'")
        if a["relation"] != NOT_APPLICABLE and a["relation"].get("type") not in RELATION_TYPES:
            raise ClaimRelationError(f"{what}: relation.type must be one of {RELATION_TYPES}")
        if _require(a, "verdict", what) not in VERDICTS:
            raise ClaimRelationError(f"{what}: verdict must be one of {VERDICTS}")
        if _require(a, "issue_type", what) not in ISSUE_TYPES:
            raise ClaimRelationError(f"{what}: issue_type must be one of {ISSUE_TYPES}")
        if not isinstance(_require(a, "rationale", what), str):
            raise ClaimRelationError(f"{what}: rationale must be a string")
    if not isinstance(raw.get("uncovered_spans", []), list):
        raise ClaimRelationError("uncovered_spans must be a list")
    return raw


def _exact(text: str, span: Mapping[str, Any]) -> bool:
    s, e, t = span.get("start"), span.get("end"), span.get("text")
    return isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text) and isinstance(t, str) and text[s:e] == t


def _locate(text: str, quote: str) -> tuple[int, int] | None:
    first = text.find(quote)
    return (first, first + len(quote)) if first >= 0 and text.find(quote, first + 1) < 0 else None


def validate_assertions(raw: Mapping[str, Any], source: str, candidate: str) -> list[Assertion]:
    """Exact-span checks. A candidate span whose offsets are wrong but whose text occurs exactly once is repaired and flagged; otherwise it is invalid.
    Evidence authenticity is recorded per quote and never changes the semantic verdict."""
    out = []
    for a in raw["assertions"]:
        flags: list[str] = []
        span = dict(a["candidate_span"])
        valid = _exact(candidate, span)
        if not valid:
            loc = _locate(candidate, span["text"])
            if loc is not None:
                span.update(start=loc[0], end=loc[1])
                valid, flags = True, [*flags, "candidate_span_offsets_repaired"]
            else:
                flags.append("candidate_span_invalid")
        evidence, authentic = [], []
        for ev in a["evidence"]:
            ev = dict(ev) if isinstance(ev, dict) else {"text": str(ev)}
            ok = _exact(source, ev)
            if not ok and isinstance(ev.get("text"), str) and ev["text"]:
                loc = _locate(source, ev["text"])
                if loc is not None:
                    ev.update(start=loc[0], end=loc[1])
                    ok = True
                    flags.append("evidence_offsets_repaired")
            evidence.append(ev)
            authentic.append(bool(ok))
        out.append(Assertion(span=span, evidence=evidence, fields={k: a[k] for k in ("actor", "time", "modality", "numeric", "relation")},
                             verdict=a["verdict"], issue_type=a["issue_type"], rationale=a["rationale"], flags=flags, span_valid=valid, evidence_authentic=authentic))
    return out


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group().lower()) for m in re.finditer(r"[A-Za-z0-9$%]+", text)]


def aggregate(assertions: Sequence[Assertion], raw: Mapping[str, Any], candidate: str) -> dict[str, Any]:
    """Code-side verdict. Span COVERAGE is measured on substantive words; it is not semantic completeness (a broad span can hide a second claim)."""
    words = [(s, e, w) for s, e, w in _word_spans(candidate) if w not in STOP and len(w) > 1]
    covered_ranges = [(a.span["start"], a.span["end"]) for a in assertions if a.span_valid]
    uncovered = [w for s, e, w in words if not any(cs <= s and e <= ce for cs, ce in covered_ranges)]
    declared = [u for u in raw.get("uncovered_spans", []) if isinstance(u, dict)]
    invalid = sum(1 for a in assertions if not a.span_valid)
    flags: list[str] = []
    broad = (len(assertions) == 1 and assertions[0].span_valid and (assertions[0].span["end"] - assertions[0].span["start"]) >= 0.9 * len(candidate)
             and len(CONJUNCTIONS.findall(candidate)) >= 1)
    if broad:
        flags.append("possible_broad_span_hiding_second_claim")
    unevidenced = [i for i, a in enumerate(assertions) if a.verdict == "supported" and not any(a.evidence_authentic)]
    if any(a.span_valid and a.verdict == "unsupported" for a in assertions):
        verdict = "unsupported"
    elif (uncovered or declared or invalid or unevidenced or broad or any(a.verdict == "uncertain" for a in assertions)):
        verdict = "uncertain"
    else:
        verdict = "supported"
    model_overall = (raw.get("overall") or {}).get("verdict") if isinstance(raw.get("overall"), dict) else None
    if uncovered:
        flags.append("unreviewed_substantive_words")
    if unevidenced:
        flags.append("supported_without_authentic_evidence")
    return {"verdict": verdict, "coverage": {"substantive_words": len(words), "uncovered_words": uncovered, "declared_uncovered_spans": len(declared),
                                             "invalid_candidate_spans": invalid, "note": "span coverage is not semantic completeness"},
            "flags": flags, "model_overall_verdict": model_overall, "model_overall_disagrees": model_overall is not None and model_overall != verdict}


# ------------------------------------------------------------------------------------------------------------- running
@dataclass(frozen=True)
class Response:
    text: str
    usage: Mapping[str, Any] | None = None  # raw provider usage, kept as given
    model: str | None = None  # model identity as reported by the provider; None when unknown


Request = Callable[[str], Response]


class RunAbort(RuntimeError):
    """Raised by a request function for a condition that makes full coverage impossible or the spend untrustworthy (cap, quota, model mismatch,
    invalid usage, cost anomaly). The runner stops at once, makes no further provider call, and withholds every comparison."""


def judge_case(case: Mapping[str, Any], request: Request) -> dict[str, Any]:
    """One request per case, no automatic retry. Inputs, the raw response, structure, usage and model identity are all recorded; unknowns stay null."""
    prompt = build_prompt(case["source"], case["candidate"], case.get("context", ""))
    record: dict[str, Any] = {"case_id": case["case_id"], "variant": VARIANT_ID, "content_hash": content_hash(), "prompt_sha256": sha256(prompt),
                              "input": {"source_sha256": sha256(case["source"]), "candidate_sha256": sha256(case["candidate"]), "source": case["source"],
                                        "candidate": case["candidate"], "context": case.get("context", "")},
                              "prompt": prompt, "raw_response": None, "usage": None, "model": None, "status": "request_error", "structured": None}
    try:
        resp = request(prompt)
    except RunAbort as exc:
        record.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        return record
    except Exception as exc:  # noqa: BLE001 - recorded, not retried
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record.update(raw_response=resp.text, usage=dict(resp.usage) if resp.usage is not None else None, model=resp.model)
    try:
        raw = parse_response(resp.text)
        assertions = validate_assertions(raw, case["source"], case["candidate"])
    except ClaimRelationError as exc:
        record.update(status="invalid_structure", error=str(exc))
        return record
    agg = aggregate(assertions, raw, case["candidate"])
    record["status"] = "ok"
    record["structured"] = {"assertions": [a.__dict__ for a in assertions], "uncovered_spans": raw.get("uncovered_spans", []), **agg}
    return record


def run_cases(cases: Sequence[Mapping[str, Any]], request: Request, out_dir: Path, *, protocol_sha256: str | None = None) -> dict[str, Any]:
    """Judge cases in order, persisting each record. FAIL-FAST: the first request error, malformed response, cap/quota/model/usage/cost anomaly stops the run
    immediately (no further provider call). Terminal status: COMPLETE only if every case is ok; ABORTED after a request/spend condition; INCOMPLETE after a
    malformed response. Cases never attempted are listed. Any non-COMPLETE terminal withholds every comparison."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ClaimRelationError("duplicate case ids")
    path = out_dir / "judgments.jsonl"
    if path.exists():
        raise ClaimRelationError("judgments.jsonl exists; refusing to overwrite (runs are not silently repeated)")
    records: list[dict[str, Any]] = []
    stop_reason = None
    with path.open("w", encoding="utf8") as fh:
        for case in cases:
            rec = judge_case(case, request)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
            if rec["status"] != "ok":
                stop_reason = {"case_id": rec["case_id"], "status": rec["status"], "error": rec.get("error")}
                break
    ok = sum(r["status"] == "ok" for r in records)
    if stop_reason is None:
        status = "COMPLETE"
    else:
        status = "INCOMPLETE" if stop_reason["status"] == "invalid_structure" else "ABORTED"
    verdicts = {v: sum(1 for r in records if r["status"] == "ok" and r["structured"]["verdict"] == v) for v in VERDICTS}
    terminal = {"status": status, "cases": len(cases), "attempted": len(records), "ok": ok, "not_attempted": [c["case_id"] for c in cases[len(records):]],
                "stopped_on": stop_reason, "verdicts": verdicts, "variant": VARIANT_ID, "content_hash": content_hash(), "protocol_sha256": protocol_sha256,
                "no_automatic_retry": True, "comparison": "withheld unless COMPLETE and every frozen check passes"}
    (out_dir / "terminal.json").write_text(json.dumps(terminal, indent=2), encoding="utf8")
    return terminal


# ------------------------------------------------------------------------------------------------------- comparison
def kappa_from_table(tp: float, fp: float, fn: float, tn: float) -> float | None:
    n = tp + fp + fn + tn
    if n <= 0:
        return None
    po, pe = (tp + tn) / n, ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return None if pe >= 1 - 1e-12 else (po - pe) / (1 - pe)


def _kappa(y: Sequence[int], p: Sequence[int]) -> float | None:
    tp = sum(1 for a, b in zip(y, p, strict=True) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p, strict=True) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p, strict=True) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, p, strict=True) if a == 0 and b == 0)
    return kappa_from_table(tp, fp, fn, tn)


def _summarize(ids: Sequence[str], labels: Mapping[str, int], baseline: Mapping[str, int], pred: Mapping[str, int], abstained: set[str],
               groups: Mapping[str, str], draws: int, seed: int) -> dict[str, Any]:
    import random

    y, base, new = [labels[i] for i in ids], [baseline[i] for i in ids], [pred[i] for i in ids]
    fam = sorted({groups[i] for i in ids})
    members = {g: [k for k, i in enumerate(ids) if groups[i] == g] for g in fam}
    rng = random.Random(seed)
    gains = []
    for _ in range(draws):
        idx = [k for g in (rng.choice(fam) for _ in fam) for k in members[g]]
        kb, kn = _kappa([y[k] for k in idx], [base[k] for k in idx]), _kappa([y[k] for k in idx], [new[k] for k in idx])
        if kb is not None and kn is not None:
            gains.append(kn - kb)
    gains.sort()
    interval = [gains[int(0.025 * len(gains))], gains[max(0, int(0.975 * len(gains)) - 1)]] if gains else None
    fixed_fr = sum(1 for a, b, c in zip(y, base, new, strict=True) if a == 1 and b == 0 and c == 1)
    fixed_fp = sum(1 for a, b, c in zip(y, base, new, strict=True) if a == 0 and b == 1 and c == 0)
    new_fr = sum(1 for a, b, c in zip(y, base, new, strict=True) if a == 1 and b == 1 and c == 0)
    new_fp = sum(1 for a, b, c in zip(y, base, new, strict=True) if a == 0 and b == 0 and c == 1)
    keep = [k for k, i in enumerate(ids) if i not in abstained]
    return {"n": len(ids), "documents": len(fam), "abstentions": sum(1 for i in ids if i in abstained), "kappa_baseline": _kappa(y, base),
            "kappa_candidate": _kappa(y, new), "false_rejections_fixed": fixed_fr, "false_passes_fixed": fixed_fp, "new_false_rejections": new_fr,
            "new_false_passes": new_fp, "errors_fixed": fixed_fr + fixed_fp, "new_errors": new_fr + new_fp,
            "gain_interval_95_document_grouped": interval, "bootstrap_draws_used": len(gains),
            "supplementary_excluding_abstentions": {"n": len(keep), "kappa_baseline": _kappa([y[k] for k in keep], [base[k] for k in keep]),
                                                    "kappa_candidate": _kappa([y[k] for k in keep], [new[k] for k in keep])}}


def compare_to_baseline(terminal: Mapping[str, Any], records: Sequence[Mapping[str, Any]], labels: Mapping[str, int], baseline: Mapping[str, int],
                        groups: Mapping[str, str], splits: Mapping[str, str], *, uncertain_as: int, draws: int = 2000, seed: int = 20261101,
                        target: float = 0.74) -> dict[str, Any]:
    """Frozen comparison. WITHHELD unless the run is COMPLETE and the case set matches labels, baseline, groups and splits exactly.
    The 128-case DEV result is primary and alone drives the improvement gate and any target statement; TRAIN is diagnostic; pooled is supplementary.
    `uncertain_as` (0 = not a pass) is a declared rule; abstentions stay in every denominator."""
    ids = sorted(labels)
    by = {r["case_id"]: r for r in records}
    same = set(by) == set(ids) and set(baseline) == set(ids) and set(groups) == set(ids) and set(splits) == set(ids) and len(by) == len(records)
    if terminal.get("status") != "COMPLETE" or not same or any(by[i]["status"] != "ok" for i in ids if i in by) or set(splits.values()) - {"train", "dev"}:
        return {"status": "WITHHELD_INCOMPLETE", "reason": "run not COMPLETE, or cases missing/extra/duplicated/failed, or population differs across files"}
    if uncertain_as not in (0, 1):
        raise ClaimRelationError("uncertain_as must be 0 or 1")
    pred = {i: {"supported": 1, "unsupported": 0, "uncertain": uncertain_as}[by[i]["structured"]["verdict"]] for i in ids}
    abstained = {i for i in ids if by[i]["structured"]["verdict"] == "uncertain"}
    dev = [i for i in ids if splits[i] == "dev"]
    train = [i for i in ids if splits[i] == "train"]
    out = {"status": "COMPLETE", "uncertain_mapping": f"uncertain->{uncertain_as}",
           "dev_primary": _summarize(dev, labels, baseline, pred, abstained, groups, draws, seed) if dev else None,
           "train_diagnostic_only": _summarize(train, labels, baseline, pred, abstained, groups, draws, seed) if train else None,
           "pooled_supplementary_not_for_gating": _summarize(ids, labels, baseline, pred, abstained, groups, draws, seed)}
    d = out["dev_primary"]
    improved = bool(d and d["errors_fixed"] > d["new_errors"] and d["gain_interval_95_document_grouped"] and d["gain_interval_95_document_grouped"][0] > 0)
    reached = bool(d and d["kappa_candidate"] is not None and d["kappa_candidate"] >= target)
    out["gate"] = {"IMPROVED_ON_DEV": improved, "criterion": "dev errors fixed > new dev errors AND 95% document-grouped lower bound of the dev kappa gain > 0",
                   "DEV_KAPPA_REACHES_0.74": reached,
                   "note": "a development result on inspected cases is not held-out confirmation and does not pass the original benchmark target; "
                           "pooled or TRAIN results never drive the gate"}
    return out


# ------------------------------------------------------------------------------------------------------- identity
def check_v18_identity() -> dict[str, Any]:
    """v18 must remain byte/identity compatible: the candidate has its own variant and hash and does not alter v18's config or template hash."""
    from . import judge
    from .run_calibration import GRID

    cfg = next(c for c in GRID if c.variant_id == "v18-reasoning-o4mini")
    got = {"v18_config_hash": cfg.config_hash, "v18_prompt_template_hash": judge.prompt_template_hash("rubric")}
    ok = got["v18_config_hash"] == V18_CONFIG_HASH and got["v18_prompt_template_hash"] == V18_PROMPT_TEMPLATE_HASH
    if VARIANT_ID in {c.variant_id for c in GRID}:
        raise ClaimRelationError("candidate variant id collides with a registered variant")
    return {**got, "unchanged": ok, "candidate_variant": VARIANT_ID, "candidate_content_hash": content_hash()}


# ------------------------------------------------------------------------------------------------------- provider (future)
def provider_request_factory(model: str, effort: str, cap_usd: float, journal_path: Path, prices: tuple[float, float]) -> Request:
    """A single-HTTP-attempt request with a fsynced spend journal. Every spend anomaly raises RunAbort so the runner stops with no further call."""
    from aisys import llm

    from . import audit2_v3 as a3
    from .audit2_v2 import Journal, SpendError
    from .multi_judge import single_request

    journal = Journal(Path(journal_path), cap_usd)

    def request(prompt: str) -> Response:
        bound_in = len(prompt.encode("utf8")) + a3.MESSAGE_OVERHEAD_TOKENS
        reserve = (bound_in * prices[0] + MAX_TOKENS * prices[1]) / 1e6
        try:
            rid = journal.reserve("claim-relation", 1, reserve)
        except SpendError as exc:  # cap would be exceeded
            raise RunAbort(f"cap: {exc}") from exc
        try:
            text, usage, served = single_request(model, effort, [{"role": "user", "content": prompt}])
        except llm.QuotaExceededError as exc:
            journal.settle(rid, reserve, None)
            raise RunAbort(f"quota: {exc}") from exc
        except Exception:
            journal.settle(rid, reserve, None)  # a failed request may have billed: charge the bound; the runner records the error and stops
            raise
        valid = a3.validated_usage(usage)
        if valid is None:
            journal.settle(rid, reserve, None)
            raise RunAbort("invalid or missing provider usage (reservation charged)")
        try:
            journal.settle(rid, reserve, (valid[0] * prices[0] + valid[1] * prices[1]) / 1e6)
        except SpendError as exc:
            raise RunAbort(f"cost anomaly: {exc}") from exc
        if valid[0] > bound_in or valid[1] > MAX_TOKENS:
            raise RunAbort("usage above the reserved token bounds")
        if served and not served.startswith(model):
            raise RunAbort(f"model mismatch: served {served!r}, requested {model!r}")
        return Response(text=text, usage=usage, model=served or None)

    return request


# ------------------------------------------------------------------------------------------------------------- inputs
INPUT_FILES = ("cases.json", "labels.json", "groups.json", "baseline_v18.json", "splits.json")
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20261101, 2000
MAX_CAP_USD = 6.0  # the ceiling written into the proposal; a protocol above it is refused
FROZEN_RULE = {"supported": 1, "unsupported": 0, "uncertain": 0, "uncertain_as": 0}
PRICE_ASSUMPTIONS = {"o4-mini": (1.10, 4.40)}  # USD per 1M input / output tokens; reasoning tokens bill as output


class ProtocolError(ClaimRelationError):
    """The frozen experiment no longer matches its manifest (altered, missing, duplicated or extra records, or changed code/identity/rule)."""


def build_cases(splits: Sequence[str], out_dir: Path) -> dict[str, Any]:
    """Label-free case file for `run`, plus SEPARATE labels / groups / splits / frozen-v18 baseline files used only by `compare`. The sealed test split
    is never loaded."""
    from .experiments import JudgmentCache, _cache_key
    from .multi_judge import DATA, load_cases
    from .run_calibration import GRID

    if "test" in splits:
        raise ClaimRelationError("the sealed test split is not available to this tool")
    cfg = next(c for c in GRID if c.variant_id == "v18-reasoning-o4mini")
    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, labels, groups, baseline, split_of = [], {}, {}, {}, {}
    for case, label, split in load_cases(list(splits)):
        j = cache.get(_cache_key(case, cfg))
        if j is None or not j.usable:
            raise ClaimRelationError(f"missing frozen v18 judgment for {case.case_id}")
        cases.append({"case_id": case.case_id, "source": case.context, "candidate": case.candidate_output.strip()})
        labels[case.case_id], groups[case.case_id], baseline[case.case_id] = label, case.group_id, int(float(j.raw_score) >= 1.0)
        split_of[case.case_id] = split
    for name, value in (("cases.json", cases), ("labels.json", labels), ("groups.json", groups), ("baseline_v18.json", baseline), ("splits.json", split_of)):
        target = out_dir / name
        if target.exists():
            raise ClaimRelationError(f"{name} exists; refusing to overwrite")
        target.write_text(json.dumps(value, sort_keys=True), encoding="utf8")
    return {"cases": len(cases), "splits": list(splits), "files": list(INPUT_FILES)}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_inputs(inputs_dir: Path) -> dict[str, Any]:
    out = {}
    for name in INPUT_FILES:
        p = Path(inputs_dir) / name
        if not p.exists():
            raise ProtocolError(f"missing input file {name}")
        out[name] = json.loads(p.read_text(encoding="utf8"))
    cases = out["cases.json"]
    if not isinstance(cases, list):
        raise ProtocolError("cases.json must be a list")
    ids = [c.get("case_id") for c in cases]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) for i in ids):
        raise ProtocolError("cases.json has duplicate or invalid case ids")
    if any(set(c) != {"case_id", "source", "candidate"} for c in cases):
        raise ProtocolError("cases.json rows must contain exactly case_id, source, candidate (no labels, predictions or other fields)")
    for name in INPUT_FILES[1:]:
        if not isinstance(out[name], dict) or set(out[name]) != set(ids):
            raise ProtocolError(f"{name} case ids differ from cases.json (missing or extra records)")
    if set(out["splits.json"].values()) - {"train", "dev"}:
        raise ProtocolError("splits.json may contain only train and dev")
    if set(out["labels.json"].values()) - {0, 1} or set(out["baseline_v18.json"].values()) - {0, 1}:
        raise ProtocolError("labels and baseline must be 0/1")
    return out


def _protocol_body(inputs_dir: Path, model: str, effort: str, cap_usd: float) -> dict[str, Any]:
    data = _load_inputs(inputs_dir)
    counts = {s: sum(1 for v in data["splits.json"].values() if v == s) for s in ("train", "dev")}
    ident = check_v18_identity()
    prices = PRICE_ASSUMPTIONS[model]
    return {
        "protocol": "claim-relation-v1-ablation", "input_sha256": {n: _file_sha(Path(inputs_dir) / n) for n in INPUT_FILES},
        "population": {"cases": len(data["cases.json"]), "per_split": counts, "case_ids_sha256": sha256(json.dumps(sorted(data["labels.json"])))},
        "candidate": {"variant": VARIANT_ID, "content_hash": content_hash(), "prompt_sha256": sha256(PROMPT), "code_sha256": _file_sha(Path(__file__))},
        "v18_identity": {k: ident[k] for k in ("v18_config_hash", "v18_prompt_template_hash")} | {"unchanged": ident["unchanged"]},
        "model": model, "reasoning_effort": effort, "decision_rule": dict(FROZEN_RULE),
        "gate": "dev errors fixed > new dev errors AND dev 95% document-grouped lower bound of kappa gain > 0; train diagnostic; pooled supplementary",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS}, "price_usd_per_1m": {"input": prices[0], "output": prices[1]}, "cap_usd": cap_usd,
        "max_tokens": MAX_TOKENS, "one_request_per_case_no_retry": True, "fail_fast": True,
    }


def freeze_protocol(inputs_dir: Path, out_path: Path, *, model: str = "o4-mini", effort: str = "medium", cap_usd: float = 6.0) -> dict[str, Any]:
    out_path = Path(out_path)
    if out_path.exists():
        raise ClaimRelationError("protocol exists; a frozen protocol is never overwritten")
    if model not in PRICE_ASSUMPTIONS:
        raise ClaimRelationError(f"no price assumption for {model!r}")
    if not 0 < cap_usd <= MAX_CAP_USD:
        raise ClaimRelationError(f"cap must be in (0, {MAX_CAP_USD}] USD")
    body = _protocol_body(Path(inputs_dir), model, effort, cap_usd)
    if not body["v18_identity"]["unchanged"]:
        raise ProtocolError("v18 identity differs from the frozen values")
    out_path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf8")
    return body


def verify_protocol(inputs_dir: Path, protocol_path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Recompute everything from the files on disk and compare with the frozen manifest. Raises ProtocolError on any difference.
    `expected_sha256` is the manifest hash printed at freeze time: it anchors the manifest itself, so a hand-edited model, effort or cap is refused."""
    if expected_sha256 is not None and _file_sha(protocol_path) != expected_sha256:
        raise ProtocolError("protocol file hash differs from the hash recorded at freeze time (the manifest itself was changed)")
    frozen = json.loads(Path(protocol_path).read_text(encoding="utf8"))
    if frozen.get("model") not in PRICE_ASSUMPTIONS:
        raise ProtocolError("model has no price assumption")
    if not 0 < frozen.get("cap_usd", 0) <= MAX_CAP_USD:
        raise ProtocolError(f"cap outside (0, {MAX_CAP_USD}]")
    current = _protocol_body(Path(inputs_dir), frozen["model"], frozen["reasoning_effort"], frozen["cap_usd"])
    for key, value in current.items():
        if frozen.get(key) != value:
            raise ProtocolError(f"frozen protocol differs from the current state at {key!r}")
    return frozen


def run_frozen(inputs_dir: Path, protocol_path: Path, out_dir: Path, request: Request, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Verify the frozen protocol BEFORE the first request, then run exactly the frozen cases."""
    verify_protocol(inputs_dir, protocol_path, expected_sha256)
    cases = _load_inputs(inputs_dir)["cases.json"]
    return run_cases(cases, request, out_dir, protocol_sha256=_file_sha(protocol_path))


def compare_run(inputs_dir: Path, run_dir: Path, protocol_path: Path, *, uncertain_as: int | None = None, expected_sha256: str | None = None) -> dict[str, Any]:
    """Re-verify the frozen protocol and the run's identity, then compare with the FROZEN decision rule. A different rule or population is refused."""
    inputs_dir, run_dir = Path(inputs_dir), Path(run_dir)
    frozen = verify_protocol(inputs_dir, protocol_path, expected_sha256)
    rule = frozen["decision_rule"]["uncertain_as"]
    if uncertain_as is not None and uncertain_as != rule:
        raise ProtocolError(f"decision rule differs from the frozen protocol (frozen uncertain->{rule}, requested uncertain->{uncertain_as})")
    terminal = json.loads((run_dir / "terminal.json").read_text(encoding="utf8"))
    if terminal.get("protocol_sha256") != _file_sha(protocol_path) or terminal.get("content_hash") != content_hash():
        raise ProtocolError("the run was not produced under this frozen protocol and candidate code")
    records = [json.loads(x) for x in (run_dir / "judgments.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
    data = _load_inputs(inputs_dir)
    target = run_dir / "comparison.json"
    if target.exists():
        raise ClaimRelationError("comparison.json exists; the comparison is written once")
    by_case = {c["case_id"]: c for c in data["cases.json"]}
    for r in records:
        c = by_case.get(r["case_id"])
        if c is None or r["input"]["source_sha256"] != sha256(c["source"]) or r["input"]["candidate_sha256"] != sha256(c["candidate"]):
            raise ProtocolError(f"run record {r['case_id']} does not match the frozen inputs (extra, altered or duplicated record)")
    result = compare_to_baseline(terminal, records, data["labels.json"], data["baseline_v18.json"], data["groups.json"], data["splits.json"],
                                 uncertain_as=rule, draws=frozen["bootstrap"]["draws"], seed=frozen["bootstrap"]["seed"])
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


# ------------------------------------------------------------------------------------------------------------- demo
def synthetic_fixtures() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "claim_relation_synthetic.json"
    return json.loads(path.read_text(encoding="utf8"))


def scripted_response(fixture: Mapping[str, Any]) -> str:
    """Builds the scripted model reply for a synthetic fixture: spans are computed from the texts so the mock is exact."""
    source, candidate = fixture["source"], fixture["candidate"]
    assertions = []
    for a in fixture["script"]["assertions"]:
        cs = candidate.find(a["span"])
        evidence = []
        for q in a.get("evidence", []):
            es = source.find(q)
            evidence.append({"start": es, "end": es + len(q), "text": q})
        assertions.append({"candidate_span": {"start": cs, "end": cs + len(a["span"]), "text": a["span"]}, "evidence": evidence,
                           "actor": a.get("actor", NOT_APPLICABLE), "time": a.get("time", NOT_APPLICABLE), "modality": a.get("modality", NOT_APPLICABLE),
                           "numeric": a.get("numeric", NOT_APPLICABLE), "relation": a.get("relation", NOT_APPLICABLE), "verdict": a["verdict"],
                           "issue_type": a["issue_type"], "rationale": a.get("rationale", "scripted synthetic rationale")})
    return json.dumps({"assertions": assertions, "uncovered_spans": fixture["script"].get("uncovered_spans", []),
                       "overall": {"verdict": fixture["script"].get("overall", "supported"), "rationale": "scripted"}})


def demo(out: Path) -> dict[str, Any]:
    fixtures = synthetic_fixtures()
    by_prompt = {sha256(build_prompt(f["source"], f["candidate"])): f for f in fixtures}

    def mock(prompt: str) -> Response:
        return Response(text=scripted_response(by_prompt[sha256(prompt)]), usage=None, model="MOCK-NO-INFERENCE")

    cases = [{"case_id": f["case_id"], "source": f["source"], "candidate": f["candidate"]} for f in fixtures]
    terminal = run_cases(cases, mock, out)
    recs = [json.loads(line) for line in (Path(out) / "judgments.jsonl").read_text(encoding="utf8").splitlines()]
    expected = {f["case_id"]: f["expected_verdict"] for f in fixtures}
    matches = sum(1 for r in recs if r["status"] == "ok" and r["structured"]["verdict"] == expected[r["case_id"]])
    return {"terminal": terminal, "scripted_expectations_met": f"{matches}/{len(fixtures)}", "identity": check_v18_identity(),
            "warning": "mock inference on SYNTHETIC fixtures tests schema and aggregation only; it says nothing about model accuracy or human agreement"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "identity", "build-cases", "freeze", "run", "compare"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--model", default="o4-mini")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--cap", type=float, default=6.0)
    parser.add_argument("--uncertain-as", type=int, choices=[0, 1])
    parser.add_argument("--confirm", default="")
    parser.add_argument("--protocol-sha256", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "identity":
            print(json.dumps(check_v18_identity(), indent=2))
        elif args.command == "demo":
            if args.out is None:
                parser.error("--out required")
            print(json.dumps(demo(args.out), indent=2))
        elif args.command == "build-cases":
            if args.out is None:
                parser.error("--out required")
            print(json.dumps(build_cases(args.splits.split(","), args.out), indent=2))
        elif args.command == "freeze":
            if args.inputs is None or args.protocol is None:
                parser.error("--inputs and --protocol required")
            body = freeze_protocol(args.inputs, args.protocol, model=args.model, effort=args.effort, cap_usd=args.cap)
            print(json.dumps({"protocol_sha256": _file_sha(args.protocol), **body}, indent=2))
        elif args.command == "compare":
            if args.inputs is None or args.run is None or args.protocol is None:
                parser.error("--inputs, --run and --protocol required")
            print(json.dumps(compare_run(args.inputs, args.run, args.protocol, uncertain_as=args.uncertain_as, expected_sha256=args.protocol_sha256), indent=2))
        else:
            if args.inputs is None or args.protocol is None or args.out is None:
                parser.error("--inputs, --protocol and --out required")
            if args.confirm != "RUN-CLAIM-RELATION-PAID":
                print("REFUSED: provider runs need --confirm RUN-CLAIM-RELATION-PAID (separate approval required)", file=sys.stderr)
                return 2
            if not args.protocol_sha256:
                print("REFUSED: --protocol-sha256 (printed at freeze) is required", file=sys.stderr)
                return 2
            frozen = verify_protocol(args.inputs, args.protocol, args.protocol_sha256)
            request = provider_request_factory(frozen["model"], frozen["reasoning_effort"], frozen["cap_usd"], Path(args.out) / "spend_journal.jsonl",
                                               (frozen["price_usd_per_1m"]["input"], frozen["price_usd_per_1m"]["output"]))
            terminal = run_frozen(args.inputs, args.protocol, args.out, request, expected_sha256=args.protocol_sha256)
            print(json.dumps(terminal, indent=2))
            return 0 if terminal["status"] == "COMPLETE" else 3
    except ClaimRelationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
