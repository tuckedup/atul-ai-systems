"""`claim-relation-v3`: a separately versioned AGGREGATION repair of `claim-relation-v2`. v1 and v2 (code, prompt, frozen protocols, measured results) are untouched.

Defect (found by reproduction on the frozen run, see docs/ROUTEBENCH_PEAK_REPRODUCTION_ANALYSIS_20261004.md): the prompt asks the model to list ANY candidate text it did
not cover, and v2.aggregate demoted a case to "uncertain" whenever that list was non-empty, including a lone full stop or whitespace. 26 of 128 frozen responses were
demoted although every assertion was supported.

Repair: a declared uncovered span makes the case uncertain only if the candidate text it points at contains a SUBSTANTIVE word (content words, any digit-bearing token,
negation incl. n't, modals, quantifiers - the same `_substantive` policy as v2). Whitespace, punctuation and ordinary function words are not factual uncertainty.
The declared text is never trusted on its own: the span is resolved against the stored candidate (verified offsets, else a unique text match); if it cannot be
resolved, the words of the DECLARED text are used, so an unverifiable span containing substantive words still demotes. A model-supplied reason can never exempt a span.
Independently of the declared list, uncovered substantive words are computed from the assertion spans themselves (unchanged from v2), so an omitted "not", a modal or a
number is still caught. All other guards (invalid offsets, unevidenced support, broad span, assertion-level uncertain/unsupported) are unchanged.

Nothing here has been run against a model. A post-hoc offline reprocessing of the 128 cached frozen responses is a development diagnosis, not an accepted experiment.
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from typing import Any

from . import claim_relation as v1
from . import claim_relation_v2 as v2
from .claim_relation import (
    Assertion,
    ClaimRelationError,
    Request,
    Response,
    build_prompt,
    sha256,
)

VARIANT_ID = "claim-relation-v3"
parse_response = v2.parse_response
validate_assertions = v2.validate_assertions


def content_hash() -> str:
    return sha256(v2.content_hash() + inspect.getsource(declared_span_has_substance) + inspect.getsource(aggregate) + VARIANT_ID)


def declared_span_has_substance(span: Mapping[str, Any], candidate: str) -> tuple[bool, str]:
    """(has_substantive_word, how_resolved). Every reading of the span is checked - the candidate slice at its offsets, a unique text match, and the declared wording -
    and ANY substantive word among them keeps the demotion. A span is never exempted merely because the model called it unimportant."""
    text = span.get("text") if isinstance(span.get("text"), str) else ""
    start, end = span.get("start"), span.get("end")
    readings: list[str] = []
    in_range = isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(candidate)
    if in_range:
        readings.append(candidate[start:end])
    how = "verified_offsets" if in_range and (not text or candidate[start:end] == text) else ""
    if not how:
        hit = v2.locate(candidate, text) if text else None
        if hit is not None and not hit[2]:
            readings.append(candidate[hit[0]:hit[1]])
            how = "unique_text_match"
        else:
            how = "unresolved_declared_text" if not in_range else "offsets_disagree_with_declared_text"
    readings.append(text)  # the declared wording is always considered as well
    own = any(v2._substantive(w) for r in readings for _s, _e, w in v2._words(r))
    return own, how


def aggregate(assertions: Sequence[Assertion], raw: Mapping[str, Any], candidate: str) -> dict[str, Any]:
    base = v2.aggregate(assertions, {**raw, "uncovered_spans": []}, candidate)  # v2 logic with the declared list set aside
    declared = [u for u in raw.get("uncovered_spans", []) if isinstance(u, dict)]
    checked = [(u, *declared_span_has_substance(u, candidate)) for u in declared]
    substantive = [u for u, own, _how in checked if own]
    flags = list(base["flags"])
    if substantive:
        flags.append("declared_uncovered_span_with_substantive_words")
    ignored = len(declared) - len(substantive)
    verdict = base["verdict"]
    if verdict == "supported" and substantive:
        verdict = "uncertain"
    cov = dict(base["coverage"]) | {"declared_uncovered_spans": len(declared), "declared_spans_substantive": len(substantive), "declared_spans_ignored_non_substantive": ignored,
                                    "declared_span_resolution": [how for _u, _own, how in checked]}
    overall = raw.get("overall") if isinstance(raw.get("overall"), dict) else {}
    model_overall = overall.get("verdict")
    return {"verdict": verdict, "coverage": cov, "flags": flags, "model_overall_verdict": model_overall, "model_overall_disagrees": model_overall is not None and model_overall != verdict}


def judge_case(case: Mapping[str, Any], request: Request) -> dict[str, Any]:
    prompt = build_prompt(case["source"], case["candidate"], case.get("context", ""))
    record: dict[str, Any] = {"case_id": case["case_id"], "variant": VARIANT_ID, "content_hash": content_hash(), "prompt_sha256": sha256(prompt),
                              "input": {"source_sha256": sha256(case["source"]), "candidate_sha256": sha256(case["candidate"]), "source": case["source"],
                                        "candidate": case["candidate"], "context": case.get("context", "")},
                              "prompt": prompt, "raw_response": None, "usage": None, "model": None, "status": "request_error", "structured": None}
    try:
        resp = request(prompt)
    except v1.RunAbort as exc:
        record.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        return record
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    return from_response(record, resp, case)


def from_response(record: dict[str, Any], resp: Response, case: Mapping[str, Any]) -> dict[str, Any]:
    record.update(raw_response=resp.text, usage=dict(resp.usage) if resp.usage is not None else None, model=resp.model)
    try:
        raw = parse_response(resp.text)
        assertions = validate_assertions(raw, case["source"], case["candidate"])
    except ClaimRelationError as exc:
        record.update(status="invalid_structure", error=str(exc))
        return record
    record["status"] = "ok"
    record["structured"] = {"assertions": [a.__dict__ for a in assertions], "uncovered_spans": raw.get("uncovered_spans", []), **aggregate(assertions, raw, case["candidate"])}
    return record


def reprocess_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive a v3 record from a stored record's raw response WITHOUT any model call. The input record is not modified."""
    case = {"case_id": record["case_id"], "source": record["input"]["source"], "candidate": record["input"]["candidate"], "context": record["input"].get("context", "")}
    fresh: dict[str, Any] = {"case_id": case["case_id"], "variant": VARIANT_ID, "content_hash": content_hash(), "prompt_sha256": record["prompt_sha256"],
                             "input": json.loads(json.dumps(record["input"])), "prompt": record["prompt"], "raw_response": None, "usage": None, "model": None,
                             "status": "request_error", "structured": None, "reprocessed_from": {"variant": record["variant"], "content_hash": record["content_hash"]}}
    return from_response(fresh, Response(record["raw_response"], record.get("usage"), record.get("model")), case)
