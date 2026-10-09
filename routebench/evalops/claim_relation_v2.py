"""`claim-relation-v2`: an OFFLINE candidate that repairs four demonstrated post-processing defects of the frozen `claim-relation-v1`.

v1 (its code, prompt and frozen manifest) is untouched and remains the measured version. v2 keeps v1's PROMPT byte-for-byte, so any later difference between
the two comes only from response handling. Nothing here has been run against a model; the repairs are verified on synthetic fixtures only and are
NOT a demonstrated accuracy improvement. A v2 evaluation would need its own frozen protocol and approval.

Repairs (see .local/overnight-20261004/cycles.md for measurements on the 257 frozen source/candidate pairs):
  1. authenticity after whitespace / quote / dash normalisation, mapped back to original offsets; a repeated quote is authentic (flagged ambiguous);
  2. the single-broad-span demotion needs >=2 substantive clauses and fewer evidence spans than clauses (v1 fired on 73% of real candidates);
  3. unambiguous not-applicable synonyms are normalised and flagged; missing keys and unknown vocabulary stay invalid;
  4. word coverage uses Unicode word characters.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import claim_relation as v1
from .claim_relation import (
    ISSUE_TYPES,
    MODALITIES,
    NOT_APPLICABLE,
    RELATION_TYPES,
    STOP,
    VERDICTS,
    Assertion,
    ClaimRelationError,
    Request,
    Response,
    _require,
    build_prompt,
    sha256,
)

VARIANT_ID = "claim-relation-v2"
# Words v1 treats as stop words but that change meaning when dropped: negation, modals, quantifiers. A reviewed assertion must cover them.
MEANING_BEARING = frozenset({"not", "no", "nor", "never", "cannot", "neither", "without", "will", "would", "can", "could", "shall", "should", "may", "might",
                             "must", "all", "any", "each", "every", "only", "both", "than"})
NA_SYNONYMS = frozenset({"", "n/a", "na", "none", "null", "not applicable", "not_applicable", "not-applicable"})
CLAUSE_SPLIT = re.compile(r"\s*(?:;|,?\s+\b(?:and|but|while|whereas)\b)\s+", re.IGNORECASE)
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}


def content_hash() -> str:
    code = "".join(inspect.getsource(f) for f in (normalise_with_map, locate, parse_response, validate_assertions, aggregate, _words, _substantive, _clauses, _na))
    return sha256(v1.PROMPT + json.dumps([VARIANT_ID, sorted(NA_SYNONYMS), CLAUSE_SPLIT.pattern, sorted(STOP), sorted(MEANING_BEARING)]) + code)


# ------------------------------------------------------------------------------------------------ repair 1: normalised location
def normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space, map curly quotes/dashes/NBSP to ASCII; return the normalised text and, per normalised char, its original index."""
    out: list[str] = []
    index: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        ch = _QUOTES.get(ch, ch)
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        out.append(ch)
        index.append(i)
    return "".join(out), index


def locate(text: str, quote: str) -> tuple[int, int, bool, bool] | None:
    """(start, end, ambiguous, normalised_match) of `quote` in `text` as ORIGINAL offsets, or None. Exact match first, then normalised."""
    if not quote:
        return None
    first = text.find(quote)
    if first >= 0:
        return first, first + len(quote), text.find(quote, first + 1) >= 0, False
    norm_text, idx = normalise_with_map(text)
    norm_quote, _ = normalise_with_map(quote.strip())
    if not norm_quote.strip():
        return None
    hit = norm_text.find(norm_quote.strip())
    if hit < 0:
        return None
    last = hit + len(norm_quote.strip()) - 1
    return idx[hit], idx[last] + 1, norm_text.find(norm_quote.strip(), hit + 1) >= 0, True


# ------------------------------------------------------------------------------------------------ repair 3: explicit synonyms
def _na(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in NA_SYNONYMS)


def parse_response(text: str) -> dict[str, Any]:
    """v1's validation with unambiguous not-applicable synonyms normalised (and flagged). Missing keys / unknown vocabulary stay invalid."""
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
        normalised = a.setdefault("_normalised_fields", [])
        for key in ("modality", "numeric", "relation"):
            if _na(a[key]) and a[key] != NOT_APPLICABLE:
                a[key] = NOT_APPLICABLE
                normalised.append(key)
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


def validate_assertions(raw: Mapping[str, Any], source: str, candidate: str) -> list[Assertion]:
    out = []
    for a in raw["assertions"]:
        flags = [f"normalised_{k}" for k in a.get("_normalised_fields", [])]
        span = dict(a["candidate_span"])
        valid = v1._exact(candidate, span)
        if not valid:
            hit = locate(candidate, span["text"])
            if hit is not None and not hit[2]:
                span.update(start=hit[0], end=hit[1])
                valid = True
                flags.append("candidate_span_normalised_match" if hit[3] else "candidate_span_offsets_repaired")
            else:
                flags.append("candidate_span_invalid" if hit is None else "candidate_span_ambiguous")
        evidence, authentic = [], []
        for ev in a["evidence"]:
            ev = dict(ev) if isinstance(ev, dict) else {"text": str(ev)}
            ok = v1._exact(source, ev)
            if not ok and isinstance(ev.get("text"), str):
                hit = locate(source, ev["text"])
                if hit is not None:
                    ev.update(start=hit[0], end=hit[1])
                    ok = True
                    flags.append("evidence_normalised_match" if hit[3] else "evidence_offsets_repaired")
                    if hit[2]:
                        flags.append("evidence_quote_occurs_more_than_once")
            evidence.append(ev)
            authentic.append(bool(ok))
        out.append(Assertion(span=span, evidence=evidence, fields={k: a[k] for k in ("actor", "time", "modality", "numeric", "relation")}, verdict=a["verdict"],
                             issue_type=a["issue_type"], rationale=a["rationale"], flags=flags, span_valid=valid, evidence_authentic=authentic))
    return out


# ------------------------------------------------------------------------------------------------ repairs 2 and 4
def _words(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group().lower()) for m in re.finditer(r"[^\W_]+(?:['\u2019][^\W_]+)*|[$%]", text)]


def _substantive(word: str) -> bool:
    """Must be covered by a reviewed assertion: content words, ANY digit-bearing token (single digits too), negation incl. n't contractions, and modals."""
    if word in MEANING_BEARING or word.endswith(("n't", "n\u2019t")) or any(ch.isdigit() for ch in word):
        return True
    return word not in STOP and len(word) > 1


def _clauses(candidate: str) -> list[str]:
    """Clauses with >=3 substantive words: the unit a single assertion would have to cover to hide a second claim."""
    parts = [p for p in CLAUSE_SPLIT.split(candidate) if p and p.strip()]
    return [p for p in parts if sum(1 for _s, _e, w in _words(p) if w not in STOP and len(w) > 1) >= 3]


def aggregate(assertions: Sequence[Assertion], raw: Mapping[str, Any], candidate: str) -> dict[str, Any]:
    words = [(s, e, w) for s, e, w in _words(candidate) if _substantive(w)]
    covered = [(a.span["start"], a.span["end"]) for a in assertions if a.span_valid]
    uncovered = [w for s, e, w in words if not any(cs <= s and e <= ce for cs, ce in covered)]
    declared = [u for u in raw.get("uncovered_spans", []) if isinstance(u, dict)]
    invalid = sum(1 for a in assertions if not a.span_valid)
    flags: list[str] = []
    clauses = _clauses(candidate)
    one = len(assertions) == 1 and assertions[0].span_valid
    broad = bool(one and (assertions[0].span["end"] - assertions[0].span["start"]) >= 0.9 * len(candidate) and len(clauses) >= 2
                 and len(assertions[0].evidence) < len(clauses))
    if broad:
        flags.append("possible_broad_span_hiding_second_claim")
    unevidenced = [i for i, a in enumerate(assertions) if a.verdict == "supported" and not any(a.evidence_authentic)]
    if any(a.span_valid and a.verdict == "unsupported" for a in assertions):
        verdict = "unsupported"
    elif uncovered or declared or invalid or unevidenced or broad or any(a.verdict == "uncertain" for a in assertions):
        verdict = "uncertain"
    else:
        verdict = "supported"
    if uncovered:
        flags.append("unreviewed_substantive_words")
    if unevidenced:
        flags.append("supported_without_authentic_evidence")
    overall = raw.get("overall") if isinstance(raw.get("overall"), dict) else {}
    model_overall = overall.get("verdict")
    return {"verdict": verdict, "coverage": {"substantive_words": len(words), "uncovered_words": uncovered, "declared_uncovered_spans": len(declared),
                                             "invalid_candidate_spans": invalid, "note": "span coverage is not semantic completeness"},
            "flags": flags, "model_overall_verdict": model_overall, "model_overall_disagrees": model_overall is not None and model_overall != verdict}


def judge_case(case: Mapping[str, Any], request: Request) -> dict[str, Any]:
    """Same record shape as v1.judge_case, same prompt; only the response handling differs. One request, no retry."""
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
    record.update(raw_response=resp.text, usage=dict(resp.usage) if resp.usage is not None else None, model=resp.model)
    try:
        raw = parse_response(resp.text)
        assertions = validate_assertions(raw, case["source"], case["candidate"])
    except ClaimRelationError as exc:
        record.update(status="invalid_structure", error=str(exc))
        return record
    record["status"] = "ok"
    record["structured"] = {"assertions": [a.__dict__ for a in assertions], "uncovered_spans": raw.get("uncovered_spans", []),
                            **aggregate(assertions, raw, case["candidate"])}
    return record


def reprocess_v1_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive a v2 record from a stored v1 record's raw response WITHOUT any model call (for a future comparison of response handling only)."""
    case = {"case_id": record["case_id"], "source": record["input"]["source"], "candidate": record["input"]["candidate"], "context": record["input"].get("context", "")}
    return judge_case(case, lambda _p: Response(record["raw_response"], record.get("usage"), record.get("model")))


def file_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
