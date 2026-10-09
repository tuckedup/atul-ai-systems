"""`typed-veto-v1`: an OPT-IN, separately versioned candidate that lets a claim-relation audit VETO a v18 pass, for six typed distinctions only.

v18 stays the judge. The final binary verdict equals v18's verdict unless ALL of the following hold, in which case a v18 PASS becomes a REJECT:
  1. v18 passed the case (a v18 rejection is never touched and is never turned into a pass);
  2. a claim-relation-v3 audit record exists for the case, with status ok, the frozen variant id and judge-code hash, and it is BOUND to this exact case: the expected case id, the
     SHA-256 of the full source and of the full candidate (checked both against the stored hashes and against the stored texts), the exact prompt (rebuilt from the case and hashed),
     the expected model identity, and a stored structure that re-derives from the stored raw response. An audit of a different document is refused even when the candidate text is identical;
  3. the audit's case-level verdict is explicit "unsupported";
  4. no assertion of the audit is "uncertain" (any uncertainty anywhere leaves v18 unchanged, so abstentions can never create a rejection);
  5. at least one assertion is explicitly "unsupported", has a type in the frozen WHITELIST, and carries a VALID candidate span (offsets re-verified here against the stored
     candidate text: 0 <= start < end <= len, candidate[start:end] == span text).
The whitelist is frozen BEFORE further evaluation and is not to be tuned on dev: the six pre-declared distinctions (tense/modality, entity/attachment, unlinked facts,
purpose vs result, authorization vs payment, share vs increase). `attribute_transfer`, `unsupported_addition` and `other` are excluded. The rule cannot reduce false
rejections of v18; it can only convert false passes into rejections, and its price is the supported claims it wrongly vetoes. Both directions are always reported.

Provenance of the whitelist, disclosed: it was chosen from the 2026-09-29 independent label audit's categories of genuine judge misses, but after the author had seen the
audit's dev outcomes. Its dev number is therefore a post-hoc, in-sample figure and is not evidence of generalisation.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import claim_relation_v3 as cr3
from .claim_relation import build_prompt, sha256

VARIANT_ID = "typed-veto-v1.1"  # v1 plus audit-to-case binding; the whitelist and decision semantics are unchanged
EXPECTED_AUDIT_MODEL = "claude-opus-5-5"
WHITELIST = frozenset({"possible_vs_confirmed", "entity_mismatch", "unlinked_facts", "purpose_vs_result", "authorization_vs_payment", "share_vs_increase"})
EXCLUDED_TYPES = frozenset({"attribute_transfer", "unsupported_addition", "other", "none", "paraphrase_ok"})
V18_CONFIG_HASH = "12fc8b550d954359"
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".local" / "typed-veto-v1"


class VetoError(ValueError):
    pass


def rule_hash() -> str:
    return hashlib.sha256((json.dumps([VARIANT_ID, sorted(WHITELIST), sorted(EXCLUDED_TYPES), cr3.VARIANT_ID, cr3.content_hash(), EXPECTED_AUDIT_MODEL]) + inspect.getsource(decide)
                           + inspect.getsource(_valid_span) + inspect.getsource(binding_problem)).encode()).hexdigest()


def _valid_span(span: Mapping[str, Any], candidate: str) -> bool:
    s, e, t = span.get("start"), span.get("end"), span.get("text")
    return isinstance(s, int) and isinstance(e, int) and not isinstance(s, bool) and not isinstance(e, bool) and 0 <= s < e <= len(candidate) and isinstance(t, str) and bool(t) \
        and candidate[s:e] == t


def binding_problem(record: Mapping[str, Any], case: Mapping[str, Any], expected_model: str = EXPECTED_AUDIT_MODEL) -> str | None:
    """None when the audit record is bound to exactly this case, else the name of the first failed check. `case` = {case_id, source, candidate[, context]}."""
    inp = record.get("input")
    if not isinstance(inp, dict):
        return "no_input_block"
    if record.get("case_id") != case["case_id"]:
        return "case_id"
    if inp.get("source_sha256") != sha256(case["source"]) or inp.get("source") != case["source"] or sha256(str(inp.get("source"))) != inp.get("source_sha256"):
        return "source_hash"
    if inp.get("candidate_sha256") != sha256(case["candidate"]) or inp.get("candidate") != case["candidate"] or sha256(str(inp.get("candidate"))) != inp.get("candidate_sha256"):
        return "candidate_hash"
    prompt = build_prompt(case["source"], case["candidate"], case.get("context", ""))
    if record.get("prompt") != prompt or record.get("prompt_sha256") != sha256(prompt):
        return "prompt"
    if record.get("model") != expected_model:
        return "model"
    if not isinstance(record.get("raw_response"), str) or not record["raw_response"]:
        return "raw_response_missing"
    again = cr3.reprocess_record(record)
    if again["status"] != record.get("status") or again.get("structured") != record.get("structured"):
        return "structure_does_not_rederive_from_raw_response"
    return None


def decide(v18_pass: int, record: Mapping[str, Any] | None, case: Mapping[str, Any], expected_model: str = EXPECTED_AUDIT_MODEL) -> dict[str, Any]:
    """Return {"final": 0|1, "flipped": bool, "reason": str, ...}. Never raises on audit problems: they leave v18 unchanged and are named in `reason`."""
    if v18_pass not in (0, 1) or isinstance(v18_pass, bool):
        raise VetoError("v18_pass must be 0 or 1")
    candidate = case["candidate"]
    keep = {"final": v18_pass, "flipped": False}
    if v18_pass == 0:
        return keep | {"reason": "baseline_rejection_preserved"}
    if record is None:
        return keep | {"reason": "audit_missing"}
    if record.get("status") != "ok" or not isinstance(record.get("structured"), dict):
        return keep | {"reason": "audit_not_ok"}
    if record.get("variant") != cr3.VARIANT_ID or record.get("content_hash") != cr3.content_hash():
        return keep | {"reason": "audit_identity_mismatch"}
    problem = binding_problem(record, case, expected_model)
    if problem:
        return keep | {"reason": "audit_binding_mismatch", "binding_check": problem}
    s = record["structured"]
    if s.get("verdict") != "unsupported":
        return keep | {"reason": "audit_verdict_" + str(s.get("verdict"))}
    assertions = s.get("assertions") or []
    if any(a.get("verdict") == "uncertain" for a in assertions):
        return keep | {"reason": "uncertainty_present_v18_unchanged"}
    unsupported = [a for a in assertions if a.get("verdict") == "unsupported"]
    allowed = [a for a in unsupported if a.get("issue_type") in WHITELIST]
    if not allowed:
        return keep | {"reason": "unsupported_type_not_whitelisted", "types": sorted({str(a.get("issue_type")) for a in unsupported})}
    valid = [a for a in allowed if _valid_span(a.get("span") or {}, candidate)]
    if not valid:
        return keep | {"reason": "whitelisted_assertion_without_valid_candidate_span"}
    return {"final": 0, "flipped": True, "reason": "veto_whitelisted_type", "types": sorted({a["issue_type"] for a in valid})}


def apply_all(baseline: Mapping[str, int], records: Mapping[str, Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """`cases[i]` = {case_id, source, candidate[, context]}; each audit is looked up by case id AND re-bound to that case's full texts."""
    return {i: decide(baseline[i], records.get(i), cases[i]) for i in baseline}


def audit_coverage(baseline: Mapping[str, int], decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Every v18 PASS needs a usable, bound audit; otherwise an evaluation of the rule is incomplete (v18 is merely left unchanged for that case)."""
    bad = {i: d["reason"] for i, d in decisions.items() if baseline[i] == 1 and d["reason"] in ("audit_missing", "audit_not_ok", "audit_identity_mismatch", "audit_binding_mismatch")}
    return {"v18_passes": sum(1 for v in baseline.values() if v == 1), "unusable_or_unbound_audits": len(bad), "complete": not bad, "cases": bad}


def summarize(labels: Mapping[str, int], baseline: Mapping[str, int], decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Counts in BOTH directions plus kappas. Requires every case to be present."""
    from .audit2_v2 import kappa

    ids = sorted(labels)
    if set(decisions) != set(ids) or set(baseline) != set(ids):
        raise VetoError("decisions/baseline/labels cover different cases")
    y, b, f = [labels[i] for i in ids], [baseline[i] for i in ids], [decisions[i]["final"] for i in ids]
    return {"n": len(ids), "audit_coverage": audit_coverage(baseline, decisions), "kappa_baseline": kappa(y, b), "kappa_final": kappa(y, f), "flips": sum(1 for i in ids if decisions[i]["flipped"]),
            "false_passes_fixed": sum(1 for a, c, d in zip(y, b, f, strict=True) if a == 0 and c == 1 and d == 0),
            "new_false_rejections": sum(1 for a, c, d in zip(y, b, f, strict=True) if a == 1 and c == 1 and d == 0),
            "false_rejections_fixed": sum(1 for a, c, d in zip(y, b, f, strict=True) if a == 1 and c == 0 and d == 1),
            "new_false_passes": sum(1 for a, c, d in zip(y, b, f, strict=True) if a == 0 and c == 0 and d == 1),
            "errors_baseline": sum(1 for a, c in zip(y, b, strict=True) if a != c), "errors_final": sum(1 for a, d in zip(y, f, strict=True) if a != d),
            "accuracy_baseline": sum(1 for a, c in zip(y, b, strict=True) if a == c) / len(ids), "accuracy_final": sum(1 for a, d in zip(y, f, strict=True) if a == d) / len(ids)}


def freeze_rule(path: Path = WORK / "rule_v1_1.json") -> dict[str, Any]:
    """Write the frozen rule once. The v18 and claim-relation identities are bound; the whitelist cannot change without a new variant."""
    from .claim_relation import check_v18_identity

    if Path(path).exists():
        raise VetoError("rule exists; a frozen rule is never overwritten")
    ident = check_v18_identity()
    if not ident["unchanged"]:
        raise VetoError("v18 identity differs from the frozen values")
    body = {"variant": VARIANT_ID, "whitelist": sorted(WHITELIST), "excluded": sorted(EXCLUDED_TYPES), "rule_hash": rule_hash(), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "expected_audit_model": EXPECTED_AUDIT_MODEL, "claim_relation_variant": cr3.VARIANT_ID, "claim_relation_content_hash": cr3.content_hash(), "v18_config_hash": ident["v18_config_hash"], "tuning_on_dev": False,
            "statement": "a v18 PASS becomes a REJECT iff the audit is ok+identity-matched, explicit case-level unsupported, has no uncertain assertion, and an unsupported assertion of a whitelisted type "
                         "has a valid candidate span; everything else is v18 verbatim"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf8")
    return body


def posthoc_dev_reproduction(inputs_dir: Path, records_path: Path, out_path: Path) -> dict[str, Any]:
    """POST-HOC, IN-SAMPLE reproduction on the 128 cached dev responses (claim-relation-v3 reprocessing of the frozen Claude run). Labels are read here deliberately."""
    inputs_dir = Path(inputs_dir)
    cases = {c["case_id"]: c for c in json.loads((inputs_dir / "cases.json").read_text(encoding="utf8"))}
    labels = json.loads((inputs_dir / "labels.json").read_text(encoding="utf8"))
    baseline = json.loads((inputs_dir / "baseline_v18.json").read_text(encoding="utf8"))
    groups = json.loads((inputs_dir / "groups.json").read_text(encoding="utf8"))
    records = {r["case_id"]: r for r in (json.loads(x) for x in Path(records_path).read_text(encoding="utf8").splitlines() if x.strip())}
    decisions = apply_all(baseline, records, cases)
    out = summarize(labels, baseline, decisions)
    out["label"] = ("POST-HOC, IN-SAMPLE reproduction on the inspected 128 dev cases; whitelist chosen after seeing the audit's dev outcomes; not validation, not held-out, "
                    "not an unbiased estimate")
    out["flips"] = [{"case_id": i, "document": groups[i], "gold": labels[i], "types": decisions[i]["types"], "outcome": "false pass fixed" if labels[i] == 0 else "NEW false rejection",
                     "candidate_head": cases[i]["candidate"][:110]} for i in sorted(decisions) if decisions[i]["flipped"]]
    out["reasons_for_not_flipping_among_v18_passes"] = {}
    for i, d in decisions.items():
        if baseline[i] == 1 and not d["flipped"]:
            out["reasons_for_not_flipping_among_v18_passes"][d["reason"]] = out["reasons_for_not_flipping_among_v18_passes"].get(d["reason"], 0) + 1
    out["rule_hash"] = rule_hash()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(out_path).exists():
        raise VetoError("reproduction exists; written once")
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf8")
    return out


def documents(flips: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    count: dict[str, int] = {}
    for f in flips:
        count[f["document"]] = count.get(f["document"], 0) + 1
    return count
