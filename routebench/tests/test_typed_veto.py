"""Tests for typed-veto-v1.1 (v1 plus audit-to-case binding). Synthetic newly authored text only; mocked audit responses; no network, no model."""
import json
from pathlib import Path

import pytest
from evalops import claim_relation as v1
from evalops import claim_relation_v3 as cr3
from evalops import typed_veto as tv

SRC = "The board approved the plan on Monday. The clerk filed forms."
CAND = "The board approved the plan."
MODEL = tv.EXPECTED_AUDIT_MODEL


def assertion(text=CAND, verdict="unsupported", issue="possible_vs_confirmed", start=None, end=None, cand=CAND, evidence=SRC):
    s = cand.index(text) if start is None else start
    e = s + len(text) if end is None else end
    return {"candidate_span": {"start": s, "end": e, "text": text}, "evidence": [{"start": 0, "end": 1, "text": evidence}], "actor": "x", "time": "not_applicable", "modality": "actual",
            "numeric": "not_applicable", "relation": "not_applicable", "verdict": verdict, "issue_type": issue, "rationale": "synthetic"}


def case_of(cand=CAND, src=SRC, cid="c"):
    return {"case_id": cid, "source": src, "candidate": cand}


def record(assertions, cand=CAND, src=SRC, cid="c", model=MODEL):
    text = json.dumps({"assertions": assertions, "uncovered_spans": [], "overall": {"verdict": "supported", "rationale": "x"}})
    return cr3.judge_case({"case_id": cid, "source": src, "candidate": cand}, lambda p: v1.Response(text, {"prompt_tokens": 1, "completion_tokens": 1}, model))


def decide(v18, assertions, cand=CAND, src=SRC):
    return tv.decide(v18, record(assertions, cand, src), case_of(cand, src))


# ---- each whitelisted type vetoes a v18 pass
@pytest.mark.parametrize("issue", sorted(tv.WHITELIST))
def test_each_allowed_type_vetoes_a_v18_pass(issue):
    assert decide(1, [assertion(issue=issue)]) == {"final": 0, "flipped": True, "reason": "veto_whitelisted_type", "types": [issue]}


def test_the_whitelist_is_exactly_the_six_frozen_types():
    assert tv.WHITELIST == frozenset({"possible_vs_confirmed", "entity_mismatch", "unlinked_facts", "purpose_vs_result", "authorization_vs_payment", "share_vs_increase"})
    assert not tv.WHITELIST & tv.EXCLUDED_TYPES and {"attribute_transfer", "unsupported_addition", "other"} <= tv.EXCLUDED_TYPES
    superseded = tv.WORK / "rule.superseded-unbound.json"
    if superseded.exists():
        assert json.loads(superseded.read_text())["whitelist"] == sorted(tv.WHITELIST)  # the whitelist did not change when binding was added


# ---- excluded types leave v18 unchanged
@pytest.mark.parametrize("issue", ["attribute_transfer", "unsupported_addition", "other"])
def test_excluded_types_leave_v18_unchanged(issue):
    d = decide(1, [assertion(issue=issue)])
    assert d["final"] == 1 and not d["flipped"] and d["reason"] == "unsupported_type_not_whitelisted" and d["types"] == [issue]


def test_a_whitelisted_type_next_to_an_excluded_one_still_vetoes_but_only_for_the_allowed_one():
    cand = "The board approved the plan on Monday."
    d = decide(1, [assertion("The board approved the plan", issue="unsupported_addition", cand=cand), assertion("on Monday", issue="possible_vs_confirmed", cand=cand)], cand)
    assert d["flipped"] and d["types"] == ["possible_vs_confirmed"]


# ---- spans must be valid
@pytest.mark.parametrize("start,end", [(5, 5), (9, 3), (-1, 4), (0, 999)])
def test_invalid_candidate_spans_never_veto(start, end):
    r = record([assertion()])
    for x in r["structured"]["assertions"]:
        x["span"] = {"start": start, "end": end, "text": CAND}
    # the stored structure no longer re-derives from the raw response, so binding refuses it; the span check is exercised directly as well
    assert not tv.decide(1, r, case_of())["flipped"]
    assert not tv._valid_span({"start": start, "end": end, "text": CAND}, CAND)


def test_a_span_whose_text_disagrees_with_the_stored_candidate_never_vetoes():
    assert not tv._valid_span({"start": 0, "end": 9, "text": "something"}, CAND)
    r2 = record([assertion()])
    assert not tv.decide(1, r2, case_of("A different candidate sentence."))["flipped"]


def test_missing_or_non_integer_span_fields_never_validate():
    for bad in ({}, {"start": "0", "end": 5, "text": "The b"}, {"start": True, "end": 5, "text": "The b"}, {"start": 0, "end": 5, "text": ""}):
        assert not tv._valid_span(bad, CAND)
    assert tv._valid_span({"start": 0, "end": 9, "text": "The board"}, CAND)


def test_a_model_span_with_wrong_offsets_is_repaired_by_the_audit_layer_and_then_accepted():
    a = assertion()
    a["candidate_span"] = {"start": 7, "end": 12, "text": CAND}
    r = record([a])
    d = tv.decide(1, r, case_of())
    assert d["flipped"] and "candidate_span_offsets_repaired" in r["structured"]["assertions"][0]["flags"]


# ---- uncertainty leaves v18 unchanged
def test_case_level_uncertainty_leaves_v18_unchanged():
    r = record([assertion(verdict="uncertain", issue="other")])
    assert r["structured"]["verdict"] == "uncertain"
    d = tv.decide(1, r, case_of())
    assert d["final"] == 1 and not d["flipped"] and d["reason"] == "audit_verdict_uncertain"


def test_an_uncertain_assertion_beside_a_whitelisted_unsupported_one_leaves_v18_unchanged():
    cand = "The board approved the plan on Monday."
    d = decide(1, [assertion("The board approved the plan", verdict="unsupported", issue="possible_vs_confirmed", cand=cand), assertion("on Monday", verdict="uncertain", issue="other", cand=cand)], cand)
    assert d["final"] == 1 and not d["flipped"] and d["reason"] == "uncertainty_present_v18_unchanged"


def test_unreviewed_substantive_text_makes_the_audit_uncertain_and_cannot_veto():
    cand = "The board did not approve the plan."
    r = record([assertion("The board did", verdict="supported", issue="paraphrase_ok", cand=cand), assertion("approve the plan.", verdict="supported", issue="paraphrase_ok", cand=cand)], cand)
    assert r["structured"]["verdict"] == "uncertain" and not tv.decide(1, r, case_of(cand))["flipped"]


# ---- baseline rejections are preserved in every audit situation
@pytest.mark.parametrize("assertions", [[assertion(issue="possible_vs_confirmed")], [assertion(verdict="supported", issue="paraphrase_ok")], [assertion(verdict="uncertain", issue="other")]])
def test_baseline_rejections_are_preserved_whatever_the_audit_says(assertions):
    assert decide(0, assertions) == {"final": 0, "flipped": False, "reason": "baseline_rejection_preserved"}
    assert tv.decide(0, None, case_of())["final"] == 0


# ---- genuine paraphrases are not vetoed
def test_supported_paraphrases_are_untouched():
    d = decide(1, [assertion(verdict="supported", issue="paraphrase_ok")])
    assert d["final"] == 1 and not d["flipped"] and d["reason"] == "audit_verdict_supported"


@pytest.mark.parametrize("cand,src", [("The clerk filed the forms.", "The clerk filed forms."), ("The council voted to approve the budget.", "The council voted in favour of the budget.")])
def test_paraphrase_pairs_pass_through_while_the_nearby_false_claim_is_vetoed(cand, src):
    ok = tv.decide(1, record([assertion(cand, verdict="supported", issue="paraphrase_ok", cand=cand, evidence=src)], cand, src), case_of(cand, src))
    assert ok["final"] == 1 and not ok["flipped"]
    bad_cand = cand.replace("approve", "reject").replace("filed the forms", "destroyed the forms")
    bad = tv.decide(1, record([assertion(bad_cand, verdict="unsupported", issue="entity_mismatch", cand=bad_cand, evidence=src)], bad_cand, src), case_of(bad_cand, src))
    assert bad["final"] == 0 and bad["flipped"]


# ---- audit availability and identity
def test_missing_failed_or_foreign_audit_records_leave_v18_unchanged():
    assert tv.decide(1, None, case_of())["reason"] == "audit_missing"
    r = record([assertion()])
    assert tv.decide(1, {**r, "status": "invalid_structure", "structured": None}, case_of())["reason"] == "audit_not_ok"
    assert tv.decide(1, {**r, "variant": "claim-relation-v2"}, case_of())["reason"] == "audit_identity_mismatch"
    assert tv.decide(1, {**r, "content_hash": "0" * 64}, case_of())["reason"] == "audit_identity_mismatch"
    with pytest.raises(tv.VetoError):
        tv.decide(2, r, case_of())
    with pytest.raises(tv.VetoError):
        tv.decide(True, r, case_of())


# ---- BINDING: every audit must belong to exactly this case
def test_an_audit_of_a_different_document_is_refused_even_when_the_candidate_text_is_identical():
    other_source = "The committee rejected the plan on Tuesday. The treasurer resigned."
    audit_of_other_document = record([assertion(issue="entity_mismatch")], CAND, other_source, cid="c")  # same case id AND same candidate text, different source
    d = tv.decide(1, audit_of_other_document, case_of(CAND, SRC, "c"))
    assert d["final"] == 1 and not d["flipped"] and d["reason"] == "audit_binding_mismatch" and d["binding_check"] == "source_hash"
    genuine = record([assertion(issue="entity_mismatch")], CAND, SRC, cid="c")
    assert tv.decide(1, genuine, case_of(CAND, SRC, "c"))["flipped"]  # the same call with the right document still works


def test_an_audit_filed_under_another_case_id_is_refused():
    d = tv.decide(1, record([assertion()], cid="other-case"), case_of(cid="c"))
    assert not d["flipped"] and d["binding_check"] == "case_id"


def test_each_binding_field_is_checked_independently():
    base = record([assertion()])
    case = case_of()

    def with_input(**kw):
        r = json.loads(json.dumps(base))
        r["input"].update(kw)
        return r

    assert tv.binding_problem(base, case) is None
    assert tv.binding_problem(with_input(source_sha256="0" * 64), case) == "source_hash"
    assert tv.binding_problem(with_input(candidate_sha256="0" * 64), case) == "candidate_hash"
    assert tv.binding_problem(with_input(source=SRC + " tampered"), case) == "source_hash"  # stored text no longer matches its stored hash
    assert tv.binding_problem(with_input(candidate=CAND + " tampered"), case) == "candidate_hash"
    assert tv.binding_problem({**base, "prompt": "another prompt"}, case) == "prompt"
    assert tv.binding_problem({**base, "prompt_sha256": "0" * 64}, case) == "prompt"
    assert tv.binding_problem({**base, "model": "gpt-4o"}, case) == "model"
    assert tv.binding_problem({**base, "model": None}, case) == "model"
    assert tv.binding_problem({**base, "raw_response": ""}, case) == "raw_response_missing"
    assert tv.binding_problem({k: v for k, v in base.items() if k != "input"}, case) == "no_input_block"


def test_a_structure_edited_after_the_fact_no_longer_matches_its_raw_response():
    r = json.loads(json.dumps(record([assertion(verdict="supported", issue="paraphrase_ok")])))
    r["structured"]["verdict"] = "unsupported"
    r["structured"]["assertions"][0].update(verdict="unsupported", issue_type="entity_mismatch")
    d = tv.decide(1, r, case_of())
    assert not d["flipped"] and d["reason"] == "audit_binding_mismatch" and d["binding_check"] == "structure_does_not_rederive_from_raw_response"


def test_a_wrong_model_identity_is_refused_and_the_expected_one_is_configurable_only_by_the_caller():
    r = record([assertion()], model="some-other-model")
    d = tv.decide(1, r, case_of())
    assert not d["flipped"] and d["binding_check"] == "model"
    assert tv.decide(1, r, case_of(), expected_model="some-other-model")["flipped"]  # explicit, caller-visible override only


def test_coverage_reports_unbound_or_missing_audits_so_an_evaluation_cannot_pass_silently():
    wrong_doc = record([assertion(issue="entity_mismatch")], CAND, "A different source document about other things entirely.")
    cases = {"a": case_of(CAND, SRC, "c"), "b": case_of(CAND, SRC, "c"), "e": case_of(CAND, SRC, "c")}
    baseline = {"a": 1, "b": 1, "e": 0}
    decisions = tv.apply_all(baseline, {"a": record([assertion(issue="entity_mismatch")])}, cases)  # "b" has no audit
    assert tv.audit_coverage(baseline, decisions) == {"v18_passes": 2, "unusable_or_unbound_audits": 1, "complete": False, "cases": {"b": "audit_missing"}}
    decisions2 = tv.apply_all(baseline, {"a": wrong_doc, "b": record([assertion()])}, cases)
    assert tv.audit_coverage(baseline, decisions2)["cases"] == {"a": "audit_binding_mismatch"}


# ---- accounting in both directions
def test_summary_reports_both_metrics_and_both_directions_and_the_veto_can_never_fix_a_false_rejection():
    labels = {"a": 0, "b": 1, "c": 1, "d": 0}
    baseline = {"a": 1, "b": 1, "c": 0, "d": 0}
    veto = record([assertion(issue="entity_mismatch")])
    cases = dict.fromkeys(labels, case_of())
    decisions = tv.apply_all(baseline, {"a": veto, "b": veto, "c": veto, "d": veto}, cases)  # c and d are v18 rejections: untouched
    s = tv.summarize(labels, baseline, decisions)
    assert (s["false_passes_fixed"], s["new_false_rejections"], s["false_rejections_fixed"], s["new_false_passes"], s["flips"]) == (1, 1, 0, 0, 2)
    assert s["errors_baseline"] == 2 and s["errors_final"] == 2 and s["accuracy_baseline"] == 0.5 and s["accuracy_final"] == 0.5 and s["audit_coverage"]["complete"]
    with pytest.raises(tv.VetoError):
        tv.summarize(labels, baseline, {"a": decisions["a"]})


# ---- identity, freeze, preservation
def test_rule_identity_and_frozen_versions():
    assert tv.VARIANT_ID == "typed-veto-v1.1" and len(tv.rule_hash()) == 64 and tv.V18_CONFIG_HASH == "12fc8b550d954359" and v1.check_v18_identity()["unchanged"]
    assert cr3.content_hash() != v1.content_hash()


def test_freeze_never_overwrites(tmp_path):
    p = tmp_path / "rule.json"
    body = tv.freeze_rule(p)
    assert body["tuning_on_dev"] is False and body["whitelist"] == sorted(tv.WHITELIST) and body["rule_hash"] == tv.rule_hash() and body["expected_audit_model"] == MODEL
    with pytest.raises(tv.VetoError, match="never overwritten"):
        tv.freeze_rule(p)


def test_module_makes_no_calls():
    src = Path(tv.__file__).read_text(encoding="utf8")
    assert "import requests" not in src and "httpx" not in src and "subprocess" not in src


# ---- POST-HOC reproduction on cached dev responses, now with every audit bound to its case
def test_posthoc_reproduction_of_the_cached_dev_result(tmp_path):
    inputs, recs = tv.ROOT / ".local" / "claim-relation-claude-v1" / "inputs", tv.ROOT / ".local" / "claim-relation-claude-v1" / "post-hoc-v3" / "judgments.jsonl"
    if not recs.exists():
        pytest.skip("cached post-hoc v3 judgments not present")
    out = tv.posthoc_dev_reproduction(inputs, recs, tmp_path / "repro.json")
    assert out["kappa_final"] == pytest.approx(0.7133825079030558) and out["kappa_baseline"] == pytest.approx(0.6745182012847966)
    assert (out["false_passes_fixed"], out["new_false_rejections"], out["false_rejections_fixed"], out["new_false_passes"], out["errors_final"]) == (3, 1, 0, 0, 17)
    assert out["audit_coverage"]["complete"] and out["audit_coverage"]["unusable_or_unbound_audits"] == 0  # all 88 v18 passes had a bound audit
    assert len(out["flips"]) == 4 and "POST-HOC, IN-SAMPLE" in out["label"]
    assert tv.documents(out["flips"]) == {"doc-4687dadb7aa7": 1, "doc-c1738eb8d4c9": 2, "doc-b651ffa82b47": 1}
    with pytest.raises(tv.VetoError, match="written once"):
        tv.posthoc_dev_reproduction(inputs, recs, tmp_path / "repro.json")


def test_every_cached_dev_audit_is_bound_to_its_own_case_and_not_to_any_other():
    inputs = tv.ROOT / ".local" / "claim-relation-claude-v1" / "inputs"
    recs_path = tv.ROOT / ".local" / "claim-relation-claude-v1" / "post-hoc-v3" / "judgments.jsonl"
    if not recs_path.exists():
        pytest.skip("cached post-hoc v3 judgments not present")
    cases = {c["case_id"]: c for c in json.loads((inputs / "cases.json").read_text(encoding="utf8"))}
    recs = {r["case_id"]: r for r in (json.loads(x) for x in recs_path.read_text(encoding="utf8").splitlines() if x.strip())}
    assert all(tv.binding_problem(recs[i], cases[i]) is None for i in cases)
    ids = sorted(cases)
    rotated = {i: recs[ids[(k + 1) % len(ids)]] for k, i in enumerate(ids)}  # every audit attached to the NEXT case instead
    assert all(tv.binding_problem(rotated[i], cases[i]) is not None for i in cases)
