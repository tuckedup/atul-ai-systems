"""Regression tests for the claim-relation-v3 aggregation repair. Newly authored SYNTHETIC text; mocked responses test behaviour, not accuracy."""
import json

import pytest
from evalops import claim_relation as v1
from evalops import claim_relation_v2 as v2
from evalops import claim_relation_v3 as v3

SRC = "The board approved the plan on Monday. The clerk filed 5 forms and will post the minutes."


def assertion(cand, text, evidence=None, **over):
    s = cand.index(text)
    a = {"candidate_span": {"start": s, "end": s + len(text), "text": text}, "evidence": [{"start": 0, "end": 1, "text": evidence or SRC}], "actor": "x", "time": "not_applicable",
         "modality": "actual", "numeric": "not_applicable", "relation": "not_applicable", "verdict": "supported", "issue_type": "paraphrase_ok", "rationale": "synthetic"}
    a.update(over)
    return a


def reply(assertions, uncovered=()):
    return json.dumps({"assertions": assertions, "uncovered_spans": list(uncovered), "overall": {"verdict": "supported", "rationale": "x"}})


def run(module, src, cand, text):
    return module.judge_case({"case_id": "c", "source": src, "candidate": cand}, lambda p: v1.Response(text, {"prompt_tokens": 1, "completion_tokens": 1}, "MOCK"))


def span(cand, text, **kw):
    s = cand.index(text)
    return {"start": s, "end": s + len(text), "text": text, **kw}


CAND = "The board approved the plan on Monday."
BODY = CAND[:-1]  # everything but the final full stop


def one(declared):
    return reply([assertion(CAND, BODY)], declared)


# ---- the reproduced defect: punctuation / whitespace / function words are not factual uncertainty
def test_an_uncovered_final_full_stop_demoted_v2_but_not_v3():
    text = one([span(CAND, ".", reason="punctuation")])
    assert run(v2, SRC, CAND, text)["structured"]["verdict"] == "uncertain"  # the defect
    r = run(v3, SRC, CAND, text)
    assert r["structured"]["verdict"] == "supported" and r["structured"]["coverage"]["declared_spans_ignored_non_substantive"] == 1


@pytest.mark.parametrize("declared_text", [".", ",", " ", " .", "\n", "...", "!", "the", "on the", "of a"])
def test_whitespace_punctuation_and_function_words_are_not_uncertainty(declared_text):
    cand = f"The board approved the plan{declared_text}" if declared_text.strip() in {".", ",", "...", "!"} or declared_text in {" ", " .", "\n"} else "The board approved the plan on Monday."
    body = "The board approved the plan"
    declared = {"start": 0, "end": 1, "text": declared_text}
    r = run(v3, SRC, cand, reply([assertion(cand, body)], [declared]))
    assert r["structured"]["coverage"]["declared_spans_substantive"] == 0
    assert "declared_uncovered_span_with_substantive_words" not in r["structured"]["flags"]


# ---- substantive uncovered material is still caught, however it is detected
@pytest.mark.parametrize("cand,covered,declared", [
    ("The board did not approve the plan.", "The board did", "not"), ("The board will approve the plan.", "The board", "will"),
    ("The clerk filed 5 forms.", "The clerk filed", "5"), ("The mayor opened the meeting and the treasurer resigned.", "The mayor opened the meeting", "the treasurer resigned"),
])
def test_declared_spans_with_negation_modal_digit_or_content_words_still_demote(cand, covered, declared):
    r = run(v3, SRC, cand, reply([assertion(cand, covered)], [span(cand, declared)]))
    assert r["structured"]["verdict"] == "uncertain"
    assert "declared_uncovered_span_with_substantive_words" in r["structured"]["flags"] and r["structured"]["coverage"]["declared_spans_substantive"] == 1


@pytest.mark.parametrize("cand,first,second,missed", [
    ("The board did not approve the plan.", "The board did", "approve the plan.", "not"), ("The council will approve the budget.", "The council", "approve the budget.", "will"),
    ("The clerk filed 5 forms.", "The clerk filed", "forms.", "5"),
])
def test_undeclared_uncovered_negation_modals_and_numbers_are_still_caught_by_coverage(cand, first, second, missed):
    r = run(v3, SRC, cand, reply([assertion(cand, first), assertion(cand, second)], []))  # the model declared nothing
    assert r["structured"]["verdict"] == "uncertain" and missed in r["structured"]["coverage"]["uncovered_words"]
    assert "unreviewed_substantive_words" in r["structured"]["flags"]


def test_a_reason_given_by_the_model_never_exempts_a_substantive_span():
    cand = "The board did not approve the plan."
    r = run(v3, SRC, cand, reply([assertion(cand, "The board did"), assertion(cand, "approve the plan.")], [span(cand, "not", reason="unimportant filler word, ignore")]))
    assert r["structured"]["verdict"] == "uncertain"


# ---- span resolution: offsets are verified, never trusted alone
def test_wrong_offsets_pointing_at_a_negation_still_demote_even_if_the_declared_text_is_punctuation():
    cand = "The board did not approve the plan."
    pos = cand.index("not")
    lying = {"start": pos, "end": pos + 3, "text": "."}  # declares '.' but the offsets cover 'not'
    r = run(v3, SRC, cand, reply([assertion(cand, "The board did"), assertion(cand, "approve the plan.")], [lying]))
    assert r["structured"]["verdict"] == "uncertain" and r["structured"]["coverage"]["declared_span_resolution"] == ["unique_text_match"]  # the slice at the offsets ('not') is what makes it substantive


def test_unverifiable_declared_span_is_judged_by_its_declared_words_both_ways():
    bad = {"start": 9999, "end": 10000, "text": "the treasurer resigned"}
    assert run(v3, SRC, CAND, one([bad]))["structured"]["verdict"] == "uncertain"
    harmless = {"start": 9999, "end": 10000, "text": ","}  # not present in the candidate: cannot be resolved
    r = run(v3, SRC, CAND, one([harmless]))
    assert r["structured"]["verdict"] == "supported" and r["structured"]["coverage"]["declared_span_resolution"] == ["unresolved_declared_text"]


def test_a_unique_text_match_repairs_wrong_offsets():
    cand = "The board approved the plan on Monday."
    r = run(v3, SRC, cand, reply([assertion(cand, "The board approved the plan")], [{"start": 1, "end": 2, "text": "on Monday"}]))
    assert r["structured"]["coverage"]["declared_span_resolution"] == ["unique_text_match"] and r["structured"]["verdict"] == "uncertain"


# ---- everything else is unchanged from v2
def test_other_guards_are_unchanged():
    # invalid assertion span, unsupported verdict, assertion-level uncertain, unevidenced support
    bad_span = assertion(CAND, BODY)
    bad_span["candidate_span"] = {"start": 0, "end": 5, "text": "wrong"}
    assert run(v3, SRC, CAND, reply([bad_span]))["structured"]["verdict"] == "uncertain"
    assert run(v3, SRC, CAND, reply([assertion(CAND, BODY, verdict="unsupported", issue_type="entity_mismatch")]))["structured"]["verdict"] == "unsupported"
    assert run(v3, SRC, CAND, reply([assertion(CAND, BODY, verdict="uncertain", issue_type="other")]))["structured"]["verdict"] == "uncertain"
    assert run(v3, SRC, CAND, reply([assertion(CAND, BODY, evidence="not in the source at all")]))["structured"]["verdict"] == "uncertain"


def test_v3_agrees_with_v2_whenever_no_spans_are_declared():
    for cand, text in [(CAND, reply([assertion(CAND, BODY)])), ("The clerk filed 5 forms.", reply([assertion("The clerk filed 5 forms.", "The clerk filed")]))]:
        a, b = run(v2, SRC, cand, text), run(v3, SRC, cand, text)
        assert a["structured"]["verdict"] == b["structured"]["verdict"] and a["structured"]["flags"] == b["structured"]["flags"]


# ---- identity, preservation, offline reprocessing
def test_identity_variants_and_hashes_are_distinct_and_frozen_versions_untouched():
    assert v3.VARIANT_ID == "claim-relation-v3" and v3.content_hash() != v2.content_hash()
    assert v1.content_hash() == "91e6870ac05c0dd622036ef02ae7f7111f461e1a02218255ee9bc5cc08a85164"
    assert v2.content_hash() == "adf2291227d854c505ca0b9f320265e81543ceb2c7b25cad3c62ec64dddb384e"
    assert v1.check_v18_identity()["unchanged"]


def test_reprocessing_keeps_the_input_record_intact_and_makes_no_call():
    rec = run(v2, SRC, CAND, one([span(CAND, ".")]))
    before = json.dumps(rec, sort_keys=True)
    out = v3.reprocess_record(rec)
    assert json.dumps(rec, sort_keys=True) == before and rec["structured"]["verdict"] == "uncertain"
    assert out["variant"] == "claim-relation-v3" and out["structured"]["verdict"] == "supported" and out["reprocessed_from"]["variant"] == "claim-relation-v2"
    assert out["raw_response"] == rec["raw_response"] and out["input"] == rec["input"]


def test_malformed_stored_responses_stay_invalid_when_reprocessed():
    rec = run(v2, SRC, CAND, "not json")
    assert v3.reprocess_record(rec)["status"] == "invalid_structure"
