"""Offline regression tests for claim-relation-v2 post-processing repairs. Newly authored SYNTHETIC text only; mocked responses test behaviour, not accuracy."""
import json

import pytest
from evalops import claim_relation as v1
from evalops import claim_relation_v2 as v2

SOURCE = ("Speaker 1: The board met on Monday.\nSpeaker 2: We “voted to approve” the\n  roof repair, and the clerk\nwill post the minutes Friday.\n"
          "Speaker 1: The audit report is due in May. The audit report is due in May.")


def reply(candidate_span, evidence, verdict="supported", issue="paraphrase_ok", **over):
    a = {"candidate_span": candidate_span, "evidence": evidence, "actor": "board", "time": "not_applicable", "modality": "actual", "numeric": "not_applicable",
         "relation": "not_applicable", "verdict": verdict, "issue_type": issue, "rationale": "synthetic"}
    a.update(over)
    return a


def respond(assertions, uncovered=()):
    return json.dumps({"assertions": assertions, "uncovered_spans": list(uncovered), "overall": {"verdict": "supported", "rationale": "x"}})


def run(module, source, candidate, text):
    return module.judge_case({"case_id": "c", "source": source, "candidate": candidate}, lambda p: v1.Response(text, {"prompt_tokens": 1, "completion_tokens": 1}, "MOCK"))


# ---- cycle 1: authenticity under real formatting (valid paraphrase) vs nearby unsupported claim (still unsupported, quote or not)
def test_cycle1_multiline_curly_quote_evidence_is_authentic_in_v2_but_not_v1():
    cand = "The board voted to approve the roof repair."
    text = respond([reply({"start": 0, "end": len(cand), "text": cand}, [{"start": 0, "end": 1, "text": '"voted to approve" the roof repair'}])])
    r1, r2 = run(v1, SOURCE, cand, text), run(v2, SOURCE, cand, text)
    assert r1["structured"]["verdict"] == "uncertain" and "supported_without_authentic_evidence" in r1["structured"]["flags"]  # the defect
    a = r2["structured"]["assertions"][0]
    assert r2["structured"]["verdict"] == "supported" and a["evidence_authentic"] == [True] and "evidence_normalised_match" in a["flags"]
    start, end = a["evidence"][0]["start"], a["evidence"][0]["end"]
    assert SOURCE[start:end].startswith("“voted to approve”") and SOURCE[start:end].endswith("repair")  # mapped back to ORIGINAL offsets


def test_cycle1_authentic_quote_does_not_rescue_an_unsupported_claim():
    cand = "The board voted to reject the roof repair."
    text = respond([reply({"start": 0, "end": len(cand), "text": cand}, [{"start": 0, "end": 1, "text": "voted to approve"}], verdict="unsupported",
                          issue="entity_mismatch")])
    r = run(v2, SOURCE, cand, text)
    assert r["structured"]["verdict"] == "unsupported" and r["structured"]["assertions"][0]["evidence_authentic"] == [True]


def test_cycle1_repeated_quote_is_authentic_but_flagged_and_fabricated_quote_is_not():
    cand = "The audit report is due in May."
    ok = respond([reply({"start": 0, "end": len(cand), "text": cand}, [{"start": 999, "end": 1000, "text": "The audit report is due in May."}])])
    r1, r2 = run(v1, SOURCE, cand, ok), run(v2, SOURCE, cand, ok)
    assert r1["structured"]["verdict"] == "uncertain"  # defect: v1 requires a unique occurrence to repair wrong offsets
    assert r2["structured"]["verdict"] == "supported" and "evidence_quote_occurs_more_than_once" in r2["structured"]["assertions"][0]["flags"]
    fake = respond([reply({"start": 0, "end": len(cand), "text": cand}, [{"start": 0, "end": 5, "text": "The audit report is due in June."}])])
    assert run(v2, SOURCE, cand, fake)["structured"]["verdict"] == "uncertain"


def test_normalise_with_map_maps_back_exactly():
    text = "a  “b”\n c"
    norm, idx = v2.normalise_with_map(text)
    assert norm == 'a "b" c' and text[idx[2]] == "“" and text[idx[-1]] == "c"
    start, end, _amb, normed = v2.locate(text, 'a "b"   c')
    assert (start, end, normed) == (0, len(text), True) and v2.locate(text, "zzz") is None and v2.locate(text, "") is None


# ---- cycle 2: broad-span rule must not fire on ordinary single claims
def test_cycle2_single_claim_with_a_conjunction_is_not_demoted_in_v2():
    src = "The clerk said the minutes were approved and filed."
    cand = "The clerk said the minutes were approved and filed."
    text = respond([reply({"start": 0, "end": len(cand), "text": cand}, [{"start": 0, "end": 1, "text": cand}])])
    assert run(v1, src, cand, text)["structured"]["verdict"] == "uncertain"  # defect: 'and' alone triggered the demotion
    assert run(v2, src, cand, text)["structured"]["verdict"] == "supported"


def test_cycle2_a_real_second_claim_hidden_in_one_span_is_still_flagged():
    src = "The mayor opened the meeting."
    cand = "The mayor opened the meeting and the treasurer resigned from the committee."
    text = respond([reply({"start": 0, "end": len(cand), "text": cand}, [{"start": 0, "end": 1, "text": "The mayor opened the meeting."}])])
    r = run(v2, src, cand, text)
    assert r["structured"]["verdict"] == "uncertain" and "possible_broad_span_hiding_second_claim" in r["structured"]["flags"]
    two = respond([reply({"start": 0, "end": 26, "text": cand[:26]}, [{"start": 0, "end": 1, "text": "The mayor opened the meeting."}]),
                   reply({"start": 31, "end": len(cand), "text": cand[31:]}, [], verdict="unsupported", issue="unsupported_addition")])
    assert run(v2, src, cand, two)["structured"]["verdict"] == "unsupported"


# ---- cycle 3: unambiguous not-applicable synonyms normalised; missing keys and bad vocabulary still invalid
@pytest.mark.parametrize("value", [None, "", "N/A", "none", "Not Applicable"])
def test_cycle3_not_applicable_synonyms_are_normalised_and_flagged(value):
    cand = "The board met on Monday."
    text = respond([reply({"start": 0, "end": len(cand), "text": cand}, [{"start": 0, "end": 1, "text": "The board met on Monday."}], numeric=value, relation=value)])
    assert run(v1, SOURCE, cand, text)["status"] == "invalid_structure"  # defect: a single odd literal aborts a fail-fast run
    r = run(v2, SOURCE, cand, text)
    a = r["structured"]["assertions"][0]
    assert r["status"] == "ok" and a["fields"]["numeric"] == "not_applicable" and {"normalised_numeric", "normalised_relation"} <= set(a["flags"])


@pytest.mark.parametrize("mutate", [lambda a: a.pop("numeric"), lambda a: a.update(modality="certain"), lambda a: a.update(verdict="maybe"),
                                    lambda a: a.update(relation={"type": "magic"}), lambda a: a.update(numeric="roughly 5")])
def test_cycle3_missing_keys_and_unknown_values_are_never_repaired(mutate):
    cand = "The board met on Monday."
    a = reply({"start": 0, "end": len(cand), "text": cand}, [])
    mutate(a)
    assert run(v2, SOURCE, cand, respond([a]))["status"] == "invalid_structure"


# ---- cycle 4: Unicode words count toward coverage
def test_cycle4_non_ascii_words_must_be_covered():
    src = "José Álvarez opened the session and Zoë Muñoz closed it."
    cand = "José Álvarez opened the session and Zoë Muñoz closed it."
    part = cand[:cand.index(" and")]
    text = respond([reply({"start": 0, "end": len(part), "text": part}, [{"start": 0, "end": 1, "text": part}])])
    r1, r2 = run(v1, src, cand, text), run(v2, src, cand, text)
    assert "zoë" in r2["structured"]["coverage"]["uncovered_words"] and r2["structured"]["verdict"] == "uncertain"
    assert "zoë" not in r1["structured"]["coverage"]["uncovered_words"]  # v1 never saw the non-ASCII words


# ---- identity and compatibility
def test_v2_keeps_the_v1_prompt_and_leaves_v1_identity_and_manifest_untouched():
    assert v2.v1.PROMPT == v1.PROMPT and v2.content_hash() != v1.content_hash() and v2.VARIANT_ID != v1.VARIANT_ID
    assert v1.check_v18_identity()["unchanged"] and v1.content_hash() == "91e6870ac05c0dd622036ef02ae7f7111f461e1a02218255ee9bc5cc08a85164"


def test_reprocessing_a_stored_v1_record_makes_no_model_call():
    cand = "The board voted to approve the roof repair."
    text = respond([reply({"start": 0, "end": len(cand), "text": cand}, [{"start": 0, "end": 1, "text": '"voted to approve" the roof repair'}])])
    rec = run(v1, SOURCE, cand, text)
    again = v2.reprocess_v1_record(rec)
    assert again["variant"] == "claim-relation-v2" and again["structured"]["verdict"] == "supported" and again["raw_response"] == rec["raw_response"]


# ---- cycle 5: omitted negation, modal words and single-digit numbers must be detected as unreviewed
def _two_spans(src, cand, first, second, verdicts=("supported", "supported")):
    """A reply whose two assertions leave the text BETWEEN them (the word under test) outside every reviewed span."""
    s1, s2 = cand.index(first), cand.index(second)
    return respond([reply({"start": s1, "end": s1 + len(first), "text": first}, [{"start": 0, "end": 1, "text": src}], verdict=verdicts[0]),
                    reply({"start": s2, "end": s2 + len(second), "text": second}, [{"start": 0, "end": 1, "text": src}], verdict=verdicts[1])])


@pytest.mark.parametrize("src,cand,first,second,missed", [
    ("The board approved the plan.", "The board did not approve the plan.", "The board did", "approve the plan.", "not"),          # omitted negation
    ("The board approved the plan.", "The board did NOT approve the plan.", "The board did", "approve the plan.", "not"),
    ("The board did approve the plan.", "The board didn't approve the plan.", "The board", "approve the plan.", "didn't"),        # contraction
    ("The board approved the plan.", "No board approved the plan.", "board approved", "the plan.", "no"),
    ("The council approved the budget.", "The council will approve the budget.", "The council", "approve the budget.", "will"),     # modals
    ("The council approved the budget.", "The council could approve the budget.", "The council", "approve the budget.", "could"),
    ("The council approved the budget.", "The council must approve the budget.", "The council", "approve the budget.", "must"),
    ("The clerk filed forms.", "The clerk filed 5 forms.", "The clerk filed", "forms.", "5"),                                      # single digit
    ("The clerk filed forms.", "The clerk filed forms in 2 batches.", "The clerk filed forms", "batches.", "2"),
])
def test_cycle5_unreviewed_meaning_bearing_words_are_detected_in_v2_but_not_v1(src, cand, first, second, missed):
    text = _two_spans(src, cand, first, second)
    r1, r2 = run(v1, src, cand, text), run(v2, src, cand, text)
    if missed in v1.STOP or len(missed) == 1:  # words v1 ignores by design; v1 already flags the others (must, didn't)
        assert missed not in r1["structured"]["coverage"]["uncovered_words"] and r1["structured"]["verdict"] == "supported"  # the defect
    assert missed in r2["structured"]["coverage"]["uncovered_words"] and r2["structured"]["verdict"] == "uncertain"
    assert "unreviewed_substantive_words" in r2["structured"]["flags"]


def test_cycle5_a_fully_covered_negation_stays_supported_and_function_words_remain_ignored():
    src, cand = "The board rejected the plan.", "The board did not approve the plan."
    full = respond([reply({"start": 0, "end": len(cand), "text": cand}, [{"start": 0, "end": 1, "text": src}], verdict="unsupported", issue="entity_mismatch")])
    assert run(v2, src, cand, full)["structured"]["verdict"] == "unsupported"
    ok = "The board met on the day of the vote."
    gap = respond([reply({"start": 0, "end": 13, "text": ok[:13]}, [{"start": 0, "end": 1, "text": ok}]),
                   reply({"start": 17, "end": len(ok), "text": ok[17:]}, [{"start": 0, "end": 1, "text": ok}])])  # skips only 'on'
    r = run(v2, ok, ok, gap)
    assert r["structured"]["coverage"]["uncovered_words"] == [] and r["structured"]["verdict"] == "supported"
