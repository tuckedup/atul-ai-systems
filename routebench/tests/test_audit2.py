import json

import pytest
from evalops import audit2 as a

SOURCE = ("Mayor Lee said on Monday that the council will vote on the budget next week. "
          "She hoped the road repairs would start in spring. Councillor Diaz opposed the plan and said it costs too much.")


def reply(**kw):
    base = {"verdict": "unsupported", "disputed_assertion": None, "error_type": None, "claim_says": None, "source_says": None,
            "evidence_quote": None, "unsupported_details": [], "explanation": "The summary asserts something that the source does not establish at all here."}
    base.update(kw)
    return json.dumps(base)


def flips(candidate, **kw):
    return a.decide(a.parse(reply(**kw)), SOURCE, candidate)


def test_prompt_is_blind_to_labels_and_prior_verdicts():
    prompt = a.build_prompt(SOURCE, "The council voted.")
    for forbidden in ("gold", "label", "v18", "previous verdict", "passed"):
        assert forbidden not in prompt.lower()
    assert SOURCE in prompt and "The council voted." in prompt


def test_genuine_paraphrase_is_not_flipped():
    assert a.decide(a.parse(json.dumps({"verdict": "supported"})), SOURCE, "The council will vote on the budget soon.") == (False, "auditor_supported")


def test_tense_change_with_authentic_related_quote_flips():
    ok, why = flips("The council voted on the budget on Monday.", disputed_assertion="The council voted on the budget",
                    error_type="tense_or_modality", claim_says="the vote already happened", source_says="the vote is planned for next week",
                    evidence_quote="the council will vote on the budget next week")
    assert (ok, why) == (True, "flip_with_authentic_related_quote")


def test_speaker_attribution_error_flips_only_with_real_quote():
    cand = "Councillor Diaz said the road repairs would start in spring."
    kw = {"disputed_assertion": "Councillor Diaz said the road repairs would start in spring", "error_type": "mis_referencing",
          "claim_says": "Diaz expects repairs in spring", "source_says": "Mayor Lee hoped repairs would start in spring"}
    assert flips(cand, evidence_quote="She hoped the road repairs would start in spring", **kw)[0] is True
    assert flips(cand, evidence_quote="Diaz promised repairs in spring", **kw) == (False, "quote_not_in_source")


def test_real_but_irrelevant_quote_is_rejected():
    ok, why = flips("The mayor cancelled the vote.", disputed_assertion="The mayor cancelled the vote", error_type="contradiction",
                    claim_says="vote cancelled", source_says="vote planned", evidence_quote="Councillor Diaz opposed the plan")
    assert (ok, why) == (False, "quote_unrelated_to_assertion")


def test_fabricated_or_missing_quote_is_rejected():
    kw = {"disputed_assertion": "The council voted", "error_type": "tense_or_modality", "claim_says": "voted", "source_says": "will vote"}
    assert flips("The council voted.", evidence_quote="the council has already voted on it", **kw) == (False, "quote_not_in_source")
    assert flips("The council voted.", **kw) == (False, "no_evidence_quote")


def test_assertion_must_come_from_the_candidate():
    assert flips("The council will vote.", disputed_assertion="The mayor resigned", error_type="contradiction", claim_says="x",
                 source_says="y", evidence_quote="Mayor Lee said on Monday")[1] == "assertion_not_in_candidate"


def test_unsupported_addition_without_quote_flips_on_absent_detail():
    cand = "The council will vote next week and the tax rate will rise to 9 percent."
    ok, why = flips(cand, disputed_assertion="the tax rate will rise to 9 percent", error_type="unsupported_addition", claim_says="tax rises to 9 percent",
                    source_says="nothing", unsupported_details=["tax rate will rise to 9 percent"])
    assert (ok, why) == (True, "flip_unsupported_addition")


def test_addition_whose_details_are_in_the_source_does_not_flip():
    ok, why = flips("The council will vote on the budget.", disputed_assertion="The council will vote on the budget", error_type="unsupported_addition",
                    claim_says="council votes", source_says="nothing", unsupported_details=["council vote budget"])
    assert (ok, why) == (False, "addition_details_present_in_source_or_not_in_candidate")


def test_short_explanation_or_identical_claim_blocks_a_flip():
    kw = {"disputed_assertion": "The council voted", "error_type": "tense_or_modality", "evidence_quote": "the council will vote on the budget"}
    assert flips("The council voted.", claim_says="voted", source_says="will vote", explanation="Wrong.", **kw)[1] == "explanation_too_short"
    assert flips("The council voted.", claim_says="Voted", source_says="voted", **kw)[1] == "claim_equals_source"


@pytest.mark.parametrize("bad", ["no json", '{"verdict": "maybe"}', '{"verdict": "unsupported", "error_type": "vibes"}'])
def test_unparseable_output_is_rejected(bad):
    with pytest.raises(a.AuditParseError):
        a.parse(bad)


def test_cost_plan_includes_reasoning_tokens_and_retries():
    plan = a.plan_cost(182)
    assert plan["calls"] == 182 and plan["expected_usd"] > 182 * 0.0065
    assert plan["worst_case_usd"] == pytest.approx(182 * 2 * a.worst_case_call_usd(), rel=0.01)


def test_cap_stops_the_run_incomplete_and_never_overshoots():
    items = [{"id": f"c{i}", "source": SOURCE, "candidate": "x"} for i in range(10)]
    cap = a.worst_case_call_usd() * 3.5
    out = a.run_stage(items, lambda prompt: (json.dumps({"verdict": "supported"}), a.worst_case_call_usd() * 0.5), cap_usd=cap)
    assert out["status"].startswith("INCOMPLETE") and out["spent_usd"] <= cap and 0 < len(out["rows"]) < 10


def test_failed_calls_are_not_flips_and_make_the_run_incomplete():
    def boom(prompt):
        raise RuntimeError("provider down")
    out = a.run_stage([{"id": "c", "source": SOURCE, "candidate": "x"}], boom, cap_usd=1.0)
    assert out["status"] == "INCOMPLETE: failed audits" and out["rows"]["c"]["flip"] is False


def test_unknown_cost_is_charged_at_its_bound_not_zero():
    meter = a.Meter(1.0)
    held = meter.reserve(0.2)
    meter.settle(held, None)
    assert meter.spent == pytest.approx(0.2)


def test_kappa_arithmetic_matches_the_review_numbers():
    # 128 dev cases: v18 has 14 false passes + 5 false rejections (kappa 0.6745). Four corrections: 0.7471 with 0 new errors, 0.7313 with 1.
    labels = [0] * 49 + [1] * 79
    v18 = [0] * 35 + [1] * 14 + [0] * 5 + [1] * 74
    ids = [f"c{i}" for i in range(128)]
    L, V = dict(zip(ids, labels, strict=True)), dict(zip(ids, v18, strict=True))
    fixed = {ids[35 + k]: True for k in range(4)}
    assert a.score_flips(L, V, {})["kappa_v18"] == pytest.approx(0.6745, abs=1e-3)
    assert a.score_flips(L, V, fixed)["kappa_audited"] == pytest.approx(0.7471, abs=1e-3)
    with_one_new = {**fixed, ids[85]: True}  # a supported case wrongly flipped
    r = a.score_flips(L, V, with_one_new)
    assert r["corrected"] == 4 and r["introduced"] == 1 and r["kappa_audited"] == pytest.approx(0.7313, abs=1e-3)
