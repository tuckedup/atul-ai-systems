"""Mocked end-to-end tests for the opt-in claim_fidelity variant: the criterion reaches the real prompt, is REQUIRED in returned judgments, survives parsing and storage, and changes the decision as
specified. v18 and its identity are preserved. The provider is replaced by a scripted double: no provider call, no GPU, no download."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from aisys import llm  # noqa: F401  (the judge module's own client is patched below)
from evalops import claim_fidelity as cf
from evalops import experiments, judge, rubrics
from evalops import judge_v19_analysis as an
from evalops.dataset import CaseRecord

FIXTURE = Path(__file__).resolve().parents[1] / "evalops" / "fixtures" / "claim_fidelity_regression_v2.json"
GOLD = Path(__file__).resolve().parents[2] / ".local" / "judge-v19-phase1" / "blind-review-v1" / "reconciliation" / "expert_explanation_audit_inputs.json"
OLD_IDS = ["no_contradiction", "claims_entailed", "specifics_grounded", "no_invented_relations"]
ALL_IDS = [*OLD_IDS, "claim_fidelity"]


def case(sentence="The council approved the budget.", source="The council voted 7-2 on Tuesday to approve the budget.", cid="synthetic") -> CaseRecord:
    return CaseRecord(case_id=cid, group_id="g", task_class="summarize", task_input="Summarise the source document faithfully.", context=source, candidate_output=sentence)


def reply(verdicts: dict[str, str], drop=(), extra=(), duplicate=()):
    entries = [{"id": c, "verdict": v, "evidence": ""} for c, v in verdicts.items() if c not in drop]
    entries += [{"id": c, "verdict": "yes", "evidence": ""} for c in extra] + [{"id": c, "verdict": "yes", "evidence": ""} for c in duplicate]
    return json.dumps({"criteria": entries, "critical_errors": [], "rationale": "scripted"})


def all_yes(**override):
    return {**{c: "yes" for c in ALL_IDS}, **override}


class Chat:
    """Scripted replacement for the judge module's client. `script` is a list of reply strings, or a callable(prompt) -> reply."""

    def __init__(self, script):
        self.script, self.prompts = script, []

    def __call__(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        text = self.script(messages[0]["content"]) if callable(self.script) else self.script[min(len(self.prompts), len(self.script)) - 1]
        return SimpleNamespace(text=text, model="o4-mini-2025-04-16", usage=SimpleNamespace(prompt_tokens=800, completion_tokens=200), cost_usd=0.001)


@pytest.fixture
def chat(monkeypatch):
    def install(script):
        c = Chat(script)
        monkeypatch.setattr(judge.llm, "chat", c)
        return c

    return install


# ------------------------------------------------------------------------------------------------------------------------------ identity: v18 untouched
def test_v18_identity_is_preserved_and_the_new_variant_differs_only_in_its_rubric():
    ident = cf.identity()
    assert (ident["v18_config_hash"], ident["v18_rubric_hash"], ident["v18_prompt_template_hash"]) == (cf.V18_IDENTITY["config_hash"], cf.V18_IDENTITY["rubric_hash"], cf.V18_IDENTITY["prompt_template_hash"])
    assert ident["claim_fidelity_config_hash"] != ident["v18_config_hash"] and ident["claim_fidelity_rubric_hash"] != ident["v18_rubric_hash"]
    assert ident["claim_fidelity_prompt_template_hash"] == ident["v18_prompt_template_hash"]  # same scaffolding: the only change is the rubric
    base, new = cf.base_config(), cf.config()
    import dataclasses

    diff = {f.name for f in dataclasses.fields(base) if getattr(base, f.name) != getattr(new, f.name)}
    assert diff == {"variant_id", "rubric_dir", "notes"} and new.model == base.model and new.reasoning_effort == base.reasoning_effort and new.mode == "rubric"
    assert new.variant_id == cf.VARIANT_ID != base.variant_id and base.variant_id == "v18-reasoning-o4mini"


def test_the_default_rubric_directory_and_bundle_are_unchanged():
    assert rubrics.for_task("summarize").hash == cf.V18_IDENTITY["rubric_hash"]
    assert rubrics.bundle_hash() == "a615d70cb0cf0c8a"  # frozen protocols record this in their code hashes
    assert [c.id for c in rubrics.for_task("summarize").criteria] == OLD_IDS


def test_the_new_rubric_is_the_old_one_plus_exactly_one_criterion():
    old, new = rubrics.for_task("summarize"), cf.rubric()
    assert new.criteria[:4] == old.criteria and [c.id for c in new.criteria] == ALL_IDS
    assert (new.pass_definition, new.boundary_examples, new.task_class, new.evidence_required) == (old.pass_definition, old.boundary_examples, old.task_class, old.evidence_required)
    added = new.criterion("claim_fidelity")
    assert not added.critical and added.weight == 2 and new.version == "1.1.0-claim-fidelity-v1" and new.hash != old.hash


def test_the_frozen_judge_v19_protocol_still_verifies_after_these_additions():
    from evalops import judge_v19_phase1 as jv

    protocol = jv.WORK / "protocol.json"
    if not protocol.exists():
        pytest.skip("frozen protocol not present")
    sha = jv._sha(protocol)
    jv.verify(jv.WORK, sha)
    assert json.loads(protocol.read_text(encoding="utf8"))["code"]["rubric_bundle"] == rubrics.bundle_hash()


# ------------------------------------------------------------------------------------------------------------------------------ the criterion reaches the real prompt
def test_the_criterion_text_reaches_the_prompt_and_the_required_ids_list_includes_it():
    c = case()
    new = judge.build_prompt(c, cf.rubric(), cf.config())
    old = judge.build_prompt(c, rubrics.for_task("summarize"), cf.base_config())
    crit = cf.rubric().criterion("claim_fidelity")
    for needle in ("- id: claim_fidelity", crit.question.strip(), crit.pass_when.strip(), crit.fail_when.strip()):
        assert needle in new and needle not in old
    assert ", ".join(sorted(ALL_IDS)) in new and ", ".join(sorted(OLD_IDS)) in old
    only_old = [x for x in old.split("\n") if x not in set(new.split("\n"))]
    assert only_old == [f"{', '.join(sorted(OLD_IDS))}"]  # the one line that changes is the required-ids list; nothing of v18's prompt is removed
    assert "claim_fidelity" not in old


def test_the_criterion_judges_evidential_status_not_speaker_authority():
    crit = cf.rubric().criterion("claim_fidelity")
    text = f"{crit.question} {crit.pass_when} {crit.fail_when}".lower()
    for needle in ("evidential status", "opinion", "prediction", "disputed", "tense", "faithful paraphrase", "whoever the speaker is", "not how authoritative the speaker is", "an official's prediction"):
        assert needle in text, needle
    assert "ordinary unqualified factual report" in text and "plainly stated fact may be restated plainly" in text


@pytest.mark.skipif(not GOLD.exists(), reason="local audit inputs not present")
def test_the_criterion_contains_no_benchmark_text_expert_explanation_or_case_specific_names():
    crit = cf.rubric().criterion("claim_fidelity")
    text = f"{crit.question} {crit.pass_when} {crit.fail_when}"
    toks = re.findall(r"[a-z0-9']+", text.lower())
    mine = {tuple(toks[i:i + 6]) for i in range(len(toks) - 5)}
    gold = json.loads(GOLD.read_text(encoding="utf8"))["items"]
    protected = set()
    for g in gold:
        for field in ("sentence", "expert_explanation", "reviewer_explanation"):
            t = re.findall(r"[a-z0-9']+", (g.get(field) or "").lower())
            protected |= {tuple(t[i:i + 6]) for i in range(len(t) - 5)}
    assert not mine & protected
    for name in ("barrett", "nra", "quds", "woodward", "newtown", "giuliani", "tofueval", "aggrefact"):
        assert name not in text.lower()


# ------------------------------------------------------------------------------------------------------------------------------ the criterion is REQUIRED and decides
def test_all_yes_passes_and_the_stored_judgment_names_every_criterion(chat):
    chat([reply(all_yes())])
    j = judge.judge_case(case(), cf.config())
    assert j.usable and j.status == "ok" and j.raw_score == 1.0 and cf.decision(j.raw_score) == 1
    assert [c["id"] for c in j.criteria] == ALL_IDS and all(c["verdict"] is True for c in j.criteria)
    assert j.rubric_version == f"summarize@1.1.0-claim-fidelity-v1+{cf.rubric().hash}" and j.config_hash == cf.config().config_hash and j.variant_id == cf.VARIANT_ID


def test_a_no_on_claim_fidelity_alone_fails_the_response_under_the_unchanged_all_criteria_rule(chat):
    chat([reply(all_yes(claim_fidelity="no"))])
    j = judge.judge_case(case(), cf.config())
    assert j.usable and j.raw_score == pytest.approx((1 + 4 + 2 + 1) / 10) and cf.decision(j.raw_score) == 0
    assert {c["id"]: c["verdict"] for c in j.criteria}["claim_fidelity"] is False and an.verdict(j.raw_score) == 0  # same decision as the frozen analysis helper


def test_every_original_criterion_still_decides_exactly_as_before(chat):
    for cid in OLD_IDS:
        chat([reply(all_yes(**{cid: "no"}))])
        j = judge.judge_case(case(), cf.config())
        assert cf.decision(j.raw_score) == 0, cid
    chat([reply(all_yes(no_contradiction="no"))])
    assert judge.judge_case(case(), cf.config()).raw_score == 0.0  # the critical criterion still zeroes the score


def test_a_missing_claim_fidelity_verdict_is_an_error_never_a_pass(chat):
    c = chat([reply(all_yes(), drop=("claim_fidelity",))])
    j = judge.judge_case(case(), cf.config())
    assert not j.usable and j.raw_score is None and j.status == "parse_error" and "claim_fidelity" in (j.error or "") and len(c.prompts) == cf.config().max_attempts
    assert j.criteria == []  # nothing was stored as if it were a verdict
    with pytest.raises(rubrics.RubricError):
        rubrics.score(cf.rubric(), {k: True for k in OLD_IDS})  # the score rule refuses an incomplete verdict set


def test_a_repair_that_supplies_the_missing_criterion_is_accepted_on_the_second_attempt(chat):
    c = chat([reply(all_yes(), drop=("claim_fidelity",)), reply(all_yes(claim_fidelity="no"))])
    j = judge.judge_case(case(), cf.config())
    assert j.usable and j.attempts == 2 and len(c.prompts) == 2 and cf.decision(j.raw_score) == 0
    assert "claim_fidelity" in c.prompts[0]  # the criterion was asked for in the first prompt


@pytest.mark.parametrize("script", [reply(all_yes(), extra=("invented_criterion",)), reply(all_yes(), duplicate=("claim_fidelity",)),
                                    reply(all_yes(claim_fidelity="perhaps")), json.dumps({"criteria": []})])
def test_unexpected_extra_duplicate_or_unreadable_claim_fidelity_output_is_rejected(chat, script):
    chat([script])
    j = judge.judge_case(case(), cf.config())
    assert not j.usable and j.raw_score is None and j.status == "parse_error"


def test_the_same_reply_is_rejected_by_v18_because_v18_has_no_such_criterion(chat):
    chat([reply(all_yes())])
    j = judge.judge_case(case(), cf.base_config())
    assert not j.usable and "claim_fidelity" in (j.error or "")
    chat([reply({c: "yes" for c in OLD_IDS})])
    assert judge.judge_case(case(), cf.base_config()).raw_score == 1.0  # v18's own contract is unchanged


# ------------------------------------------------------------------------------------------------------------------------------ persistence and cache isolation
def test_the_criterion_survives_storage_and_reload(chat, tmp_path):
    chat([reply(all_yes(claim_fidelity="no"))])
    c, config = case(), cf.config()
    j = judge.judge_case(c, config)
    path = tmp_path / "cache.jsonl"
    cache = experiments.JudgmentCache(path)
    key = experiments._cache_key(c, config)
    cache.put(key, j)
    again = experiments.JudgmentCache(path).get(key)
    assert again is not None and again.usable and again.raw_score == j.raw_score and cf.decision(again.raw_score) == 0
    assert [(x["id"], x["verdict"]) for x in again.criteria] == [(i, i != "claim_fidelity") for i in ALL_IDS] and again.rubric_version == j.rubric_version


def test_cache_keys_never_let_a_v18_judgment_answer_for_the_new_variant(chat, tmp_path):
    c = case()
    chat([reply({k: "yes" for k in OLD_IDS})])
    v18_judgment = judge.judge_case(c, cf.base_config())
    cache = experiments.JudgmentCache(tmp_path / "cache.jsonl")
    cache.put(experiments._cache_key(c, cf.base_config()), v18_judgment)
    assert experiments._cache_key(c, cf.base_config()) != experiments._cache_key(c, cf.config())
    assert cache.get(experiments._cache_key(c, cf.config())) is None
    assert cache.get(experiments._cache_key(c, cf.base_config())) is not None


# ------------------------------------------------------------------------------------------------------------------------------ the synthetic fixture drives the whole pipeline
def test_every_fixture_item_flows_through_prompt_parse_storage_and_decision_as_specified(chat, tmp_path):
    items = json.loads(FIXTURE.read_text(encoding="utf8"))["items"]
    by_text = {(i["sentence"], i["source"]): i for i in items}

    def scripted_judge(prompt: str) -> str:
        sentence = prompt.split("<<<BEGIN_CANDIDATE\n", 1)[1].split("\nEND_CANDIDATE>>>", 1)[0]
        source = prompt.split("=== SOURCE MATERIAL (the only admissible support) ===\n", 1)[1].split("\n\n", 1)[0]
        item = by_text[(sentence, source)]  # the prompt carried exactly this item's text
        return reply(all_yes(claim_fidelity="no" if item["expected"] == "fails_claim_fidelity" else "yes"))

    chat(scripted_judge)
    cache = experiments.JudgmentCache(tmp_path / "cache.jsonl")
    for i in items:
        c = case(sentence=i["sentence"], source=i["source"], cid=i["id"])
        j = judge.judge_case(c, cf.config())
        assert j.usable, i["id"]
        cache.put(experiments._cache_key(c, cf.config()), j)
        stored = experiments.JudgmentCache(tmp_path / "cache.jsonl").get(experiments._cache_key(c, cf.config()))
        assert (cf.decision(stored.raw_score) == 1) == (i["expected"] == "passes"), i["id"]
    assert len(items) >= 30
