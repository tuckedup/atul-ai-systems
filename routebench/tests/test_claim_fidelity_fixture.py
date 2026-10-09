"""Structure and contamination checks for the synthetic claim-fidelity regression fixture v2 (no model, no benchmark text)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "evalops" / "fixtures" / "claim_fidelity_regression_v2.json"
GOLD = Path(__file__).resolve().parents[2] / ".local" / "judge-v19-phase1" / "blind-review-v1" / "reconciliation" / "expert_explanation_audit_inputs.json"
VIOLATION_MODES = {"attribution", "hedge", "tense", "strength", "merged", "prediction", "disputed"}


def load():
    return json.loads(FIXTURE.read_text(encoding="utf8"))


def test_fixture_is_synthetic_labelled_and_unmodelled():
    d = load()
    assert d["version"] == "claim-fidelity-regression-v2" and "SYNTHETIC" in d["status"] and "NOT been checked against any model" in d["status"] and "EVIDENTIAL STATUS" in d["status"]
    ids = [i["id"] for i in d["items"]]
    assert len(ids) == len(set(ids)) >= 30
    for i in d["items"]:
        assert set(i) == {"id", "pair", "source", "sentence", "expected", "mode"} and i["expected"] in {"fails_claim_fidelity", "passes"} and i["mode"] in VIOLATION_MODES | {"none"}
        assert i["source"].strip() and i["sentence"].strip()


def test_every_violation_has_a_faithful_control_on_the_same_source_and_each_mode_is_covered():
    items = load()["items"]
    by_pair = {}
    for i in items:
        if i["pair"]:
            by_pair.setdefault(i["pair"], []).append(i)
    assert len(by_pair) == 14
    for pair, group in by_pair.items():
        assert sorted(g["expected"] for g in group) == ["fails_claim_fidelity", "passes"], pair
        assert len({g["source"] for g in group}) == 1 and len({g["sentence"] for g in group}) == 2
    assert {i["mode"] for i in items if i["expected"] == "fails_claim_fidelity"} == VIOLATION_MODES
    guards = [i for i in items if i["mode"] == "none"]
    assert len(guards) >= 7 and all(g["expected"] == "passes" for g in guards)


def test_status_not_authority_both_kinds_of_speaker_are_covered_for_predictions_disputes_and_plain_facts():
    ids = {i["id"] for i in load()["items"]}
    for needed in ("prediction-official-violation", "prediction-nonofficial-violation", "disputed-official-report-violation", "disputed-official-claim-violation", "disputed-nonofficial-report-violation",
                   "authority-plain-fact-guard", "nonauthority-plain-fact-guard", "official-qualified-prediction-kept-guard"):
        assert needed in ids, needed
    by_id = {i["id"]: i for i in load()["items"]}
    assert by_id["prediction-official-violation"]["expected"] == "fails_claim_fidelity" and by_id["authority-plain-fact-guard"]["expected"] == "passes"  # no blanket exemption, no blanket penalty


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def grams(s: str, n: int = 6) -> set[tuple[str, ...]]:
    t = tokens(s)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


@pytest.mark.skipif(not GOLD.exists(), reason="local audit inputs not present")
def test_no_text_is_copied_from_reviewed_sentences_sources_or_expert_explanations():
    gold = json.loads(GOLD.read_text(encoding="utf8"))["items"]
    protected = set()
    for g in gold:
        for field in ("sentence", "expert_explanation", "reviewer_explanation"):
            protected |= grams(g.get(field) or "")
        for p in g.get("source_passages_for_sentence", []) + g.get("source_passages_for_objection", []):
            protected |= grams(p)
    for i in load()["items"]:
        assert not (grams(i["source"]) | grams(i["sentence"])) & protected, i["id"]
