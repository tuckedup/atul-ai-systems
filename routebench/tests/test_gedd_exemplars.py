import json

import pytest
from evalops.dataset import Annotation, CaseRecord, Corpus, LabelProvenance
from evalops.exemplars import validate_for_corpus
from evalops.gedd_exemplars import build
from evalops.splits import SplitPlan


def fixture():
    cases, annotations, notes, raw = [], [], [], []
    for cid, split, label in (("good", "train", 1), ("bad", "train", 0),
                              ("dev", "dev", 0), ("test", "test", 0)):
        case = CaseRecord(case_id=cid, group_id=cid, task_class="summarize", task_input="Check",
                          context=f"Source {cid}", candidate_output=cid,
                          source="LLM-AggreFact/TofuEval-MediaS")
        cases.append(case)
        annotations.append(Annotation(case_id=cid, annotator_id="human", label=label,
                                      provenance=LabelProvenance.HUMAN_EXPERT))
        notes.append({"subset": "TofuEval-MediaS", "summ_sent": cid,
                      "sent_label": "yes" if label else "no", "exp": f"Expert reason {cid}",
                      "type": "" if label else "Contradiction"})
        raw.append({"subset": "TofuEval-MediaS", "claim": cid, "doc": case.context})
    plan = SplitPlan({"good": "train", "bad": "train", "dev": "dev", "test": "test"},
                     {"train": .5, "dev": .25, "test": .25}, 7)
    return Corpus(cases, annotations), plan, notes, raw


def test_uses_only_train_and_preserves_published_reason_and_full_source():
    result = build(*fixture())
    examples = result["exemplars"]["summarize"]
    assert {e["case_id"] for e in examples} == {"good", "bad"}
    failed = next(e for e in examples if e["verdict"] == "FAIL")
    assert failed["context"] == "Source bad"
    assert failed["why"] == "Contradiction: Expert reason bad"
    assert result["source_split"] == "train" and result["annotations_hash"]


@pytest.mark.parametrize("problem", ["ambiguous", "conflicting_label", "cropped"])
def test_refuses_bad_training_evidence(problem):
    corpus, plan, notes, raw = fixture()
    if problem == "ambiguous":
        notes.append(dict(notes[1]))
    elif problem == "conflicting_label":
        notes[1]["sent_label"] = "yes"
    else:
        raw[1]["doc"] += " Evidence missing from supplied example."
    with pytest.raises(ValueError, match="both classes"):
        build(corpus, plan, notes, raw)


@pytest.mark.parametrize("tamper", ["binding", "test_case", "verdict", "context"])
def test_preflight_rejects_stale_or_test_exemplars(tmp_path, tamper):
    corpus, plan, notes, raw = fixture()
    packet = build(corpus, plan, notes, raw)
    path = tmp_path / "examples.json"
    path.write_text(json.dumps(packet), encoding="utf-8")
    validate_for_corpus(path, corpus, plan)
    if tamper == "binding":
        packet["split_membership_hash"] = "stale"
    else:
        ex = packet["exemplars"]["summarize"][0]
        ex[{"test_case": "case_id", "verdict": "verdict", "context": "context"}[tamper]] = {
            "test_case": "test", "verdict": "PASS", "context": "cut off",
        }[tamper]
    path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_for_corpus(path, corpus, plan)
