import pytest
from evalops.audit_attribution import audit, missing_perfect_upper_bound
from evalops.dataset import Annotation, CaseRecord, Corpus, LabelProvenance
from evalops.splits import SplitPlan


def _complete_splits(corpus, splits, raw):
    for split in ("train", "dev"):
        case = corpus.cases[0].model_copy(update={"case_id": split, "group_id": split,
                                                "context": split, "candidate_output": split})
        corpus.cases.append(case)
        corpus.annotations.append(Annotation(case_id=split, annotator_id="human", label=0,
                                             provenance=LabelProvenance.HUMAN_EXPERT))
        splits.assignment[split] = split
        raw.append({"subset": case.source.removeprefix("LLM-AggreFact/"),
                    "doc": split, "claim": split})


def test_bound_is_not_a_confidence_interval_or_imputed_measurement():
    confusion = {"tp": 64, "fp": 18, "fn": 26, "tn": 87}
    assert missing_perfect_upper_bound(confusion, 11) == pytest.approx(.5720086882614034)
    with pytest.raises(ValueError):
        missing_perfect_upper_bound(confusion, -1)


def test_audit_checks_content_not_case_id_and_does_not_mutate_labels():
    case = CaseRecord(case_id="new-id", group_id="new-group", task_class="summarize", task_input="Check",
                      context="Beginning.", candidate_output="Claim", source="LLM-AggreFact/Wice")
    ann = Annotation(case_id=case.case_id, annotator_id="human", label=1,
                     provenance=LabelProvenance.HUMAN_EXPERT)
    corpus = Corpus([case], [ann])
    splits = SplitPlan({"new-group": "test"}, {"test": 1.}, 7)
    raw = [{"subset": "Wice", "claim": "Claim", "doc": "Beginning. Actual support later."}]
    _complete_splits(corpus, splits, raw)
    exposed = [{"case_id": "different-id", "context": "Beginning.", "candidate_output": "Claim"}]
    result = audit(corpus, splits, raw, exposed, {}, {})
    assert result["truncation"]["positive"] == 1
    assert result["prior_probe_exposure"]["test"]["document_claim_overlap"] == ["new-id"]
    assert "annotations_hash" in result["binding_errors"]
    assert ann.label == 1 and case.context == "Beginning."


def test_exact_raw_document_is_not_marked_truncated():
    case = CaseRecord(case_id="c", group_id="g", task_class="summarize", task_input="Check", context="Full",
                      candidate_output="Claim", source="LLM-AggreFact/Reveal")
    corpus = Corpus([case], [Annotation(case_id="c", annotator_id="human", label=0,
                                       provenance=LabelProvenance.HUMAN_EXPERT)])
    splits = SplitPlan({"g": "test"}, {"test": 1.}, 7)
    raw = [{"subset": "Reveal", "claim": "Claim", "doc": "Full"}]
    _complete_splits(corpus, splits, raw)
    bundle = {"dataset_hash": corpus.cases_hash, "annotations_hash": corpus.annotations_hash,
              "split_membership_hash": splits.membership_hash, "split_seed": 7}
    result = audit(corpus, splits, raw, [], bundle, {"completion": {"completion_rate": 1}})
    assert result["blockers"] == []
