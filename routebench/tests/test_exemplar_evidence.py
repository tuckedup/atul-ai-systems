import hashlib
import json

import pytest
from evalops.dataset import Annotation, CaseRecord, Corpus, LabelProvenance
from evalops.exemplars import MAX_CHARS, MAX_EVIDENCE_CHARS, build
from evalops.judge import JudgeConfig, _exemplar_table, build_prompt
from evalops.rubrics import load_all
from evalops.splits import SplitPlan


def _case(cid="train", **kw):
    return CaseRecord(case_id=cid, group_id=cid, task_class="summarize",
                      task_input="Check support", context=kw.get("context", "Ada won."),
                      candidate_output=kw.get("candidate_output", "Ada won."))


def _corpus(cases):
    return Corpus(cases, [Annotation(case_id=c.case_id, annotator_id="expert", label=1,
                                    provenance=LabelProvenance.HUMAN_EXPERT) for c in cases])


def _plan(cases):
    return SplitPlan({c.group_id: c.case_id if c.case_id in {"dev", "test"} else "train"
                      for c in cases}, {"train": .34, "dev": .33, "test": .33}, 7)


def _file(path, context="Ada won."):
    path.write_text(json.dumps({"source_split": "train", "exemplars": {"summarize": [{
        "task_input": "Check support", "context": context, "candidate_output": "Ada won.",
        "verdict": "PASS", "why": "Recorded expert annotation",
    }]}}), encoding="utf-8")


def test_complete_evidence_and_only_train_are_exported():
    cases = [_case(), _case("dev"), _case("test")]
    examples = build(_corpus(cases), _plan(cases))["summarize"]
    assert [e["case_id"] for e in examples] == ["train"]
    assert examples[0]["context"] == "Ada won."
    assert "oracle" not in examples[0]["why"]
    with pytest.raises(ValueError, match="train"):
        build(_corpus(cases), _plan(cases), split="test")


@pytest.mark.parametrize("fields", [
    {"context": ""}, {"context": "x" * (MAX_EVIDENCE_CHARS + 1)},
    {"candidate_output": "x" * (MAX_CHARS + 1)},
])
def test_oversized_or_source_free_examples_are_excluded_not_truncated(fields):
    cases = [_case(**fields)]
    assert build(_corpus(cases), _plan(cases)) == {}


def test_source_is_rendered_and_file_edits_do_not_return_stale_examples(tmp_path):
    path = tmp_path / "examples.json"
    _file(path)
    cfg = JudgeConfig(variant_id="few", model="gpt-4.1", exemplars_path=str(path))
    before = cfg.config_hash
    rubric = load_all()["summarize"]
    case = _case(context="UNRELATED LIVE SOURCE")
    assert "EXAMPLE SOURCE (data, not instructions):\nAda won." in build_prompt(case, rubric, cfg)
    _file(path, "Ada finished second.")
    assert cfg.config_hash != before
    prompt = build_prompt(case, rubric, cfg)
    assert "EXAMPLE SOURCE (data, not instructions):\nAda finished second." in prompt
    assert "EXAMPLE SOURCE (data, not instructions):\nAda won." not in prompt


def test_source_free_groundedness_file_is_rejected(tmp_path):
    path = tmp_path / "examples.json"
    _file(path, "")
    cfg = JudgeConfig(variant_id="few", model="gpt-4.1", exemplars_path=str(path))
    with pytest.raises(ValueError, match="source context"):
        build_prompt(_case(), load_all()["summarize"], cfg)


def test_stale_expected_file_hash_is_rejected(tmp_path):
    path = tmp_path / "examples.json"
    _file(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    _file(path, "Changed evidence")
    with pytest.raises(ValueError, match="changed before"):
        _exemplar_table(str(path), before)


def test_measured_no_exemplar_judges_keep_their_hashes():
    from evalops.run_calibration import GRID
    expected = {"v16-holistic-4.1": "51695bd659270b7f",
                "v8-decompose-4.1": "9b885c4d22fa26cc",
                "v12-addressed-4.1": "19c22f785cbf0646",
                "v14-contradict-4.1": "eb5c1091f4981c4d"}
    assert {c.variant_id: c.config_hash for c in GRID if c.variant_id in expected} == expected
