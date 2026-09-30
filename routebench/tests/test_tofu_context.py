import json
from pathlib import Path

import pytest
from evalops.dataset import Corpus
from evalops.judge import JudgeConfig, build_prompt
from evalops.rubrics import for_task

DATA = Path(__file__).parent.parent / "evalops" / "data"


def _cases():
    if not (DATA / "tofu_context.json").exists():
        pytest.skip("optional local-only summary context artifact is not redistributed")
    corpus = Corpus.load(DATA)
    ctx = json.loads((DATA / "tofu_context.json").read_text(encoding="utf8"))["entries"]
    return {c.case_id: c for c in corpus.cases}, ctx


def test_every_entry_matches_its_case_sentence():
    cases, ctx = _cases()
    assert len(ctx) > 300
    for cid, e in ctx.items():
        assert e["sentences"][e["target_index"]] == cases[cid].candidate_output.strip()


def test_context_prompt_marks_target_and_leaves_others_identical():
    cases, ctx = _cases()
    base = JudgeConfig(variant_id="a", model="o4-mini", restrict_tasks=("summarize",))
    withctx = JudgeConfig(variant_id="b", model="o4-mini", restrict_tasks=("summarize",),
                          context_path=str(DATA / "tofu_context.json"))
    cid = "gnd-406e875c9ec93511"
    rubric = for_task("summarize", None)
    p = build_prompt(cases[cid], rubric, withctx)
    assert "GRADE THIS SENTENCE" in p and "Giuliani" in p.split("ORIGINAL SUMMARY CONTEXT")[1]
    plain = next(c for k, c in cases.items() if k not in ctx and c.task_class == "summarize")
    assert build_prompt(plain, rubric, base) == build_prompt(plain, rubric, withctx)
    assert base.config_hash != withctx.config_hash


def test_unset_context_keeps_old_hash_shape():
    a = JudgeConfig(variant_id="a", model="gpt-4.1")
    assert "context_path" not in a.as_dict() or a.context_path == ""
