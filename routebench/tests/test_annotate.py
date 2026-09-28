from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from evalops import annotate
from evalops.dataset import Annotation, CaseRecord, Corpus, LabelProvenance, read_jsonl
from evalops.splits import SplitPlan

QUEUE_KEYS = {
    "case_id", "task_class", "task_input", "context", "reference",
    "candidate_output", "rubric_id", "rubric_version",
}


def _case(case_id: str, task_class: str) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        group_id=f"grp-{case_id}",
        task_class=task_class,
        task_input=f"input for {case_id}",
        context=f"context for {case_id}",
        reference=f"reference for {case_id}",
        candidate_output=f"candidate for {case_id}",
        generator_id="unit-test",
        source="unit-test",
    )


def _synthetic_corpus(
    n_per_task: int = 20, tasks: tuple[str, ...] = ("code", "sql")
) -> tuple[list[CaseRecord], list[Annotation], SplitPlan]:
    """A small corpus with two task classes and alternating existing labels, all assigned to
    the "test" split, so `export_queue(..., split="test")` sees the whole thing as eligible.
    """
    cases: list[CaseRecord] = []
    annotations: list[Annotation] = []
    assignment: dict[str, str] = {}
    for task in tasks:
        for i in range(n_per_task):
            case_id = f"{task}-{i:03d}"
            case = _case(case_id, task)
            cases.append(case)
            assignment[case.group_id] = "test"
            annotations.append(
                Annotation(
                    case_id=case_id,
                    annotator_id="oracle",
                    label=i % 2,
                    provenance=LabelProvenance.GOLD_ORACLE,
                )
            )
    sp = SplitPlan(
        assignment=assignment, weights={"train": 0.34, "dev": 0.33, "test": 0.33}, seed=1
    )
    return cases, annotations, sp


def _read_queue_file(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def test_export_produces_requested_count_and_is_stratified(tmp_path):
    cases, annotations, sp = _synthetic_corpus(n_per_task=20)
    out = tmp_path / "queue.jsonl"

    counts = annotate.export_queue(
        cases, annotations, sp, split="test", n=20, seed=42, out_path=out
    )

    rows = _read_queue_file(out)
    assert len(rows) == 20
    assert sum(counts.values()) == 20
    assert set(counts) == {"code", "sql"}
    # Two equally-sized task pools should split ~evenly.
    assert abs(counts["code"] - counts["sql"]) <= 1


def test_export_queue_rows_have_no_label_field(tmp_path):
    cases, annotations, sp = _synthetic_corpus(n_per_task=10)
    out = tmp_path / "queue.jsonl"

    annotate.export_queue(cases, annotations, sp, split="test", n=10, seed=1, out_path=out)

    rows = _read_queue_file(out)
    assert rows, "export produced no rows to check"
    for row in rows:
        assert set(row.keys()) == QUEUE_KEYS, (
            "queue row carries an unexpected key (possibly a leaked label): "
            f"{set(row.keys()) - QUEUE_KEYS}"
        )


def test_export_is_deterministic_for_fixed_seed(tmp_path):
    cases, annotations, sp = _synthetic_corpus(n_per_task=15)
    out1 = tmp_path / "q1.jsonl"
    out2 = tmp_path / "q2.jsonl"

    annotate.export_queue(cases, annotations, sp, split="test", n=12, seed=99, out_path=out1)
    annotate.export_queue(cases, annotations, sp, split="test", n=12, seed=99, out_path=out2)

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_export_excludes_already_human_local_labelled_cases(tmp_path):
    cases, annotations, sp = _synthetic_corpus(n_per_task=5)
    for c in cases:
        if c.task_class == "code":
            annotations.append(
                Annotation(
                    case_id=c.case_id, annotator_id="alice", label=1,
                    provenance=LabelProvenance.HUMAN_LOCAL,
                )
            )
    out = tmp_path / "queue.jsonl"

    counts = annotate.export_queue(
        cases, annotations, sp, split="test", n=10, seed=1, out_path=out
    )

    assert "code" not in counts
    assert counts.get("sql", 0) == 5


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------
def _queue_row(case_id: str, task_class: str = "code") -> dict:
    rubric = annotate.for_task(task_class)
    return {
        "case_id": case_id, "task_class": task_class,
        "task_input": f"input {case_id}", "context": "", "reference": "",
        "candidate_output": f"candidate {case_id}",
        "rubric_id": rubric.id, "rubric_version": rubric.version,
    }


def _ticker(prefix: str = "2026-01-01T00:00:") -> Callable[[], str]:
    def gen():
        i = 0
        while True:
            yield f"{prefix}{i:02d}+00:00"
            i += 1
    g = gen()
    return lambda: next(g)


def test_label_writes_expected_annotation_rows(tmp_path):
    queue_rows = [_queue_row("code-000"), _queue_row("code-001")]
    keys = iter(["p", "f"])
    reasons = iter(["looks right", ""])
    out = tmp_path / "annotations.jsonl"

    counts = annotate.run_labeling_session(
        queue_rows, annotator_id="alice", out_path=out,
        get_key=lambda: next(keys), get_reason=lambda: next(reasons), now=_ticker(),
    )

    assert counts == {"pass": 1, "fail": 1, "skip": 0}
    rows = list(read_jsonl(out, Annotation))
    assert len(rows) == 2
    assert rows[0].case_id == "code-000" and rows[0].label == 1
    assert rows[0].reason == "looks right"
    assert rows[1].case_id == "code-001" and rows[1].label == 0
    for r in rows:
        assert r.provenance == LabelProvenance.HUMAN_LOCAL
        assert r.annotator_id == "alice"


def test_verdict_persisted_immediately_on_abort(tmp_path):
    queue_rows = [_queue_row(f"code-{i:03d}") for i in range(3)]
    keys = iter(["p", "f"])  # exhausts before the 3rd item -- simulates a mid-session crash
    reasons = iter(["", ""])
    out = tmp_path / "annotations.jsonl"

    with pytest.raises(StopIteration):
        annotate.run_labeling_session(
            queue_rows, annotator_id="alice", out_path=out,
            get_key=lambda: next(keys), get_reason=lambda: next(reasons), now=_ticker(),
        )

    rows = list(read_jsonl(out, Annotation))
    assert [r.case_id for r in rows] == ["code-000", "code-001"]


def test_undo_removes_previous_verdict_without_stale_row(tmp_path):
    queue_rows = [_queue_row("code-000"), _queue_row("code-001")]
    # pass item0 -> undo (back to item0) -> fail item0 -> pass item1 -> loop ends naturally
    keys = iter(["p", "u", "f", "p"])
    reasons = iter(["", "", ""])
    out = tmp_path / "annotations.jsonl"

    counts = annotate.run_labeling_session(
        queue_rows, annotator_id="alice", out_path=out,
        get_key=lambda: next(keys), get_reason=lambda: next(reasons), now=_ticker(),
    )

    rows = list(read_jsonl(out, Annotation))
    assert [(r.case_id, r.label) for r in rows] == [("code-000", 0), ("code-001", 1)]
    assert counts == {"pass": 1, "fail": 1, "skip": 0}


def test_label_resumes_and_skips_already_labelled_by_same_annotator(tmp_path):
    out = tmp_path / "annotations.jsonl"
    queue_rows = [_queue_row("code-000"), _queue_row("code-001")]
    keys1 = iter(["p"])
    annotate.run_labeling_session(
        queue_rows[:1], annotator_id="alice", out_path=out,
        get_key=lambda: next(keys1), get_reason=lambda: "", now=_ticker(),
    )

    # Re-run against the full queue: code-000 must not be asked about again.
    keys2 = iter(["f"])
    counts = annotate.run_labeling_session(
        queue_rows, annotator_id="alice", out_path=out,
        get_key=lambda: next(keys2), get_reason=lambda: "", now=_ticker("2026-01-02T00:00:"),
    )

    rows = list(read_jsonl(out, Annotation))
    assert [r.case_id for r in rows] == ["code-000", "code-001"]
    assert counts == {"pass": 0, "fail": 1, "skip": 0}


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def test_status_computes_interannotator_kappa_correctly():
    cases = [_case(cid, "code") for cid in ("c1", "c2", "c3", "c4")]
    alice = {"c1": 1, "c2": 1, "c3": 0, "c4": 0}
    bob = {"c1": 1, "c2": 0, "c3": 0, "c4": 0}
    annotations = [
        Annotation(case_id=cid, annotator_id="alice", label=lab, provenance=LabelProvenance.HUMAN_LOCAL)
        for cid, lab in alice.items()
    ] + [
        Annotation(case_id=cid, annotator_id="bob", label=lab, provenance=LabelProvenance.HUMAN_LOCAL)
        for cid, lab in bob.items()
    ]
    corpus = Corpus(cases=cases, annotations=annotations)

    report = annotate.compute_status(corpus, queue_rows=[])

    assert report["n_cases_with_human_local"] == 4
    assert report["by_task_class"] == {"code": 4}
    assert report["by_label_value"] == {"0": 5, "1": 3}

    pair = report["inter_annotator"]
    assert pair is not None
    assert {pair["annotator_a"], pair["annotator_b"]} == {"alice", "bob"}
    assert pair["n_shared_cases"] == 4
    assert pair["defined"] is True
    # By hand: tp=1, fn=1, tn=2, fp=0 -> po=0.75, pe=0.5 -> kappa=0.5
    assert pair["kappa"] == pytest.approx(0.5)
    assert pair["observed_agreement"] == pytest.approx(0.75)
