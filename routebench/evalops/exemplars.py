"""Build per-task few-shot exemplars from the TRAIN split only.

Two constraints make this safe:

*   **Train only.** Exemplars are drawn from `train`, never `dev` (which selects the threshold and
    the variant) and never `test`. `build()` takes a `SplitPlan` and asserts it. An exemplar drawn
    from test would put a test label directly into the judge's prompt.
*   **Per task.** Exemplars are keyed by task class and only a case's own task's exemplars are
    rendered into its prompt. A SQL example inside a code judgment is not a hint, it is noise --
    and worse, it dilutes the rubric it is meant to illustrate.

Exemplars are chosen to show the *decision boundary*, not typical cases. For each task the
selection prefers a failing candidate that superficially looks fine and a passing candidate that
superficially looks wrong, because those are where a judge's prior overrides the rubric. They are
selected by label and length only -- no judge output is consulted, so this cannot become a way of
teaching the judge the answers to items it got wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import Annotation, CaseRecord, Corpus
from .splits import Split, SplitPlan, select

DATA = Path(__file__).parent / "data"
DEFAULT_PATH = DATA / "exemplars.json"

#: Keep exemplars short: they are prepended to every judge prompt, so a long one is paid for on
#: every call and crowds out the material being graded.
MAX_CHARS = 700
MAX_EVIDENCE_CHARS = 6000


def _render(case: CaseRecord, annotation: Annotation) -> dict[str, str]:
    return {
        "case_id": case.case_id,
        "task_input": case.task_input,
        "context": case.context,
        "reference": case.reference,
        "candidate_output": case.candidate_output,
        "verdict": "PASS" if annotation.label == 1 else "FAIL",
        "why": annotation.reason[:240] or (
            "the recorded annotation accepts this response"
            if annotation.label == 1 else "the recorded annotation rejects this response"
        ),
    }


def build(
    corpus: Corpus,
    plan: SplitPlan,
    *,
    per_class: int = 2,
    split: Split = "train",
) -> dict[str, list[dict[str, str]]]:
    if split != "train":
        raise ValueError(
            f"exemplars must come from the train split, not {split!r}: dev selects the threshold "
            "and the variant, and test is the frozen measurement"
        )
    cases = select(corpus.cases, plan, split)
    labels = corpus.resolved_labels()
    out: dict[str, list[dict[str, str]]] = {}
    for task in sorted({c.task_class for c in cases}):
        pool = [
            (c, labels[c.case_id]) for c in cases
            if c.task_class == task and c.case_id in labels
            # Never cut off the evidence or the part of a response that caused its label.
            # Exclude oversized demonstrations instead of teaching a corrupted example.
            and len(c.candidate_output) <= MAX_CHARS
            and len(c.task_input) <= MAX_CHARS
            and len(c.context) <= MAX_EVIDENCE_CHARS
            and len(c.reference) <= MAX_EVIDENCE_CHARS
            and (c.task_class != "summarize" or bool(c.context.strip()))
        ]
        chosen: list[dict[str, str]] = []
        for want in (0, 1):  # failing example first: it carries more information
            # Shortest first -- a compact example is easier to generalise from and cheaper to send.
            bucket = sorted(
                (x for x in pool if x[1].label == want),
                key=lambda x: len(x[0].candidate_output),
            )
            chosen += [_render(c, a) for c, a in bucket[:per_class]]
        if chosen:
            out[task] = chosen
    return out


def load(path: str | Path | None = None) -> dict[str, list[dict[str, str]]]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.get("exemplars", raw).items()}


def validate_for_corpus(path: str | Path, corpus: Corpus, plan: SplitPlan) -> None:
    """Reject stale or non-train demonstrations before a calibration provider call."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"source_split": "train", "dataset_hash": corpus.cases_hash,
                "annotations_hash": corpus.annotations_hash,
                "split_membership_hash": plan.membership_hash, "split_seed": plan.seed}
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"exemplar {key} missing or changed; rebuild from train")
    train = {c.case_id: c for c in select(corpus.cases, plan, "train")}
    labels = corpus.resolved_labels()
    for task, examples in raw.get("exemplars", {}).items():
        for ex in examples:
            case = train.get(ex.get("case_id"))
            if case is None or case.task_class != task:
                raise ValueError("exemplar is not in the declared task's train split")
            for key in ("task_input", "context", "reference", "candidate_output"):
                if ex.get(key, "") != getattr(case, key):
                    raise ValueError(f"exemplar {key} differs from its complete train case")
            expected_verdict = "PASS" if labels[case.case_id].label == 1 else "FAIL"
            if ex.get("verdict") != expected_verdict:
                raise ValueError("exemplar verdict differs from the recorded train annotation")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    args = ap.parse_args(argv)

    corpus = Corpus.load(DATA)
    plan = SplitPlan.load(DATA / "splits.json")
    exemplars = build(corpus, plan, per_class=args.per_class)
    payload: dict[str, Any] = {
        "source_split": "train",
        "per_class": args.per_class,
        "split_seed": plan.seed,
        "dataset_hash": corpus.cases_hash,
        "annotations_hash": corpus.annotations_hash,
        "split_membership_hash": plan.membership_hash,
        "selection_rule": "shortest candidate per label class; no judge output consulted",
        "exemplars": exemplars,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for task, items in sorted(exemplars.items()):
        verdicts = [i["verdict"] for i in items]
        print(f"  {task:10s} {len(items)} exemplars {verdicts}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
