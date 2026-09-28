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


def _render(case: CaseRecord, annotation: Annotation) -> dict[str, str]:
    return {
        "case_id": case.case_id,
        "task_input": case.task_input[:MAX_CHARS],
        "candidate_output": case.candidate_output[:MAX_CHARS],
        "verdict": "PASS" if annotation.label == 1 else "FAIL",
        "why": annotation.reason[:240] or (
            "the deterministic oracle accepted this response"
            if annotation.label == 1 else "the deterministic oracle rejected this response"
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
            and len(c.candidate_output) <= MAX_CHARS * 2
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
