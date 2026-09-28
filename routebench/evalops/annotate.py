"""Human-in-the-loop pass/fail labelling for the calibration corpus.

Cohen's kappa compares two graders. It is only a measurement of judge quality if both
graders are answering the same question. If the human screen shows a looser or different
notion of "pass" than the rubric the judge prompt is built from -- a criterion worded
differently, a boundary case left out, a pass definition paraphrased instead of quoted --
kappa mostly measures that mismatch, not the judge. So every screen this module renders is
built from `rubrics.for_task(...)` and `Rubric.as_prompt_block()`, the exact objects
`judge.py` builds its prompt from. There is deliberately no second copy of rubric text
anywhere in this file.

The second defect this file exists to prevent is losing an evening of labelling to a crash.
Every verdict is appended to `annotations.jsonl` (via `dataset.append_jsonl`) the instant it
is entered, never buffered in memory for a final write at the end of the session.

The third defect: label leakage into the queue. `export` reads the pre-existing label only to
stratify the sample (so the queue is not all easy passes), then drops it. A queue row carries
exactly eight fields, none of which is a label. If the human annotator could see how a
published dataset or an oracle already scored a case, their verdict would be anchored by it,
and the two label sources would stop being independent measurements of the same response --
which is the whole point of collecting a second, local label.

Three subcommands:
    export  -- build data/annotation_queue.jsonl, stratified by task class and, within each
               task class, evenly across whichever label already exists for that case.
    label   -- the interactive terminal loop a human runs against that queue.
    status  -- labelling progress, plus an inter-annotator kappa diagnostic when two
               annotators have labelled overlapping cases. That number is NOT the same
               measurement as judge-vs-adjudicated-reference kappa -- see `compute_status`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import (
    Annotation,
    CaseRecord,
    Corpus,
    LabelProvenance,
    append_jsonl,
    read_jsonl,
)
from .metrics import agreement
from .rubrics import Rubric, for_task
from .splits import SplitPlan, select, verify
from .taxonomy import canonical

try:
    import msvcrt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only off Windows
    msvcrt = None  # type: ignore[assignment]

DATA = Path(__file__).parent / "data"

#: The exact, and only, fields a queue row may carry. Anything else -- most importantly a
#: label -- must never appear here; `_queue_row` is the single place rows are constructed.
QUEUE_FIELDS: tuple[str, ...] = (
    "case_id", "task_class", "task_input", "context", "reference", "candidate_output",
    "rubric_id", "rubric_version",
)

#: How much of context/reference to show before requiring 'm' to expand. Long enough to read
#: the shape of the material, short enough that ten items fit on one pleasant screen.
TRUNCATE_CHARS = 600


class AnnotateError(ValueError):
    """Raised for a labelling precondition that must not be silently worked around."""


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def _queue_row(case: CaseRecord) -> dict[str, Any]:
    """Build one queue row. The rubric fields come from `rubrics.for_task`, not from the case
    record, because that is the same lookup `judge.py` performs -- the guarantee that the
    annotator and the judge are shown the identical rubric, not a stale copy on the case.
    """
    rubric = for_task(case.task_class)
    return {
        "case_id": case.case_id,
        "task_class": case.task_class,
        "task_input": case.task_input,
        "context": case.context,
        "reference": case.reference,
        "candidate_output": case.candidate_output,
        "rubric_id": rubric.id,
        "rubric_version": rubric.version,
    }


def _select_within_task(
    pool: list[CaseRecord], resolved: dict[str, Annotation], k: int, rng: random.Random
) -> list[CaseRecord]:
    """Take `k` cases from one task class, spread evenly across existing label values.

    Without this, a random sample of a corpus that is mostly-pass reproduces a mostly-pass
    queue, and a human annotator spends the evening confirming easy passes instead of
    labelling the boundary cases that actually inform kappa.
    """
    if k >= len(pool):
        return list(pool)
    buckets: dict[str, list[CaseRecord]] = {}
    for c in pool:
        key = str(resolved[c.case_id].label) if c.case_id in resolved else "unlabelled"
        buckets.setdefault(key, []).append(c)
    bucket_keys = sorted(buckets)
    base, extra = divmod(k, len(bucket_keys))
    take = {key: base for key in bucket_keys}
    for key in bucket_keys[:extra]:
        take[key] += 1

    chosen: list[CaseRecord] = []
    leftover: list[CaseRecord] = []
    for key in bucket_keys:
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        want = take[key]
        chosen.extend(bucket[:want])
        leftover.extend(bucket[want:])
    shortfall = k - len(chosen)
    if shortfall > 0:
        rng.shuffle(leftover)
        chosen.extend(leftover[:shortfall])
    return chosen


def _select_stratified(
    eligible: list[CaseRecord], resolved: dict[str, Annotation], n: int, seed: int
) -> list[CaseRecord]:
    """Apportion `n` across task classes proportional to the eligible pool, largest-remainder
    method, then sample within each task class across label values. Fully deterministic: pools
    are sorted by `case_id` before any shuffling, so the only source of randomness is the seeded
    RNG advancing in a fixed order.
    """
    rng = random.Random(seed)
    by_task: dict[str, list[CaseRecord]] = {}
    for c in sorted(eligible, key=lambda c: c.case_id):
        by_task.setdefault(c.task_class, []).append(c)

    total = len(eligible)
    target = min(n, total)
    tasks = sorted(by_task)
    raw = {t: target * len(by_task[t]) / total for t in tasks}
    quota = {t: int(raw[t]) for t in tasks}
    remainder = target - sum(quota.values())
    # Largest fractional remainder gets the leftover seats; ties broken by task name so this
    # never depends on dict iteration order.
    order = sorted(tasks, key=lambda t: (-(raw[t] - quota[t]), t))
    for t in order[:remainder]:
        quota[t] += 1

    selected: list[CaseRecord] = []
    for t in tasks:
        selected.extend(_select_within_task(by_task[t], resolved, quota[t], rng))
    return selected


def _write_queue(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_queue(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def export_queue(
    cases: Sequence[CaseRecord],
    annotations: Sequence[Annotation],
    split_plan: SplitPlan,
    *,
    split: str,
    n: int,
    seed: int,
    out_path: Path,
    tasks: Sequence[str] | None = None,
) -> dict[str, int]:
    """Build the stratified queue and write it to `out_path`. Returns per-task-class counts.

    Cases that already carry a HUMAN_LOCAL annotation are excluded up front, so calling this
    again after a partial labelling session naturally produces the remaining work rather than
    re-asking for cases someone already labelled.

    `tasks` restricts the queue to specific task classes. Human attention is the scarcest input
    in this pipeline, so it is worth spending where no human label exists yet: `summarize`
    already carries 520 published expert annotations, while the crisp-correctness families carry
    none. Restricting the queue also keeps the hand-labelled track's scope aligned with whatever
    claim it is meant to support -- and that scope has to be stated wherever the number is
    quoted, because a kappa over one task mix is not a kappa over another.
    """
    split_cases = select(cases, split_plan, split)  # type: ignore[arg-type]
    if tasks:
        wanted = {canonical(x) for x in tasks}
        split_cases = [c for c in split_cases if c.task_class in wanted]
    already_local = {a.case_id for a in annotations if a.provenance is LabelProvenance.HUMAN_LOCAL}
    eligible = [c for c in split_cases if c.case_id not in already_local]
    if not eligible:
        _write_queue([], out_path)
        return {}

    # Read only to stratify -- `resolved` is never written to the queue.
    resolved = Corpus(cases=list(cases), annotations=list(annotations)).resolved_labels()
    chosen = _select_stratified(eligible, resolved, n, seed)
    rows = [_queue_row(c) for c in chosen]
    random.Random(f"{seed}:order").shuffle(rows)
    _write_queue(rows, out_path)
    return dict(sorted(Counter(row["task_class"] for row in rows).items()))


def cmd_export(args: argparse.Namespace) -> int:
    corpus = Corpus.load(DATA)
    sp = SplitPlan.load(DATA / "splits.json")
    verify(corpus.cases, sp)
    out_path = Path(args.out)
    counts = export_queue(
        corpus.cases, corpus.annotations, sp,
        split=args.split, n=args.n, seed=args.seed, out_path=out_path,
        tasks=(None if args.tasks.strip().lower() in ('', 'all')
               else [x.strip() for x in args.tasks.split(',')]),
    )
    total = sum(counts.values())
    if total < args.n:
        print(f"note: requested n={args.n} but only {total} eligible case(s) remained; "
              f"exported {total}.")
    print(f"wrote {total} queue items to {out_path}")
    for task in sorted(counts):
        print(f"  {task:12s} {counts[task]}")
    return 0


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------
def get_keypress() -> str:
    """Read one keypress without requiring Enter, where the platform supports it.

    Falls back to line-buffered `input()` everywhere else, so the loop still runs -- and a
    test can still drive it by supplying an explicit `get_key` callable -- on platforms
    without `msvcrt`.
    """
    if msvcrt is not None:
        raw = msvcrt.getch()
        try:
            return raw.decode("utf-8", errors="ignore").lower()
        except (UnicodeDecodeError, AttributeError):
            return ""
    try:
        line = input().strip().lower()
    except EOFError:
        return "q"
    return line[0] if line else ""


def _prompt_reason() -> str:
    try:
        return input("reason (optional, Enter to skip): ").strip()
    except EOFError:
        return ""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clear_screen() -> None:
    if not sys.stdout.isatty():
        return
    # A failure here (e.g. no controlling terminal) should never take down a labelling
    # session; the worst outcome is an un-cleared screen, not a lost verdict.
    with contextlib.suppress(OSError):
        os.system("cls" if os.name == "nt" else "clear")


def _truncated(text: str, expanded: bool) -> str:
    if expanded or len(text) <= TRUNCATE_CHARS:
        return text
    hidden = len(text) - TRUNCATE_CHARS
    return text[:TRUNCATE_CHARS] + f"\n... [{hidden} more characters -- press 'm' to expand]"


def _render_item(
    item: dict[str, Any], *, index: int, total: int, counts: dict[str, int],
    expanded: bool, elapsed: float, rubric: Rubric,
) -> None:
    _clear_screen()
    done = counts["pass"] + counts["fail"] + counts["skip"]
    rate = elapsed / done if done else 0.0
    print(f"item {index + 1}/{total}   task_class={item['task_class']}")
    print(f"pass={counts['pass']}  fail={counts['fail']}  skip={counts['skip']}   "
          f"elapsed={elapsed:.0f}s  ({rate:.1f}s/item)")
    print("=" * 78)
    print(rubric.as_prompt_block())
    print("=" * 78)
    print("TASK INPUT:")
    print(item["task_input"])
    if item.get("context"):
        print("-" * 78)
        print("CONTEXT:")
        print(_truncated(item["context"], expanded))
    if item.get("reference"):
        print("-" * 78)
        print("REFERENCE:")
        print(_truncated(item["reference"], expanded))
    print("=" * 78)
    print(">>> CANDIDATE OUTPUT <<<")
    print(item["candidate_output"])
    print(">>> END CANDIDATE OUTPUT <<<")
    print("=" * 78)
    toggle = "show less" if expanded else "show more"
    print(f"[p] pass  [f] fail  [s] skip  [m] {toggle}  [u] undo  [q] save & quit")


def _annotator_case_ids(out_path: Path, annotator_id: str) -> set[str]:
    """Cases this annotator already has a HUMAN_LOCAL verdict for, so resuming after a crash
    (same --annotator, same --out) does not re-ask about already-answered items.
    """
    if not out_path.exists():
        return set()
    return {
        a.case_id
        for a in read_jsonl(out_path, Annotation)
        if a.annotator_id == annotator_id and a.provenance is LabelProvenance.HUMAN_LOCAL
    }


def _undo_annotation(path: Path, case_id: str, annotator_id: str, timestamp: str) -> None:
    """Remove exactly the annotation line just appended, identified by case/annotator/timestamp
    rather than by "last line", so a concurrent writer's line is never at risk.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    for i in range(len(lines) - 1, -1, -1):
        row = json.loads(lines[i])
        if (row.get("case_id"), row.get("annotator_id"), row.get("timestamp")) == (
            case_id, annotator_id, timestamp,
        ):
            del lines[i]
            break
    else:
        raise AnnotateError(
            f"undo could not find the annotation it expected to remove for case {case_id!r} "
            f"(annotator {annotator_id!r}, timestamp {timestamp!r}); refusing to guess which "
            "line to delete"
        )
    text = "\n".join(lines)
    path.write_text(text + ("\n" if lines else ""), encoding="utf-8")


def _check_rubric_freshness(queue_rows: list[dict[str, Any]]) -> None:
    """Refuse to run if the rubric changed since this queue was exported.

    Labelling against a stale rubric text would show the human a different pass definition
    than the judge is graded against, silently reproducing the exact mismatch this module
    exists to prevent.
    """
    mismatches: list[str] = []
    for row in queue_rows:
        current = for_task(row["task_class"])
        if (row.get("rubric_id"), row.get("rubric_version")) != (current.id, current.version):
            mismatches.append(
                f"{row['case_id']}: queue has {row.get('rubric_id')}@{row.get('rubric_version')}"
                f", current rubric is {current.id}@{current.version}"
            )
    if mismatches:
        raise AnnotateError(
            "the rubric changed since this queue was exported; the annotator would be shown "
            f"different criteria than the judge is graded against. Re-run `export`. "
            f"first mismatches: {mismatches[:5]}"
        )


def run_labeling_session(
    queue_rows: list[dict[str, Any]],
    *,
    annotator_id: str,
    out_path: Path,
    limit: int | None = None,
    get_key: Callable[[], str] | None = None,
    get_reason: Callable[[], str] | None = None,
    now: Callable[[], str] | None = None,
) -> dict[str, int]:
    """Drive one labelling session.

    Every verdict is appended to `out_path` before the next item is shown (never buffered), so
    a crash mid-session loses at most whatever is on screen right now, never a prior item.
    `get_key`/`get_reason`/`now` are looked up as bare names when not supplied, so
    `monkeypatch.setattr(annotate, "get_keypress", ...)` also works without passing anything.
    """
    _check_rubric_freshness(queue_rows)
    key_fn = get_key if get_key is not None else get_keypress
    reason_fn = get_reason if get_reason is not None else _prompt_reason
    now_fn = now if now is not None else _utcnow_iso

    already = _annotator_case_ids(out_path, annotator_id)
    worklist = [row for row in queue_rows if row["case_id"] not in already]
    if limit is not None:
        worklist = worklist[:limit]

    counts = {"pass": 0, "fail": 0, "skip": 0}
    history: list[dict[str, Any]] = []
    rubric_cache: dict[str, Rubric] = {}
    start = time.time()
    index = 0
    last_index = -1
    expanded = False

    while index < len(worklist):
        if index != last_index:
            expanded = False
            last_index = index
        item = worklist[index]
        rubric = rubric_cache.setdefault(item["task_class"], for_task(item["task_class"]))
        _render_item(item, index=index, total=len(worklist), counts=counts,
                     expanded=expanded, elapsed=time.time() - start, rubric=rubric)

        key = (key_fn() or "").strip().lower()[:1]

        if key == "q":
            break
        if key == "m":
            expanded = not expanded
            continue
        if key == "u":
            if not history:
                print("nothing to undo yet.")
                continue
            last = history.pop()
            if last["action"] in ("pass", "fail"):
                _undo_annotation(out_path, last["case_id"], annotator_id, last["timestamp"])
            counts[last["action"]] -= 1
            index -= 1
            continue
        if key == "s":
            history.append({"case_id": item["case_id"], "action": "skip", "timestamp": None})
            counts["skip"] += 1
            index += 1
            continue
        if key in ("p", "f"):
            reason = (reason_fn() or "").strip()
            timestamp = now_fn()
            record = Annotation(
                case_id=item["case_id"],
                annotator_id=annotator_id,
                label=1 if key == "p" else 0,
                provenance=LabelProvenance.HUMAN_LOCAL,
                rubric_version=f"{item['rubric_id']}@{item['rubric_version']}",
                reason=reason,
                timestamp=timestamp,
            )
            append_jsonl(out_path, [record])
            action = "pass" if key == "p" else "fail"
            counts[action] += 1
            history.append({"case_id": item["case_id"], "action": action, "timestamp": timestamp})
            index += 1
            continue
        print(f"unrecognised key {key!r}; use p/f/s/m/u/q.")

    return counts


def cmd_label(args: argparse.Namespace) -> int:
    queue_path = Path(args.queue)
    rows = read_queue(queue_path)
    if not rows:
        print(f"queue {queue_path} is empty or missing; run `export` first.", file=sys.stderr)
        return 1
    out_path = Path(args.out)
    try:
        counts = run_labeling_session(
            rows, annotator_id=args.annotator, out_path=out_path, limit=args.limit,
        )
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted; every verdict entered so far is already saved.")
        return 0
    print(f"\nsaved. pass={counts['pass']} fail={counts['fail']} skip={counts['skip']}")
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def _best_annotator_pair(local: list[Annotation]) -> tuple[str, str, list[str]] | None:
    """The pair of annotators with the largest case_id overlap, or None if fewer than two
    annotators have any HUMAN_LOCAL labels. Deterministic: annotator ids are sorted first, so
    ties are always resolved the same way regardless of file order.
    """
    per_annotator: dict[str, set[str]] = {}
    for a in local:
        per_annotator.setdefault(a.annotator_id, set()).add(a.case_id)
    annotators = sorted(per_annotator)
    if len(annotators) < 2:
        return None
    best: tuple[str, str, list[str]] | None = None
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a1, a2 = annotators[i], annotators[j]
            shared = sorted(per_annotator[a1] & per_annotator[a2])
            if shared and (best is None or len(shared) > len(best[2])):
                best = (a1, a2, shared)
    return best


def _inter_annotator(local: list[Annotation], by_id: dict[str, CaseRecord]) -> dict[str, Any] | None:
    pair = _best_annotator_pair(local)
    if pair is None:
        return None
    a1, a2, shared = pair
    per_annotator: dict[str, dict[str, int]] = {}
    for a in local:
        per_annotator.setdefault(a.annotator_id, {})[a.case_id] = a.label
    human = [per_annotator[a1][cid] for cid in shared]
    judge = [per_annotator[a2][cid] for cid in shared]
    groups = [by_id[cid].group_id for cid in shared]
    tasks = [by_id[cid].task_class for cid in shared]
    result = agreement(human, judge, groups=groups, tasks=tasks, bootstrap=0)
    return {
        "annotator_a": a1,
        "annotator_b": a2,
        "n_shared_cases": len(shared),
        "kappa": result.kappa,
        "defined": result.defined,
        "observed_agreement": result.observed_agreement,
        "reason": result.reason,
    }


def compute_status(corpus: Corpus, queue_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = corpus.by_id()
    local = [a for a in corpus.annotations if a.provenance is LabelProvenance.HUMAN_LOCAL]
    labelled_case_ids = sorted({a.case_id for a in local})
    by_task = Counter(by_id[cid].task_class for cid in labelled_case_ids)
    by_label_value = Counter(a.label for a in local)
    remaining = [row for row in queue_rows if row["case_id"] not in set(labelled_case_ids)]

    return {
        "n_cases_with_human_local": len(labelled_case_ids),
        "by_task_class": dict(sorted(by_task.items())),
        "by_label_value": {str(k): v for k, v in sorted(by_label_value.items())},
        "n_queue_total": len(queue_rows),
        "n_queue_remaining": len(remaining),
        "inter_annotator": _inter_annotator(local, by_id),
    }


def _print_status(report: dict[str, Any]) -> None:
    print(f"HUMAN_LOCAL labels: {report['n_cases_with_human_local']} cases")
    print("  by task class:")
    for task, count in report["by_task_class"].items():
        print(f"    {task:12s} {count}")
    print("  by label value:")
    for label, count in report["by_label_value"].items():
        print(f"    {label:>5s} {count}")
    print(f"queue: {report['n_queue_remaining']}/{report['n_queue_total']} items remaining")

    print()
    print("INTER-ANNOTATOR DIAGNOSTIC -- this measures how well two people agreed with each")
    print("other. It is NOT a ceiling on, or a substitute for, judge agreement against an")
    print("adjudicated reference label.")
    pair = report["inter_annotator"]
    if pair is None:
        print("  fewer than two annotators have overlapping HUMAN_LOCAL labels yet.")
        return
    print(f"  {pair['annotator_a']} vs {pair['annotator_b']}   n={pair['n_shared_cases']}")
    print(f"  raw agreement: {pair['observed_agreement']:.3f}")
    if pair["defined"]:
        print(f"  cohen's kappa: {pair['kappa']:.4f}")
    else:
        print(f"  cohen's kappa: undefined ({pair['reason']})")


def cmd_status(args: argparse.Namespace) -> int:
    corpus = Corpus.load(DATA)
    queue_rows = read_queue(Path(args.queue))
    _print_status(compute_status(corpus, queue_rows))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="build a stratified annotation queue")
    p_export.add_argument("--n", type=int, default=100)
    p_export.add_argument("--split", default="test")
    p_export.add_argument("--seed", type=int, default=20260927)
    p_export.add_argument("--out", default=str(DATA / "annotation_queue.jsonl"))
    p_export.add_argument(
        "--tasks", default="code,sql,reason,tool_use",
        help="comma-separated task classes to draw from. Defaults to the crisp-correctness "
             "families, which have no human labels yet; pass "
             "code,sql,reason,tool_use,summarize for a representative mix, or 'all'.")
    p_export.set_defaults(fn=cmd_export)

    p_label = sub.add_parser("label", help="interactive terminal labelling loop")
    p_label.add_argument("--queue", default=str(DATA / "annotation_queue.jsonl"))
    p_label.add_argument("--annotator", required=True, help="the annotator's id string")
    p_label.add_argument("--out", default=str(DATA / "annotations.jsonl"))
    p_label.add_argument("--limit", type=int, default=None)
    p_label.set_defaults(fn=cmd_label)

    p_status = sub.add_parser("status", help="labelling progress and inter-annotator agreement")
    p_status.add_argument("--queue", default=str(DATA / "annotation_queue.jsonl"))
    p_status.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except AnnotateError as e:
        print(f"annotate error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
