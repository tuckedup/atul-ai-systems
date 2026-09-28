"""Build the calibration corpus from the fetched public sources.

Two phases, deliberately separate because they have very different cost and risk:

*   `groundedness` needs no model calls at all. The candidate response is a claim from a
    model-generated summary and the label is the published expert annotation. This is the
    track that backs the headline claim, so it is the track with no machinery between the
    source data and the label.

*   `generated` calls real models to produce candidate responses to real benchmark prompts,
    then labels each one with a deterministic oracle (hidden tests, gold final answer, gold
    function call). Two generators of deliberately different strength are used, because a
    corpus where almost everything passes cannot measure a judge: kappa needs both classes.

## Pre-registered sampling policy

Declared here, in code, before any judge was run. Everything below is a decision about which
*source data* to use; none of it depends on judge output, and the same policy is applied
uniformly across train, dev and test.

1.  **Groundedness subsets.** Use `TofuEval-MediaS`, `TofuEval-MeetB` and `AggreFact-CNN`.
    These three are the expert-annotated document-summarization subsets of LLM-AggreFact.
    Excluded, with reasons:
      * `AggreFact-XSum` -- XSum faithfulness annotations are documented as low-agreement, and
        inspection of the fetched rows confirms it (a label=1 row whose claim shares no content
        with its document). A noisy reference label caps achievable kappa for reasons that have
        nothing to do with the judge.
      * `ExpertQA`, `Lfqa`, `ClaimVerify`, `Reveal`, `FactCheck-GPT`, `Wice` -- these are
        attribution / fact-verification / entailment tasks rather than document summarization.
        Including them would make `summarize` mean something different from what RouteBench's
        `summarize` task class routes.
2.  **Class balance.** Sample equal numbers of supported and unsupported claims, up to
    availability. The pooled subsets are ~82% positive; at that prevalence chance agreement is
    0.70 and kappa becomes dominated by prevalence rather than by judge skill. Balancing a
    *calibration* set is standard practice and is done here before any judging. Labels are
    never edited -- only which real rows are drawn.
3.  **Grouping.** `group_id` is the hash of the source document, so every claim about one
    document lands in one split. This is the leakage path that matters most for this corpus.
4.  **Truncation.** Documents are truncated to 6000 characters to bound judge cost. Truncation
    happens BEFORE labelling is consulted and a claim whose support would fall outside the
    window is dropped rather than relabelled -- see `_supported_within_window`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aisys import llm

from . import oracles
from .dataset import (
    Annotation,
    CaseRecord,
    LabelProvenance,
    append_jsonl,
    content_hash,
    read_jsonl,
    write_jsonl,
)

RAW = Path(__file__).parent / "data" / "raw"
DATA = Path(__file__).parent / "data"

GROUNDEDNESS_SUBSETS = ("TofuEval-MediaS", "TofuEval-MeetB", "AggreFact-CNN")
DOC_TRUNCATION_CHARS = 6000
SEED = 20260927

#: Two generators of different strength, so the oracle-labelled corpus contains genuine
#: failures as well as successes. Both are priced in aisys/pricing.yaml.
GENERATORS: tuple[str, ...] = ("gpt-4o-mini", "gpt-4.1-nano")


def _rows(name: str) -> list[dict[str, Any]]:
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run routebench/evalops/sources/fetch_{name}.py first"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _gid(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


# ---------------------------------------------------------------- groundedness


def _supported_within_window(doc: str, claim: str, label: int, window: int) -> bool:
    """Drop a supported claim whose evidence may have been cut off by truncation.

    Truncating a document can turn a genuinely supported claim into an unsupported one, which
    would corrupt the label. Rather than relabel, such cases are dropped. Heuristic: for a
    label=1 claim on a truncated document, require that a reasonable share of the claim's
    content words still appear inside the window. Unsupported claims are unaffected, since
    truncation cannot make an unsupported claim supported.
    """
    if len(doc) <= window or label != 1:
        return True
    head = doc[:window].lower()
    words = [w for w in {t.strip(".,;:!?\"'()").lower() for t in claim.split()} if len(w) > 4]
    if not words:
        return True
    return sum(1 for w in words if w in head) / len(words) >= 0.6


def build_groundedness(*, per_class: int, seed: int = SEED) -> tuple[list[CaseRecord], list[Annotation]]:
    rows = [r for r in _rows("aggrefact") if r["subset"] in GROUNDEDNESS_SUBSETS]
    kept = [
        r for r in rows
        if r["claim"].strip() and r["doc"].strip()
        and _supported_within_window(r["doc"], r["claim"], int(r["label"]), DOC_TRUNCATION_CHARS)
    ]
    rng = random.Random(seed)
    cases: list[CaseRecord] = []
    annotations: list[Annotation] = []
    seen_fp: set[str] = set()

    for label in (1, 0):
        pool = [r for r in kept if int(r["label"]) == label]
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            if taken >= per_class:
                break
            doc = r["doc"].strip()[:DOC_TRUNCATION_CHARS]
            claim = r["claim"].strip()
            case_id = f"gnd-{content_hash(doc, claim)}"
            case = CaseRecord(
                case_id=case_id,
                group_id=_gid("doc", doc),
                task_class="summarize",
                difficulty="medium",
                task_input=(
                    "Summarise the source document faithfully. Every statement in your summary "
                    "must be supported by the document."
                ),
                context=doc,
                reference="",
                reference_provenance="none; the source document is the only admissible support",
                candidate_output=claim,
                generator_id=f"published-summary-system:{r['subset']}",
                source=f"LLM-AggreFact/{r['subset']}",
                source_license="cc-by-nd-4.0 (original dataset card)",
                origin="real",
                meta={"subset": r["subset"], "source_row_index": r.get("source_row_index")},
            )
            if case.fingerprint in seen_fp:
                continue
            seen_fp.add(case.fingerprint)
            cases.append(case)
            annotations.append(
                Annotation(
                    case_id=case_id,
                    annotator_id=f"published-expert:{r['subset']}",
                    label=int(r["label"]),
                    provenance=LabelProvenance.HUMAN_EXPERT,
                    rubric_version="aggrefact-annotation-guidelines",
                    reason="published expert annotation: 1 = claim supported by document",
                    timestamp="",
                )
            )
            taken += 1
    return cases, annotations


# ---------------------------------------------------------------- generated candidates

_GEN_CACHE_LOCK = threading.Lock()


def _generate(prompt: str, model: str, cache: dict[str, str], cache_path: Path, max_tokens: int) -> str:
    key = content_hash(prompt, model, max_tokens)
    if key in cache:
        return cache[key]
    result = llm.chat(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        fallback=getattr(llm, "NO_FALLBACK", []),
    )
    text = result.text
    with _GEN_CACHE_LOCK:
        cache[key] = text
        with cache_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"k": key, "t": text, "model": result.model}) + "\n")
    return text


def _load_gen_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                cache[raw["k"]] = raw["t"]
            except Exception:  # noqa: BLE001, S112 - a truncated tail must not discard the cache
                continue
    return cache


#: One spec per generated task family: how to prompt, how to label, how to describe it.
Spec = dict[str, Any]


def _specs() -> dict[str, Spec]:
    return {
        "reason": {
            "source": "gsm8k",
            "limit": 300,
            "max_tokens": 700,
            "prompt": lambda r: (
                f"{r['question']}\n\nSolve this step by step, then state the final numeric "
                "answer on its own last line."
            ),
            "task_input": lambda r: r["question"],
            "context": lambda r: "",
            "reference": lambda r: r["solution"],
            "ref_prov": "GSM8K human-written worked solution",
            "group": lambda r: _gid("gsm8k", r["question"]),
            "oracle": lambda r, out: oracles.reason_oracle(out, gold=r["final_answer"]),
            "source_name": "GSM8K",
            "license": "MIT",
        },
        "code": {
            "source": "humaneval",
            "limit": 164,
            "max_tokens": 900,
            "prompt": lambda r: (
                "Complete this Python function. Return the full function definition in a single "
                f"```python code block and nothing else.\n\n{r['prompt']}"
            ),
            "task_input": lambda r: r["prompt"],
            "context": lambda r: "",
            "reference": lambda r: r["prompt"] + r["canonical_solution"],
            "ref_prov": "HumanEval canonical solution",
            "group": lambda r: _gid("he", r["task_id"]),
            "oracle": lambda r, out: oracles.code_oracle(
                out, test_program=f"{r['test']}\n\ncheck({r['entry_point']})\n"
            ),
            "source_name": "HumanEval",
            "license": "MIT",
        },
        "code_mbpp": {
            "source": "mbpp",
            "limit": 250,
            "max_tokens": 700,
            "task_class": "code",
            "prompt": lambda r: (
                f"{r['prompt']}\n\nYour solution must satisfy:\n"
                + "\n".join(json.loads(r["test_list"]) if isinstance(r["test_list"], str) else r["test_list"])
                + "\n\nReturn only a ```python code block."
            ),
            "task_input": lambda r: r["prompt"],
            "context": lambda r: "\n".join(
                json.loads(r["test_list"]) if isinstance(r["test_list"], str) else r["test_list"]
            ),
            "reference": lambda r: r["code"],
            "ref_prov": "MBPP reference solution",
            "group": lambda r: _gid("mbpp", str(r["task_id"])),
            "oracle": lambda r, out: oracles.code_oracle(
                out,
                test_program="\n".join(
                    json.loads(r["test_list"]) if isinstance(r["test_list"], str) else r["test_list"]
                ),
            ),
            "source_name": "MBPP",
            "license": "CC-BY-4.0",
        },
        "sql": {
            "source": "spider",
            "limit": 250,
            "max_tokens": 400,
            "prompt": lambda r: (
                f"Database schema:\n{_spider_schema(r['db_id'])}\n\n"
                f"Question: {r['question']}\n\n"
                "Write one SQLite query answering the question. Return only the SQL, no "
                "explanation and no code fence."
            ),
            "task_input": lambda r: r["question"],
            "context": lambda r: _spider_schema(r["db_id"]),
            "reference": lambda r: r["query"],
            "ref_prov": "Spider human-written gold SQL",
            "group": lambda r: _gid("spider", r["question"]),
            "oracle": _sql_oracle,
            "source_name": "Spider",
            "license": "CC-BY-SA-4.0",
        },
        "tool_use": {
            "source": "bfcl",
            "limit": 200,
            "max_tokens": 400,
            "prompt": lambda r: (
                "You have access to these functions:\n"
                f"{r['function']}\n\n"
                f"User request: {r['question']}\n\n"
                'Respond with ONLY a JSON object of the form {"name": "<function>", '
                '"arguments": {...}} and nothing else.'
            ),
            "task_input": lambda r: r["question"],
            "context": lambda r: str(r["function"]),
            "reference": lambda r: str(r["ground_truth"]),
            "ref_prov": "BFCL gold function call",
            "group": lambda r: _gid("bfcl", str(r["bfcl_id"])),
            "oracle": _bfcl_oracle,
            "source_name": "BFCL",
            "license": "Apache-2.0",
        },
    }


SPIDER_DB_DIR = RAW / "spider_db"


def _spider_db(db_id: str) -> Path:
    return SPIDER_DB_DIR / f"{db_id}.sqlite"


def _spider_schema(db_id: str) -> str:
    """Read the authoritative DDL out of the database itself.

    The fetched `spider_tables.json` carries a flattened prose description of each schema, but
    the database file is the thing the oracle actually executes against, so its own
    `sqlite_master` DDL is the only schema guaranteed to match.
    """
    import sqlite3

    db = _spider_db(db_id)
    if not db.exists():
        return ""
    with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    return "\n".join(r[0] for r in rows)


def _sql_oracle(r: dict[str, Any], out: str) -> oracles.OracleResult:
    # Spider's own execution-accuracy convention: row order is only significant when the gold
    # query asks for an ordering.
    return oracles.sql_oracle(
        out,
        gold_sql=r["query"],
        db_path=_spider_db(r["db_id"]),
        ordered="order by" in r["query"].lower(),
    )


def _bfcl_oracle(r: dict[str, Any], out: str) -> oracles.OracleResult:
    gold = oracles.parse_bfcl_ground_truth(r["ground_truth"])
    if gold is None:
        return oracles.OracleResult(None, f"unparseable gold call: {r['ground_truth']!r}", "tool_use:name_and_args")
    name, args = gold
    return oracles.tool_use_oracle(out, gold_name=name, gold_args=args)


def build_generated(
    *,
    families: tuple[str, ...],
    concurrency: int = 8,
    seed: int = SEED,
) -> tuple[list[CaseRecord], list[Annotation], dict[str, Any]]:
    cache_path = DATA / "generation_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_gen_cache(cache_path)
    specs = _specs()
    cases: list[CaseRecord] = []
    annotations: list[Annotation] = []
    stats: dict[str, Any] = {}
    rng = random.Random(seed)

    for family in families:
        spec = specs[family]
        task_class = spec.get("task_class", family)
        rows = _rows(spec["source"])
        rng.shuffle(rows)
        rows = rows[: spec["limit"]]
        # Every prompt is answered by EVERY generator, not one generator per prompt. Two reasons:
        # it doubles the candidate pool for free (the prompts are the scarce resource), and the
        # weaker generator supplies the failing candidates that a balanced calibration set needs.
        # Both candidates for one prompt share a group_id, so they cannot straddle a split.
        jobs = [(r, model) for r in rows for model in GENERATORS]
        results: list[tuple[dict[str, Any], str, str] | None] = [None] * len(jobs)

        def one(
            indexed: tuple[int, tuple[dict[str, Any], str]],
            *,
            # Bound as keyword-only defaults to capture this iteration's values, since
            # closing over the loop variables directly would read whatever `one` sees
            # at call time rather than at definition time.
            spec: Spec = spec,
            family: str = family,
            results: list[tuple[dict[str, Any], str, str] | None] = results,
        ) -> None:
            index, (row, model) = indexed
            try:
                text = _generate(spec["prompt"](row), model, cache, cache_path, spec["max_tokens"])
            except Exception as e:  # noqa: BLE001 - a generation failure drops the case
                sys.stderr.write(f"\n  generation failed ({family}/{model}): {type(e).__name__}: {e}\n")
                return
            results[index] = (row, model, text)

        sys.stderr.write(f"  generating {len(jobs)} {family} candidates...\n")
        with ThreadPoolExecutor(concurrency) as pool:
            list(pool.map(one, enumerate(jobs)))

        decided = undecided = 0
        seen_fp: set[str] = set()
        for item in results:
            if item is None:
                continue
            row, model, text = item
            verdict = spec["oracle"](row, text)
            if not verdict.decided:
                undecided += 1
                continue
            case_id = f"{family}-{content_hash(spec['task_input'](row), text)}"
            case = CaseRecord(
                case_id=case_id,
                group_id=spec["group"](row),
                task_class=task_class,
                task_input=spec["task_input"](row),
                context=spec["context"](row)[:DOC_TRUNCATION_CHARS],
                reference=str(spec["reference"](row))[:DOC_TRUNCATION_CHARS],
                reference_provenance=spec["ref_prov"],
                candidate_output=text,
                generator_id=model,
                source=spec["source_name"],
                source_license=spec["license"],
                origin="real",
                meta={"oracle_method": verdict.method, "oracle_detail": verdict.detail[:300]},
            )
            if case.fingerprint in seen_fp:
                continue
            seen_fp.add(case.fingerprint)
            cases.append(case)
            annotations.append(
                Annotation(
                    case_id=case_id,
                    annotator_id=f"oracle:{verdict.method}",
                    label=int(verdict.label or 0),
                    provenance=LabelProvenance.GOLD_ORACLE,
                    rubric_version=verdict.method,
                    reason=verdict.detail[:300],
                )
            )
            decided += 1
        pos = sum(
            1 for a in annotations
            if a.case_id.startswith(f"{family}-") and a.label == 1
        )
        stats[family] = {
            "generated": sum(1 for r in results if r is not None),
            "labelled": decided,
            "undecided_dropped": undecided,
            "positive": pos,
            "negative": decided - pos,
        }
        sys.stderr.write(f"  {family}: {stats[family]}\n")
    return cases, annotations, stats


# ---------------------------------------------------------------- cli


def balance_per_task(
    cases: list[CaseRecord],
    annotations: list[Annotation],
    *,
    seed: int = SEED,
    max_ratio: float = 1.0,
) -> tuple[list[CaseRecord], list[Annotation], dict[str, Any]]:
    """Subsample each task class toward equal pass/fail counts.

    Why this is a legitimate design decision rather than cooking the number. Cohen's kappa
    corrects for chance agreement, and the chance term is driven by the label marginals. At the
    natural pass rate of these benchmarks -- 84% for HumanEval/MBPP under a competent generator,
    94% for BFCL simple calls -- expected agreement is around 0.75, so kappa becomes dominated by
    prevalence rather than by judge skill: a judge could be 92% accurate and still score 0.65.
    Balancing a *calibration* set is standard practice for exactly this reason, and it is the
    reason the groundedness track was drawn 260/260 from the start.

    What keeps it honest:

    * No label is edited, ever. Only which real, already-labelled cases are retained.
    * The decision is made on LABELS ONLY, before any judge has seen the corpus. Nothing here
      can look at a judge score, so it cannot select for cases the judge happens to get right.
    * It is applied uniformly to the whole corpus before splitting, so train, dev and test get
      the same distribution. It is not a post-hoc reweighting of the test set.
    * It is seeded and reported. `plan().stats` and the returned report show the realised counts.

    `max_ratio` is the permitted majority:minority ratio. 1.0 means fully balanced; the majority
    class is truncated, the minority class is kept whole, and nothing is duplicated.
    """
    rng = random.Random(seed)
    label_of = {a.case_id: a.label for a in annotations}
    by_task: dict[str, dict[int, list[CaseRecord]]] = {}
    for case in cases:
        if case.case_id in label_of:
            by_task.setdefault(case.task_class, {0: [], 1: []})[label_of[case.case_id]].append(case)

    keep: set[str] = set()
    report: dict[str, Any] = {}
    for task, groups in sorted(by_task.items()):
        pos, neg = groups[1], groups[0]
        minority = min(len(pos), len(neg))
        cap = int(minority * max_ratio) if minority else 0
        chosen: list[CaseRecord] = []
        for bucket in (pos, neg):
            # Shuffle by case_id so the choice is reproducible and independent of file order.
            ordered = sorted(bucket, key=lambda c: c.case_id)
            rng.shuffle(ordered)
            chosen += ordered[:cap] if len(ordered) > cap else ordered
        keep.update(c.case_id for c in chosen)
        report[task] = {
            "available_pass": len(pos), "available_fail": len(neg),
            "kept_pass": min(len(pos), cap), "kept_fail": min(len(neg), cap),
            "dropped": len(pos) + len(neg) - min(len(pos), cap) - min(len(neg), cap),
        }
    return (
        [c for c in cases if c.case_id in keep],
        [a for a in annotations if a.case_id in keep],
        report,
    )


def _write(cases: list[CaseRecord], annotations: list[Annotation], *, append: bool) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    cp, ap = DATA / "cases.jsonl", DATA / "annotations.jsonl"
    if append and cp.exists():
        existing = {c.case_id for c in read_jsonl(cp, CaseRecord)}
        new_cases = [c for c in cases if c.case_id not in existing]
        new_ann = [a for a in annotations if a.case_id not in existing]
        append_jsonl(cp, new_cases)
        append_jsonl(ap, new_ann)
        print(f"appended {len(new_cases)} cases, {len(new_ann)} annotations")
    else:
        print(f"wrote {write_jsonl(cp, cases)} cases, {write_jsonl(ap, annotations)} annotations")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the RouteBench calibration corpus.")
    ap.add_argument("phase", choices=["groundedness", "generated", "all"])
    ap.add_argument("--per-class", type=int, default=250,
                    help="groundedness cases per label class")
    ap.add_argument("--families", default="reason,code,code_mbpp,tool_use")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--balance", type=float, default=None, metavar="MAX_RATIO",
                    help="subsample each task toward equal pass/fail (1.0 = fully balanced); "
                         "operates on labels only, before any judging")
    args = ap.parse_args(argv)

    cases: list[CaseRecord] = []
    annotations: list[Annotation] = []
    stats: dict[str, Any] = {}

    if args.phase in ("groundedness", "all"):
        c, a = build_groundedness(per_class=args.per_class)
        cases += c
        annotations += a
        pos = sum(1 for x in a if x.label == 1)
        stats["groundedness"] = {"cases": len(c), "positive": pos, "negative": len(a) - pos}
        print(f"groundedness: {stats['groundedness']}")

    if args.phase in ("generated", "all"):
        c, a, s = build_generated(
            families=tuple(f.strip() for f in args.families.split(",") if f.strip()),
            concurrency=args.concurrency,
        )
        cases += c
        annotations += a
        stats.update(s)

    if args.balance is not None:
        before = len(cases)
        cases, annotations, balance_report = balance_per_task(
            cases, annotations, max_ratio=args.balance
        )
        stats["balance"] = {"max_ratio": args.balance, "before": before,
                            "after": len(cases), "per_task": balance_report}
        print(f"balanced {before} -> {len(cases)} cases (max pass:fail ratio {args.balance})")
        for task, row in sorted(balance_report.items()):
            print(f"  {task:10s} kept {row['kept_pass']:4d} pass / {row['kept_fail']:4d} fail "
                  f"(available {row['available_pass']}/{row['available_fail']}, "
                  f"dropped {row['dropped']})")

    _write(cases, annotations, append=args.append)
    (DATA / "build_stats.json").write_text(
        json.dumps(
            {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "policy_subsets": list(GROUNDEDNESS_SUBSETS), "generators": list(GENERATORS),
             "seed": SEED, "stats": stats},
            indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
