"""Local MiniCheck feature job for judge-v19 Phase 1 (GPU INFERENCE ONLY, needs its own approval; nothing here has been run).

Stand-alone on purpose: it runs in the MiniCheck virtualenv (`.local/mc-venv`: upstream `minicheck`, torch with CUDA; no scikit-learn), so it imports only `checkers`, `dataset` and the
standard library. It reads the frozen protocol and its input files, verifies their hashes, and never opens a label file or a result file.

Order of work (fail-closed):
  1. protocol and input hashes must match the freeze;
  2. EQUIVALENCE CHECK: recompute the cached RouteBench features for EQUIVALENCE_ROWS original cases (hash-ordered) and compare with the cached values - the generator of the cached scores is
     not in the repository, so the match is demonstrated, not assumed. Any sentence-split difference or an absolute difference above EQUIVALENCE_TOL stops the job BEFORE any RAGTruth row is scored;
  3. only then score the 1,050 RAGTruth fitting sentences (whole-claim probability and one probability per candidate sentence).
No download: the Hugging Face offline switches are forced on, so absent weights stop the job instead of fetching them. Weights terms: the MIT model card is recorded in
docs/ROUTEBENCH_TRAINING_PERMISSION_STATUS.md (downstream ANLI/synthetic-data terms unresolved; private research pilot only); setting `weights_licence_checked` here is the operator's step, not a clearance.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EQUIVALENCE_ROWS, EQUIVALENCE_TOL = 24, 2e-3
MINICHECK_CONFIRM = "RUN-JV19-MINICHECK-GPU"
CHECKER_ID = "minicheck-flan-t5-large-upstream"
OUTPUT = "minicheck_ragtruth.json"


class MiniCheckJobError(ValueError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def minicheck_pair(checker: Any, source: str, candidate: str, splitter: Any = None) -> tuple[float, list[float]]:
    """Whole-claim probability and one probability per candidate sentence (the cached RouteBench `full` / `sent` fields)."""
    from .checkers import split_claim_sentences

    full = float(checker.score([source], [candidate])[0])
    sents = split_claim_sentences(candidate, splitter=splitter)
    return full, ([float(checker.score([source], [s])[0]) for s in sents] if sents else [full])


def _verified_inputs(work: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    work = Path(work)
    if _sha(work / "protocol.json") != expected_sha256:
        raise MiniCheckJobError("protocol hash differs from the hash recorded at freeze time")
    frozen = json.loads((work / "protocol.json").read_text(encoding="utf8"))
    for n in ("cases_routebench.jsonl", "cases_ragtruth.jsonl", "minicheck_cached_routebench.json"):
        if _sha(work / n) != frozen["input_sha256"][n]:
            raise MiniCheckJobError(f"input file {n} changed after the freeze")
    mine = frozen["code"].get("judge_v19_minicheck.py")
    if mine != _sha(Path(__file__)):
        raise MiniCheckJobError("this module differs from the frozen copy")
    from .dataset import CaseRecord

    def load(name: str) -> dict[str, Any]:
        out = {}
        for line in (work / name).read_text(encoding="utf8").splitlines():
            c = CaseRecord.model_validate_json(line)
            out[c.case_id] = c
        return out

    return frozen, load("cases_routebench.jsonl"), load("cases_ragtruth.jsonl")


def run(work: Path, expected_sha256: str, confirm: str, checker: Any = None, splitter: Any = None) -> dict[str, Any]:
    if confirm != MINICHECK_CONFIRM:
        raise MiniCheckJobError(f"the GPU feature job needs --confirm {MINICHECK_CONFIRM}")
    work = Path(work)
    frozen, rb_cases, rt_cases = _verified_inputs(work, expected_sha256)
    if frozen["minicheck"]["equivalence_rows"] != EQUIVALENCE_ROWS or frozen["minicheck"]["tolerance"] != EQUIVALENCE_TOL:
        raise MiniCheckJobError("equivalence constants differ from the freeze")
    out = work / OUTPUT
    if out.exists():
        raise MiniCheckJobError(f"{OUTPUT} exists; the job is written once")
    if checker is None:
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[var] = "1"  # absent weights stop the job; nothing is downloaded
        ckpts = Path(__file__).resolve().parents[2] / ".local" / "ckpts"
        os.environ.setdefault("HF_HUB_CACHE", str(ckpts))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(ckpts))
        from .checkers import DEFAULT_SPECS, load_checker

        spec = dataclasses.replace(next(s for s in DEFAULT_SPECS if s.checker_id == CHECKER_ID), weights_licence_checked=True)
        checker = load_checker(spec, device=0, batch_size=8)
    cached = json.loads((work / "minicheck_cached_routebench.json").read_text(encoding="utf8"))
    probe = sorted(rb_cases, key=lambda c: _h(f"equivalence|{c}"))[:EQUIVALENCE_ROWS]
    worst = 0.0
    for cid in probe:
        full, sents = minicheck_pair(checker, rb_cases[cid].context, rb_cases[cid].candidate_output, splitter)
        if len(sents) != len(cached[cid]["sent"]):
            raise MiniCheckJobError(f"equivalence check failed on {cid}: the sentence split differs from the cached features; RAGTruth is not scored")
        worst = max(worst, abs(full - cached[cid]["full"]), *(abs(a - b) for a, b in zip(sents, cached[cid]["sent"], strict=True)))
    if worst > EQUIVALENCE_TOL:
        raise MiniCheckJobError(f"equivalence check failed: largest difference {worst:.5f} > {EQUIVALENCE_TOL}; RAGTruth is not scored (the feature schema cannot be matched)")
    feats = {}
    for cid, c in rt_cases.items():
        full, sents = minicheck_pair(checker, c.context, c.candidate_output, splitter)
        feats[cid] = {"full": full, "sent": sents}
    tmp = work / (OUTPUT + ".tmp")
    tmp.write_text(json.dumps({"protocol_sha256": expected_sha256, "producer_sha256": _sha(Path(__file__)), "equivalence_rows": len(probe), "equivalence_worst_abs_difference": worst,
                               "features": feats}, sort_keys=True), encoding="utf8")
    os.replace(tmp, out)
    return {"equivalence_worst_abs_difference": worst, "rows": len(feats)}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--confirm", default="")
    a = p.parse_args(argv)
    try:
        print(json.dumps(run(a.work, a.protocol_sha256, a.confirm), indent=2))
    except MiniCheckJobError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
