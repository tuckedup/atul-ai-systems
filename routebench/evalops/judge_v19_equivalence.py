"""MiniCheck feature-equivalence check ONLY (GPU inference, no training, no RAGTruth scoring, no API).

The frozen judge-v19 feature job (`judge_v19_minicheck.run`) recomputes cached RouteBench rows first and then, if they match, scores 1,050 RAGTruth rows. This module performs just the first step and writes a
SEPARATE report, so the frozen job and its protocol are untouched and no RAGTruth feature file is produced. It answers one question: does the MiniCheck pipeline available here reproduce the cached RouteBench
`full` and per-sentence scores (the generator of the cached scores is not in the repository)?

Run it in the MiniCheck virtualenv (`.local/mc-venv`: upstream `minicheck`, torch + CUDA). Hugging Face offline switches are forced on, so absent weights stop the run instead of downloading.
Memory: loading the model needs roughly 3 GB of free system RAM and about 2 GB of GPU memory; the command refuses to start below a RAM floor (default 3,000 MB) instead of pushing the machine into paging.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import judge_v19_minicheck as mc

CONFIRM = "RUN-JV19-MINICHECK-EQUIVALENCE-ONLY"
REPORT = "minicheck_equivalence_report.json"
DEFAULT_MIN_FREE_MB = 3000


class EquivalenceError(ValueError):
    pass


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise EquivalenceError(f"{name} must be a finite probability in [0, 1]")
    return float(value)


def free_ram_mb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 1048576
    except ImportError:
        return None


def check(work: Path, protocol_sha: str, checker: Any, splitter: Any = None, *, rows: int | None = None, tolerance: float | None = None) -> dict[str, Any]:
    """Recompute the hash-ordered cached RouteBench rows and compare. Pure given a checker; writes nothing."""
    frozen, rb_cases, _rt = mc._verified_inputs(work, protocol_sha)
    n = frozen["minicheck"]["equivalence_rows"] if rows is None else rows
    tol = tolerance if tolerance is not None else frozen["minicheck"]["tolerance"]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0 or n > len(rb_cases):
        raise EquivalenceError("rows must be a positive integer no greater than the verified population")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol < 0:
        raise EquivalenceError("tolerance must be finite and non-negative")
    cached = json.loads((Path(work) / "minicheck_cached_routebench.json").read_text(encoding="utf8"))
    probe = sorted(rb_cases, key=lambda c: mc._h(f"equivalence|{c}"))[:n]
    details, worst, split_mismatch = [], 0.0, []
    for cid in probe:
        full, sents = mc.minicheck_pair(checker, rb_cases[cid].context, rb_cases[cid].candidate_output, splitter)
        ref = cached[cid]
        full = _probability(full, f"{cid} recomputed full")
        reference_full = _probability(ref["full"], f"{cid} cached full")
        sents = [_probability(v, f"{cid} recomputed sentence") for v in sents]
        reference_sent = [_probability(v, f"{cid} cached sentence") for v in ref["sent"]]
        if len(sents) != len(reference_sent):
            split_mismatch.append(cid)
            details.append({"case_id": cid, "sentence_split_matches": False, "recomputed_sentences": len(sents), "cached_sentences": len(ref["sent"])})
            continue
        diffs = [abs(full - reference_full), *(abs(a - b) for a, b in zip(sents, reference_sent, strict=True))]
        worst = max(worst, *diffs)
        details.append({"case_id": cid, "sentence_split_matches": True, "max_abs_difference": max(diffs),
                        "full": [full, reference_full], "recomputed_sent": sents, "cached_sent": reference_sent})
    passed = not split_mismatch and worst <= tol
    return {"passed": passed, "rows_checked": len(probe), "worst_abs_difference": worst, "tolerance": tol, "sentence_split_mismatches": split_mismatch, "details": details}


def run(work: Path, protocol_sha: str, confirm: str, *, min_free_mb: float = DEFAULT_MIN_FREE_MB, checker: Any = None, splitter: Any = None) -> dict[str, Any]:
    if confirm != CONFIRM:
        raise EquivalenceError(f"needs --confirm {CONFIRM}")
    work = Path(work)
    if (work / REPORT).exists():
        raise EquivalenceError(f"{REPORT} exists; the check is written once")
    if (isinstance(min_free_mb, bool) or not isinstance(min_free_mb, (int, float))
            or not math.isfinite(min_free_mb) or min_free_mb < DEFAULT_MIN_FREE_MB):
        raise EquivalenceError(f"RAM floor must be finite and at least {DEFAULT_MIN_FREE_MB} MB")
    # Reject drift before allocating a real model, not only inside the scoring loop.
    mc._verified_inputs(work, protocol_sha)
    real_model = checker is None
    if checker is None:
        free = free_ram_mb()
        if (isinstance(free, bool) or not isinstance(free, (int, float))
                or not math.isfinite(free) or free < 0):
            raise EquivalenceError("cannot measure free RAM (psutil missing); refusing to load the model without a memory check")
        if free < min_free_mb:
            raise EquivalenceError(f"only {free:,.0f} MB of RAM is free; at least {min_free_mb:,.0f} MB is required to load the model safely. Close other programs and retry; nothing was loaded")
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[var] = "1"  # absent weights stop the run; nothing is downloaded
        ckpts = Path(__file__).resolve().parents[2] / ".local" / "ckpts"
        os.environ.setdefault("HF_HUB_CACHE", str(ckpts))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(ckpts))
        from .checkers import DEFAULT_SPECS, load_checker

        spec = dataclasses.replace(next(s for s in DEFAULT_SPECS if s.checker_id == mc.CHECKER_ID), weights_licence_checked=True)  # operator step; not a rights clearance
        checker = load_checker(spec, device=0, batch_size=8)
    started = time.time()
    result = check(work, protocol_sha, checker, splitter)
    result |= {"schema_version": 2, "protocol_sha256": protocol_sha, "producer_sha256": mc._sha(Path(__file__)),
               "execution": "local_model" if real_model else "injected_checker",
               "wall_seconds": round(time.time() - started, 1), "scope": "equivalence check only; no RAGTruth row scored; no training; no API call"}
    result["peak_gpu_memory_mb"] = None
    if real_model:
        try:
            import torch

            result["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1048576, 1) if torch.cuda.is_available() else None
        except ImportError:
            pass
    tmp = work / (REPORT + ".tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf8")
    os.replace(tmp, work / REPORT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--confirm", default="")
    p.add_argument("--min-free-mb", type=float, default=DEFAULT_MIN_FREE_MB)
    a = p.parse_args(argv)
    try:
        out = run(a.work, a.protocol_sha256, a.confirm, min_free_mb=a.min_free_mb)
        print(json.dumps({k: v for k, v in out.items() if k != "details"}, indent=1))
        return 0 if out["passed"] else 3
    except (EquivalenceError, mc.MiniCheckJobError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
