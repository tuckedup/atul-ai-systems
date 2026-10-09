"""Test-collection policy for research-protocol tests that need unpublished local inputs.

Some tests exercise frozen research protocols (srcint, claim-fidelity, judge-v19, matched v16/v18, fresh-eval, ...)
against the author's local artifacts under `.local/` and unpublished judgment/data files. Those inputs are
deliberately NOT part of the repository (third-party-derived or uncleared data, private judgment caches), so on a
fresh clone these modules cannot run. They are skipped with an explicit reason instead of erroring; nothing is
loosened when the inputs exist, and the skip is keyed to a single sentinel (the frozen judge-v19 protocol file).
Every other test, including the whole gateway/budget/auth/streaming/rollout suite, runs unconditionally.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SENTINEL = ROOT / ".local" / "judge-v19-phase1" / "protocol.json"

NEEDS_LOCAL_RESEARCH_INPUTS = {
    "test_audit2_v2.py",
    "test_claim_fidelity_eval.py",
    "test_claim_fidelity_eval_v2.py",
    "test_claim_fidelity_eval_v2_cost.py",
    "test_claim_relation_frozen.py",
    "test_dq_claim_relation.py",
    "test_fresh_eval.py",
    "test_judge_v19_phase1.py",
    "test_judge_v19_recovery.py",
    "test_matched_v16_v18.py",
    "test_srcint_eval.py",
    "test_srcint_eval_v2.py",
    "test_srcint_eval_v3.py",
}


def pytest_collection_modifyitems(config, items):
    if SENTINEL.exists():
        return
    skip = pytest.mark.skip(reason="requires the author's unpublished local research artifacts under .local/ and unpublished data files (not part of the repository)")
    for item in items:
        if Path(str(item.fspath)).name in NEEDS_LOCAL_RESEARCH_INPUTS:
            item.add_marker(skip)
