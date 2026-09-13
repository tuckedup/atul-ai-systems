"""Execute one ForgeBench case through the complete ForgeCode graph.

This module owns the benchmark-specific trust boundaries around the graph:

* the agent works in a disposable git repository created from a fixture;
* approval is correlated by the graph thread's deterministic approval key;
* hidden tests stay outside the disposable repository until the graph stops; and
* the return value uses the metric keys consumed by :func:`aisys.evals.run`.

Provider selection, tracing, auditing, checkpointing, and tool execution remain in
the production graph and shared ``aisys`` primitives. The benchmark does not
replace any of those paths with a benchmark-only implementation.
"""
from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aisys.approval import ApprovalStore
from aisys.evals import EvalCase
from aisys.settings import settings

from . import graph
from .sandbox import SubprocessSandbox

FORGEBENCH = Path(__file__).parents[1] / "forgebench"


def _pending_approval(thread_id: str, store: ApprovalStore) -> dict[str, Any] | None:
    """Find only the pending patch approval associated with this graph thread.

    ApprovalStore is shared by projects, so selecting the first pending row would
    risk resuming an unrelated run. ForgeCode's graph writes approval keys in the
    form ``forgecode:<thread_id>:<action>``; matching that prefix supplies the
    correlation boundary.
    """
    prefix = f"forgecode:{thread_id}:"
    for item in store.pending():
        key = str(item["action"].get("approval_key", ""))
        if key.startswith(prefix):
            return item
    return None


def _install_hidden_tests(source: Path, workdir: Path) -> None:
    """Copy a case's hidden-test tree into its disposable repository.

    Relative paths are preserved so a hidden test can include package fixtures or
    configuration alongside the test module. The source must be a directory under
    ``forgebench/hidden`` resolved by the caller.
    """
    if not source.is_dir():
        raise ValueError(f"hidden-test directory does not exist: {source}")
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        target = workdir / item.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def execute_agent_case(
    case: EvalCase,
    *,
    workspace_root: Path,
    approve: bool = True,
    graph_run: Callable[[str, str, str, str], dict[str, Any]] = graph.run,
    graph_resume: Callable[[str, str], dict[str, Any]] = graph.resume,
) -> dict[str, object]:
    """Run one full-agent case and return metrics in the shared eval-runner shape.

    ``case.input`` must provide ``fixture``, ``task``, and ``hidden_tests`` and may
    override ``test_command``. When the graph pauses, ``approve=True`` records an
    explicit benchmark-operator decision in the durable ApprovalStore and resumes
    the same checkpointed thread. ``approve=False`` leaves the pending decision in
    place and reports the case as an error through the outer eval runner.

    A case passes only when its post-run hidden tests pass *and* the repository has
    a non-empty git diff. Requiring both prevents an already-green fixture or a
    no-op agent response from being counted as a successful repair.

    ``graph_run`` and ``graph_resume`` are injectable solely so unit tests can
    verify orchestration without spending provider tokens. Production callers use
    the real graph defaults.
    """
    payload = case.input
    fixture = (FORGEBENCH / "fixtures" / payload["fixture"]).resolve()
    hidden = (FORGEBENCH / "hidden" / payload["hidden_tests"]).resolve()
    test_command = str(payload.get("test_command", "python -m pytest -q"))
    started = time.perf_counter()
    thread_id = f"bench-{case.id}-{uuid.uuid4().hex[:12]}"
    interventions = 0

    # SubprocessSandbox copies the fixture and initializes a private baseline
    # commit. Its context manager removes the task copy after scoring.
    with SubprocessSandbox(fixture, workspace_root) as sandbox:
        # Run the production graph unchanged. At this point the hidden tests are
        # physically absent, so neither retrieval nor model tool calls can inspect
        # them.
        outcome = graph_run(str(sandbox.workdir), str(payload["task"]), thread_id, test_command)

        # A successful edit reaches the patch approval gate. The benchmark records
        # the durable decision before resuming the exact checkpoint, mirroring the
        # CLI/operator flow while making the intervention visible in metrics.
        pending = _pending_approval(thread_id, ApprovalStore(settings.database_url))
        if pending is not None:
            interventions = 1
            if not approve:
                raise RuntimeError(f"agent paused for approval {pending['id']}")
            store = ApprovalStore(settings.database_url)
            store.decide(str(pending["id"]), "forgebench-operator", "approved")
            outcome = graph_resume(thread_id, "approved")

        # The graph has now exited; install the held-out tests and score the patch.
        _install_hidden_tests(hidden, sandbox.workdir)
        verification = sandbox.run(test_command)
        diff = sandbox.run("git diff --no-ext-diff").stdout
        passed = verification.returncode == 0 and bool(diff.strip())

        # aisys.evals.run consumes these six standard metric fields. test_output is
        # retained as diagnostic context in direct calls but is intentionally not
        # part of the compact CaseResult persisted by the shared runner.
        return {
            "output": "pass" if passed else "fail",
            "tokens": int(outcome.get("tokens", 0)),
            "cost_usd": float(outcome.get("cost_usd", 0.0)),
            "latency_ms": (time.perf_counter() - started) * 1_000,
            "steps": len(outcome.get("steps", [])),
            "human_interventions": interventions,
            "test_output": (verification.stdout + verification.stderr)[-4_000:],
        }
