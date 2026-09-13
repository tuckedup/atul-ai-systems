"""CLI for the full-agent ForgeBench suite.

This is intentionally separate from ``run.py``, which measures context retrieval
without calling a model. Full-agent cases are run serially: each case can create a
durable approval/checkpoint and consume provider quota, so concurrency would make
cost control and failure attribution harder. The emitted JSON is a shared
``aisys.evals.RunResult`` suitable for later baseline comparison.
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from aisys.evals import EvalSuite, run
from aisys.settings import settings
from forgecode.benchmark import execute_agent_case

HERE = Path(__file__).parent
WORKSPACE_ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run held-out local repair cases through the complete ForgeCode graph."
    )
    parser.add_argument("--limit", type=int, default=0, help="Run the first N cases; zero runs all cases.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".local/forgebench-agent.json"),
        help="Write the machine-readable RunResult JSON to this path.",
    )
    parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="Stop cases at the approval gate instead of recording a benchmark-operator approval.",
    )
    args = parser.parse_args()
    if not settings.openai_api_key:
        raise SystemExit("ForgeBench agent mode requires OPENAI_API_KEY or AISYS_OPENAI_API_KEY")

    suite = EvalSuite.load(HERE / "agent_tasks")
    if args.limit > 0:
        suite = EvalSuite(suite.cases[: args.limit])

    # Serial execution makes each provider charge, trace, approval, and failure
    # attributable to one case. It also avoids concurrent writes to the local
    # SQLite checkpointer used by the degraded-host configuration.
    execute = partial(execute_agent_case, workspace_root=WORKSPACE_ROOT, approve=not args.no_auto_approve)
    result = run(suite, execute, name="forgecode-full-agent", concurrency=1)
    print(result.table())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if any(case.error is not None or case.score < 1.0 for case in result.results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
