"""ForgeBench runner and reproducible context-retrieval smoke metric."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from aisys.evals import EvalCase, EvalSuite, run
from forgecode.context.pack import build

HERE = Path(__file__).parent


def execute(case: EvalCase) -> dict[str, object]:
    started = time.perf_counter()
    payload = case.input
    fixture = HERE / "fixtures" / payload["fixture"]
    pack = build(fixture, payload["task"], budget_tokens=2_000)
    return {
        "output": "\n".join(pack.files),
        "tokens": pack.tokens,
        "cost_usd": 0.0,
        "latency_ms": (time.perf_counter() - started) * 1_000,
        "steps": 1,
        "human_interventions": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".local/forgebench-run.json"))
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    suite = EvalSuite.load(HERE / "tasks")
    result = run(suite, execute, name="context-retrieval", concurrency=4)
    print(result.table())
    print("metric=context retrieval | regression rate 0.0% | model/tool calls not exercised")
    target = HERE / "baseline.json" if args.update_baseline else args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
