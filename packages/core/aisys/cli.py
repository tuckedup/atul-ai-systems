"""`aisys` CLI: smoke commands used by the prompts' verification steps."""
from __future__ import annotations

import json
from typing import Any

import typer
from rich import print

from . import audit as audit_mod, evals, llm, tracing
from .settings import settings

app = typer.Typer(no_args_is_help=True)


@app.command("tracing-smoke")
def tracing_smoke() -> None:
    """Emit one traced LLM call so you can see it in Phoenix."""
    tracing.init_tracing("aisys-smoke")
    tracing.new_trace_id()
    r = llm.chat([{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=5)
    print(f"[green]{r.model}[/] via {r.provider}: {r.text!r} tokens={r.usage.total} cost={r.cost_usd} {r.latency_ms:.0f}ms")


@app.command("audit-verify")
def audit_verify(dsn: str = settings.database_url) -> None:
    broken = audit_mod.AuditLog(dsn).verify()
    print("[green]audit chain intact[/]" if broken is None else f"[red]chain broken at seq {broken}[/]")
    raise typer.Exit(code=0 if broken is None else 1)


@app.command("evals-compare")
def evals_compare(baseline: str, candidate: str, max_drop: float = 0.02) -> None:
    b = evals.RunResult.model_validate_json(open(baseline).read())
    c = evals.RunResult.model_validate_json(open(candidate).read())
    rep = evals.compare(b, c, max_drop=max_drop)
    print(rep.model_dump())
    raise typer.Exit(code=1 if rep.blocked else 0)


@app.command("evals-demo")
def evals_demo() -> None:
    """Run a small eval suite demo and print a results table."""
    from . import evals

    suite = evals.EvalSuite([
        evals.EvalCase(id="demo-1", input="capital of France?", expected="Paris", tags=["qa"]),
        evals.EvalCase(id="demo-2", input="2+2", expected="4", tags=["math"]),
        evals.EvalCase(id="demo-3", input="reverse 'hello'", expected="'olleh'", tags=["code"]),
    ])

    def stub_agent(c: evals.EvalCase) -> dict[str, Any]:
        return {
            "output": str(c.expected),
            "tokens": 50,
            "cost_usd": 0.002,
            "latency_ms": 120.0,
            "steps": 1,
            "human_interventions": 0,
        }

    result = evals.run(suite, stub_agent, name="demo", concurrency=1)
    print(result.table())
