"""`aisys` CLI: smoke commands used by the prompts' verification steps."""
from pathlib import Path

import typer
from rich import print

from . import audit as audit_mod
from . import evals, llm, tracing
from .approval import ApprovalStore
from .settings import settings

app = typer.Typer(no_args_is_help=True)
evals_app = typer.Typer(no_args_is_help=True)
app.add_typer(evals_app, name="evals")


@app.command("local-init")
def local_init() -> None:
    """Initialize degraded-host persistence without external services."""
    audit_mod.default_audit_log()
    ApprovalStore(settings.database_url)
    tracing.JsonlSpanExporter(settings.trace_jsonl_path)
    print(f"[green]local runtime ready[/]: {settings.database_url}; traces={settings.trace_jsonl_path}")


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
    b = evals.RunResult.model_validate_json(Path(baseline).read_text(encoding="utf-8"))
    c = evals.RunResult.model_validate_json(Path(candidate).read_text(encoding="utf-8"))
    rep = evals.compare(b, c, max_drop=max_drop)
    print(rep.model_dump())
    raise typer.Exit(code=1 if rep.blocked else 0)


@evals_app.command("demo")
def evals_demo() -> None:
    """Run a deterministic smoke suite and print its result table."""
    suite = evals.EvalSuite([
        evals.EvalCase(id="exact", input="2+2", expected="4"),
        evals.EvalCase(id="contains", input="greeting", expected="hello", grader="contains"),
    ])
    outputs = {"exact": "4", "contains": "hello world"}
    result = evals.run(suite, lambda case: {"output": outputs[case.id]}, name="demo", concurrency=1)
    print(result.table())
