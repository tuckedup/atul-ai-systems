"""ForgeCode command-line interface."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

import typer
from aisys.approval import ApprovalStore, Decision
from aisys.settings import settings
from rich import print

from . import graph
from .sandbox import SubprocessSandbox

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def run_command(
    repo: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    task: Annotated[str, typer.Option()],
    test_command: Annotated[str, typer.Option()] = "python -m pytest -q",
) -> None:
    root = Path.cwd().resolve()
    sandbox = SubprocessSandbox(repo, root)
    thread_id = uuid.uuid4().hex
    print(f"thread_id={thread_id} task_copy={sandbox.workdir}")
    result = graph.run(str(sandbox.workdir), task, thread_id, test_command)
    print(result)


@app.command("approve")
def approve(approval_id: str, approver: str = "cli-operator", reject: bool = False) -> None:
    store = ApprovalStore(settings.database_url)
    pending = {item["id"]: item for item in store.pending()}
    if approval_id not in pending:
        raise typer.BadParameter("approval id is not pending")
    verdict: Decision = "rejected" if reject else "approved"
    store.decide(approval_id, approver, verdict)
    key = str(pending[approval_id]["action"].get("approval_key", ""))
    thread_id = key.split(":")[1] if key.startswith("forgecode:") else ""
    if not thread_id:
        raise typer.BadParameter("approval is not associated with a ForgeCode thread")
    print(graph.resume(thread_id, verdict))
