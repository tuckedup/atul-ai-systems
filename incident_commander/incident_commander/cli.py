import typer
from aisys.approval import ApprovalStore, Decision
from aisys.audit import AuditLog
from aisys.settings import settings
from rich import print

from .agents.graph import resume_incident

app = typer.Typer(no_args_is_help=True)


@app.command("approve")
def approve(approval_id: str, approver: str = "sre", reject: bool = False) -> None:
    verdict: Decision = "rejected" if reject else "approved"
    pending = {str(item["id"]): item for item in ApprovalStore(settings.database_url).pending()}
    if approval_id not in pending:
        raise typer.BadParameter("approval id is not pending")
    approval_key = str(pending[approval_id]["action"].get("approval_key", ""))
    parts = approval_key.split(":", 3)
    if len(parts) < 3 or parts[0] != "incident" or not parts[1]:
        raise typer.BadParameter("approval is not associated with an Incident Commander thread")
    thread_id = parts[1]
    result = resume_incident(
        thread_id=thread_id,
        decision=verdict,
        approver=approver,
        database_url=settings.database_url,
    )
    print(f"{approval_id}: {verdict}; thread_id={thread_id}; phase={result.get('phase', 'unknown')}")


@app.command("audit-verify")
def audit_verify() -> None:
    broken = AuditLog(settings.database_url).verify()
    print("audit chain intact" if broken is None else f"audit chain broken at {broken}")
    raise typer.Exit(0 if broken is None else 1)
