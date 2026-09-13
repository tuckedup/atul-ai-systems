"""Durable Incident Commander graph with a human-approval interrupt.

The seven investigation roles retain their explicit context projections. After
the risk reviewer, the graph checkpoints and pauses in ``approval_gate``. A later
process can reopen the SQLite checkpointer with the same ``thread_id``, supply a
decision via ``Command(resume=...)``, and continue at that exact node.

Degraded mode uses ``durability="sync"`` for every invocation. LangGraph therefore
writes each super-step before advancing, trading throughput for the strongest
local durability guarantee.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from aisys import llm, tracing
from aisys.approval import ApprovalPolicy, ApprovalStore, Decision, Gate
from aisys.audit import AuditLog
from aisys.settings import settings
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .scoping import prompt

INVESTIGATION_NODES = [
    "coordinator",
    "logs_agent",
    "dependency_agent",
    "code_agent",
    "root_cause_agent",
    "remediation_agent",
    "risk_reviewer",
]

# Node output is copied into the scoped context key consumed by the next stage.
# The full ``outputs`` map remains in state for auditability and resume tests.
CONTEXT_OUTPUT_KEYS = {
    "coordinator": "investigation_plan",
    "logs_agent": "logs_summary",
    "dependency_agent": "dependency_summary",
    "code_agent": "code_summary",
    "root_cause_agent": "top_hypothesis",
    "remediation_agent": "proposal",
    "risk_reviewer": "risk_review",
}

HIGH_RISK_ACTIONS = {"restart", "scale", "rollback", "db_modify", "merge_pr"}


class IncidentState(TypedDict, total=False):
    incident_id: str
    thread_id: str
    context: dict[str, Any]
    outputs: dict[str, str]
    errors: list[str]
    phase: str
    approval: str
    execution: dict[str, Any]


Responder = Callable[[str, str, IncidentState], str]
ActionExecutor = Callable[[IncidentState], dict[str, Any]]


def _model_responder(_name: str, rendered_prompt: str, _state: IncidentState) -> str:
    """Production responder; deterministic tests inject a local replacement."""
    return llm.chat([{"role": "user", "content": rendered_prompt}]).text


def _degraded_subprocess_executor(state: IncidentState) -> dict[str, Any]:
    """Cross the execution boundary in a child process without Docker/kubectl.

    This is durability evidence, not a claim that a Kubernetes remediation ran.
    The child echoes the approved action as JSON so tests can prove execution
    occurred only after resume and capture a concrete subprocess return code.
    """
    context = state["context"]
    payload = json.dumps({
        "incident_id": state["incident_id"],
        "action": context.get("remediation_action", "rollback"),
        "resource": context.get("remediation_resource", "deployment/demo"),
    }, sort_keys=True)
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", payload],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return {
        "mode": "degraded-subprocess",
        "status": "executed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "payload": json.loads(completed.stdout.strip()) if completed.stdout.strip() else {},
        "stderr": completed.stderr[-2_000:],
    }


def node(name: str, responder: Responder = _model_responder) -> Callable[[IncidentState], dict[str, Any]]:
    """Build one scoped investigation node with checkpointable error state."""

    @tracing.traced(kind="agent", name=name)
    def execute(state: IncidentState) -> dict[str, Any]:
        context = dict(state["context"])
        outputs = dict(state.get("outputs", {}))
        errors = list(state.get("errors", []))
        try:
            response = responder(name, prompt(name, context), state)
        except Exception as error:  # noqa: BLE001 - persist the failed super-step as graph state
            response = f"ERROR[{name}]: {type(error).__name__}: {error}"
            errors.append(response)
        outputs[name] = response
        context[CONTEXT_OUTPUT_KEYS[name]] = response
        return {"context": context, "outputs": outputs, "errors": errors, "phase": name}

    return execute


def _approval_node(gate: Gate) -> Callable[[IncidentState], dict[str, Any]]:
    @tracing.traced(kind="agent", name="approval_gate")
    def approval_gate(state: IncidentState) -> dict[str, Any]:
        context = state["context"]
        action = str(context.get("remediation_action", "rollback"))
        risk = "high" if action in HIGH_RISK_ACTIONS else "medium"
        decision = gate.check(
            {
                "tool": action,
                "risk": risk,
                "args": {"resource": context.get("remediation_resource", "deployment/demo")},
                "sandboxed": False,
                "approval_key": f"incident:{state['thread_id']}:{state['incident_id']}:{action}",
            },
            agent="risk_reviewer",
        )
        return {"approval": decision, "phase": "approval_decided"}

    return approval_gate


def _after_approval(state: IncidentState) -> Literal["execute_remediation", "end"]:
    return "execute_remediation" if state.get("approval") in {"auto", "approved"} else "end"


def _execution_node(executor: ActionExecutor, audit: AuditLog) -> Callable[[IncidentState], dict[str, Any]]:
    @tracing.traced(kind="tool", name="execute_remediation")
    def execute_remediation(state: IncidentState) -> dict[str, Any]:
        result = executor(state)
        audit.append({
            "type": "remediation_executed",
            "incident_id": state["incident_id"],
            "thread_id": state["thread_id"],
            "action": state["context"].get("remediation_action", "rollback"),
            "result": result,
            "trace_id": tracing.current_trace_id.get(),
        })
        return {"execution": result, "phase": "completed"}

    return execute_remediation


def build_graph(
    *,
    gate: Gate | None = None,
    audit: AuditLog | None = None,
    responder: Responder = _model_responder,
    executor: ActionExecutor = _degraded_subprocess_executor,
) -> StateGraph[IncidentState]:
    """Build the graph, defaulting to the configured degraded-mode stores."""
    if audit is None:
        _sqlite_path(settings.database_url)
        audit = AuditLog(settings.database_url)
    if gate is None:
        gate = Gate(ApprovalPolicy.default(), ApprovalStore(settings.database_url), audit)
    workflow = StateGraph(IncidentState)
    for name in INVESTIGATION_NODES:
        workflow.add_node(name, cast(Any, node(name, responder)))
    workflow.add_node("approval_gate", cast(Any, _approval_node(gate)))
    workflow.add_node("execute_remediation", cast(Any, _execution_node(executor, audit)))
    workflow.set_entry_point(INVESTIGATION_NODES[0])
    for first, second in pairwise(INVESTIGATION_NODES):
        workflow.add_edge(first, second)
    workflow.add_edge(INVESTIGATION_NODES[-1], "approval_gate")
    workflow.add_conditional_edges(
        "approval_gate",
        _after_approval,
        {"execute_remediation": "execute_remediation", "end": END},
    )
    workflow.add_edge("execute_remediation", END)
    return workflow


def _sqlite_path(database_url: str) -> Path:
    if not database_url.startswith("sqlite:///"):
        raise ValueError("degraded mode requires a sqlite:/// database URL")
    path = Path(database_url.removeprefix("sqlite:///"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _runtime(
    database_url: str,
    *,
    responder: Responder = _model_responder,
    executor: ActionExecutor = _degraded_subprocess_executor,
) -> Iterator[Any]:
    """Open a fresh saver/runtime, modeling a new operating-system process."""
    path = _sqlite_path(database_url)
    audit = AuditLog(database_url)
    gate = Gate(ApprovalPolicy.default(), ApprovalStore(database_url), audit)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield build_graph(gate=gate, audit=audit, responder=responder, executor=executor).compile(
            checkpointer=saver
        )


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def start_incident(
    context: dict[str, Any],
    *,
    incident_id: str,
    thread_id: str,
    database_url: str,
    responder: Responder = _model_responder,
    executor: ActionExecutor = _degraded_subprocess_executor,
) -> dict[str, Any]:
    """Start an incident and synchronously checkpoint through the interrupt."""
    tracing.init_tracing("incident-commander")
    tracing.new_trace_id()
    with _runtime(database_url, responder=responder, executor=executor) as app:
        return cast(dict[str, Any], app.invoke(
            {
                "incident_id": incident_id,
                "thread_id": thread_id,
                "context": dict(context),
                "outputs": {},
                "errors": [],
                "phase": "received",
            },
            config=_config(thread_id),
            durability="sync",
        ))


def _pending_for_thread(store: ApprovalStore, thread_id: str) -> dict[str, Any] | None:
    prefix = f"incident:{thread_id}:"
    for item in store.pending():
        if str(item["action"].get("approval_key", "")).startswith(prefix):
            return item
    return None


def resume_incident(
    *,
    thread_id: str,
    decision: Decision,
    approver: str,
    database_url: str,
    responder: Responder = _model_responder,
    executor: ActionExecutor = _degraded_subprocess_executor,
) -> dict[str, Any]:
    """Reopen the database and resume the exact checkpoint for ``thread_id``."""
    with _runtime(database_url, responder=responder, executor=executor) as app:
        config = _config(thread_id)
        snapshot = app.get_state(config)
        if not snapshot.values:
            raise ValueError(f"no checkpoint exists for thread_id={thread_id}")
        if snapshot.values.get("thread_id") != thread_id:
            raise ValueError("checkpoint thread_id does not match the requested thread")
        pending = _pending_for_thread(ApprovalStore(database_url), thread_id)
        if pending is None:
            raise ValueError(f"no pending approval exists for thread_id={thread_id}")
        ApprovalStore(database_url).decide(str(pending["id"]), approver, decision)
        return cast(dict[str, Any], app.invoke(
            Command(resume=decision),
            config=config,
            durability="sync",
        ))


def inspect_thread(database_url: str, thread_id: str) -> dict[str, Any]:
    """Return durable state, next node, and checkpoint count for verification."""
    path = _sqlite_path(database_url)
    config = _config(thread_id)
    audit = AuditLog(database_url)
    gate = Gate(ApprovalPolicy.default(), ApprovalStore(database_url), audit)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        app = build_graph(gate=gate, audit=audit).compile(checkpointer=saver)
        snapshot = app.get_state(config)
        return {
            "values": dict(snapshot.values),
            "next": tuple(snapshot.next),
            "checkpoint_count": len(list(saver.list(config))),
        }
