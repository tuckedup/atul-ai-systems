"""ForgeCode's planner → implementer → tester → reviewer → approval graph."""
from __future__ import annotations

import json
import operator
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, cast

from aisys import approval, llm, tracing
from aisys.audit import default_audit_log
from aisys.settings import settings
from aisys.tools import registry
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from . import tools as _forge_tools  # noqa: F401 - import registers tools
from .context.pack import build_context_pack
from .router import pick_model

MAX_ATTEMPTS = 4
PROMPTS = Path(__file__).parent / "prompts"


class ForgeState(TypedDict, total=False):
    repo: str
    task: str
    thread_id: str
    test_command: str
    plan: str
    context_pack: str
    diff: str
    test_output: str
    tests_pass: bool
    attempt: int
    review: str
    review_ok: bool
    approval: str
    steps: Annotated[list[str], operator.add]
    tokens: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]


_audit = default_audit_log()
_gate = approval.Gate(approval.ApprovalPolicy.default(), approval.ApprovalStore(settings.database_url), _audit)
registry.audit = _audit


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def _acc(result: llm.ChatResult, step: str) -> dict[str, Any]:
    return {"tokens": result.usage.total, "cost_usd": result.cost_usd or 0.0, "steps": [step]}


def scoped_tool_arguments(repo: str, tool_name: str, raw_arguments: str) -> dict[str, Any]:
    """Parse model arguments and inject the harness-owned task-copy path.

    Repository identity is a security boundary. Models may emit placeholders or
    arbitrary paths, so every repo-scoped ForgeCode tool receives the trusted path
    from graph state instead of the model-supplied value.
    """
    arguments = cast(dict[str, Any], json.loads(raw_arguments))
    if "repo" in registry.tools[tool_name].model.model_fields:
        arguments["repo"] = repo
    return arguments


@tracing.traced(kind="agent", name="planner")
def planner(state: ForgeState) -> dict[str, Any]:
    pack = build_context_pack(state["repo"], state["task"], budget_tokens=12_000)
    result = llm.chat(
        [
            {"role": "system", "content": _prompt("planner")},
            {"role": "user", "content": f"TASK:\n{state['task']}\n\nCONTEXT:\n{pack}"},
        ],
        model=pick_model("plan"),
    )
    return {"plan": result.text, "context_pack": pack, "attempt": 0, **_acc(result, "plan")}


@tracing.traced(kind="agent", name="implementer")
def implementer(state: ForgeState) -> dict[str, Any]:
    attempt = state.get("attempt", 0)
    model = pick_model("edit", attempt=attempt, last_failure=state.get("test_output"))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _prompt("implementer")},
        {"role": "user", "content": (
            f"PLAN:\n{state['plan']}\n\nCONTEXT:\n{state['context_pack']}\n\n"
            f"PREVIOUS TEST OUTPUT:\n{state.get('test_output', '')}"
        )},
    ]
    result: llm.ChatResult | None = None
    total_tokens = 0
    total_cost_usd = 0.0
    for _ in range(40):
        result = llm.chat(messages, model=model, tools=registry.schema())
        total_tokens += result.usage.total
        total_cost_usd += result.cost_usd or 0.0
        if not result.tool_calls:
            break
        messages.append({
            "role": "assistant",
            "content": result.text or None,
            "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in result.tool_calls
            ],
        })
        for call in result.tool_calls:
            arguments = scoped_tool_arguments(state["repo"], call.name, call.arguments)
            output = registry.call(call.name, arguments, agent="implementer")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(output)[:8000]})
    if result is None:
        raise RuntimeError("implementer produced no model result")
    diff = registry.call("git_diff", {"repo": state["repo"]}, agent="implementer")
    return {
        "diff": diff,
        "attempt": attempt + 1,
        "tokens": total_tokens,
        "cost_usd": total_cost_usd,
        "steps": [f"implement#{attempt + 1}"],
    }


@tracing.traced(kind="agent", name="tester")
def tester(state: ForgeState) -> dict[str, Any]:
    output = registry.call(
        "run_tests",
        {"repo": state["repo"], "command": state.get("test_command", "python -m pytest -q")},
        agent="tester",
    )
    combined = f"{output['stdout']}\n{output['stderr']}"
    return {"test_output": combined[-6000:], "tests_pass": output["returncode"] == 0, "steps": ["test"]}


def after_tests(state: ForgeState) -> Literal["reviewer", "implementer", "give_up"]:
    if state["tests_pass"]:
        return "reviewer"
    return "implementer" if state["attempt"] < MAX_ATTEMPTS else "give_up"


def reviewer_messages(state: ForgeState) -> list[dict[str, str]]:
    """Expose the isolation boundary for direct verification."""
    return [
        {"role": "system", "content": _prompt("reviewer")},
        {"role": "user", "content": (
            f"TASK:\n{state['task']}\n\nDIFF:\n{state['diff']}\n\nTESTS:\n{state['test_output']}\n\n"
            "Reply with VERDICT: APPROVE or VERDICT: REJECT, then reasons."
        )},
    ]


@tracing.traced(kind="agent", name="reviewer")
def reviewer(state: ForgeState) -> dict[str, Any]:
    result = llm.chat(reviewer_messages(state), model=pick_model("review"))
    return {"review": result.text, "review_ok": "VERDICT: APPROVE" in result.text, **_acc(result, "review")}


def after_review(state: ForgeState) -> Literal["approval_gate", "implementer"]:
    return "approval_gate" if state["review_ok"] else "implementer"


@tracing.traced(kind="agent", name="approval_gate")
def approval_gate(state: ForgeState) -> dict[str, Any]:
    risk = "high" if len(state["diff"]) > 20_000 or "migration" in state["diff"].lower() else "medium"
    decision = _gate.check(
        {
            "tool": "create_patch",
            "risk": risk,
            "args": {"repo": state["repo"]},
            "sandboxed": False,
            "approval_key": f"forgecode:{state['thread_id']}:patch",
        },
        agent="harness",
    )
    return {"approval": decision, "steps": [f"approval:{decision}"]}


def after_gate(state: ForgeState) -> Literal["create_patch", "end"]:
    return "create_patch" if state["approval"] in ("auto", "approved") else "end"


@tracing.traced(kind="agent", name="create_patch")
def create_patch(state: ForgeState) -> dict[str, Any]:
    registry.call("create_patch", {"repo": state["repo"], "message": state["task"][:72]}, agent="harness")
    return {"steps": ["patch"]}


def give_up(state: ForgeState) -> dict[str, Any]:
    _audit.append({"type": "give_up", "attempts": state["attempt"], "trace_id": tracing.current_trace_id.get()})
    return {"steps": ["give_up"]}


def build_graph() -> StateGraph[ForgeState]:
    graph = StateGraph(ForgeState)
    for name, node in [
        ("planner", planner), ("implementer", implementer), ("tester", tester), ("reviewer", reviewer),
        ("approval_gate", approval_gate), ("create_patch", create_patch), ("give_up", give_up),
    ]:
        graph.add_node(name, node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "implementer")
    graph.add_edge("implementer", "tester")
    graph.add_conditional_edges("tester", after_tests)
    graph.add_conditional_edges("reviewer", after_review)
    graph.add_conditional_edges("approval_gate", after_gate, {"create_patch": "create_patch", "end": END})
    graph.add_edge("create_patch", END)
    graph.add_edge("give_up", END)
    return graph


@contextmanager
def _checkpointer() -> Iterator[Any]:
    if settings.database_url.startswith("sqlite:///"):
        path = settings.database_url.removeprefix("sqlite:///")
        with SqliteSaver.from_conn_string(path) as saver:
            yield saver
    else:
        with PostgresSaver.from_conn_string(settings.database_url) as saver:
            saver.setup()
            yield saver


def run(repo: str, task: str, thread_id: str, test_command: str = "python -m pytest -q") -> dict[str, Any]:
    tracing.init_tracing("forgecode")
    tracing.new_trace_id()
    with _checkpointer() as saver:
        app = build_graph().compile(checkpointer=saver)
        return app.invoke(
            {
                "repo": repo, "task": task, "thread_id": thread_id, "test_command": test_command,
                "steps": [], "tokens": 0, "cost_usd": 0.0,
            },
            config={"configurable": {"thread_id": thread_id}},
        )


def resume(thread_id: str, decision: str) -> dict[str, Any]:
    """Resume exactly at the persisted approval interrupt."""
    with _checkpointer() as saver:
        app = build_graph().compile(checkpointer=saver)
        return app.invoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})
