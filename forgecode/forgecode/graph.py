"""ForgeCode harness: planner -> implementer -> tester -> (loop) -> reviewer -> approval gate -> patch.

The shape is the deliverable. Prompt 02 fills in context/, tools/, sandbox/, router.py.
Key properties this skeleton already enforces:
  * reviewer sees only (task, diff, test_output) — never implementer reasoning
  * approval gate uses aisys.approval.Gate -> LangGraph interrupt -> SQLite checkpoint -> resumable
  * every node is @traced so trace mining can query failed trajectories by node
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from aisys import approval, audit, llm, tracing
from aisys.settings import settings
from aisys.tools import registry

from .router import pick_model  # tier selection: task class + attempt + failure history
from .context.pack import build_context_pack

MAX_ATTEMPTS = 4


class ForgeState(TypedDict, total=False):
    repo: str
    task: str
    plan: str
    context_pack: str
    diff: str
    test_output: str
    tests_pass: bool
    attempt: int
    review: str
    review_ok: bool
    approval: str
    steps: Annotated[list[str], operator.add]     # append-only trajectory summary for trace mining
    tokens: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]


_audit = audit.AuditLog(settings.database_url)
_gate = approval.Gate(approval.ApprovalPolicy.default(), approval.ApprovalStore(settings.database_url), _audit)
registry.audit = _audit


def _acc(r: llm.ChatResult, step: str) -> dict:
    return {"tokens": r.usage.total, "cost_usd": r.cost_usd or 0.0, "steps": [step]}


@tracing.traced(kind="agent", name="planner")
def planner(s: ForgeState) -> dict:
    pack = build_context_pack(s["repo"], s["task"], budget_tokens=12_000)
    r = llm.chat([{"role": "system", "content": open("forgecode/prompts/planner.md").read()},
                  {"role": "user", "content": f"TASK:\n{s['task']}\n\nCONTEXT:\n{pack}"}],
                 model=pick_model("plan", attempt=0))
    return {"plan": r.text, "context_pack": pack, "attempt": 0, **_acc(r, "plan")}


@tracing.traced(kind="agent", name="implementer")
def implementer(s: ForgeState) -> dict:
    model = pick_model("edit", attempt=s.get("attempt", 0), last_failure=s.get("test_output"))
    msgs = [{"role": "system", "content": open("forgecode/prompts/implementer.md").read()},
            {"role": "user", "content": f"PLAN:\n{s['plan']}\n\nCONTEXT:\n{s['context_pack']}\n\n"
                                        f"PREVIOUS TEST OUTPUT:\n{s.get('test_output', '')}"}]
    # tool loop: the model calls read_file/search_code/write_file until it says done
    for _ in range(40):
        r = llm.chat(msgs, model=model, tools=registry.schema())
        if not r.tool_calls:
            break
        msgs.append({"role": "assistant", "content": r.text or None,
                     "tool_calls": [{"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}} for t in r.tool_calls]})
        for t in r.tool_calls:
            out = registry.call(t.name, t.arguments, agent="implementer")
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": str(out)[:8000]})
    diff = registry.call("git_diff", {"repo": s["repo"]}, agent="implementer")
    return {"diff": diff, "attempt": s.get("attempt", 0) + 1, **_acc(r, f"implement#{s.get('attempt', 0) + 1}")}


@tracing.traced(kind="agent", name="tester")
def tester(s: ForgeState) -> dict:
    out = registry.call("run_tests", {"repo": s["repo"]}, agent="tester")  # runs inside the docker sandbox
    return {"test_output": out["stdout"][-6000:], "tests_pass": out["returncode"] == 0, "steps": ["test"]}


def after_tests(s: ForgeState) -> Literal["reviewer", "implementer", "give_up"]:
    if s["tests_pass"]:
        return "reviewer"
    return "implementer" if s["attempt"] < MAX_ATTEMPTS else "give_up"


@tracing.traced(kind="agent", name="reviewer")
def reviewer(s: ForgeState) -> dict:
    # Isolation: the reviewer is NOT given plan, context pack, or implementer messages.
    r = llm.chat([{"role": "system", "content": open("forgecode/prompts/reviewer.md").read()},
                  {"role": "user", "content": f"TASK:\n{s['task']}\n\nDIFF:\n{s['diff']}\n\nTESTS:\n{s['test_output']}\n\n"
                                              "Reply with VERDICT: APPROVE or VERDICT: REJECT, then reasons."}],
                 model=pick_model("review", attempt=0))
    return {"review": r.text, "review_ok": "VERDICT: APPROVE" in r.text, **_acc(r, "review")}


def after_review(s: ForgeState) -> Literal["approval_gate", "implementer"]:
    return "approval_gate" if s["review_ok"] else "implementer"


@tracing.traced(kind="agent", name="approval_gate")
def approval_gate(s: ForgeState) -> dict:
    risk = "high" if len(s["diff"]) > 20_000 or "migration" in s["diff"] else "medium"
    decision = _gate.check({"tool": "create_patch", "risk": risk, "args": {"repo": s["repo"]}, "sandboxed": False}, agent="harness")
    return {"approval": decision, "steps": [f"approval:{decision}"]}


def after_gate(s: ForgeState) -> Literal["create_patch", "end"]:
    return "create_patch" if s["approval"] in ("auto", "approved") else "end"


@tracing.traced(kind="agent", name="create_patch")
def create_patch(s: ForgeState) -> dict:
    registry.call("create_patch", {"repo": s["repo"], "message": s["task"][:72]}, agent="harness")
    return {"steps": ["patch"]}


def give_up(s: ForgeState) -> dict:
    _audit.append({"type": "give_up", "attempts": s["attempt"], "trace_id": tracing.current_trace_id.get()})
    return {"steps": ["give_up"]}


def build_graph():
    g = StateGraph(ForgeState)
    for name, fn in [("planner", planner), ("implementer", implementer), ("tester", tester), ("reviewer", reviewer),
                     ("approval_gate", approval_gate), ("create_patch", create_patch), ("give_up", give_up)]:
        g.add_node(name, fn)
    g.set_entry_point("planner")
    g.add_edge("planner", "implementer")
    g.add_edge("implementer", "tester")
    g.add_conditional_edges("tester", after_tests)
    g.add_conditional_edges("reviewer", after_review)
    g.add_conditional_edges("approval_gate", after_gate, {"create_patch": "create_patch", "end": END})
    g.add_edge("create_patch", END)
    g.add_edge("give_up", END)
    return g


def run(repo: str, task: str, thread_id: str):
    tracing.init_tracing("forgecode")
    tracing.new_trace_id()
    db_path = settings.database_url.replace("sqlite:///", "")
    with SqliteSaver.from_conn_string(db_path) as saver:
        saver.setup()
        app = build_graph().compile(checkpointer=saver)
        return app.invoke({"repo": repo, "task": task, "steps": [], "tokens": 0, "cost_usd": 0.0},
                          config={"configurable": {"thread_id": thread_id}})


def resume(thread_id: str, decision: str):
    """`forgecode approve <thread_id>` after the operator decides; picks up at the interrupted node."""
    db_path = settings.database_url.replace("sqlite:///", "")
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = build_graph().compile(checkpointer=saver)
        return app.invoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})
