from aisys.llm import ChatResult, ToolCall, Usage
from forgecode.context.evict import estimate_tokens
from forgecode.context.trajectory import summarize
from forgecode.graph import ForgeState, reviewer_messages, scoped_tool_arguments

from forgecode import graph as graph_module


def test_reviewer_context_is_isolated():
    state: ForgeState = {
        "task": "fix expiry",
        "diff": "+ return now >= expiry",
        "test_output": "1 passed",
        "plan": "PRIVATE PLAN",
        "context_pack": "PRIVATE SOURCE",
    }
    rendered = str(reviewer_messages(state))
    assert "fix expiry" in rendered and "1 passed" in rendered
    assert "PRIVATE PLAN" not in rendered and "PRIVATE SOURCE" not in rendered


def test_twenty_step_summary_stays_under_budget():
    steps = [f"step {number}: " + "diagnostic output " * 80 for number in range(20)]
    summary = summarize(steps, budget_tokens=300)
    assert estimate_tokens(summary) <= 300
    assert "step 19" in summary


def test_harness_overrides_model_supplied_repository_path():
    arguments = scoped_tool_arguments(
        "C:/trusted/task-copy",
        "read_file",
        '{"repo": "REPO", "path": "auth.py"}',
    )
    assert arguments == {"repo": "C:/trusted/task-copy", "path": "auth.py"}


def test_implementer_aggregates_every_tool_loop_model_call(monkeypatch):
    responses = iter([
        ChatResult(
            text="",
            tool_calls=[ToolCall(id="call-1", name="read_file", arguments='{"repo":"REPO","path":"auth.py"}')],
            usage=Usage(prompt_tokens=10, completion_tokens=1),
            model="test-model",
            provider="test",
            latency_ms=1.0,
            cost_usd=0.01,
        ),
        ChatResult(
            text="done",
            usage=Usage(prompt_tokens=20, completion_tokens=2),
            model="test-model",
            provider="test",
            latency_ms=1.0,
            cost_usd=0.02,
        ),
    ])
    monkeypatch.setattr(graph_module.llm, "chat", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(graph_module.registry, "schema", list)

    def fake_call(name, arguments, agent):
        assert agent == "implementer"
        if name == "read_file":
            assert arguments["repo"] == "C:/trusted/task-copy"
            return "source"
        assert name == "git_diff"
        return "diff --git a/auth.py b/auth.py"

    monkeypatch.setattr(graph_module.registry, "call", fake_call)
    state: ForgeState = {
        "repo": "C:/trusted/task-copy",
        "plan": "fix boundary",
        "context_pack": "auth.py",
        "attempt": 0,
    }

    result = graph_module.implementer(state)

    assert result["tokens"] == 33
    assert result["cost_usd"] == 0.03
