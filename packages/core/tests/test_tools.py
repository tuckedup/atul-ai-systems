"""M4 verification: tools register with risk, schema() returns OpenAI tool JSON,
and registry.call() validates + executes + audits."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aisys.tools import registry, tool, Tool, Risk


# ---- reset between tests ----
@pytest.fixture(autouse=True)
def _clear():
    registry.tools.clear()
    registry.audit = None
    yield
    registry.tools.clear()
    registry.audit = None


def test_tool_decorator_registers():
    @tool(risk="low")
    def read_file(path: str) -> str:
        """Read a file."""
        return f"content of {path}"

    assert "read_file" in registry.tools
    t = registry.tools["read_file"]
    assert t.risk == "low"
    assert t.name == "read_file"
    assert t.description == "Read a file."
    assert t.fn(1) == "content of 1"


def test_schema_returns_openai_tool_defs():
    @tool(risk="medium")
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        return "ok"

    schema = registry.schema()
    assert len(schema) == 1
    fn = schema[0]["function"]
    assert fn["name"] == "write_file"
    assert fn["description"] == "Write a file."
    assert "properties" in fn["parameters"]
    assert "path" in fn["parameters"]["properties"]
    assert "content" in fn["parameters"]["properties"]


def test_call_validates_and_executes():
    @tool(risk="low")
    def echo(value: str) -> str:
        return value.upper()

    out = registry.call("echo", {"value": "hello"})
    assert out == "HELLO"


def test_call_rejects_invalid_args():
    @tool(risk="low")
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(ValueError, match="invalid arguments"):
        registry.call("add", {"a": "not-int", "b": 2})


def test_call_with_str_args():
    @tool(risk="low")
    def greet(name: str) -> str:
        return f"hi {name}"

    out = registry.call("greet", json.dumps({"name": "alice"}))
    assert out == "hi alice"


def test_effective_risk_with_predicate():
    from aisys.tools import sensitive_path

    @tool(risk="low", risk_predicate=sensitive_path)
    def write_file(path: str, content: str) -> str:
        return "ok"

    # normal path stays low
    assert registry.tools["write_file"].effective_risk({"path": "src/app.py"}) == "low"
    # sensitive path escalates
    assert registry.tools["write_file"].effective_risk({"path": "migrations/001.sql"}) == "high"


def test_audit_row_emitted_when_audit_attached(tmp_path):
    from aisys.audit import AuditLog

    log = AuditLog(f"sqlite:///{tmp_path}/a.db")
    registry.audit = log

    @tool(risk="low")
    def noop(x: str) -> str:
        return x

    out = registry.call("noop", {"x": "y"})
    assert out == "y"
    rows = log.rows()
    assert rows
    last = rows[-1]
    assert last[2]["type"] == "tool_call"
    assert last[2]["tool"] == "noop"
    assert last[2]["risk"] == "low"
    assert last[2]["result_preview"] == "y"


def test_tool_model_round_trip(tmp_path):
    """One registered tool is callable via a fake MCP round trip: schema -> parse -> call."""
    from aisys.tools import registry as reg

    @tool(risk="low")
    def ping(message: str) -> str:
        """reply with the message"""
        return message

    reg.audit = None
    schema = reg.schema()
    assert schema
    tool_def = schema[0]
    # parse the Pydantic model from the schema — this is what an MCP client would do
    model = reg.tools["ping"].model
    validated = model.model_validate({"message": "hello"})
    assert validated.message == "hello"
    result = reg.tools["ping"].fn(**validated.model_dump())
    assert result == "hello"
