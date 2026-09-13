"""Tests for aisys.tools — verify tool registration, schema, and MCP round trip."""
import asyncio
import os
import sys
import tempfile
import textwrap

import pytest

from aisys.tools import Tool, registry, sensitive_path, tool


def test_tool_decorator_registers() -> None:
    """@tool decorator registers a tool with correct metadata."""

    @tool(risk="low")
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert "add" in registry.tools
    t = registry.tools["add"]
    assert t.risk == "low"
    assert t.description == "Add two numbers."


def test_schema_produces_openai_format() -> None:
    """registry.schema() returns a list of OpenAI-style tool definitions."""

    @tool(risk="medium", name="greet")
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}"

    schemas = registry.schema()
    assert len(schemas) >= 1
    greet_schema = next(s for s in schemas if s["function"]["name"] == "greet")
    assert greet_schema["type"] == "function"
    assert "parameters" in greet_schema["function"]
    props = greet_schema["function"]["parameters"]["properties"]
    assert "name" in props


def test_registry_call_validates_args() -> None:
    """registry.call raises on invalid args."""

    @tool(risk="low")
    def multiply(x: int, y: int) -> int:
        """Multiply."""
        return x * y

    with pytest.raises(ValueError, match="invalid arguments"):
        registry.call("multiply", {"x": "not_an_int", "y": 2})


def test_registry_call_returns_result() -> None:
    """registry.call executes the tool and returns the result."""

    @tool(risk="low")
    def subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    result = registry.call("subtract", {"a": 10, "b": 3})
    assert result == 7


def test_sensitive_path_escalation() -> None:
    """sensitive_path predicate escalates risk for sensitive paths."""
    assert sensitive_path({"path": "src/migration/001.sql"}) == "high"
    assert sensitive_path({"path": ".env.production"}) == "high"
    assert sensitive_path({"path": "src/main.py"}) is None


def test_effective_risk_with_predicate() -> None:
    """Tool.effective_risk uses predicate when provided."""
    from pydantic import BaseModel

    class EmptyModel(BaseModel):
        pass

    t = Tool(
        name="test",
        fn=lambda: None,
        risk="low",
        description="",
        model=EmptyModel,
        risk_predicate=lambda args: "high" if args.get("dangerous") else None,
    )
    assert t.effective_risk({"dangerous": True}) == "high"
    assert t.effective_risk({}) == "low"


# MCP round-trip test using subprocess
_MCP_SERVER_SCRIPT = textwrap.dedent("""\
    import sys
    sys.path.insert(0, r"{core_path}")
    from aisys.tools import tool, registry

    @tool(risk="low")
    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    registry.serve_mcp("test-server")
""")


def test_mcp_round_trip() -> None:
    """A registered tool is callable via MCP client round trip."""
    core_path = os.path.join(os.path.dirname(__file__), "..")
    script = _MCP_SERVER_SCRIPT.format(core_path=os.path.abspath(core_path))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=os.path.abspath(core_path)) as f:
        f.write(script)
        script_path = f.name

    try:
        async def run_client() -> None:
            from mcp import ClientSession, stdio_client
            from mcp.client.stdio import StdioServerParameters

            params = StdioServerParameters(
                command=sys.executable,
                args=[script_path],
            )
            async with (
                stdio_client(params) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()

                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                assert "add" in tool_names

                result = await session.call_tool(
                    "add",
                    arguments={"a": 3, "b": 4},
                )
                assert len(result.content) == 1
                first = result.content[0]
                assert hasattr(first, "text")
                assert "7" in first.text

        asyncio.run(run_client())
    finally:
        os.unlink(script_path)
