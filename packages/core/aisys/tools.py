"""Tool registry: Pydantic-typed tools with a risk level, exportable as OpenAI tool JSON or as an MCP server.

    @tool(risk="medium")
    def write_file(path: str, content: str) -> str: ...

    registry.schema()          -> list of OpenAI-style tool definitions
    registry.call(name, args)  -> validated, traced, audited execution
    registry.serve_mcp()       -> exposes every tool over MCP (stdio)
"""
from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, create_model

from .audit import AuditLog
from .tracing import current_trace_id, traced

Risk = Literal["low", "medium", "high"]


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    risk: Risk
    description: str
    model: type[BaseModel]
    risk_predicate: Callable[[dict[str, Any]], Risk | None] | None = None  # escalate risk by argument

    def effective_risk(self, args: dict[str, Any]) -> Risk:
        if self.risk_predicate:
            return self.risk_predicate(args) or self.risk
        return self.risk

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.model.model_json_schema()},
        }


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    audit: AuditLog | None = None

    def register(self, t: Tool) -> None:
        self.tools[t.name] = t

    def schema(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self.tools.values()]

    @traced(kind="tool")
    def call(self, name: str, args: dict[str, Any] | str, agent: str = "unknown") -> Any:
        t = self.tools[name]
        parsed = json.loads(args) if isinstance(args, str) else args
        try:
            validated = t.model(**parsed)
        except ValidationError as e:
            raise ValueError(f"{name}: invalid arguments: {e}") from e
        result = t.fn(**validated.model_dump())
        if self.audit:
            self.audit.append({
                "type": "tool_call", "tool": name, "risk": t.effective_risk(parsed), "agent": agent,
                "args": parsed, "trace_id": current_trace_id.get(), "result_preview": str(result)[:500],
            })
        return result

    def serve_mcp(self, server_name: str = "aisys-tools") -> None:
        """Expose all tools via the official MCP SDK over stdio."""
        from mcp.server.mcpserver import MCPServer

        mcp = MCPServer(name=server_name)
        for t in self.tools.values():
            mcp.tool(name=t.name, description=t.description)(t.fn)
        import asyncio
        asyncio.run(mcp.run_stdio_async())


registry = Registry()


def tool(risk: Risk = "low", name: str | None = None,
         risk_predicate: Callable[[dict[str, Any]], Risk | None] | None = None) -> Callable:
    """Decorator: derive a Pydantic model from the signature, register with a risk level."""
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        fields = {
            p.name: (p.annotation if p.annotation is not inspect.Parameter.empty else Any,
                     ... if p.default is inspect.Parameter.empty else p.default)
            for p in sig.parameters.values()
        }
        model = create_model(f"{fn.__name__}_args", **fields)  # type: ignore[call-overload]
        registry.register(Tool(
            name=name or fn.__name__, fn=fn, risk=risk, model=model,
            description=(fn.__doc__ or "").strip(), risk_predicate=risk_predicate,
        ))
        return fn
    return deco


# Example escalation predicate used by ForgeCode: touching secrets/migrations/deps is always high risk.
SENSITIVE = ("migration", ".env", "secret", "credential", "requirements", "pyproject", "package.json", "lock")


def sensitive_path(args: dict[str, Any]) -> Risk | None:
    p = str(args.get("path", "")).lower()
    return "high" if any(s in p for s in SENSITIVE) else None
