"""Human-in-the-loop approval gate.

Policy maps (risk, predicate) -> auto | require_approval | deny.
Pending approvals persist in SQLite (or Postgres) so a LangGraph run can `interrupt()`, the process can die,
and `resume` picks up exactly where it stopped once an operator decides.

    policy = ApprovalPolicy.default()
    store  = ApprovalStore(settings.database_url)
    gate   = Gate(policy, store, audit)

    # inside a LangGraph node:
    verdict = gate.check({"tool": "run_terminal", "args": {...}, "risk": "high", "agent": "remediation"})
    if verdict == "deny": ...
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.types import interrupt

from .audit import AuditLog
from .tracing import current_trace_id, traced

Verdict = Literal["auto", "require_approval", "deny"]
Decision = Literal["approved", "rejected"]

DDL_PG = """
CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY, trace_id TEXT, agent TEXT, action JSONB NOT NULL, risk TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', approver TEXT, reason TEXT,
  created_at TIMESTAMPTZ DEFAULT now(), decided_at TIMESTAMPTZ);
"""
DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY, trace_id TEXT, agent TEXT, action TEXT, risk TEXT,
  status TEXT DEFAULT 'pending', approver TEXT, reason TEXT,
  created_at REAL, decided_at REAL
);
"""


@dataclass
class Rule:
    verdict: Verdict
    risk: str | None = None
    when: Callable[[dict[str, Any]], bool] | None = None

    def matches(self, action: dict[str, Any]) -> bool:
        return (self.risk is None or action.get("risk") == self.risk) and (self.when is None or self.when(action))


@dataclass
class ApprovalPolicy:
    rules: list[Rule] = field(default_factory=list)

    def evaluate(self, action: dict[str, Any]) -> Verdict:
        for r in self.rules:
            if r.matches(action):
                return r.verdict
        return "require_approval"  # fail closed

    @classmethod
    def default(cls) -> ApprovalPolicy:
        return cls([
            Rule("deny", when=lambda a: a.get("tool") in {"drop_database", "delete_namespace"}),
            Rule("auto", risk="low"),
            Rule("auto", risk="medium", when=lambda a: a.get("sandboxed", False)),
            Rule("require_approval", risk="medium"),
            Rule("require_approval", risk="high"),
        ])


class ApprovalStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.sqlite = not dsn.startswith("postgresql://") and not dsn.startswith("postgres://")
        if self.sqlite:
            db_path = dsn.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self._sq = sqlite3.connect(db_path, check_same_thread=False)
            self._sq.execute(DDL_SQLITE)
        else:
            import psycopg
            with psycopg.connect(dsn) as c:
                c.execute(DDL_PG)

    def request(self, action: dict[str, Any], agent: str) -> str:
        aid = uuid.uuid4().hex[:12]
        if self.sqlite:
            self._sq.execute(
                "INSERT INTO approvals (id, trace_id, agent, action, risk, created_at) VALUES (?,?,?,?,?,?)",
                (aid, current_trace_id.get(), agent, json.dumps(action), action.get("risk", "high"), __import__('time').time()),
            )
            self._sq.commit()
        else:
            import psycopg
            with psycopg.connect(self.dsn) as c:
                c.execute(
                    "INSERT INTO approvals (id, trace_id, agent, action, risk) VALUES (%s,%s,%s,%s,%s)",
                    (aid, current_trace_id.get(), agent, json.dumps(action), action.get("risk", "high")),
                )
        return aid

    def decide(self, aid: str, approver: str, decision: Decision, reason: str = "") -> None:
        if self.sqlite:
            self._sq.execute(
                "UPDATE approvals SET status=?, approver=?, reason=?, decided_at=? WHERE id=? AND status='pending'",
                (decision, approver, reason, __import__('time').time(), aid),
            )
            self._sq.commit()
        else:
            import psycopg
            with psycopg.connect(self.dsn) as c:
                c.execute(
                    "UPDATE approvals SET status=%s, approver=%s, reason=%s, decided_at=now() WHERE id=%s AND status='pending'",
                    (decision, approver, reason, aid),
                )

    def status(self, aid: str) -> tuple[str, str | None]:
        if self.sqlite:
            row = self._sq.execute("SELECT status, approver FROM approvals WHERE id=?", (aid,)).fetchone()
        else:
            import psycopg
            with psycopg.connect(self.dsn) as c:
                row = c.execute("SELECT status, approver FROM approvals WHERE id=%s", (aid,)).fetchone()
        return (row[0], row[1]) if row else ("missing", None)

    def pending(self) -> list[dict[str, Any]]:
        if self.sqlite:
            rows = self._sq.execute(
                "SELECT id, trace_id, agent, action, risk, created_at FROM approvals WHERE status='pending' ORDER BY created_at"
            ).fetchall()
        else:
            import psycopg
            with psycopg.connect(self.dsn) as c:
                rows = c.execute(
                    "SELECT id, trace_id, agent, action, risk, created_at FROM approvals WHERE status='pending' ORDER BY created_at"
                ).fetchall()
        return [dict(id=r[0], trace_id=r[1], agent=r[2], action=r[3], risk=r[4], created_at=r[5]) for r in rows]


@dataclass
class Gate:
    policy: ApprovalPolicy
    store: ApprovalStore
    audit: AuditLog | None = None

    @traced(kind="approval")
    def check(self, action: dict[str, Any], agent: str = "unknown") -> Decision | Literal["auto", "denied"]:
        """Call from inside a LangGraph node. Auto-runs, denies, or interrupts until decided."""
        verdict = self.policy.evaluate(action)
        self._log("policy_verdict", action, agent, verdict=verdict)
        if verdict == "auto":
            return "auto"
        if verdict == "deny":
            return "denied"
        aid = self.store.request(action, agent)
        self._log("approval_requested", action, agent, approval_id=aid)
        # First execution: raises GraphInterrupt, state is checkpointed with the approval id.
        # On resume with Command(resume=...), returns the decision payload.
        resumed = interrupt({"approval_id": aid, "action": action, "agent": agent})
        decision: Decision = resumed if resumed in ("approved", "rejected") else self.store.status(aid)[0]  # type: ignore[assignment]
        self._log("approval_decided", action, agent, approval_id=aid, decision=decision)
        return decision

    def _log(self, event: str, action: dict[str, Any], agent: str, **kw: Any) -> None:
        if self.audit:
            self.audit.append({"type": event, "action": action, "agent": agent, "trace_id": current_trace_id.get(), **kw})
