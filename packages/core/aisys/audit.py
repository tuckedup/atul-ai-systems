"""Append-only, hash-chained audit log. Any edit or deletion breaks verify().

Postgres in production; SQLite for tests. Each row: hash = sha256(prev_hash + canonical_json(event)).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any

import psycopg

DDL_PG = """
CREATE TABLE IF NOT EXISTS audit (
  seq BIGSERIAL PRIMARY KEY, ts DOUBLE PRECISION NOT NULL, event JSONB NOT NULL,
  prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
REVOKE UPDATE, DELETE ON audit FROM PUBLIC;
"""
DDL_SQLITE = "CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY, ts REAL, event TEXT, prev_hash TEXT, hash TEXT)"
GENESIS = "0" * 64


def _canon(event: dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)


def _h(prev: str, ts: float, event: dict[str, Any]) -> str:
    return hashlib.sha256(f"{prev}|{ts!r}|{_canon(event)}".encode()).hexdigest()


class AuditLog:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.sqlite = dsn.startswith("sqlite")
        if self.sqlite:
            self._sq = sqlite3.connect(dsn.replace("sqlite:///", ""), check_same_thread=False)
            self._sq.execute(DDL_SQLITE)
        else:
            with psycopg.connect(dsn) as c:
                c.execute(DDL_PG)

    def _last_hash(self, cur: Any) -> str:
        row = cur.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS

    def append(self, event: dict[str, Any]) -> str:
        ts = time.time()
        if self.sqlite:
            cur = self._sq.cursor()
            prev = self._last_hash(cur)
            h = _h(prev, ts, event)
            cur.execute("INSERT INTO audit (ts, event, prev_hash, hash) VALUES (?,?,?,?)", (ts, _canon(event), prev, h))
            self._sq.commit()
            return h
        with psycopg.connect(self.dsn) as c:
            c.execute("LOCK TABLE audit IN EXCLUSIVE MODE")  # serialize appends so the chain is linear
            prev = self._last_hash(c)
            h = _h(prev, ts, event)
            c.execute("INSERT INTO audit (ts, event, prev_hash, hash) VALUES (%s,%s,%s,%s)", (ts, _canon(event), prev, h))
        return h

    def rows(self) -> list[tuple[int, float, dict[str, Any], str, str]]:
        q = "SELECT seq, ts, event, prev_hash, hash FROM audit ORDER BY seq"
        if self.sqlite:
            raw = self._sq.execute(q).fetchall()
            return [(r[0], r[1], json.loads(r[2]), r[3], r[4]) for r in raw]
        with psycopg.connect(self.dsn) as c:
            raw = c.execute(q).fetchall()
        return [(r[0], r[1], r[2] if isinstance(r[2], dict) else json.loads(r[2]), r[3], r[4]) for r in raw]

    def verify(self) -> int | None:
        """Return seq of the first broken row, or None if the chain is intact."""
        prev = GENESIS
        for seq, ts, event, prev_hash, h in self.rows():
            if prev_hash != prev or _h(prev_hash, ts, event) != h:
                return seq
            prev = h
        return None

    def for_trace(self, trace_id: str) -> list[dict[str, Any]]:
        return [e for _, _, e, _, _ in self.rows() if e.get("trace_id") == trace_id]
