"""Long-term memory — cross-incident store with similarity search for past incidents."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any

from aisys.tracing import traced


_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", ".local", "ic_memory.db")

DDL = """
CREATE TABLE IF NOT EXISTS long_term (
    incident_id TEXT PRIMARY KEY,
    service TEXT NOT NULL,
    alert_hash TEXT NOT NULL,
    alert_summary TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    resolution TEXT NOT NULL,
    risk_score REAL,
    tags TEXT NOT NULL,
    embedding TEXT,
    created_at REAL NOT NULL
);
"""


def _alert_hash(alert: dict[str, Any]) -> str:
    """Create a deterministic hash from alert for similarity matching."""
    key_parts = [
        alert.get("alertname", ""),
        alert.get("service", ""),
        alert.get("severity", ""),
    ]
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _simple_embedding(text: str) -> list[float]:
    """Create a simple bag-of-words embedding for similarity search.
    
    In production, use a real embedding model. This is sufficient for
    Degraded Mode testing and demonstrates the recall mechanism.
    """
    words = text.lower().split()
    # Use a fixed vocabulary of common incident-related terms
    vocab = [
        "error", "timeout", "connection", "database", "service", "deploy",
        "memory", "cpu", "disk", "network", "payment", "user", "api",
        "restart", "rollback", "scale", "fix", "regression", "failure",
        "latency", "rate", "high", "low", "critical", "warning",
    ]
    embedding = [0.0] * len(vocab)
    for word in words:
        for i, v in enumerate(vocab):
            if v in word:
                embedding[i] += 1.0
    # Normalize
    total = sum(embedding)
    if total > 0:
        embedding = [e / total for e in embedding]
    return embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LongTermMemory:
    """Cross-incident memory store with similarity-based recall."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or _DEFAULT_DB
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(DDL)
        self._conn.commit()

    @traced(kind="tool")
    def store_incident(
        self,
        incident_id: str,
        service: str,
        alert: dict[str, Any],
        root_cause: str,
        hypothesis: str,
        resolution: str,
        risk_score: float = 0.0,
        tags: list[str] | None = None,
    ) -> None:
        """Store a completed incident for future recall."""
        embedding = _simple_embedding(f"{alert.get('summary', '')} {root_cause} {hypothesis}")
        self._conn.execute(
            """INSERT OR REPLACE INTO long_term
            (incident_id, service, alert_hash, alert_summary, root_cause, hypothesis,
             resolution, risk_score, tags, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                incident_id,
                service,
                _alert_hash(alert),
                alert.get("summary", ""),
                root_cause,
                hypothesis,
                resolution,
                risk_score,
                json.dumps(tags or []),
                json.dumps(embedding),
                time.time(),
            ),
        )
        self._conn.commit()

    @traced(kind="tool")
    def recall_similar(
        self, alert: dict[str, Any], service: str | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Recall similar past incidents based on alert similarity."""
        query_embedding = _simple_embedding(alert.get("summary", "") + " " + alert.get("alertname", ""))
        alert_h = _alert_hash(alert)

        # Get all incidents (or filter by service)
        if service:
            rows = self._conn.execute(
                "SELECT * FROM long_term WHERE service = ? ORDER BY created_at DESC",
                (service,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM long_term ORDER BY created_at DESC"
            ).fetchall()

        # Compute similarity and rank
        results = []
        for row in rows:
            incident_id, svc, a_hash, summary, root_cause, hypothesis, resolution, risk, tags_json, emb_json, ts = row
            stored_embedding = json.loads(emb_json) if emb_json else []
            similarity = _cosine_similarity(query_embedding, stored_embedding)

            # Boost exact alert hash matches
            if a_hash == alert_h:
                similarity = min(1.0, similarity + 0.5)

            results.append({
                "incident_id": incident_id,
                "service": svc,
                "alert_summary": summary,
                "root_cause": root_cause,
                "hypothesis": hypothesis,
                "resolution": resolution,
                "risk_score": risk,
                "tags": json.loads(tags_json),
                "similarity": round(similarity, 4),
                "created_at": ts,
            })

        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    @traced(kind="tool")
    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        """Get a specific incident by ID."""
        row = self._conn.execute(
            "SELECT * FROM long_term WHERE incident_id = ?", (incident_id,)
        ).fetchone()
        if row is None:
            return None
        incident_id, svc, a_hash, summary, root_cause, hypothesis, resolution, risk, tags_json, emb_json, ts = row
        return {
            "incident_id": incident_id,
            "service": svc,
            "alert_summary": summary,
            "root_cause": root_cause,
            "hypothesis": hypothesis,
            "resolution": resolution,
            "risk_score": risk,
            "tags": json.loads(tags_json),
            "created_at": ts,
        }

    @traced(kind="tool")
    def list_incidents(self, service: str | None = None) -> list[dict[str, Any]]:
        """List all stored incidents."""
        if service:
            rows = self._conn.execute(
                "SELECT incident_id, service, alert_summary, created_at FROM long_term WHERE service = ? ORDER BY created_at DESC",
                (service,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT incident_id, service, alert_summary, created_at FROM long_term ORDER BY created_at DESC"
            ).fetchall()
        return [{"incident_id": r[0], "service": r[1], "alert_summary": r[2], "created_at": r[3]} for r in rows]

    def clear(self) -> None:
        """Clear all long-term memory."""
        self._conn.execute("DELETE FROM long_term")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
