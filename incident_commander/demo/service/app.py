"""Demo service — tiny FastAPI app with a chaos flag that raises error rate."""
from __future__ import annotations

import os
import random
from fastapi import FastAPI

app = FastAPI(title="demo-service")

CHAOS_ENABLED = os.environ.get("CHAOS_ENABLED", "false").lower() == "true"
ERROR_RATE_TARGET = float(os.environ.get("ERROR_RATE_TARGET", "0.01"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/users")
def get_users() -> dict[str, list[str]]:
    if CHAOS_ENABLED and random.random() < ERROR_RATE_TARGET:
        raise RuntimeError("simulated error: dependency timeout")
    return {"users": ["alice", "bob", "charlie"]}


@app.get("/api/v1/orders")
def get_orders() -> dict[str, list[dict[str, str]]]:
    if CHAOS_ENABLED and random.random() < ERROR_RATE_TARGET:
        raise RuntimeError("simulated error: database connection refused")
    return {"orders": [{"id": "1", "status": "shipped"}, {"id": "2", "status": "pending"}]}
