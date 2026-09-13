"""Prometheus HTTP integration; live alert proof is Docker-blocked on this host."""
from typing import cast

import httpx


def query(base_url: str, expression: str) -> dict[str, object]:
    response = httpx.get(f"{base_url.rstrip('/')}/api/v1/query", params={"query": expression}, timeout=10)
    response.raise_for_status()
    return cast(dict[str, object], response.json())
