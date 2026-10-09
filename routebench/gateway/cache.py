"""Exact and conservative prefix response cache over the shared cache interface."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from aisys.cache import Cache


class ResponseCache:
    def __init__(self, cache: Cache):
        self.cache = cache
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        payload: dict[str, Any],
        *,
        tenant: str = "",
        policy_fingerprint: str = "",
    ) -> str:
        # A response is valid only for both the authenticated billing identity and
        # the exact routing rollout.  Including these values prevents one tenant
        # from observing another's cached answer and prevents a weight/policy
        # change from continuing to serve stale backend choices.
        namespaced = {
            "tenant": tenant,
            "policy_fingerprint": policy_fingerprint,
            "payload": payload,
        }
        stable = json.dumps(namespaced, sort_keys=True, separators=(",", ":"), default=str)
        return "response:" + hashlib.sha256(stable.encode()).hexdigest()

    def get(
        self,
        payload: dict[str, Any],
        *,
        tenant: str = "",
        policy_fingerprint: str = "",
    ) -> dict[str, Any] | None:
        value = self.cache.get(self.key(payload, tenant=tenant, policy_fingerprint=policy_fingerprint))
        if isinstance(value, dict):
            self.hits += 1
            return value
        self.misses += 1
        return None

    def set(
        self,
        payload: dict[str, Any],
        value: dict[str, Any],
        ttl_s: float = 300,
        *,
        tenant: str = "",
        policy_fingerprint: str = "",
    ) -> None:
        self.cache.set(
            self.key(payload, tenant=tenant, policy_fingerprint=policy_fingerprint),
            value,
            ttl_s,
        )

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.hits + self.misses)

