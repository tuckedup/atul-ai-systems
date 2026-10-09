"""Optional bearer-key authentication that derives the tenant from the key, so `x-tenant-id` cannot be spoofed.

Without it the gateway trusts `x-tenant-id` from an authenticated edge (documented, unchanged). With it, a request must carry `Authorization: Bearer <key>`; the tenant is the one registered for that key,
and an `x-tenant-id` naming a different tenant is refused. Only SHA-256 digests of keys are stored (configuration never contains a usable key). Comparison is constant-time over every registered digest.
Keys must be high-entropy random secrets (a bare digest is not a password hash and offers no protection for guessable keys). This is authentication for tenant isolation, not authorisation, rate limiting,
key rotation or TLS: terminate TLS at the edge and rotate keys by redeploying the digest map.

Create a digest without printing the key: `python -m gateway.auth` reads the key from standard input.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import sys
from collections.abc import Mapping

_FORMAT = re.compile(r"^sha256:[0-9a-f]{64}$")


class AuthConfigError(ValueError):
    pass


def hash_key(key: str) -> str:
    if not isinstance(key, str) or len(key) < 24:
        raise AuthConfigError("API keys must be random strings of at least 24 characters")
    return "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()


class TenantAuth:
    def __init__(self, key_hashes: Mapping[str, str]) -> None:
        if not key_hashes:
            raise AuthConfigError("authentication needs at least one tenant key digest")
        seen: set[str] = set()
        for tenant, digest in key_hashes.items():
            if not isinstance(tenant, str) or not tenant.strip():
                raise AuthConfigError("tenant ids must be non-empty strings")
            if not isinstance(digest, str) or not _FORMAT.match(digest):
                raise AuthConfigError(f"tenant {tenant!r}: key digest must look like 'sha256:<64 hex>'")
            if digest in seen:
                raise AuthConfigError("two tenants share one key digest")
            seen.add(digest)
        self._digests = dict(key_hashes)

    @property
    def tenants(self) -> frozenset[str]:
        return frozenset(self._digests)

    def resolve(self, authorization: str | None) -> str | None:
        """The tenant for a 'Bearer <key>' header, or None. Never raises on bad input; compares against every digest."""
        if not authorization:
            return None
        scheme, _, key = authorization.partition(" ")
        if scheme.lower() != "bearer" or not key.strip():
            return None
        presented = "sha256:" + hashlib.sha256(key.strip().encode("utf-8")).hexdigest()
        match: str | None = None
        for tenant, digest in self._digests.items():
            if hmac.compare_digest(presented, digest):
                match = tenant
        return match


def main() -> int:
    key = sys.stdin.readline().strip()
    print(hash_key(key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
