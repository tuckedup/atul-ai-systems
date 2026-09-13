def is_fresh(stored_at: float, ttl_seconds: float, now: float) -> bool:
    """Return whether a cached value may still be served."""
    # Injected bug: an entry is stale when age is exactly the TTL.
    return now - stored_at <= ttl_seconds
