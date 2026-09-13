from datetime import datetime, timedelta, timezone


def token_expired(expires_at: datetime, now: datetime | None = None) -> bool:
    """Return whether a token needs refreshing."""
    current = now or datetime.now(timezone.utc)
    # Injected bug: equality must count as expired.
    return current > expires_at


def refresh_deadline(now: datetime) -> datetime:
    return now + timedelta(minutes=30)

