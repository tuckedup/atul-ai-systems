from datetime import datetime, timezone

from auth import token_expired


def test_past_token_is_expired():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert token_expired(datetime(2025, 1, 1, tzinfo=timezone.utc), now)

