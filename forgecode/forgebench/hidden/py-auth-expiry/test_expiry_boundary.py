from datetime import datetime, timezone

from auth import token_expired


def test_token_is_expired_at_exact_expiry_boundary():
    boundary = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert token_expired(boundary, boundary)
