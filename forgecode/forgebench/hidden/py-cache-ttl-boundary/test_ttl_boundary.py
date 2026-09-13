from cache import is_fresh


def test_entry_expires_at_exact_ttl():
    assert not is_fresh(100.0, 30.0, 130.0)
