from cache import is_fresh


def test_younger_entry_is_fresh():
    assert is_fresh(100.0, 30.0, 129.0)
