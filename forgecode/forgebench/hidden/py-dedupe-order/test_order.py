from dedupe import unique


def test_first_occurrence_order_is_preserved():
    assert unique(["beta", "alpha", "beta", "gamma", "alpha"]) == ["beta", "alpha", "gamma"]
