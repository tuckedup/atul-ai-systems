from dedupe import unique


def test_duplicates_are_removed():
    assert unique(["same", "same"]) == ["same"]
