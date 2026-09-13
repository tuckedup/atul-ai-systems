from windows import windows


def test_oversized_window_has_no_results():
    assert windows([1, 2], 3) == []
