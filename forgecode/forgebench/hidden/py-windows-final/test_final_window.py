from windows import windows


def test_final_valid_window_is_included():
    assert windows([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]
