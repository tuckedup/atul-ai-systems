from average import mean


def test_fractional_mean_is_not_truncated():
    assert mean([1, 2]) == 1.5
