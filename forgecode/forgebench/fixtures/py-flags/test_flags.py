from flags import enabled


def test_true_values_are_enabled():
    assert enabled(True)
    assert enabled("true")
