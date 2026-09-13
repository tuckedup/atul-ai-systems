from flags import enabled


def test_false_spellings_are_disabled():
    assert not enabled(False)
    assert not enabled("false")
    assert not enabled("0")
    assert not enabled("off")
