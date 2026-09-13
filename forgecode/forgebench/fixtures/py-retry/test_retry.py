from retry import retry_delays


def test_cap_is_respected():
    assert retry_delays(5, 3, 6)[-1] == 6
