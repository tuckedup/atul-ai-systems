from retry import retry_delays


def test_sequence_starts_at_configured_base():
    assert retry_delays(1, 4, 10) == [1, 2, 4, 8]
