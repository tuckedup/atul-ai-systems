from config import merge_config


def test_overrides_win_without_mutating_inputs():
    defaults = {"timeout": 30, "retries": 2}
    overrides = {"timeout": 5}
    assert merge_config(defaults, overrides) == {"timeout": 5, "retries": 2}
    assert defaults == {"timeout": 30, "retries": 2}
    assert overrides == {"timeout": 5}
