from config import merge_config


def test_default_keys_are_retained():
    assert merge_config({"timeout": 30}, {}) == {"timeout": 30}
