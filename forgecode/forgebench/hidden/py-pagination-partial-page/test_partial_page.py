from pagination import page_count


def test_partial_page_is_counted():
    assert page_count(21, 10) == 3
