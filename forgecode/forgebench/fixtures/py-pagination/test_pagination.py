from pagination import page_count


def test_exact_pages():
    assert page_count(20, 10) == 2
