from uploads import allowed_upload


def test_lowercase_extension_is_allowed():
    assert allowed_upload("report.pdf", {".pdf", ".csv"})
