from uploads import allowed_upload


def test_uppercase_extension_matches_allowed_lowercase():
    assert allowed_upload("REPORT.PDF", {".pdf", ".csv"})
