"""No-op success is forbidden; the inspection CLI never starts model calls."""
from evalops import offline


def test_default_command_reports_blocked_instead_of_success(capsys):
    assert offline.main([]) == 2
    assert "BLOCKED" in capsys.readouterr().out


def test_missing_artifact_reports_blocked(tmp_path, capsys):
    assert offline.main(["--check-artifact", str(tmp_path / "missing.json")]) == 2
    assert "BLOCKED" in capsys.readouterr().out
