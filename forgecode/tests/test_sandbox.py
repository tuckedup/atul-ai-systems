from pathlib import Path

from forgecode.sandbox import SubprocessSandbox

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "forgecode" / "forgebench" / "fixtures" / "py-auth"


def test_command_runs_in_fixture_copy_with_timeout():
    with SubprocessSandbox(FIXTURE, ROOT, timeout_s=2, memory_mb=128) as sandbox:
        result = sandbox.run("python -c \"from pathlib import Path; print(Path.cwd())\"")
        assert result.returncode == 0
        assert Path(result.stdout.strip()).resolve() == sandbox.workdir.resolve()
        assert sandbox.workdir != ROOT
        timeout = sandbox.run("python -c \"import time; time.sleep(2)\"", timeout_s=0.05)
        assert timeout.timed_out


def test_repository_root_is_rejected():
    try:
        SubprocessSandbox(ROOT, ROOT)
    except ValueError as error:
        assert "repository root" in str(error)
    else:
        raise AssertionError("root must be rejected")


def test_tool_round_trip_runs_inside_task_copy():
    import forgecode.tools  # noqa: F401
    from aisys.tools import registry

    with SubprocessSandbox(FIXTURE, ROOT) as sandbox:
        result = registry.call(
            "run_terminal",
            {"repo": str(sandbox.workdir), "command": "python -c \"print('inside')\""},
            agent="test",
        )
        assert result["returncode"] == 0 and result["stdout"].strip() == "inside"
