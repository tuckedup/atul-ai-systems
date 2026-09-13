"""Per-task subprocess isolation for hosts without a container runtime."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import psutil  # type: ignore[import-untyped]


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    memory_exceeded: bool = False


class SubprocessSandbox:
    """Copy a fixture, pin cwd to it, and enforce wall-clock and RSS limits."""

    def __init__(
        self,
        fixture: str | Path,
        workspace_root: str | Path,
        timeout_s: float = 60.0,
        memory_mb: int = 512,
    ) -> None:
        self.fixture = Path(fixture).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.timeout_s = timeout_s
        self.memory_bytes = memory_mb * 1024 * 1024
        if not self.fixture.is_dir():
            raise ValueError(f"fixture is not a directory: {self.fixture}")
        if self.fixture == self.workspace_root:
            raise ValueError("the repository root cannot be used as a task fixture")
        task_root = self.workspace_root / ".local" / "forgecode" / "tasks"
        task_root.mkdir(parents=True, exist_ok=True)
        self.workdir = task_root / uuid.uuid4().hex
        shutil.copytree(self.fixture, self.workdir)
        subprocess.run(["git", "init", "-q"], cwd=self.workdir, check=True)
        subprocess.run(["git", "config", "user.email", "forgecode@local"], cwd=self.workdir, check=True)
        subprocess.run(["git", "config", "user.name", "ForgeCode"], cwd=self.workdir, check=True)
        subprocess.run(["git", "add", "."], cwd=self.workdir, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture baseline"], cwd=self.workdir, check=True)

    def _kill_tree(self, process: subprocess.Popen[str]) -> None:
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.Error:
            process.kill()

    def run(self, command: str, timeout_s: float | None = None) -> CommandResult:
        resolved = self.workdir.resolve()
        if resolved == self.workspace_root or self.workspace_root not in resolved.parents:
            raise RuntimeError("sandbox cwd escaped the task workspace")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=resolved,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=flags,
        )
        memory_exceeded = threading.Event()
        finished = threading.Event()

        def watch_memory() -> None:
            while not finished.wait(0.02):
                try:
                    root = psutil.Process(process.pid)
                    rss = root.memory_info().rss + sum(
                        child.memory_info().rss for child in root.children(recursive=True)
                    )
                    if rss > self.memory_bytes:
                        memory_exceeded.set()
                        self._kill_tree(process)
                        return
                except psutil.Error:
                    return

        watcher = threading.Thread(target=watch_memory, daemon=True)
        watcher.start()
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_s or self.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(process)
            stdout, stderr = process.communicate()
        finally:
            finished.set()
            watcher.join(timeout=0.2)
        return CommandResult(
            command,
            process.returncode,
            stdout,
            stderr,
            timed_out=timed_out,
            memory_exceeded=memory_exceeded.is_set(),
        )

    def cleanup(self) -> None:
        task_root = (self.workspace_root / ".local" / "forgecode" / "tasks").resolve()
        target = self.workdir.resolve()
        if target == task_root or task_root not in target.parents:
            raise RuntimeError("refusing to remove a path outside the task directory")
        if target.exists():
            def make_writable(function: Any, path: str, _: Any) -> None:
                Path(path).chmod(stat.S_IWRITE)
                function(path)

            shutil.rmtree(target, onexc=make_writable)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()
