"""Deterministic pass/fail oracles.

These produce the ground-truth label for the task families whose correctness is objectively
checkable: run the hidden tests, compare the final answer, compare the result set, compare the
extracted object, compare the function call. The oracle is *independent of the judge under
test* -- it never calls a model -- which is what makes it admissible as a label source.

Provenance honesty: an oracle label is not a human reading the response. It is a human-authored
ground truth (the hidden tests, the gold SQL, the gold final answer) mechanically applied. That
is recorded as `LabelProvenance.GOLD_ORACLE` and reported as its own track, separate from the
`HUMAN_EXPERT` and `HUMAN_LOCAL` labels that back the headline claim. Conflating the two would
be the same category of overclaim as the fixture kappa this project is replacing.

Subprocess policy (per the standing infrastructure amendment in DECISIONS.md): every execution
runs in a per-case temporary directory, never the repository root, with a hard wall-clock
timeout and a memory cap where the platform supports one. On POSIX the cap is `resource.
setrlimit(RLIMIT_AS, ...)`, applied in the child via `preexec_fn`. `resource.setrlimit` does
not exist on Windows, so there the cap is enforced with a Win32 **job object**: the child is
assigned to a job created with `JOB_OBJECT_LIMIT_PROCESS_MEMORY` (capping its committed memory
at `MEM_LIMIT_BYTES`) and `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (so a child that somehow escapes
supervision still dies when the job handle is closed). If the job object cannot be created or
the child cannot be assigned to it -- permissions, an existing job on the process, anything --
the oracle does NOT fall back to running uncapped: it raises `SandboxUnavailable`, which
`code_oracle` turns into an undecided (`label=None`) result. An undecided case is dropped by
the corpus builder; a silently-uncapped execution is not an acceptable substitute. This is
resource limiting, not isolation -- fixtures must stay curated.
"""
from __future__ import annotations

import ast
import ctypes
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MEM_LIMIT_BYTES = 1024 * 1024 * 1024  # 1 GiB
DEFAULT_TIMEOUT_S = 20


class SandboxUnavailable(RuntimeError):
    """The platform memory-cap mechanism (job object on Windows, rlimit on POSIX) could not be
    established for a subprocess call. Callers must treat this as an undecided result -- never
    proceed to run the child without the cap."""


@dataclass(frozen=True)
class OracleResult:
    """`label` is None when the oracle could not decide; that case is dropped, never guessed."""

    label: int | None
    detail: str
    method: str
    stdout: str = ""
    stderr: str = ""

    @property
    def decided(self) -> bool:
        return self.label is not None


# ---------------------------------------------------------------- code extraction


def extract_code(text: str) -> str:
    """Pull Python source out of a model response: fenced block if present, else the whole text."""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fences:
        return max(fences, key=len).strip()
    return text.strip()


def _preexec() -> Any:
    """Address-space cap on POSIX. Returns None on Windows, where the API does not exist and
    the memory cap is instead enforced by a job object -- see `_run_python_windows` below."""
    if os.name != "posix":
        return None

    def limit() -> None:
        # Guarded by os.name == 'posix' above; the stubs are platform-specific.
        import resource

        resource.setrlimit(  # type: ignore[attr-defined]
            resource.RLIMIT_AS,  # type: ignore[attr-defined]
            (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES),
        )

    return limit


# ---------------------------------------------------------------- Windows job-object memory cap

# JOBOBJECT_EXTENDED_LIMIT_INFORMATION.BasicLimitInformation.LimitFlags bits (winnt.h).
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
# JOBOBJECTINFOCLASS enum value for JobObjectExtendedLimitInformation (winnt.h).
_JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


def _win_job_structs() -> tuple[type, type]:
    """Build the ctypes structs for `SetInformationJobObject` on demand, so importing this
    module off Windows never touches `ctypes.wintypes`.

    Layout (winnt.h), verified empirically against `ctypes.sizeof` (64 / 48 / 144 bytes):

        JOBOBJECT_BASIC_LIMIT_INFORMATION {
            LARGE_INTEGER PerProcessUserTimeLimit;   // 8 bytes
            LARGE_INTEGER PerJobUserTimeLimit;       // 8 bytes
            DWORD         LimitFlags;                // 4 bytes (+4 pad, next field is 8-aligned)
            SIZE_T        MinimumWorkingSetSize;      // 8 bytes
            SIZE_T        MaximumWorkingSetSize;      // 8 bytes
            DWORD         ActiveProcessLimit;         // 4 bytes (+4 pad)
            ULONG_PTR     Affinity;                   // 8 bytes
            DWORD         PriorityClass;               // 4 bytes
            DWORD         SchedulingClass;             // 4 bytes
        }                                             // = 64 bytes on x64
        IO_COUNTERS { ULONGLONG x6; }                  // = 48 bytes
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
            JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            IO_COUNTERS IoInfo;
            SIZE_T ProcessMemoryLimit;
            SIZE_T JobMemoryLimit;
            SIZE_T PeakProcessMemoryUsed;
            SIZE_T PeakJobMemoryUsed;
        }                                             // = 64 + 48 + 32 = 144 bytes on x64

    Getting the IO_COUNTERS block in the middle wrong (e.g. omitting it) silently misaligns
    `ProcessMemoryLimit` onto other fields and the cap does nothing -- ctypes' default
    (natural/MSVC) struct alignment reproduces this layout exactly, which is why the class
    bodies below list fields in this same order rather than packing memory fields first.
    """
    from ctypes import wintypes

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return _BasicLimitInformation, _ExtendedLimitInformation


def _create_memory_capped_job(mem_limit_bytes: int) -> int:
    """Create a job object that kills its processes when closed and caps each process's
    committed memory at `mem_limit_bytes`. Raises `SandboxUnavailable` on any failure -- the
    caller must not proceed to run a child uncapped."""
    _, extended_cls = _win_job_structs()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]

    hjob = kernel32.CreateJobObjectW(None, None)
    if not hjob:
        raise SandboxUnavailable(f"CreateJobObjectW failed: error {ctypes.get_last_error()}")

    info = extended_cls()
    info.BasicLimitInformation.LimitFlags = (
        _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    info.ProcessMemoryLimit = mem_limit_bytes

    ok = kernel32.SetInformationJobObject(
        ctypes.c_void_p(hjob),
        _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        err = ctypes.get_last_error()
        kernel32.CloseHandle(ctypes.c_void_p(hjob))
        raise SandboxUnavailable(f"SetInformationJobObject failed: error {err}")
    return int(hjob)


def _assign_process_to_job(hjob: int, hprocess: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ok = kernel32.AssignProcessToJobObject(ctypes.c_void_p(hjob), ctypes.c_void_p(hprocess))
    if not ok:
        raise SandboxUnavailable(
            f"AssignProcessToJobObject failed: error {ctypes.get_last_error()}"
        )


def _run_python_windows(
    script: Path, *, timeout_s: int, workdir: Path, mem_limit_bytes: int
) -> tuple[int, str, str]:
    """Run the candidate under a job object that enforces `mem_limit_bytes`.

    The child is started with a normal (non-suspended) `Popen` and assigned to the job
    immediately afterward, rather than started `CREATE_SUSPENDED` and resumed once assigned.
    That leaves a small window in which the child could allocate before the cap applies.
    Doing this race-free needs the child's thread id to call `OpenThread`/`ResumeThread`, and
    `subprocess.Popen` does not expose it: CPython's `_execute_child` closes the thread handle
    `CreateProcess` returns and never surfaces the thread id, so getting it would need a
    `CreateToolhelp32Snapshot`/`Thread32First` walk. Given the interpreter starts with `-I -S`
    (no site import) and does essentially nothing before running the candidate, this window is
    a few milliseconds of Python startup, not enough for a candidate to have allocated past
    `mem_limit_bytes` before assignment lands -- an acceptable, documented trade-off, not a
    silent gap.
    """
    try:
        hjob = _create_memory_capped_job(mem_limit_bytes)
    except SandboxUnavailable as e:
        return 126, "", f"sandbox could not be established: {e}"

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", str(script)],
                cwd=str(workdir),  # never the repository root
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as e:
            return 125, "", f"could not start subprocess: {e}"
        with proc:
            try:
                # subprocess.Popen on Windows keeps the raw process HANDLE from CreateProcess
                # on `_handle`; that handle carries PROCESS_ALL_ACCESS (CreateProcess grants
                # the creator full access), so it can be used directly here without a second
                # OpenProcess call. Not a documented public API, hence the ignore + comment.
                _assign_process_to_job(hjob, int(proc._handle))  # type: ignore[attr-defined]
            except SandboxUnavailable as e:
                proc.kill()
                proc.communicate()
                return 126, "", f"sandbox could not be established: {e}"
            try:
                out, err = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return 124, "", f"timeout after {timeout_s}s"
            return proc.returncode, out[-4000:], err[-4000:]
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(hjob))


def _run_python(
    source: str, *, timeout_s: int, workdir: Path, mem_limit_bytes: int = MEM_LIMIT_BYTES
) -> tuple[int, str, str]:
    script = workdir / "candidate_check.py"
    script.write_text(source, encoding="utf-8")

    if os.name == "nt":
        return _run_python_windows(
            script, timeout_s=timeout_s, workdir=workdir, mem_limit_bytes=mem_limit_bytes
        )
    if os.name != "posix":
        # No supported sandbox mechanism on this platform: undecided, never run uncapped.
        return 126, "", f"no supported memory-cap mechanism for os.name={os.name!r}"

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", str(script)],
            cwd=str(workdir),  # never the repository root
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            preexec_fn=_preexec(),  # type: ignore[arg-type]
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout_s}s"
    except OSError as e:
        return 125, "", f"could not start subprocess: {e}"
    return proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:]


# ---------------------------------------------------------------- code


def code_oracle(
    candidate: str,
    *,
    test_program: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    mem_limit_bytes: int = MEM_LIMIT_BYTES,
) -> OracleResult:
    """Label a candidate implementation by executing the dataset's own hidden tests.

    `test_program` is the full runnable program: the candidate source is prepended, so the
    caller composes HumanEval's `test` + `check(entry_point)` or MBPP's assert list.

    `mem_limit_bytes` overrides the module default `MEM_LIMIT_BYTES` cap (POSIX `RLIMIT_AS`,
    Windows job object) -- mainly so callers/tests can use a smaller, faster-to-hit limit
    without touching the module-level constant used by production runs.
    """
    source = extract_code(candidate)
    if not source:
        return OracleResult(0, "empty candidate", "code:exec")
    try:
        ast.parse(source)
    except SyntaxError as e:
        # A syntax error is a genuine, decidable failure -- the tests could not pass.
        return OracleResult(0, f"syntax error: {e}", "code:exec")

    workdir = Path(tempfile.mkdtemp(prefix="rb_code_"))
    try:
        code, out, err = _run_python(
            source + "\n\n" + test_program,
            timeout_s=timeout_s,
            workdir=workdir,
            mem_limit_bytes=mem_limit_bytes,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if code in (125, 126):
        # 125: could not start the subprocess. 126: the memory-cap sandbox could not be
        # established (see `SandboxUnavailable`). Neither is a verdict on the candidate, and
        # 126 must never be silently treated as "ran uncapped" -- undecided is the safe
        # outcome; the corpus builder drops it.
        return OracleResult(None, err, "code:exec", out, err)
    if code == 124:
        # A hung candidate is a failure, not an undecided case: the tests did not pass.
        return OracleResult(0, "candidate did not terminate", "code:exec", out, err)
    return OracleResult(int(code == 0), "tests passed" if code == 0 else f"exit {code}", "code:exec", out, err)


# ---------------------------------------------------------------- reason


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[str]:
    return [m.group(0).replace(",", "") for m in _NUM.finditer(text)]


def final_answer(text: str) -> str | None:
    """Best-effort FINAL numeric answer, mirroring the GSM8K convention.

    Every rule here reads from the END of the response, which is the whole point. An earlier
    version of this function used `re.search` for an answer phrase, which returns the FIRST
    match -- and in a step-by-step solution the first "= <number>" is an intermediate result.
    It extracted 19.50 from a response whose stated answer was 26, and mislabelled roughly
    130 of 150 correct GSM8K responses as wrong. Worth stating plainly: a broken oracle does
    not produce a low kappa, it produces a *meaningless* one, because the judge is then being
    scored against noise.

    Preference order:
      1. the LAST `#### x` marker
      2. the last number on the last line that contains one (models were asked to put the
         final answer on its own last line)
      3. the LAST answer-phrase match
      4. the last number anywhere

    Applied to the candidate only. The gold side is read from the dataset's own `final_answer`
    field, so this heuristic can never move a label.
    """
    marked = re.findall(r"####\s*(-?[\d,]+\.?\d*)", text)
    if marked:
        return marked[-1].replace(",", "")

    # Walk backwards through the closing lines: the final answer is normally alone on the last
    # non-empty line, possibly decorated ("**243**", "$243", "48 \text{ g}").
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in reversed(lines[-4:]):
        stripped = re.sub(r"\\(?:boxed|text|mathrm)\s*\{([^}]*)\}", r"\1", line)
        stripped = stripped.replace("*", "").replace("$", "").replace("\\", "")
        nums = _numbers(stripped)
        if nums:
            return nums[-1]

    phrases = re.findall(
        r"(?:answer|result|total)\s*(?:is|:|=)?\s*\$?\s*(-?[\d,]+\.?\d*)",
        text,
        flags=re.IGNORECASE,
    )
    if phrases:
        return phrases[-1].replace(",", "")
    nums = _numbers(text)
    return nums[-1] if nums else None


def reason_oracle(candidate: str, *, gold: str) -> OracleResult:
    got = final_answer(candidate)
    if got is None:
        return OracleResult(0, "no numeric answer found in response", "reason:final_answer")
    want = str(gold).replace(",", "").strip().rstrip(".")
    try:
        ok = math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-9)
    except ValueError:
        ok = got.strip() == want
    return OracleResult(int(ok), f"candidate={got} gold={want}", "reason:final_answer")


# ---------------------------------------------------------------- extract


def _json_from(text: str) -> Any:
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, depth = cleaned.find("{"), 0
        if start < 0:
            return None
        for index in range(start, len(cleaned)):
            depth += (cleaned[index] == "{") - (cleaned[index] == "}")
            if depth == 0:
                try:
                    return json.loads(cleaned[start : index + 1])
                except json.JSONDecodeError:
                    return None
        return None


def _semantically_equal(a: Any, b: Any) -> bool:
    """Value equality after light coercion: 3, 3.0 and "3" agree; key order never matters."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_semantically_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_semantically_equal(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b)
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(a).strip().casefold() == str(b).strip().casefold()


def extract_oracle(candidate: str, *, gold: Any) -> OracleResult:
    got = _json_from(candidate)
    if got is None:
        return OracleResult(0, "response contained no parseable JSON object", "extract:semantic_eq")
    want = json.loads(gold) if isinstance(gold, str) else gold
    ok = _semantically_equal(got, want)
    return OracleResult(int(ok), "semantic field equality" if ok else f"differs from gold: {json.dumps(want)[:200]}", "extract:semantic_eq")


# ---------------------------------------------------------------- sql


def _normalise_rows(rows: list[tuple[Any, ...]], *, ordered: bool) -> Any:
    def cell(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, 6)
        return str(v).strip().casefold() if isinstance(v, str) else v

    norm = [tuple(cell(v) for v in r) for r in rows]
    return norm if ordered else sorted(norm, key=lambda r: tuple(str(x) for x in r))


def sql_oracle(
    candidate: str,
    *,
    gold_sql: str,
    db_path: str | Path,
    ordered: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> OracleResult:
    """Result-set equivalence against the gold query on the actual database.

    Textual SQL comparison would be the wrong oracle: two correct queries rarely look alike,
    and a judge told to predict a text match would be predicting noise.
    """
    db = Path(db_path)
    if not db.exists():
        return OracleResult(None, f"database not available: {db}", "sql:result_set")
    sql = re.sub(r"^\s*```(?:sql)?|```\s*$", "", candidate.strip(), flags=re.MULTILINE).strip()
    sql = sql.split(";")[0].strip() or sql
    if not sql:
        return OracleResult(0, "empty query", "sql:result_set")
    # Read-only connection: a candidate query must not be able to mutate the fixture.
    uri = f"file:{db.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=timeout_s) as conn:
            conn.execute(f"PRAGMA busy_timeout = {timeout_s * 1000}")
            want = _normalise_rows(conn.execute(gold_sql).fetchall(), ordered=ordered)
    except sqlite3.Error as e:
        # If the GOLD query fails, the fixture is broken; that is undecided, not a candidate fail.
        return OracleResult(None, f"gold query failed on fixture: {e}", "sql:result_set")
    try:
        with sqlite3.connect(uri, uri=True, timeout=timeout_s) as conn:
            got = _normalise_rows(conn.execute(sql).fetchall(), ordered=ordered)
    except sqlite3.Error as e:
        return OracleResult(0, f"candidate query failed: {e}", "sql:result_set")
    ok = got == want
    return OracleResult(
        int(ok), "result sets match" if ok else f"got {len(got)} rows, gold {len(want)} rows",
        "sql:result_set",
    )


# ---------------------------------------------------------------- tool use


def _parse_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Accept either a JSON call object or Python call syntax, both of which models emit."""
    data = _json_from(text)
    if isinstance(data, dict):
        name = data.get("name") or data.get("function") or data.get("function_name")
        args = data.get("arguments", data.get("args", data.get("parameters")))
        if isinstance(args, str):
            args = _json_from(args)
        if name and isinstance(args, dict):
            return str(name), args
    match = re.search(r"([A-Za-z_][\w.]*)\s*\((.*)\)", text, flags=re.DOTALL)
    if match:
        try:
            call = ast.parse(f"_f({match.group(2)})", mode="eval").body
            kwargs = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in call.keywords  # type: ignore[attr-defined]
                if kw.arg is not None
            }
            return match.group(1), kwargs
        except (SyntaxError, ValueError):
            return None
    return None


def parse_bfcl_ground_truth(raw: Any) -> tuple[str, dict[str, list[Any]]] | None:
    """Normalise BFCL's gold-call format.

    BFCL stores a gold call as a single-element list wrapping a name -> args mapping, and each
    argument maps to a LIST of acceptable values rather than one value:

        [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]

    The list is a genuine equivalence class, not a value. `"unit": ["units", ""]` means the
    argument may be `"units"` or omitted entirely -- so comparing the candidate against the
    list itself (which is what a naive oracle does) fails every correct call. An empty string
    or `None` inside the list marks the argument as optional.
    """
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]
    if not isinstance(data, dict) or len(data) != 1:
        return None
    name, args = next(iter(data.items()))
    if not isinstance(args, dict):
        return None
    return str(name), {k: (v if isinstance(v, list) else [v]) for k, v in args.items()}


def tool_use_oracle(
    candidate: str,
    *,
    gold_name: str,
    gold_args: dict[str, Any],
) -> OracleResult:
    """Compare a candidate call to a gold call by function name plus argument values.

    `gold_args` values may be a single value or a BFCL-style list of acceptable values; a list
    containing `""` or `None` marks the argument optional.
    """
    parsed = _parse_call(candidate)
    if parsed is None:
        return OracleResult(0, "no parseable function call in response", "tool_use:name_and_args")
    name, args = parsed
    if name.split(".")[-1] != str(gold_name).split(".")[-1]:
        return OracleResult(0, f"called {name!r}, gold {gold_name!r}", "tool_use:name_and_args")

    for key, raw_want in gold_args.items():
        accepted = raw_want if isinstance(raw_want, list) else [raw_want]
        optional = any(v in ("", None) for v in accepted)
        if key not in args:
            if optional:
                continue
            return OracleResult(0, f"missing required argument {key!r}", "tool_use:name_and_args")
        if not any(_semantically_equal(args[key], v) for v in accepted):
            return OracleResult(
                0, f"argument {key!r}: got {args[key]!r}, gold accepts {accepted!r}",
                "tool_use:name_and_args",
            )
    extra = {k: v for k, v in args.items() if k not in gold_args and v not in (None, "", [], {})}
    if extra:
        return OracleResult(0, f"invented arguments {sorted(extra)}", "tool_use:name_and_args")
    return OracleResult(1, "function and arguments match gold", "tool_use:name_and_args")
