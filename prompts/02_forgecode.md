# PROMPT 02 — ForgeCode: autonomous coding agent + ForgeBench (Phase 1)

## Goal
An autonomous coding agent that, given a natural-language task on a real repository, produces a patch with
passing tests, under bounded autonomy, with a benchmark and a CI regression gate. The valuable part is
the **harness**, not a UI. Reuse `aisys.*` for everything cross-cutting.

## Non-negotiable design
- Loop: understand repo → build context pack → plan → inspect/search → edit → run tests → read failures →
  revise → independent review → approval gate → final patch.
- Four roles, LangGraph nodes, not "agents chatting": `planner`, `implementer`, `tester`, `reviewer`.
  The reviewer receives only (task, diff, test output) — never the implementer's reasoning.
- Context engine, not context dumping. Token-budget controller with eviction.
- All shell/test execution inside a Docker sandbox per task.
- Model routing tiers: small (classify, summarize), medium (edits), frontier (planning, failed retries).
- Trace mining closes the loop: failed runs become new eval cases.

## Folder layout to create
```
forgecode/
├── forgecode/
│   ├── graph.py            # LangGraph StateGraph, ForgeState, nodes, edges, checkpointer (Postgres)
│   ├── context/            # repo_map.py (tree-sitter symbols), code_graph.py (imports/callers),
│   │                       # retrieval.py (bm25 + embeddings), pack.py (budgeted context pack), evict.py
│   ├── tools/              # read_file write_file search_code list_symbols git_diff git_history
│   │                       # run_terminal run_tests run_linter create_patch — all via aisys.tools with risk levels
│   ├── sandbox/            # docker.py: per-task container, mounted repo, resource limits, timeouts
│   ├── router.py           # tier selection: task class + attempt count + failure history → model
│   ├── prompts/            # one .md per role, versioned
│   └── cli.py              # `forgecode run --repo <path> --task "<text>"` and `forgecode approve <id>`
├── forgebench/
│   ├── tasks/              # ≥100 YAML cases: repo fixture, task text, hidden tests, tags
│   ├── fixtures/           # small real repos (Python + one TS) with injected bugs / missing features
│   ├── run.py              # runs suite through aisys.evals with grader=unit_tests
│   ├── mine.py             # pulls failed trajectories from Phoenix, clusters by failure reason,
│   │                       # writes new tasks/ cases + a MINED.md report
│   └── baseline.json       # committed baseline results
├── tests/
├── Makefile                # run, bench, mine-failures, test, lint
├── STATUS.md  DECISIONS.md  README.md
```

## Milestones
**M1 — sandbox + tools.** Tools registered with risk: `write_file`/`run_terminal` medium, `create_patch`
high, delete/dep-change/migration/credential-touching paths → high via predicate. Verify: tool round-trip
tests inside sandbox.

**M2 — context engine.** `pack.build(repo, task, budget_tokens)` returns ranked file slices + repo map +
recent git context; unit test proves it never exceeds budget and includes the file that contains the
injected bug on ≥80% of fixtures. Verify: `pytest tests/test_context.py`.

**M3 — graph.** Full loop with Postgres checkpointer; `interrupt()` at approval gate; `forgecode approve`
resumes. Verify: run a fixture task end to end, kill the process mid-run, resume, get the same patch.

**M4 — router + trajectory summaries.** Rolling summaries keep the implementer under budget across
≥15 steps. Verify: a 20-step task completes with context never exceeding budget (assert in test).

**M5 — ForgeBench.** ≥100 tasks across tags: bugfix, feature, tests, refactor, api-migration, dep-error,
multi-file, ambiguous. `make bench` prints: success rate, tests passed, regression rate, tool-call
success, steps/task, tokens/task, cost/task, latency/task, human-intervention rate — overall and per tag.
Also run a 20-task subset of SWE-bench Verified with the official harness and report the number honestly.

**M6 — trace mining + CI gate.** `make mine-failures` produces ≥10 new cases from failures.
`.github/workflows/forgebench-gate.yml`: on PR touching `forgecode/`, run a 30-task smoke subset, compare
to `baseline.json` via `aisys.evals.compare`, fail if success rate drops >2 points.

## README (write last, two ways, with the M5 table pasted in)
Harness & Evals: agent loop, context strategy and budget, model selection methodology (why each tier),
reviewer isolation rationale, eval design, trace mining. Governance & Lineage: what triggers approval, what
is audited, how to reproduce any run from its trace_id.

## DEFINITION OF DONE
- `forgecode run --repo forgebench/fixtures/py-auth --task "refresh tokens intermittently fail on expiry; find, fix, add regression tests"` produces an approved, tested patch.
- `make bench` table exists in README with real numbers; `baseline.json` committed.
- CI gate blocks a deliberately-broken PR (prove it: include the failing run URL or log in STATUS.md).
- STATUS.md M1–M6 checked.
