# ForgeCode

ForgeCode is a bounded coding harness: planner → implementer/tool loop → sandboxed tests → independent reviewer → human approval → patch. It operates only on a copied fixture repository and rejects the portfolio root as an execution cwd.

## Measured smoke benchmark

The committed 100-case smoke run measures whether the context engine finds each injected bug file under a 2,000-token budget. It is deliberately labeled separately from full-agent task success; no external model call is represented by these numbers.

| Metric | Result |
|---|---:|
| Cases | 100 |
| Bug-file retrieval | 100.0% |
| Mean context tokens/task | 143 |
| Mean latency/task | 42 ms |
| Human interventions | 0 |

The official SWE-bench Verified subset has not been run because its harness requires Docker. A ten-case local end-to-end suite has been measured separately below; it validates this harness against deliberately small fixtures but is not presented as an official SWE-bench result.

## M5 full-agent benchmark runner

The full-agent runner is deliberately separate from the retrieval smoke benchmark. `forgebench/run.py` answers “did retrieval find the likely file within budget?” without calling a model. `forgebench/run_agent.py` answers “did the production graph create a real patch that passes tests it could not see?” and therefore consumes provider quota.

### Execution lifecycle

For each YAML case, the runner performs these steps:

1. Resolve the named fixture under `forgebench/fixtures/` and create a unique disposable copy under `.local/forgecode/tasks/`.
2. Initialize that copy as an independent git repository with a baseline commit. Tools cannot target the portfolio repository or another task copy.
3. Invoke the production ForgeCode graph with a unique `bench-<case-id>-<nonce>` thread ID. The normal planner, implementer/tool loop, tester, isolated reviewer, tracing, audit, and checkpoint paths are used unchanged.
4. Correlate any pending patch approval by its `forgecode:<thread-id>:` key. In the default benchmark mode, record an approval by `forgebench-operator`, count one intervention, and resume the same durable graph checkpoint. With `--no-auto-approve`, leave the approval pending and record the case error instead.
5. Only after the graph exits, copy the case’s held-out test tree from `forgebench/hidden/` into the disposable repository.
6. Run the configured test command again. Count success only when the hidden tests pass and `git diff` is non-empty; this prevents an unchanged, already-green fixture from receiving credit.
7. Emit the shared `aisys.evals.RunResult` JSON and remove the disposable task copy.

The runner executes cases serially. This keeps every provider charge, trace, approval, and failure attributable to one task and avoids competing writes to the SQLite checkpointer used by the degraded local configuration.

### Case manifest

Full-agent cases live in `forgebench/agent_tasks/`. A minimal case is:

```yaml
- id: local-py-auth-expiry
  input:
    fixture: py-auth
    task: refresh tokens fail exactly on expiry; fix the boundary
    hidden_tests: py-auth-expiry
    test_command: python -m pytest -q
  expected: pass
  tags: [bugfix, boundary, python]
  grader: exact
```

`fixture` and `hidden_tests` are directory names relative to their respective ForgeBench roots. `test_command` runs once inside the graph and again after held-out tests are installed. The expected value is `pass`; execution returns `fail` when tests fail or the agent produces no diff. Exceptions—provider errors, invalid manifests, missing hidden-test directories, and approval pauses in no-auto-approve mode—are captured as failed `CaseResult.error` values by `aisys.evals.run` rather than aborting the suite.

Keep hidden tests outside the fixture tree. Putting them in the fixture invalidates the held-out boundary because the context engine and read tools can expose them to the model. Tests may include support files and nested directories; their relative paths are preserved when installed.

### Commands

From a POSIX environment with `make`:

```text
make -C forgecode bench          # retrieval-only, no provider call
make -C forgecode bench-agent    # all local full-agent cases
```

The direct commands are equivalent and work when `make` is unavailable:

```text
uv run --project forgecode python forgecode/forgebench/run.py
uv run --project forgecode --env-file .env python forgecode/forgebench/run_agent.py
```

Useful full-agent options:

- `--limit N` runs only the first N manifest cases; `0` means all cases.
- `--output PATH` selects the persisted result JSON, defaulting to `.local/forgebench-agent.json`.
- `--no-auto-approve` verifies the pause boundary and intentionally does not resume the patch.

The full-agent command requires either `OPENAI_API_KEY` or `AISYS_OPENAI_API_KEY`. Standard and `AISYS_`-prefixed OpenAI base-URL names are also accepted. Do not place secrets in a case, fixture, result file, or committed `.env`.

### Metrics and interpretation

The output table reports overall and per-tag success plus mean tokens, cost, latency, steps, and approval interventions per task. The JSON preserves one result per case with its score, tags, metrics, and any captured error.

| Field | Meaning |
|---|---|
| `score` | `1.0` only when hidden tests pass and the patch diff is non-empty; otherwise `0.0` |
| `tokens` | Aggregate prompt and completion tokens reported by the graph |
| `cost_usd` | Aggregate modeled provider cost; unknown model pricing remains explicit in the shared client |
| `latency_ms` | Wall time for graph execution, approval resume, and held-out verification |
| `steps` | Number of graph trajectory step labels in the final state |
| `human_interventions` | `1` when the benchmark operator resumes a pending approval, otherwise `0` |
| `error` | Exception type and message captured for a failed case |

“Benchmark operator” is an automation identity, not a claim that a human clicked an approval button. The metric measures how often the policy required the human-in-the-loop boundary. A real interactive approval demonstration is tracked separately.

### Measured local full-agent run

On 2026-09-12, provider access was restored and all ten committed local cases ran through planning, the implementation tool loop, sandboxed visible tests, independent review, durable approval/resume, patch creation, and post-run held-out tests. The cases cover ten distinct Python bug types and all held-out tests passed. This local suite is not a replacement for the Docker-dependent official SWE-bench evaluation.

| Metric | Measured result |
|---|---:|
| Cases | 10 |
| Success / held-out tests passed | 10/10 (100%) |
| Total tokens | 68,718 |
| Mean / median tokens per task | 6,871.8 / 6,908.5 |
| Total cost | $0.0878563 |
| Mean / median cost per task | $0.00878563 / $0.00882595 |
| Mean / median end-to-end latency | 24.204 s / 22.280 s |
| Graph steps | 6.0 per task |
| Approval interventions | 10 total; 1.0 per task |
| Errors | 0 |

The persisted result is `forgebench/results/local-agent-10-2026-09-12.json`. The run produced one independent result per case, and the runner returned exit code 0 only after all ten had score 1.0 with no captured exception. On this Windows host, GNU Make is unavailable, so the Makefile target's exact `uv run --project forgecode --env-file .env python forgecode/forgebench/run_agent.py` command produced these results.

The first live attempt also exposed a security-boundary bug: the model emitted the placeholder repository value `REPO`. The tool layer correctly rejected that path, but the harness should never delegate repository identity to the model. The graph now parses model arguments and overwrites `repo` with the trusted task-copy path for every repo-scoped tool. A deterministic regression test covers this behavior. A second reconciliation found that only the final implementer-loop model call was initially counted; the graph now sums every call in the loop, with a regression test proving aggregation.

### Verification boundary and next run

Local orchestration tests prove that hidden tests are absent during graph execution, are installed before final scoring, and that a pending approval is decided and resumed on the same case. The measured 1/1 run proves this path can complete against a live provider, but one result does not establish general model quality.

The official 20-task SWE-bench Verified run remains a separate Docker-dependent acceptance item and must not be inferred from the local fixture result.

## M6 trace mining and CI gate

`forgebench/mine.py` reads the append-only JSONL exporter at `.local/traces.jsonl` and groups spans by the application correlation field `aisys.trace_id`. This matters because one agent trajectory may contain several OpenTelemetry trace IDs. A trajectory is mined when any correlated span has `status: ERROR` or an `aisys.error` attribute. `GraphInterrupt` is excluded because it represents the expected durable approval pause rather than an execution failure.

The miner writes one deterministic YAML regression seed per failed trajectory to `forgebench/mined_tasks/mined.yaml` and a count/reason summary to `forgebench/MINED.md`. Mined cases contain span names, kinds, status, bounded error summaries, latency, and a hash of the first traced input. They deliberately exclude raw `aisys.input` and `aisys.output`, which can contain prompts, source code, local paths, or credentials. Keeping mined tasks outside `forgebench/tasks/` also prevents unfinished replay seeds from contaminating the committed 100-case retrieval baseline.

Run it from the workspace root:

```text
uv run python forgecode/forgebench/mine.py
uv run python forgecode/forgebench/mine.py --source .local/traces.jsonl --limit 10
```

The measured 2026-09-12 mine parsed 2,532 spans with zero malformed lines, excluded 400 expected approval-interrupt spans, and found one genuine failed trajectory containing two `ValueError` spans. It therefore wrote exactly one case. The prior ten-organic-failure target is not claimed: nine additional genuine failures are still unavailable, and synthetic failures were not substituted.

`.github/workflows/eval-gate.yml` runs entirely in degraded deterministic mode. It installs the locked workspace, runs all ForgeCode tests (including miner and workflow regression tests), creates a fresh 100-case retrieval candidate, and compares it with `forgebench/baseline.json`. `aisys evals-compare --max-drop 0.02` exits nonzero when overall success drops by more than two percentage points; the shared evaluator also enforces its per-tag drop limit. The workflow needs neither Docker/Postgres nor an OpenAI key and preserves the candidate JSON as a CI artifact.

The local CI-equivalent verification passed 100/100 cases. Its comparison reported an overall delta of 0.0, every tag delta at 0.0, and `blocked: false`. A hosted failing-run URL cannot be reported until the workflow executes on a remote branch or pull request.

## Harness & Evals

The context pack combines a symbol map, import graph, lexical ranking, and recent git context, then evicts lower-priority material to a hard token ceiling. Small, medium, and frontier tiers are selected from task class and failure history. The reviewer sees only task, diff, and test output. Failed OpenTelemetry JSONL spans feed a deterministic trace miner, and CI blocks overall success drops above two points.

## Governance & Lineage

Read-only inspection is low risk; file writes and sandbox commands are medium risk; sensitive paths and patch creation are high risk. Every model/tool decision is traced and hash-chain audited with a shared trace ID. Approval rows and graph checkpoints use the single configured database DSN, allowing a stopped process to resume its exact thread after an operator decision.
