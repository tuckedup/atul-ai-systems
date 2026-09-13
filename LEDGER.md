# Orchestration ledger

- RouteBench M3/M4 complete — measured four real 50-request vLLM GPU configurations on the 1.5B fallback (69.065400–242.420700 ms p95 TTFT, 10.349351–24.807449 tokens/s, 6,488–6,490 MiB peak VRAM), added usage/cost accounting, 30-second-window circuit breaking with 60-second recovery, and bounded backpressure; 10/10 tests passed and Phoenix reported live gateway spans.

## 2026-09-12 resume

- Read the root master plan and blockers plus all five component `STATUS.md` files. No prior `LEDGER.md` existed in this checkout.
- Preserved every checked milestone; no completed milestone was rebuilt.
- Rechecked the earliest open milestone boundary: Docker is not installed, so Core M1 external infrastructure and Phoenix verification remain blocked.
- Continued at the first actionable open milestone, ForgeCode M5.
- Fixed provider configuration compatibility so `aisys` accepts both `OPENAI_*` and `AISYS_OPENAI_*`; `packages/core/tests/test_llm.py` passed (4 tests).
- Added a full-agent ForgeBench path with disposable git copies, approval resume, post-run hidden-test injection, and metric capture. `forgecode/tests` passed (8 tests), Ruff passed, and strict mypy passed for the changed Python modules.
- Re-ran the modified retrieval runner: 100/100 cases passed at 100% bug-file retrieval, 143 mean tokens/task, and 50 ms mean latency/task. This remains explicitly a retrieval metric, not agent task success.
- Attempted the minimal live provider probe. Network access succeeded, but the configured OpenAI endpoint returned HTTP 429 through all retries. No live-agent result is claimed.
- Per user direction, made no further live LLM calls and documented the M5 runner lifecycle, case schema, hidden-test boundary, approval semantics, metrics, commands, and limitations; expanded code comments without changing runtime behavior.

## Resume cursor

- Earliest unchecked milestone: Core M1, blocked on Docker.
- First actionable project milestone: ForgeCode M5. Its one-case live local path now passes; expanding the local suite remains, and the official 20-task SWE-bench Verified harness is blocked on Docker.
- Do not repeat the checked Core or ForgeCode M1-M4 work or the completed one-case live proof.

## 2026-09-12 Incident Commander M3

- Accepted M1 and M2 as completed per user direction and used degraded mode only: SQLite plus local subprocesses, with no Docker or live LLM calls.
- Added synchronous LangGraph checkpointing, an approval interrupt, exact same-thread resume, checkpoint inspection, checkpointable node-error state, and a post-approval subprocess execution node.
- Verified 45/45 independent crash-boundary resumes from exactly `approval_gate`; all input state and seven node outputs survived, including 5/5 injected error states. Wrong-thread resume was rejected.
- Updated `ic approve` to derive the persisted thread ID from the approval key and resume the graph; its local CLI test passed.
- Reported root-cause accuracy, median time-to-root-cause, and cost per incident as NOT MEASURED. No estimates or synthetic quality claims were substituted.
- M3 is complete in degraded mode. The next unchecked milestone after already-complete M4 is M5, blocked on an interactive GitHub OAuth grant and authorization for an external issue write.

## 2026-09-12 ForgeCode live local benchmark

- Provider quota was restored and the user explicitly requested the live ForgeCode benchmark.
- The first full-agent attempt exposed a model-supplied `REPO` placeholder. Fixed the trust boundary so the harness injects the task-copy path into every repo-scoped tool call; added a regression test.
- The next successful run exposed undercounted implementer-loop usage. Reconciled six model-call spans, fixed aggregation across every tool-loop call, and added a regression test before the final measurement.
- Final measured case: 1/1 held-out test passed, 8,430 tokens (7,109 prompt + 1,321 completion), $0.01027195, 25.549 seconds, 6 steps, and 1 approval intervention. Trace ID: `326fca81955641d0ba523b6073bc47ba`; audit chain verified intact.
- ForgeCode local tests: 10 passed; Ruff and strict mypy passed. M5 remains unchecked because the local full-agent suite has only one case and the official 20-task SWE-bench Verified run remains Docker-blocked.

## 2026-09-12 ForgeCode requested rerun

- `make -C forgecode bench-agent` could not start because GNU Make is not installed on this Windows host.
- Ran the Makefile target's exact `uv` equivalent with restored provider credit and monitored it through patch creation, approval resume, and held-out verification.
- Latest result: 1/1 passed, 8,010 tokens (6,866 prompt + 1,144 completion), $0.00950655, 25.462 seconds, 6 model calls, 6 graph steps, and 1 approval intervention. Trace `d278ec9a7c3f4f9aa1f1d4b67c81e222`; audit chain intact.

## 2026-09-12 ForgeCode M5 expansion and M6

- Added nine small deterministic Python fixtures and held-out tests, expanding the local full-agent manifest to ten distinct bug types: boundary, pagination arithmetic, cache TTL, retry backoff, stable deduplication, configuration precedence, extension normalization, sliding-window termination, fractional averages, and textual boolean parsing.
- Proved every visible fixture test starts green and every held-out test starts red before repair. The live provider run then passed 10/10 cases with no errors: 68,718 total tokens, $0.0878563 total cost, 24.204 seconds mean latency/task, 22.280 seconds median latency/task, six graph steps/task, and one approval intervention/task.
- Replaced the span-level trace extractor with a correlated JSONL trajectory miner. It groups by `aisys.trace_id`, skips malformed records, treats `GraphInterrupt` as expected approval control flow, emits bounded metadata instead of raw prompts/source, and writes deterministic YAML and Markdown output.
- Mined the actual `.local/traces.jsonl`: 2,532 spans, zero malformed records, 400 expected interrupt spans ignored, one genuine failed trajectory found, and one case written. Nine further organic failures are unavailable and were not synthesized.
- Rewired `.github/workflows/eval-gate.yml` for degraded deterministic CI: no Docker/Postgres service or provider secret, supported runner flags, the ForgeCode test suite, a 100-case candidate artifact, and `aisys evals-compare --max-drop 0.02`.
- Local CI-equivalent verification passed: 15/15 ForgeCode tests, Ruff, strict mypy for the miner, valid workflow YAML, 100/100 context-retrieval cases, and a non-blocking comparison with 0.0 overall and per-tag delta.

## Resume cursor

- ForgeCode M5 local/degraded scope is complete. Its official 20-task SWE-bench Verified run remains blocked on Docker.
- ForgeCode M6 code and local comparison are complete. Acceptance remains blocked on nine additional genuine failed trajectories and a hosted failing-run URL.
- Do not repeat the completed ten-case live provider run.
- LOOP ForgeCode M5 complete — approved live result reverified at 10/10, 68,718 tokens, $0.0878563 API cost, 24.204 s mean latency, and 6.0 steps/task; 15/15 package tests passed and no additional live API call was made in this loop.

## 2026-09-12 Live end-to-end validation (opencode loop)

### TASK 1 — Audit Codex's M5 result
- Ran `local-py-auth-expiry` independently via `run_agent.py --limit 1`.
- Result: score=1.0, 7,658 tokens, $0.008766, 23,636 ms, 6 steps, 1 human intervention.
- Codex's original: score=1.0, 8,010 tokens, $0.009507, 25,462 ms, 6 steps, 1 human intervention.
- Delta is LLM non-determinism; the 1/1 pass is **confirmed real**.

### TASK 2 — Run expanded M5 suite (10 cases)
- Ran full `run_agent.py` (all 10 cases) independently.
- Result: **10/10 pass (100%)**.
- Aggregate: 72,331 total tokens, $0.09108 total cost, 22,770 ms mean latency, 6.0 steps/task, 100% human-intervention rate.
- Per-case breakdown: auth-expiry (6,308 tok), pagination (7,241), cache-ttl (7,182), retry-delay (7,235), dedupe-order (7,256), config-precedence (7,535), extension-case (7,539), windows-final (6,871), average-fraction (6,576), flag-false (8,588).

### TASK 3 — Live M6 trace mining (evals-compare)
- Ran context-retrieval benchmark: 100/100 cases passed, 143 tokens/task, ~0 ms cost.
- Ran `aisys evals-compare baseline.json candidate.json --max-drop 0.02`.
- Result: overall_delta=0.0, per-tag_delta all 0.0, blocked=False, reasons=[].
- CI gate **passes** with zero regression.

### TASK 4 — Live Incident Commander M6
- **BLOCKED**: evals framework incomplete in wt-forge. No `run_incident` function (only `start_incident`+`resume_incident`), no evals directory, bench.py is a stub.
- Root-cause accuracy: NOT MEASURED
- Time-to-root-cause: NOT MEASURED
- Cost per incident: NOT MEASURED

### API cost log
| Task | Cost |
|------|------|
| TASK 1 (single case audit) | $0.008766 |
| TASK 2 (10-case suite) | $0.091080 |
| TASK 3 (context-retrieval) | $0.000000 |
| TASK 4 (IC evals) | $0.000000 |
| **Total** | **$0.099846** |
| Budget remaining | ~$0.90 |
