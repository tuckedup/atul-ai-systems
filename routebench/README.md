# RouteBench

RouteBench is an OpenAI-compatible inference gateway that routes each request using **measured** quality, price, latency, availability, SLA, context length and tenant budget, plus the evaluation harness (rubric judge, calibration protocol, release gate) that is meant to produce those measurements.

> **Read this first.** The gateway and control plane are engineered, tested and hardened (below). The groundedness judge that would feed it production quality numbers has **not** reached the original target (binary Cohen's κ ≥ 0.74 on the mixed benchmark) and the sealed held-out test has never been run. Everything marked *mocked* uses fake providers; nothing here is a live-provider benchmark. The separate **Reveal** probe (κ = 0.828125, 128 cases) was measured on a previously inspected historical dataset and is **not** independent validation or a pass of the original benchmark.

## Verified engineering wins

Each row names what the tests actually exercise. "Mocked" means fake providers or fake streaming backends on temporary SQLite databases; PostgreSQL code paths exist but were **never executed**.

| Capability | What it does | Evidence scope |
|---|---|---|
| **Measured-data routing** | Excludes disabled, open-circuit, already-tried, unmeasured, over-budget, context-incompatible and SLA-incompatible backends; missing quality is never replaced by a default; canary cohorts are deterministic and weighted | mocked unit/integration tests |
| **Durable tenant budgets** | A unique reservation is persisted **before** any provider call; settlement is exact-once; unknown or missing usage is charged at the conservative bound; overshoots are recorded and counted; reservations that were never settled stay held across restarts until a named operator resolves them (`gateway/budget_admin.py`); two runtimes on one database cannot jointly exceed a cap | mocked; SQLite executed, 24-thread race test; PostgreSQL unexecuted |
| **Fail-closed budget books** | Startup and every balance change validate each persisted reservation by state (finite bounds, booked charges, known-vs-unknown actual cost, breach flags, resolutions, tenant totals). A malformed row is refused and **never repaired, refunded or discarded automatically** | mocked; 25 tampered-row shapes, many whose aggregates still balance |
| **Tenant authentication** | Bearer keys stored only as SHA-256 digests, constant-time comparison, `x-tenant-id` cannot select another tenant once auth is on, cache isolated per tenant | mocked |
| **Native token streaming (opt-in)** | Chunks are forwarded as they arrive; usage is priced from the final chunk; a missing-usage, mid-stream failure, client disconnect or bookkeeping failure settles at the bound, closes the upstream connection and releases admission exactly once, and never triggers another provider call | mocked streaming backends and `httpx.MockTransport`, including against the durable database budgets |
| **Request-root tracing** | One trace parents router, backend and cache spans; Console + JSONL exporters only (no collector); request, token, cost and cache metrics | mocked |
| **Rollout controller** | Persisted shadow → 5% → 25% → full transitions, exact baseline restoration, resume after restart, an observation-worker loop (a library, not started by the application); gated by the calibration artifact | simulated; **not** a measured production rollback |
| **Quality-publication safety** | Refuses uncalibrated, fixture-only, drifted, undersampled or incomplete cells; generic `llm_judge` / `unit_tests` grading is refused rather than faked | mocked |
| **Honest evaluation harness** | Rubric judge that never emits a score directly, frozen protocols with hashes, document-grouped bootstrap, reserve-before-request spend journals with hard caps, and every negative result preserved | see the reports under `../docs/` |

Latest verified suite results are recorded in [Verification](#verification).

## Architecture

```text
 client (OpenAI SDK shape)
        │  Authorization: Bearer …            x-tenant-id (trusted edge only)
        ▼
 ┌─ gateway/app.py ────────────────────────────────────────────────────────────┐
 │ auth ─▶ cache (per tenant + policy) ─▶ router ─▶ budget reserve ─▶ backend  │
 │                                          ▲            │ settle      │       │
 │       measured (model, task) quality ────┘            ▼             ▼       │
 │       breakers · SLA · context · weights      BudgetStore        buffered or │
 │                                       (configured DB: reservations,  native │
 │                                        events, operator resolve)    stream  │
 └──────────────┬──────────────────────────────────────────────────────────────┘
                │ ModelState (single DSN: weights, quality, rollout, audit chain)
                ▼
 evalops/  corpus → rubric judge → calibration (dev) → freeze → sealed test → release gate
                                   └───────────────▶ rollout_controller / rollout_worker
```

## Quick start

Commands below were run on this checkout (Windows, CPU only). Tests need no provider key and make no network or GPU calls.

```text
# from the repository root, with the project virtualenv in .venv
$env:PYTHONPATH = 'packages/core;routebench'
.venv\Scripts\python.exe -m pytest routebench/tests -q -p no:cacheprovider
.venv\Scripts\ruff.exe check routebench
```

The repository's declared workflow is `uv sync` then `make test` / `make lint` from the root; `routebench/Makefile` targets must be run from the **repository root** (the `.env` that carries provider credentials is only resolved there). Credentials are never committed (`.env` is ignored).

Operating the durable budgets (operator CLI, reads `settings.database_url`):

```text
python -m gateway.budget_admin unresolved
python -m gateway.budget_admin snapshot --tenant acme
python -m gateway.budget_admin resolve --reservation <id> --operator alice --reason "process died; no billing confirmed" [--charge-usd 0.0004]
python -m gateway.budget_admin set-limit --tenant acme --limit 50 --operator alice --reason "monthly cap raised"
```

`resolve` without `--charge-usd` charges the full bound (nothing is refunded). Validated startup (`ROUTEBENCH_RUNTIME_CONFIG`) uses the database budgets by default; `ephemeral_budgets` and the single-host file ledger are explicit, disclosed alternatives. Details, limits and the review history: [durable budgets](../docs/ROUTEBENCH_DURABLE_BUDGETS_DB_20261007.md).

## Judge calibration status

- 1,232 eval cases from public sources with recorded provenance (`evalops/data/raw/MANIFEST.json`), across groundedness/summarization, coding, SQL, reasoning and tool use.
- The baseline rubric judge (v18, o4-mini) reached κ 0.6216–0.6771 on the development split over five stored runs and about 0.37–0.46 on the training split; both splits are exposed, so neither is independent confirmation. **The original target is unmet.**
- Every improvement attempt tested was negative or inconclusive (fine-tunes, HHEM/MiniCheck stacking, second-pass auditors, claim-relation and typed-veto audits, source-integrity repair, majority vote, a claim-fidelity criterion). The MiniCheck equivalence check on 24 cached rows reproduced exactly (a numerical prerequisite, not evidence of better judgment); Stage 1b and the classifier line remain closed.
- AI-assisted blind reviews are diagnostics only, not human gold. The next steps are owner decisions (annotation policy, independent human review, a new benchmark, or accepting the target as unmet): see the [closeout](../docs/ROUTEBENCH_PROJECT_CLOSEOUT_20261006.md) and the [annotation-policy proposal](../docs/ROUTEBENCH_ANNOTATION_POLICY_ADDENDUM_PROPOSAL_20261007.md).

### Judge design

The judge is never asked for a score. It answers a small set of observable yes/no questions from a versioned, task-specific rubric, and `evalops/rubrics.py::score` — ordinary Python — turns those verdicts into a number. Each rubric has exactly one `critical` criterion which forces 0.0 on failure, while the remaining weighted criteria leave a gradient for threshold tuning. Rubrics are written to predict their own oracle and explicitly declare out of scope everything the oracle ignores.

Judge outputs are validated, not parsed optimistically: a missing criterion, an invented criterion id, an unreadable verdict or a truncated response is an *error* with a status, excluded from the metric. The release gate requires ≥98% completion so coverage cannot be traded for agreement. The candidate response is delimited and declared untrusted data, and the generator's identity is withheld.

### Calibration protocol

Threshold and variant are selected on the dev split only; the winning bundle is frozen with its model, rubric hashes, dataset hash and split seed; the test split is then scored exactly once. `run_calibration test` refuses to run before `freeze` and refuses to re-select anything. Splits are assigned by hashing `(seed, group_id)` (groups are source documents or prompt families), so sibling items cannot straddle a split. The confidence interval bootstraps over groups and declines to produce an interval when the sample cannot support one. Label provenance is explicit per track; synthetic fixture labels are refused outright.

### Corrections kept on the record

An early revision claimed "50 committed calibration labels; κ = 1.0". That was an artefact (one variable written into both columns, so agreement was guaranteed and no model was called). The file is kept only as the negative fixture the gate is tested against; `docs/KAPPA_DESIGN.md` documents that defect and the others found in review. One earlier fix is worth knowing: eval suites tagged cases `coding`/`extraction`/`summarization` while the router looked up `code`/`extract`/`summarize`, so every lookup silently fell back to 0.5; `evalops/taxonomy.py` now owns the canonical names and raises on unknown ones.

## Governance and lineage

Each route and backend call shares a trace ID with hash-chained audit events. Per-tenant budgets fail closed (HTTP 402 when exhausted, 503 when the budget state cannot be trusted), queue saturation returns 429 with `Retry-After`, and backend failures open a breaker and reroute. Model traffic weights and quality are persisted through the single DSN-backed state adapter, so decisions can be reproduced from request metadata and the recorded matrix.

## Not claimed

- Live-provider quality, cost or latency numbers for three backends; the inference matrix printed by `make bench` is configuration data used to exercise the router, not a fresh measurement. Docker-dependent inference and Phoenix verification are blocked on this host.
- Any figure about 12K live requests, percentage spend savings, a quality-gap in points, TTFT/throughput improvements or a real 90-second rollback: no retained measurements support them.
- PostgreSQL behaviour of the durable budgets, a real installed OpenAI SDK smoke test, a CI quality regression gate bound to a real calibration artifact.

## Verification

Run on 2026-10-09 (CPU only, no provider calls, no GPU, temporary SQLite, fake providers):

- Repository-wide suite (`packages/core/tests forgecode/tests routebench/tests incident_commander/tests operator_api/test_app.py`): **1864 passed, 5 warnings, 675.95 s**.
- `routebench/tests/test_gateway_db_budgets.py` alone: **120 passed** (durable budgets, per-reservation validation, streaming against the database budgets).
- `ruff check routebench`: clean.
- Frozen protocols and reports (judge-v19, MiniCheck equivalence report, claim-fidelity v1/v2, srcint-v3) were re-hashed and unchanged; see `../BLOCKERS.md`.

The 1864 figure was measured in the author's full working tree. Thirteen research-protocol test modules (srcint, claim-fidelity, judge-v19, matched v16/v18, fresh-eval, …) need unpublished local artifacts and data under `.local/`; on a fresh clone `routebench/tests/conftest.py` skips them with an explicit reason, and the clean-checkout result is given below. These counts describe this revision only and are mocked-provider evidence.

- Clean-checkout run (staged tree exported to a fresh directory, same suite): **1492 passed, 372 skipped, 5 warnings, exit 0, 218.76 s**. The skips are the research-protocol modules above plus existing data-presence guards; no test was edited or loosened.

## More

[Engineering status and open acceptance items](../docs/ROUTEBENCH_ENGINEERING_COMPLETION_STATUS.md) · [Phase-2 offline engineering](../docs/ROUTEBENCH_ENGINEERING_PHASE2_OFFLINE_20261007.md) · [MiniCheck equivalence narrative correction](../docs/ROUTEBENCH_MINICHECK_EQ_NARRATIVE_CORRECTION_20261007.md) · [Reveal result](../docs/ROUTEBENCH_REVEAL_COMPLETION_RESULT_20261006.md) · [status](STATUS.md) · [blockers](../BLOCKERS.md)
