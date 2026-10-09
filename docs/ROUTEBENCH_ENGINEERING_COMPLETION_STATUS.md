# RouteBench: local engineering handoff

The local engineering phase is implemented. **The complete RouteBench acceptance contract is not passed.** Unit and mocked integration tests prove code behavior, not live inference performance or independent judge quality.

## Update 2026-10-09: later offline phase supersedes parts of the limits below

The paragraph "Remaining implementation limits" and rows marked OPEN for durable budgets, authentication, native streaming and rollout worker/recovery describe the **first** engineering phase (2026-10-07 morning). A later offline phase (fake providers, temporary SQLite, zero spend) added: durable tenant budgets in the configured database with fail-closed per-reservation validation ([report](ROUTEBENCH_DURABLE_BUDGETS_DB_20261007.md)), bearer-key tenant authentication, opt-in native token streaming with bound-based settlement, rollout resume and an observation worker ([report](ROUTEBENCH_ENGINEERING_PHASE2_OFFLINE_20261007.md)). Still true and still open: all of it is mocked evidence; PostgreSQL was never executed; there is no real OpenAI-SDK smoke test, no live three-backend measurement, no real rollback timing, no calibrated quality artifact, and the original mixed-benchmark κ >= 0.74 remains UNMET. Statements below that budgets are process-local, that streaming is only buffered, or that no observation worker exists are historical for that first phase.

## Implemented in this phase

- Routing now excludes disabled, open-circuit, previously attempted, unmeasured, over-budget, context-incompatible and SLA-incompatible backends. Missing quality is not replaced with 0.5. Canary selection uses deterministic weighted cohorts among eligible backends; those shares are relative, not guaranteed percentages of all traffic.
- Tenant funds are atomically reserved before each provider call. Failed or missing-usage attempts consume the conservative bound; overshoots are recorded, not discarded. Cache entries are isolated by tenant and routing-policy fingerprint. Unknown tenants are rejected. `x-tenant-id` requires an authenticated trusted edge; it is not authentication itself.
(superseded in part 2026-10-09; see update above) - Tool-call history and supported OpenAI request fields are preserved. Upstream calls explicitly disable streaming; downstream SSE is **buffered completed-response delivery, not native token streaming**. Multi-completion and unsupported paid features are refused before provider I/O.
- Optional startup through `ROUTEBENCH_RUNTIME_CONFIG` binds the actual calibration artifact, dataset, rubric, measured registry cells and explicit tenant limits. Startup rejects invalid backend metadata using the router's shared validation. Default startup is not production-ready; `/readyz` returns 503 without validated state and at least one statically eligible positive-weight measured backend. This is static configuration readiness, not a provider health check or a guarantee for every task. No production artifact was invented to unlock it.
- Request-root OpenTelemetry tracing parents router/backend spans and restores caller context. It uses the existing ConsoleSpanExporter and `.local/traces.jsonl` exporter. Request, token, conservative cost and cache metrics are exposed without starting any collector or telemetry service.
- A new caller-driven `evalops.rollout_controller.RolloutController` persists shadow/5%/25%/full transitions and exact baseline restoration with audit events. Its callback can apply weights to the active router. Calibration and dataset identities, minimum samples, bounded quality values, unique observations and a single active rollout are required. Simulated regression and deadline tests exercise rollback; this is **not proof of a real 90-second production rollback**.
- Quality publication refuses uncalibrated, fixture-only, drifted, undersampled and incomplete cells. Generic `llm_judge` and `unit_tests` grading are refused before backend calls because their adapters do not implement the calibrated structured judge or pinned sandbox. The offline CLI now reports BLOCKED instead of silently succeeding without doing work.

## Verification boundary

Parent verification: **112 targeted tests passed** in 7.81 seconds, with one Starlette deprecation warning; scoped Ruff and `git diff --check` passed (line-ending warnings only). The first fresh acceptance review found inconsistent readiness for invalid backend metadata; the parent corrected it with shared static validation and seven regression cases, then reran the combined checks. The final fresh acceptance review is recorded at handoff. Tests use local fixtures, temporary databases and fake providers; no held-out evaluation or provider calls were authorized in this engineering phase. The full repository suite was not rerun. Existing dirty research changes are preserved, not adopted as new engineering evidence.

Final fresh review (2026-10-07): `/root/engineering_acceptance_final` returned **ASTRA REVIEW: ship**, with no material findings. It independently passed **87 acceptance tests**, scoped Ruff and diff checks. Its first test invocation encountered sandbox-denied temporary-fixture setup; using a workspace-local pytest temporary directory resolved that infrastructure issue. No reviewer edited source. This verdict accepts the local engineering phase only. Requested review model/effort was `gpt-5.6-sol/high`; actual runtime model/effort is unobservable. One reviewer turn failed on account usage limits; a later live account check showed ordinary usage allowed, and the same read-only review was retried successfully. No reset credits were consumed or purchased.

(superseded in part 2026-10-09; see update above) Remaining implementation limits: tenant budgets are process-local and reset across restart; deploy one worker only until durable/multi-worker accounting is implemented. The controller is caller-driven, has no automatic observation worker or restart-resume reconstruction, and requires an operator to tick/check deadlines. SQL and router callback changes are not a distributed atomic commit; callback failure attempts baseline restoration. Legacy `evalops.promote` remains a simulation API, not the operational controller. SDK-shaped tests do not establish compatibility with a real installed OpenAI SDK. Gateway latency is not TTFT/TPOT.

## Acceptance work still open

| Criterion | Current status | Evidence still needed |
|---|---|---|
| Original mixed groundedness judge κ >= 0.74 | UNMET | An honestly qualified, independently evaluated calibration artifact; no exclusions based on model mistakes |
| Reveal historical probe | Separately measured κ = 0.828125 | Does not substitute for the mixed benchmark or independent production calibration |
| Three live backends and quality/cost trade-off | UNVERIFIED | Authorized backend runs, complete measured cells and online evidence bound to the same judge |
| M2/M3 Docker-dependent inference verification | BLOCKED | requires docker |
| Phoenix UI / localhost:6006 verification | BLOCKED | requires docker |
| Native token streaming and real SDK smoke | OPEN | Real adapter implementation and isolated SDK validation | *(partly superseded 2026-10-09: native streaming, observation-worker loop and rollout resume now exist as mocked code; real SDK smoke and live rollback evidence remain OPEN)*
| Live shadow, online observation worker and rollback deadline | OPEN | Worker/recovery implementation plus authorized end-to-end measurements | *(partly superseded 2026-10-09: native streaming, observation-worker loop and rollout resume now exist as mocked code; real SDK smoke and live rollback evidence remain OPEN)*
| CI quality regression merge gate | BLOCKED | Real bound calibration/quality baseline and candidate artifacts; never fake a passing bundle |
| Résumé: 12K requests, 38% spend, 1.5-point quality gap, TTFT/throughput figures | UNVERIFIED | Saved live-run protocol, raw observations and reproducible reports |

The dashboard JSON is already versioned source, not rendered or live-verified. Historical status entries do not override this evidence boundary. No Docker, external telemetry services, external provider API calls, downloads, GPU jobs, purchases, label edits or test-split evaluations were performed in this engineering phase. No commit, push or PR was created.

## Safe next sequence

1. Finish production mechanics (durable budgets/authenticated deployment, native streaming, observation worker/recovery, actual SDK integration) with offline verification.
2. Obtain a separately approved calibration/data plan and resources. Preserve the original benchmark and distinguish the Reveal result.
3. Only after valid calibration exists, run bounded live backend quality and load measurements under a new explicit spend/compute authorization.
4. Drive a real canary regression, retain timestamps and exact rollback evidence, then wire the measured baseline/candidate into CI. Report only observed results.

Forty GPU hours or repeated paid calls do not guarantee κ >= 0.74. There is no unattended paid/GPU loop scheduled.

## API-EQUIVALENT COST RECEIPT

Scope: this engineering phase, including parent work, gap audits, implementers, fix-first review and retried final review. Native tools expose no per-call token totals, so routed USD, same-token Astra repricing and price difference are **unavailable** for the whole phase; account usage percentages are not token telemetry. No delegation savings or all-Astra counterfactual is claimed. There were no new external provider API requests or GPU jobs in this phase. The previous $0.50 Reveal authorization is a separate completed experiment and was not reused.

The orchestration skill required bounded delegation, parent verification and fresh read-only acceptance; OpenAI Docs guided the live usage-status check after the reviewer failure. Neither skill makes the production acceptance target complete.
