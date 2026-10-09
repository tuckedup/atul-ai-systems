# Environment decisions

- 2026-09-11: Use the single `AISYS_DATABASE_URL` setting, defaulting to `sqlite:///.local/aisys.db`, so PostgreSQL can be restored by changing one value.
- 2026-09-11: Use `AISYS_CACHE_BACKEND=memory` with an in-process dictionary implementing the shared cache interface; a Redis adapter can replace it without changing consumers.
- 2026-09-11: Keep OpenTelemetry spans and attributes unchanged while exporting every span to the console and `AISYS_TRACE_JSONL_PATH` (default `./.local/traces.jsonl`).
- 2026-09-11: ForgeCode task commands run in a subprocess with a hard timeout, memory cap where the host supports it, and a fixture-copy working directory; the repository root is rejected as a command working directory.
- 2026-09-11: Grafana dashboards are committed as JSON artifacts only and are not rendered on this host.

- RouteBench local engineering phase: keep the existing single-DSN state adapter and memory-cache interface. Request-root tracing uses the existing Console + JSONL exporters; no collector/service is started. Budget reservations are process-local, so deployment is limited to one worker until durable restart-safe accounting exists. Tenant headers must come from an authenticated trusted edge. SSE delivery remains buffered and is not reported as native token streaming. **Superseded in part (2026-10-07/08):** tenant budgets are now durable in the configured database (reserve-before-call, exact-once settlement, fail-closed ledger validation; PostgreSQL unexecuted), bearer-key tenant authentication exists, and native token streaming is available behind an opt-in flag; buffered SSE remains the default. See `docs/ROUTEBENCH_DURABLE_BUDGETS_DB_20261007.md`.
- RouteBench acceptance remains evidence-based: the separately measured Reveal historical probe does not substitute for original mixed-benchmark calibration. No fabricated bundle/quality matrix, weakened CI gate, excluded hard samples or synthetic live-performance proof is permitted. The new rollout controller is caller-driven and local simulated rollback tests do not establish a real 90-second production result.

