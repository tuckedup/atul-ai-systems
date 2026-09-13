# Environment decisions

- 2026-09-11: Use the single `AISYS_DATABASE_URL` setting, defaulting to `sqlite:///.local/aisys.db`, so PostgreSQL can be restored by changing one value.
- 2026-09-11: Use `AISYS_CACHE_BACKEND=memory` with an in-process dictionary implementing the shared cache interface; a Redis adapter can replace it without changing consumers.
- 2026-09-11: Keep OpenTelemetry spans and attributes unchanged while exporting every span to the console and `AISYS_TRACE_JSONL_PATH` (default `./.local/traces.jsonl`).
- 2026-09-11: ForgeCode task commands run in a subprocess with a hard timeout, memory cap where the host supports it, and a fixture-copy working directory; the repository root is rejected as a command working directory.
- 2026-09-11: Grafana dashboards are committed as JSON artifacts only and are not rendered on this host.

