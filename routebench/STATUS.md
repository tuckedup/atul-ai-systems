# RouteBench status

- [x] M1 gateway parity — request, streaming SSE, model list, health, metrics, trace/audit integration, and OpenAI wire-shape tests pass
- [x] M2 backends + vLLM — provider adapters and live self-hosted fallback-model proof completed; vLLM 0.29 was incompatible with Docker Desktop/WSL UVA, so measurements used v0.10.2's V0 engine
- [x] M3 inference experiments — four 50-request GPU configurations measured; p95 TTFT ranged from 69.065400 to 242.420700 ms, throughput from 10.349351 to 24.807449 tokens/s, TPOT from 18.141688 to 20.138633 ms, and peak VRAM from 6,488 to 6,490 MiB
- [x] M4 reliability — cumulative `/v1/usage` cost/token accounting, per-model audit events, three-failures-in-30-seconds circuit breaking with 60-second recovery, fallback routing, 50-request admission threshold, queue timeout, and HTTP 429 `Retry-After`; full suite passes 10/10
- [ ] M5 eval control plane — 200 cases, quality-matrix writer, online rolling quality, and 50-label calibration pipeline exist; live per-model offline/online runs remain
- [x] M6 eval-driven promotion — regression injection automatically rolls back and writes a verified audit event; CI offline gate committed

Phoenix verification: yes. Phoenix GraphQL reported five live spans in the `default` project, including successful `router.route` and `llm.chat` gateway spans at 2026-09-13 13:58:55 UTC. The Windows UI automation helper could not initialize (`failed to write kernel assets`), so trace presence was confirmed against Phoenix's live API rather than by an automated visual inspection.
