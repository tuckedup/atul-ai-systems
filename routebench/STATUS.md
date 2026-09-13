# RouteBench status

- [x] M1 gateway parity — request, streaming SSE, model list, health, metrics, trace/audit integration, and OpenAI wire-shape tests pass
- [ ] M2 backends + vLLM — provider adapters are implemented; live self-hosted proof BLOCKED: requires docker
- [ ] M3 inference experiments — `inference/RESULTS.md` records BLOCKED: requires docker; no synthetic measurements substituted
- [x] M4 reliability — rate, admission, budget, circuit-breaker, and 100-request failure/reroute chaos test pass with 0 failures
- [ ] M5 eval control plane — 200 cases, quality-matrix writer, online rolling quality, and 50-label calibration pipeline exist; live per-model offline/online runs remain
- [x] M6 eval-driven promotion — regression injection automatically rolls back and writes a verified audit event; CI offline gate committed

