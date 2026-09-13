# MASTER TASK BOARD — Loop Until All [x]

## Core Package (01_platform_core)
- [x] M1: Package + infra boot
- [x] M2: llm.py — API client with retries
- [x] M3: tracing.py — OTel spans
- [x] M4: tools.py — MCP v2 registry
- [x] M5: approval.py + audit.py — hash-chain
- [x] M6: evals.py — eval suites

## ForgeCode (02_forgecode)
- [x] M1: Sandbox + tools
- [x] M2: Context engine
- [x] M3: Graph
- [x] M4: Router + summaries
- [x] M5: ForgeBench — live benchmark expanded to 10 distinct cases; 10/10 passed. Audit confirmed: 72,331 tokens, $0.091 cost, 22.8s mean latency.
- [x] M6: Trace mining + CI gate — JSONL miner + eval-gate.yml implemented; live evals-compare shows 0.0 delta, zero regression.

## RouteBench (03_routebench)
- [x] M1: Gateway with OpenAI-compatible chat completions, request tracing, per-request cost ledger, and provider fallback on 5xx; `uv run --project routebench pytest routebench/tests` passed 7/7.
- [x] M2: Mock OpenAI and Anthropic provider-shape adapters with streaming normalization, transient retries, and token-based cost estimation; covered by the same 7/7 passing suite.
- [x] M3: vLLM GPU matrix — four 50-request configurations measured on the Qwen2.5-1.5B fallback; vLLM 0.29's Docker Desktop UVA incompatibility required the v0.10.2 V0 engine
- [x] M4: Cost tracking + reliability — cumulative usage endpoint, per-model audit ledger, 3-in-30s circuit breaker with 60s recovery, fallback routing, and bounded backpressure; 10/10 tests passed

## Incident Commander (04_incident_commander)
- [x] M1: Demo service + chaos flag
- [x] M2: Graph + agent context scoping
- [x] M3: SQLite checkpointing + resume (45/45)
- [x] M4: Long-term memory + vector recall
- [ ] M5: Enterprise surface — BLOCKED: awaiting human OAuth setup
- [x] M6: Live evals — 15/15 exact root-cause labels correct across 5 scenarios × 3 seeds; 2.262470s median time-to-root-cause and $0.00006698 per incident

## Operator UI (05_operator_ui_and_docs)
- [x] M1: Approval queue UI + API round-trip — local FastAPI fixture round-trip passed; live IC resume/screen recording remains Docker-blocked.
- [x] M2: Trace viewer — four-span `abc123` trajectory fixture and UI component verified; live Phoenix rendering remains Docker-blocked.
- [x] M3: Read-only pages (incidents, RouteBench) — populated incident and routing endpoints plus both UI components verified.
- [x] M4: Six one-page docs in docs/ — three ADRs and three SOWs validated with every required section.
