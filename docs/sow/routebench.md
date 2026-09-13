# SOW: LLM cost and quality control plane

## Problem as stated

“Our LLM bill tripled and nobody knows which model is actually better.”

## Problem as understood

Create comparable quality, cost, and latency evidence and enforce it in routing and version rollout without client rewrites.

## Scope / out of scope

Scope: OpenAI-compatible gateway, three backend adapters, reliability controls, 200-case eval suite, judge calibration, online sampling, and gated promotion. Out: provider contract negotiation and a custom dashboard application.

## Success metrics

20% cost reduction at no more than a 2-point quality drop, κ ≥ 0.6, under 1% failure during one-backend loss, p95 SLA compliance, and automatic rollback within one canary window.

## Phases with weeks

Weeks 1–2: gateway and telemetry. Weeks 3–4: inference experiments and reliability. Weeks 5–6: evals, calibration, promotion, and handoff.

## Risks

Unrepresentative evals, judge bias, provider rate limits, traffic shifts, and stale pricing.

## What we need from you

Traffic samples, provider access, budgets, SLAs, human raters, and an approved self-hosted model/GPU.

## Assumptions

Clients can set a base URL and production sampling is permitted.

