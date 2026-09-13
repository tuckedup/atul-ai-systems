# SOW: Coding-ticket automation

## Problem as stated

“Our engineers spend half their time on bug tickets. Can an agent do the easy ones?”

## Problem as understood

Reduce bounded, repetitive ticket effort without trading review quality, repository safety, or reproducibility for raw automation.

## Scope / out of scope

Scope: repository context, plan/edit/test/review loop, risk approval, a 100-case benchmark, trace mining, and regression gating. Out: autonomous production deployment, credential changes, and unreviewed high-risk patches.

## Success metrics

At least 70% success on the agreed easy-ticket set, zero host-root command execution, 100% high-risk approval coverage, under 2-point allowed CI regression, and per-run tokens/cost/latency.

## Phases with weeks

Weeks 1–2: fixtures and safety boundary. Weeks 3–4: harness and context. Week 5: benchmark and failure mining. Week 6: pilot and handoff.

## Risks

Weak tests, repository-specific build systems, model drift, and over-broad write authority.

## What we need from you

Representative repositories, ticket history, test commands, risk policy, model credentials, and reviewers.

## Assumptions

Repositories have repeatable tests and legal approval for model processing.

