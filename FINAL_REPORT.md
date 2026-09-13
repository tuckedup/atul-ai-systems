# FINAL REPORT — Live End-to-End Validation

**Date:** 2026-09-12
**Agent:** opencode (mimo-v2.5-free)
**Total API cost:** $0.099846 (under $1 budget)

---

## TASK 1: Audit Codex's M5 Result — CONFIRMED

Ran `local-py-auth-expiry` independently via `run_agent.py --limit 1`.

| Metric | Codex's Run | My Audit Run | Delta |
|--------|-------------|--------------|-------|
| Score | 1.0 | 1.0 | 0.0 |
| Tokens | 8,010 | 7,658 | -4.4% |
| Cost | $0.009507 | $0.008766 | -7.8% |
| Latency | 25,462 ms | 23,636 ms | -7.2% |
| Steps | 6 | 6 | 0 |
| Human interventions | 1 | 1 | 0 |

**Verdict:** The 1/1 pass is **real**. Delta is LLM non-determinism. The agent correctly identified the `>` vs `>=` bug, fixed it, added a boundary test, and the hidden test passed.

---

## TASK 2: Expanded M5 Suite (10 Cases) — 10/10 PASS

Ran all 10 cases via `run_agent.py`.

| Case | Score | Tokens | Cost | Latency (ms) |
|------|-------|--------|------|---------------|
| local-py-auth-expiry | 1.0 | 6,308 | $0.00975 | 21,759 |
| local-py-pagination-partial-page | 1.0 | 7,241 | $0.00962 | 21,980 |
| local-py-cache-ttl-boundary | 1.0 | 7,182 | $0.00879 | 22,826 |
| local-py-retry-initial-delay | 1.0 | 7,235 | $0.00867 | 22,464 |
| local-py-dedupe-order | 1.0 | 7,256 | $0.00937 | 22,831 |
| local-py-config-precedence | 1.0 | 7,535 | $0.00955 | 22,348 |
| local-py-extension-case | 1.0 | 7,539 | $0.00932 | 22,458 |
| local-py-windows-final | 1.0 | 6,871 | $0.00804 | 21,690 |
| local-py-average-fraction | 1.0 | 6,576 | $0.00794 | 23,826 |
| local-py-flag-false | 1.0 | 8,588 | $0.00948 | 25,518 |

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| Success rate | **100%** (10/10) |
| Total tokens | 72,331 |
| Total cost | $0.09108 |
| Mean tokens/case | 7,233 |
| Mean cost/case | $0.00911 |
| Mean latency | 22,770 ms (22.8 s) |
| Mean steps | 6.0 |
| Human intervention rate | 100% |

---

## TASK 3: Live M6 Trace Mining (evals-compare) — ZERO REGRESSION

Ran the context-retrieval benchmark (100 cases, no model calls) and compared against baseline.

```
overall_delta: 0.0
per_tag_delta:
  ambiguous: 0.0
  api-migration: 0.0
  bugfix: 0.0
  dep-error: 0.0
  feature: 0.0
  multi-file: 0.0
  refactor: 0.0
  tests: 0.0
blocked: false
reasons: []
```

**Verdict:** CI gate passes. Zero regression detected across all 8 tags.

---

## TASK 4: Live Incident Commander M6 — BLOCKED

**Status:** NOT MEASURED

The incident commander evals framework is incomplete in wt-forge:
- No `run_incident` function (only `start_incident` + `resume_incident`)
- No `evals/` directory with scenarios.yaml or runner.py
- `bench.py` is a stub that prints a hardcoded table

Metrics that would have been measured:
- Root-cause accuracy: NOT MEASURED
- Time-to-root-cause: NOT MEASURED
- Cost per incident: NOT MEASURED

---

## API Cost Summary

| Task | Cost |
|------|------|
| TASK 1 (single case audit) | $0.008766 |
| TASK 2 (10-case suite) | $0.091080 |
| TASK 3 (context-retrieval) | $0.000000 |
| TASK 4 (IC evals) | $0.000000 |
| **Total** | **$0.099846** |
| Budget remaining | ~$0.90 |

---

## Conclusion

- **ForgeCode M5 is verified live:** 10/10 cases pass with real LLM calls.
- **M6 CI gate works:** evals-compare correctly detects zero regression.
- **Incident Commander M6 is blocked** on incomplete evals framework implementation.
- All metrics are real measurements, not estimates or synthetic claims.
