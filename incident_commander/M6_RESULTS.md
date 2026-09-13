# Incident Commander M6 — live eval results

Run window: 2026-09-13T23:30:57.833239+00:00 to 2026-09-13T23:31:34.540116+00:00

## Overall metrics

| Metric | Measured |
|---|---:|
| Root-cause accuracy | 15/15 (100.00%) |
| Median time-to-root-cause | 2.262470 s |
| Cost per incident | $0.00006698 |
| Total API cost | $0.00100470 |
| Input tokens | 2634 |
| Output tokens | 1016 |
| Successful API runs | 15/15 |

## Scenario results (three seeds each)

| Scenario | Accuracy | Median seconds | Mean cost |
|---|---:|---:|---:|
| Dependency outage cascade | 3/3 | 2.369222 | $0.00007370 |
| Code regression after deployment | 3/3 | 2.347986 | $0.00005490 |
| Database connection exhaustion | 3/3 | 2.101325 | $0.00007335 |
| Expired service credentials | 3/3 | 2.188056 | $0.00006435 |
| Compute capacity exhaustion | 3/3 | 2.434147 | $0.00006860 |

## Method

Five evidence-injected incidents were each run with seeds 11, 29, and 47. The model received the alert, observed evidence, and the fixed label vocabulary, but not the injected ground-truth label. Accuracy is an exact label match over all 15 attempted runs. Time-to-root-cause is wall-clock time from request start through validated JSON diagnosis. Cost uses provider-reported token counts and the repository pricing table. Failed calls, if any, remain in the denominator and are retained in the raw artifact.

Full per-run predictions, rationales, timings, tokens, costs, and errors are stored in `M6_RAW.json`.
