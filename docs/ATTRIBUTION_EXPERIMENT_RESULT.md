# Claim-level attribution experiment: result

> **Audit correction (2026-09-29):** This is a preserved historical report, not a
> clean independent confirmation. The stored bundle has empty annotation and
> split-membership bindings; 14 test pairs appeared in prior probes; 96/600 sources
> were truncated. The claims below of complete bindings and cross-experiment
> leak freedom are superseded. The cause of the dev/test drop is not established.
> Same-config retries are not inherently model reselection, but this old artifact
> cannot be silently rebound. Even perfect completion of all 11 missing items
> yields kappa at most 0.5721. See `ROUTEBENCH_GEDD_AUDIT.md` and the machine-readable
> `ATTRIBUTION_INTEGRITY_AUDIT.json`. Original measurements remain unchanged.

Date: 2026-09-28. Protocol: `docs/ATTRIBUTION_EXPERIMENT_DECLARATION.md`, written before the corpus
was built. Data: `routebench/evalops/data/attribution/`.

## Outcome

**κ ≥ 0.74 is NOT achieved.** Held-out unweighted binary Cohen's κ = **0.5431**,
95% group-bootstrap CI **[0.420, 0.654]**, n = 195 over 171 document groups.

| stage | κ | n |
|---|---:|---:|
| dev, `v16-holistic-4.1` | 0.6348 | 174 |
| dev, `v8-decompose-4.1` (selected) | 0.6563 | 174 |
| **held-out test, frozen `v8-decompose-4.1` @ t=0.80** | **0.5431** | 195 |

Raw agreement 0.7744. Confusion: TP 64, FP 18, FN 26, TN 87. Human prevalence 0.462, judge 0.421.

The 0.11 drop from dev to test is selection optimism — the threshold and the variant were chosen on
dev, and the test split priced that in. This is the whole reason the test split is measured once and
only after freezing.

## Per-subset, held-out, at the single frozen threshold

| subset | n | κ | raw agreement | confusion |
|---|---:|---:|---:|---|
| Reveal | 40 | 0.8000 | 0.900 | tp 19 / fp 3 / fn 1 / tn 17 |
| Lfqa | 45 | 0.7321 | 0.867 | tp 18 / fp 3 / fn 3 / tn 21 |
| ClaimVerify | 28 | 0.5648 | 0.786 | tp 9 / fp 4 / fn 2 / tn 13 |
| Wice | 20 | 0.4792 | 0.750 | tp 5 / fp 1 / fn 4 / tn 10 |
| FactCheck-GPT | 31 | 0.4294 | 0.710 | tp 7 / fp 0 / fn 9 / tn 15 |
| **ExpertQA** | 31 | **0.0726** | 0.548 | tp 6 / fp 7 / fn 7 / tn 11 |

One judge, one threshold, one prompt: κ spans **0.07 to 0.80**. ExpertQA sits at raw agreement
0.548 — indistinguishable from chance on a balanced set. Whatever its annotators were deciding,
this judge is not modelling it at all.

This is the same conclusion the source probe reached, now on held-out data: the dominant term in
this metric is the annotation source, not the judge. A single pooled κ across six sources with
incompatible annotation granularity is an average over incommensurable tasks.

## Limitation: coverage

Completion was **94.7%** (195 of 206 labelled test items; 11 `provider_error`). The declared release
gate requires 100%, so this run does not satisfy the gate on coverage grounds either.

The missing 11 cannot change the conclusion. The CI upper bound is 0.654, and 11 additional items
on n=195 cannot move a point estimate of 0.543 to 0.74. The failures were deliberately **not**
retried: the bundle is measured, and re-running a measured bundle to move a number is the behaviour
the freeze exists to prevent. The incompleteness is recorded instead of fixed.

## What was and was not done

Held: labels unedited; splits group-aware and verified leak-free; threshold and variant selected on
dev only; bundle frozen with model, prompt-template, rubric, dataset, annotations and
split-membership bindings before any test call; test measured exactly once; both dev candidates
reported including the loser; the ranking flip caused by coverage reported (see below).

Not done: no source was dropped after seeing its score; no relabelling; no re-selection against
test; no per-subset thresholds fitted; no learned combiner.

### The coverage flip, recorded because it nearly caused a wrong freeze

The first dev pass ran at 92.5% / 94.3% completion and ranked **holistic above decompose**
(0.6537 vs 0.6355). After a serial retry brought both to 100%, the ranking **reversed**
(holistic 0.6348, decompose 0.6563). The ≥98% coverage precondition on freeze eligibility is what
caught it. A partial-coverage leaderboard is not a leaderboard.

## Spend

| stage | USD |
|---|---:|
| source probe (Wice + Reveal, 256 judgments) | 1.1843 |
| Wice completion retry | ~0.09 |
| attribution dev, 2 candidates x 174 cases | 1.3107 |
| attribution dev serial retry to 100% | 0.1357 |
| attribution held-out test, 206 cases | 0.5562 |
| **total this experiment** | **~3.28** |

Token-priced estimate from the configured table, not an account invoice.

## Where 0.74 actually lives

On measured evidence, two single sources clear or approach it under this unchanged judge:
Reveal κ = 0.800 (held-out n=40; the earlier 128-case probe gave 0.8388, CI [0.734, 0.921]) and
Lfqa κ = 0.7321 (held-out n=40). Both are small samples at a threshold tuned for the pooled set.

A defensible ≥0.74 claim would therefore be **single-source and narrow**, and would need its own
pre-registered declaration, its own group-aware splits, its own frozen bundle and its own
one-shot held-out measurement — not a re-slice of this test split, which has now been spent.

What is NOT supportable by anything measured here: κ = 0.74 on groundedness of summarization
(dev 0.6532), κ = 0.74 on pooled claim-level attribution (held-out 0.5431), or any claim involving
locally hand-labelled items — that count is still **zero**.
