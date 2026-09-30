# RouteBench: completed four-variant dev evaluation

Date: 2026-09-28. Execution code: `994e3da` on
`codex/routebench-finish-audit`. This is the historical four-variant report;
see `ROUTEBENCH_PROJECT_BRIEF.md` for later results and the publication exclusions.

## Outcome

All 128 authorized development cases were judged by all four variants: 512/512
usable results, 100% coverage. **The target kappa >= 0.74 is not achieved.**
Best single-judge dev kappa is **0.6532**. The best inspected simple combination
reaches **0.6719**. These are development measurements, not held-out results.
No calibration bundle was frozen, no held-out case was submitted to a provider,
and no human annotation label was included in a judge prompt.

There is now actual current-prompt evidence; missing credentials, missing dev
judgments and stale legacy hashes no longer explain this experiment's shortfall.
No labels, cases, splits, prompts, model settings or scoring code changed during
the run. Docker and the prohibited service endpoints were not checked or used.

## Single-judge results

Binary unweighted Cohen kappa; threshold selected on the pre-existing grid
0.05, 0.10, ..., 1.00. Every row uses the same 128 cases and 46 document groups.
All successful judgments report `gpt-4.1-2025-04-14` as the served model.

| Variant | Dev threshold | Dev kappa | Kappa at 0.5 | TP / FP / FN / TN |
| --- | ---: | ---: | ---: | --- |
| v16-holistic-4.1 | 0.35 | 0.6532 | 0.6373 | 75 / 16 / 4 / 33 |
| v8-decompose-4.1 | 0.80 | 0.6154 | 0.4276 | 75 / 18 / 4 / 31 |
| v12-addressed-4.1 | 0.80 | 0.6091 | 0.4167 | 71 / 15 / 8 / 34 |
| v14-contradict-4.1 | 0.50 | 0.5872 | 0.5872 | 72 / 17 / 7 / 32 |

The holistic judge agrees on 108/128 cases (84.375%), but accuracy is not kappa.
Its fixed-threshold 95% document-bootstrap interval is [0.3604, 0.7905], from
2,000 replicates, seed 20260927. The interval is NOT adjusted for threshold or
variant selection on dev and is NOT a release confidence claim.

## Offline combination check

The three available component sets were already declared in
`run_calibration.ENSEMBLE_CANDIDATES`. The existing `MeanScore` and `MajorityVote`
implementations were evaluated locally with their default parameters. No extra
provider calls or learned coefficients were used. Threshold selection still uses
dev labels, so this comparison also has selection optimism. Member threshold for
vote scores is 0.5; the final threshold acts on the fraction of positive votes.

| Components | Method | Threshold | Dev kappa |
| --- | --- | ---: | ---: |
| holistic + decomposed | mean | 0.60 | 0.6719 |
| holistic + decomposed | vote fraction | 0.55 | 0.6560 |
| holistic + addressed | mean | 0.55 | 0.6560 |
| holistic + addressed | vote fraction | 0.55 | 0.6246 |
| holistic + addressed + contradiction | mean | 0.55 | 0.6403 |
| holistic + addressed + contradiction | vote fraction | 0.50 | 0.6560 |

All six cover 128/128 cases. The best combination makes 19 errors versus the
best single judge's 20: only one net improvement, not evidence of a robust gain.
Its fixed-threshold group-bootstrap interval is [0.3810, 0.8043]. The legacy
three-rubric negative control was not evaluated: current-config outputs were
not purchased for its components. No trained logistic combiner was fitted.

## Why these mechanisms did not close the gap

1. **Errors are shared.** At their selected thresholds, all four single judges
   disagree with the human labels on the same 16 cases (14 false passes, two
   false rejections). Their different prompts do not provide enough independent
   correction. Case IDs are in `dev_four_variant_analysis.json`.
2. **False passes dominate.** The strongest single judge falsely passes 16/49
   human-failed examples. Changing the threshold alone cannot repair confident
   wrong scores of 1.0. Four human-passed cases are also rejected.
3. **Dataset/source variation is substantial.** At the holistic judge's SAME
   0.35 threshold, MediaS is 0.7763 (n=63), CNN is 0.4554 (n=41), and MeetB is
   0.4000 (n=24). The favorable source subset is not the project-wide score;
   dropping the other sources to claim 0.74 would change the target population.
4. **Some disagreements need annotation/provenance review.** For example,
   `gnd-6d0a6a0bc1619eb9` and `gnd-468f88d2d69779b2` have negative labels despite
   candidate claims that appear supported by the stored documents. The positive
   `gnd-c956780fd0b574bb` contains "SB 145" in a noisy transcript also referring
   to SB one. These are review flags, NOT established annotation errors and NOT
   permission to relabel. All 128 local labels and document/claim mappings match
   the stored raw mirror rows. That check does not establish the mirror's exact
   equivalence to the original benchmark or resolve ambiguous annotations.
5. **Infrastructure failures were repaired, not hidden.** The initial pass had
   46 provider errors and two parser failures. All 48 recovered on a serial retry
   using the same configuration and prompts. Final missing/error count is zero.
   No model-quality conclusion relies on the initial incomplete subsets.

None of the 128 dev documents reaches the builder's 6,000-character truncation
boundary (maximum 5,906), so builder truncation is not evidenced as the cause of
these dev errors. This does not prove that the raw upstream documents were complete.

## Execution and cumulative spend

| Stage | Recorded token-priced USD |
| --- | ---: |
| Tiny API health probe | 0.000036 |
| First pass: 512 case/variant attempts, concurrency 3 | 2.229768 |
| Retry: only 48 failed judgments, concurrency 1 | 0.272670 |
| Total | 2.502474 |

Authorized cap: $12. First-pass meter cap was $11.99; retry cap was reduced to
$9.76, not reset to $12. Both meters report no reservation breaches and no
cap exceedance. Failed parsed outputs' accrued costs remain in the first-pass
ledger. Latest cache rows alone would undercount these replaced failures.
No further paid work is running. About $9.50 of the authorized ceiling is unused;
this is not an API credit balance or authority to send different cases/variants.

Pricing is the configured token-based estimate, not an account invoice; cached
input discounts are not applied. The measured inference stages took about
31.7 minutes total (first pass 1,709.2 seconds; retries 190.9 seconds), excluding
the health probe and analysis. This duration is not an ETA to achieve 0.74.

For retries, the driver's `run_variant` call was wrapped in memory to pass
`concurrency=1`. No source or JudgeConfig hash was changed. Runtime concurrency
is recorded as `execution_concurrency: 1` in each retry run summary. This matters
because the CLI's existing `--concurrency` is otherwise superseded by these
variants' configured value of 3. Successful cached results were never rerun.

## Artifacts and verification

- `routebench/evalops/data/judgment_cache.jsonl`: append-only history; 560 added
  case/variant records (512 initial + 48 retries), with 512 current usable entries.
- `routebench/evalops/data/dev_experiments.json`: final full-coverage results;
  its spend field is the RETRY pass only, not the cumulative experiment cost.
- `dev_experiments.previous.20260928T180550372310Z.json` in the same directory:
  first-pass results and $2.229768 ledger.
- `dev_experiments.previous.20260928T180211361238Z.json`: preserved pre-run record.
- `dev_four_variant_analysis.json`: full confusion matrices, error case IDs,
  source breakdowns, bootstrap intervals, combination scores and cumulative cost.

Bindings: dataset `a5387f856111ac92`; annotations `1eadae861774a167`; split
membership `c25ee9a0091b00e8`; split seed 20260927. `calibrate verify` confirms
every recorded variant can be reconstructed from this tree. Raw-source matching
was inspected for dev only; no held-out labels were analyzed to select a judge.

Final checks: 541 core/RouteBench tests passed (five existing warnings), whole
RouteBench Ruff checks passed, and `git diff --check` passed. Three assertions
that assumed the old dev record was still current were updated: the legacy
formula tests now use the preserved pre-run archive, and record verification
checks both the current reproducible and archived unreproducible records. No
production judge/scoring code was changed. Tests used a fresh workspace-local
temporary directory after Windows denied access to pytest's default temp root.

Independent scikit-learn calculations reproduce all four single-judge kappas.
The cumulative cost reconciles to both run ledgers plus the probe. The old cache
is an unchanged prefix of the new cache, and every one of the 560 appended records
belongs to the authorized dev cases and four variants. Offline `plan` reports
zero missing current judgments for all four variants.

Offline verification from the worktree in PowerShell:

```powershell
$env:PYTHONPATH = 'packages/core;routebench'
$rbPython = '../wt-forge/.venv/Scripts/python.exe'
& $rbPython -m evalops.run_calibration verify
& $rbPython -m evalops.run_calibration plan --select-on headline --variants v16-holistic-4.1 v8-decompose-4.1 v12-addressed-4.1 v14-contradict-4.1
```

## Next decision

Do not claim success, spend on the full grid, repeat successful calls hoping for
better numbers, change labels to match predictions, or tune against held-out data.
The useful next step is a blinded source/annotation audit of the common dev
disagreements against authoritative dataset provenance. Then choose ONE genuinely
different checker/backend or a train-fitted combiner under a new declared protocol.
Any corrections must be versioned and invalidate old data bindings; ad hoc
label corrections on these model-selected errors are not valid acceptance evidence.

Real MiniCheck/AlignScore integration and train-split inference are still separate
unfinished work. New provider variants, training examples, or the 263-case held-out
evaluation require their own transmission/budget authorization. Even a dev score
above 0.74 would still require a frozen candidate and independent evaluation.
Docker-only milestones and other projects' blockers remain unchanged.
