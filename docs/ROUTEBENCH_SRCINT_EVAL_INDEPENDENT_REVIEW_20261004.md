# Independent review: source-integrity evaluation v1

Verdict: **FIX FIRST. No paid run authorized by this review.**

The prepared comparison is real, but not executed. I verified the protocol SHA-256
`a52d3716d5b6ba71559302a0f0b196eca50725b30bf6f78f1bd7d0416e185c29`,
the 257-case population and the repaired-copy identity. In the actual work directory
there is no results file, spend journal, terminal report or comparison report.
The completeness check returns `WITHHELD_INCOMPLETE`, with zero valid judgments.
Claude's 14 existing tests pass; targeted Ruff passes. I did not rerun the claimed
1,167-test full suite, query account credits, or make a provider call.

## What is correct

- All 257 retained TRAIN/DEV cases are planned, with new IDs, separate results and
  the original cached v18 judgments used only as the comparison baseline.
- The frozen rule is the original v18 all-criteria-pass rule, without threshold
  searching. DEV is primary and TRAIN diagnostic.
- Cases, population, baseline and label snapshot are hash-bound. In the run phase
  the separate label file is only hashed, not parsed for scoring. Incomplete
  coverage normally withholds the comparison.
- Normal provider attempts use the single-request client path and each is reserved
  and settled in a durable spend journal. The $1.97 expectation is explicitly an
  estimate, not a cost guarantee. This protocol does not include typed-veto audits.
- No original benchmark result should change from this comparison.

## Reproduced execution/integrity gaps

Seven offline safety regressions fail on v1, representing five defect categories:

1. **Results are not validated on resume/report.** `srcint_eval.py:138` and `:183`
   trust records in `results.jsonl`. NaN, out-of-range 2.0, and boolean scores all
   reach the label-join path rather than being rejected. Stored records do not bind themselves
   to the frozen protocol, case content, prompt/config or reservation. These tests
   demonstrate malformed result acceptance, not that any real result was forged.
2. **Conflicting duplicate IDs silently overwrite earlier records.** A resumed
   run returns COMPLETE with no provider calls on these synthetic inputs. IDs and
   record consistency need validation before building a last-row-wins dictionary.
3. **An already-over-cap journal can return COMPLETE.** On scratch data, a provider
   charge of $3.10 against a $3 cap is correctly recorded and raises SpendError.
   When all result IDs are present, restarting skips calls and returns COMPLETE
   with that same $3.10 committed. `compare` also never checks journal/terminal
   validity. Normal call admission still checks the cap; the reproduced defect is
   terminal/restart acceptance of a previously aborted financial state.
4. **A runtime dependency is absent from the freeze.**
   `matched_v16_v18.code_hashes` (`:103`) omits `audit2_v3.py`, although `judge_pair`
   uses its token-overhead bound and raw-usage validator. `multi_judge.py`, which
   implements the freeze loader, is also omitted. Freeze the actual dependency
   path, not just the wrapper file. Never overwrite the existing protocol to do so.
5. **Held-out annotation values are processed by the freeze loader.**
   `srcint_eval.build_population` (`:52`) calls `multi_judge.load_cases`.
   That loader validates every annotation and resolves all labels (`:42-43`)
   BEFORE excluding test cases (`:47`). A guard blocking typed validation of test
   annotation IDs stops this loader. Returned evaluation items exclude test, and
   no original test judgments were made; nevertheless, the statement "no held-out
   access" is too broad for this code path. The new run's separate label-file
   safeguard does not fix the earlier freeze-stage access.
6. **Additional static identity gap:** `matched_v16_v18.py:258` accepts an empty
   served-model value (`if served and ...`). Require trustworthy recorded model
   identity or mark the judgment unverifiable. This finding was inspected in code;
   it is not one of the seven reproduced regression failures above.

## Statistical correction

The claim that case-level Fisher testing distinguishes a repair effect from noise
is too strong. The changed cases are concentrated in seven documents, overwhelmingly
MediaSum, whereas unchanged cases include different datasets and difficulty profiles.
Sentences share documents, and the changed/unchanged allocation is not randomized.
Pooling TRAIN and DEV flip counts compounds that confounding. A difference in flip
rates can reflect source mix, difficulty, clustering or provider drift, not just
the repairs. Unchanged-input reruns are useful reproducibility diagnostics, but are
not an exchangeable causal control group.

Keep those counts descriptive. The paired document-bootstrap DEV interval is a
reasonable description of the whole changed-input pipeline comparison, not a clean
causal estimate of punctuation repair. A stronger repair-attribution experiment
would run original and repaired versions of the SAME changed cases contemporaneously,
with a frozen model snapshot and predeclared paired/document-level analysis. That
would be a different scope and require separate budgeting; do not automatically add
calls to the proposed run.

## Paste into Claude

```text
Keep srcint-eval-v1 paused. Read docs/ROUTEBENCH_SRCINT_EVAL_INDEPENDENT_REVIEW_20261004.md
and .local/srcint-eval-review/test_safety_regressions.py. Do one bounded OFFLINE fix pass:

1. Validate all saved result rows before resume/comparison: exact expected IDs,
   no duplicates/foreign IDs, valid statuses, finite nonboolean scores in [0,1],
   criterion/score consistency, and verified model identity. Bind rows to the
   protocol, case-content/prompt/config/rubric hashes and journal attempt IDs;
   retain enough raw response data to verify parsing. Reject stale/mismatched rows.
2. Fail closed on journal/terminal inconsistencies, over-cap committed spend,
   unsettled attempts and an ABORTED run. Completeness of ID coverage alone is not
   sufficient. Require validated settlement and durable output; permit only one
   active runner for this journal. Do not discard charges or reset the ledger.
3. Freeze actual runtime dependencies, including audit2_v3.py and the loader path.
4. Exclude held-out case IDs BEFORE typed annotation validation or label resolution.
   Use a bounded TRAIN/DEV loader; add a guard test. Disclose the old loader behavior
   accurately, without claiming original test inference happened.
5. Remove causal/noise-isolation claims for pooled case-level Fisher comparisons.
   Keep flip counts descriptive and the paired document-bootstrap pipeline interval.
   Do not add extra calls, populations, thresholds or candidates.

Run the existing tests plus the safety regressions, lint and calibration verification.
Preserve v1 and all original cases/labels/splits/caches/results. Create a new isolated
v2 protocol and report its new hash; do not edit the old freeze. Return changed files,
actual checks, protocol identity and any remaining blocker. No paid calls, downloads,
vendor contact or original acceptance-test access. Wait for separate run approval.
```

These fixes concern trustworthy execution and reporting, not model improvement.
The source repair still has no measured kappa. A repaired-copy DEV score of 0.74,
if obtained, would not certify the original benchmark or establish held-out quality.
Given that only four existing DEV errors are in changed inputs and those changes
are punctuation-only, this should be justified as a source-fidelity/reproducibility
check, not sold as a likely route to the original 0.74 target.

No provider spend, source-data edits or real protocol/result mutations were made.
Only scratch reproductions and this independent review were added. API-equivalent
review cost is unavailable: native tools did not expose observed token usage.
