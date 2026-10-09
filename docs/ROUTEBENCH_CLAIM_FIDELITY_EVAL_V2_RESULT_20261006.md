# claim-fidelity-eval-v2: result (2026-10-06): a completed negative result

**Descriptive result on already-inspected, evaluation-only original TRAIN + dev inputs. Not held-out, not a pass of the 0.74 target, no threshold search, no follow-up run.** Protocol `619d5547ce4c47f808663a10e36c8ec8609046df2744096b96655c6b27d0a00f`, run once, standalone, from the repository root. Artifacts preserved in `.local/claim-fidelity-eval-v2/` (results, spend journal, terminal record, `comparison.json`); the unrun v1 freeze is untouched.

## Run
COMPLETE: **257 of 257 valid judgments**, 0 failed, 0 retried, 0 interrupted, 0 unsettled; one attempt per case. Cost from the journal: **$2.2336** committed of the $3.00 cap, all of it usage-based known cost (257 attempts; no unknown-cost bound charges); $0.00869 per judgment against the $2.11 estimate ($2.23 actual, 6% above). Usage-based cost is provider-reported tokens at the frozen prices, not an invoice. No memory-related stop occurred.

## Results (kappa against the published labels; v24 = `claim_fidelity` variant)
| Population | v24 κ [document-grouped 95% interval] | v18 run 1 (cached) | v18 runs 2-5 | v18 five-run range (mean) |
|---|---|---|---|---|
| **dev** (128 cases, 46 docs) | **0.5938** [0.295, 0.779] | 0.6745 | 0.6560, 0.6771, 0.6216, 0.6588 | 0.6216-0.6771 (0.6576) |
| **TRAIN** (129 cases, 40 docs) | **0.3889** [0.222, 0.585] | 0.4476 | 0.3683, 0.4160, 0.3852, 0.4630 | 0.3683-0.4630 (0.4160) |
| pooled (supplementary) | 0.4889 [0.329, 0.632] | 0.5545 | 0.5039, 0.5390, 0.4973, 0.5545 | 0.4973-0.5545 (0.5299) |

v24's dev kappa is below **all five** v18 runs. On TRAIN it is below three of five and slightly above two (0.3683 and 0.3852), inside v18's own range. Pooled it is below all five.

## False passes, false rejections, corrected and new errors (v24 against each v18 run)
| dev: v24 FP 15, FR 9 | v18 FP / FR | corrected by v24 | new errors | verdicts changed | κ(v24) - κ(run) [95%] |
|---|---|---|---|---|---|
| run 1 | 14 / 5 | 2 | 7 | 9 | [-0.206, +0.051] |
| run 2 | 15 / 5 | 1 | 5 | 6 | [-0.148, +0.038] |
| run 3 | 13 / 6 | 1 | 6 | 7 | [-0.199, +0.036] |
| run 4 | 16 / 6 | 2 | 4 | 6 | [-0.130, +0.091] |
| run 5 | 14 / 6 | 2 | 6 | 8 | [-0.189, +0.066] |

| TRAIN: v24 FP 27, FR 12 | v18 FP / FR | corrected by v24 | new errors | verdicts changed | κ(v24) - κ(run) [95%] |
|---|---|---|---|---|---|
| run 1 | 31 / 4 | 7 | 11 | 18 | [-0.232, +0.065] |
| run 2 | 34 / 6 | 10 | 9 | 19 | [-0.166, +0.131] |
| run 3 | 32 / 5 | 7 | 9 | 16 | [-0.185, +0.088] |
| run 4 | 32 / 7 | 7 | 7 | 14 | [-0.111, +0.094] |
| run 5 | 31 / 3 | 7 | 12 | 19 | [-0.207, +0.049] |
Pooled: v24 FP 42, FR 21 against v18 FP 45-49, FR 9-13; new errors exceed corrected errors against all five v18 runs in the pooled comparison (corrected / new: 9/18, 11/14, 8/15, 9/11, 9/18).

## What the new criterion did
- **It lowered false passes modestly and raised false rejections more:** v24 has 3-7 fewer false passes than v18 (pooled 42 vs 45-49) but about 8-12 more false rejections (21 vs 9-13); on dev FR rose from 5-6 to 9, on TRAIN from 3-7 to 12.
- **`claim_fidelity` alone caused few of the changes.** Fidelity-only failures (all four original criteria yes, `claim_fidelity` no): **11 of 257** (dev 2, TRAIN 9). All 11 were cases v18's run 1 had passed. **7 of the 11 are published-supported (wrong rejections) and 4 are published-unsupported (correct rejections).** `claim_fidelity` failed 53 responses in total (dev 22, TRAIN 31), but 42 of those also failed another criterion.
- **Most verdict changes involve the other four criteria, not `claim_fidelity` alone.** v24 changed 6-9 dev and 14-19 TRAIN verdicts against any v18 run, while v18's own runs differ from run 1 by 2-5 (dev) and 6-7 (TRAIN); v24 failed `claims_entailed` on 75 and `no_invented_relations` on 59 of 257. These are **descriptive counts only**: the paired κ intervals below all include zero, no test was run on the counts, v18's spread rests on four run-to-run comparisons, and v24 was run once. **This document does not claim that the changes exceed run-to-run noise** (an earlier draft and my summary said so; that was an overstatement and is withdrawn). A reasonable reading is that the new prompt changed many verdicts, mostly without gain, and that the evidence cannot separate a prompt effect from ordinary run-to-run variation.

## Uncertainty and reading
- **Every κ-difference interval includes 0** (dev -0.206 to +0.091; TRAIN -0.232 to +0.131; pooled -0.169 to +0.073), so the shortfall is not statistically distinguishable from run-to-run variation on 46 and 40 documents. The point estimates, however, point the wrong way on dev against all five runs and pooled against all five. The larger false-rejection counts are likewise descriptive (not tested), and given the intervals none of these differences can be called real effects.
- The data are exposed and evaluation-only; the criterion text was written after reading the audited cases, so this could not have been an independent test of improvement even if positive.
- A single run of v24 has its own run-to-run spread (unmeasured, since only one v24 run was authorised); v18's spread was 0.05-0.06 kappa, comparable to some of these differences.

## Conclusion and stop
**Completed negative result:** adding `claim_fidelity` as a fifth criterion did not improve agreement with the published labels (point estimates below v18 on dev and pooled, within v18's range on TRAIN, every paired interval including zero), shifted errors from false passes toward false rejections in the descriptive counts, and changed many verdicts involving the original criteria. It does not support another paid run of this variant, and nothing here approaches 0.74. No prompt change, additional experiment, Stage 1b, GPU work, top-up or held-out access was made or is implied; the cap was not expanded.

## Closure (owner's decision, 2026-10-06)
- **v24 is closed as a completed negative result.** Every artifact is preserved unchanged: `.local/claim-fidelity-eval-v2/` (protocol, inputs, results, spend journal, terminal record, `comparison.json`), the unrun v1 freeze `.local/claim-fidelity-eval-v1/`, the `claim_fidelity` module, rubric directory, fixtures and tests, and all earlier frozen protocols and reports.
- **v18 remains the baseline** (`v18-reasoning-o4mini`, config `12fc8b550d954359`, rubric `d72496fbb2991071`); nothing was changed in it.
- **The original acceptance target is UNMET.** The target was an honest binary Cohen's kappa of at least 0.74 against the human labels for the groundedness judge. No judge or method tested reached it: v18's best single-run dev kappa is 0.6745 (an optimistic draw; its five-run dev mean is 0.6576 and its TRAIN mean 0.4160), and v24, the last candidate, scored 0.5938 on dev. The sealed original held-out test remains untouched and unrun.
- **Closed to further work without a new approval:** paid runs, GPU jobs, exclusions, relabelling, and further variants or minor prompt tweaks of this kind.
- **How further quality work could proceed:** only under a separately approved plan grounded in an independently verified annotation policy and independent evidence (for example, human adjudication of the residual disagreements under a written policy and independently supplied source metadata), not another automatic prompt tweak. Nothing is scheduled or implied.
