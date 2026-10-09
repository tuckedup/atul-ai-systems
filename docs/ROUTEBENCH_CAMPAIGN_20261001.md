# Campaign log 2026-10-01 (after "do whatever it takes"): stronger judges, learned combiners, in-distribution fine-tuning

All numbers are development evidence (TRAIN/dev or the TofuEval upstream-dev census), not the sealed test. The original benchmark, labels and splits were not changed (the request to change them was declined; see ROUTEBENCH_TRAINING_PERMISSION_STATUS.md).

| Experiment | Result |
|---|---|
| Claude Opus as judge, v18 prompt/rubric, via isolated sub-agents | dev κ 0.6515 (FP 11, FN 10); TRAIN κ 0.5467 (FP 20, FN 9). v18: dev 0.6745, TRAIN 0.4476 |
| v18 AND Opus (both must pass), pooled TRAIN+dev | κ 0.615 vs v18 0.555, Opus 0.598; 29 false passes remain shared |
| GPT-5.4 judge | 63 successful judgments, then `credit_balance_exhausted` (HTTP 429); no complete comparison. Shared campaign accounting: $1.56813 including unknown-cost reservations, not an invoice. GPT-5.5 was attempted but produced zero successful judgments. No automatic spending authorization follows. |
| Learned combiner (v18 criteria bits + score + MiniCheck + length), corrected doc-grouped CV diagnostic on 554 of 560 rows | Six rows have unavailable criterion features and were excluded from this diagnostic only. v18 κ 0.4684; v18+MiniCheck κ about 0.52 at 0.5, about 0.53 with a post-hoc tuned threshold. Not a full-population or independent benchmark result. Earlier 560-row feature results are superseded. |
| MiniCheck zero-shot on the full 3,437-sentence / 70-document census | AUC 0.77, κ 0.36-0.40 |
| MiniCheck-Flan-T5-L LoRA (r16, q/k/v/o, 2 epochs) fine-tuned on 1,712 in-distribution TofuEval sentences, evaluated on the other 35 documents (1,725 sentences) | AUC 0.76035 → 0.76020; κ at unchanged 0.5 cutoff 0.35584 → 0.35337. No improvement. 2.44 h elapsed GPU-run time. Previously quoted best-threshold numbers were post-hoc, not a frozen decision-rule comparison. Training-use restriction remains unresolved. |

Conclusion: the completed experiments have not demonstrated κ >= 0.74. The GPT experiments are incomplete and do not establish the performance of those judges. Failed configurations and partly shared errors do not establish a general accuracy ceiling. Fold 1 of the fine-tune was not run (fold 0 showed no gain).

## Corrections after independent review (2026-10-01)
1. **Criterion reconstruction.** Failed criteria are recoverable from v18's score only for scores above 0; a score of 0 is forced by the critical `no_contradiction` failure whatever the other criteria did. The earlier combiner treated those other criteria as passing. Re-run with the 6 score-0 rows (of the 240 whose criteria were not saved) excluded: stacker κ at 0.5 is 0.42 (v18 bits+score), 0.43-0.52 (MiniCheck / v18+MiniCheck), best-threshold (optimistic) 0.51-0.53; v18 alone 0.468. Conclusion unchanged. The error-cause audit's "claims_entailed failed in 31 of 37" counts score-0 rows as critical-only failures, so treat that as a lower bound for the criterion split (the 3 critical cases may also have failed others). The separate fine-tune result is unaffected.
2. **Runner accounting.** `multi_judge run` reported `judged` = cases planned, not completed. Fixed: it now prints `planned` and `completed_ok_total_in_file`. Actual GPT-5.4 coverage was 63 successful judgments (no κ is reported below full coverage); GPT-5.5 produced none.
3. **Spend.** The $1.568 is the journal's accounting: about $1.000 of reported usage costs plus conservative reservation charges for three attempts with unknown cost. It is not a provider invoice. Opus judging ran through sub-agents (cost to the session, not the API key); the exact Opus version is not recorded in the outputs.
4. **Fine-tune numbers** (reviewer's recomputation, accepted): AUC 0.7604 → 0.7602; κ at the unchanged 0.5 cutoff 0.3558 → 0.3534; no gain. My table quoted best-threshold κ.
5. **Data eligibility.** The 70 TofuEval upstream-dev documents (including the 40 + 30 used as fresh evaluations) were consumed as training/development data under the user's broadened instruction; the dataset's own README prohibits training use, and the permission record is an interpretation of that instruction, not a rights clearance. They can no longer support an independent final claim. Still eligible as unseen evidence: the sealed RouteBench test split only (needs separate authorization). Not eligible: RouteBench TRAIN/dev (repeatedly inspected), first 40-document sample, the 30 reserved documents, the 70-document census. Everything from this campaign is uncommitted in the checkout.

## Offline quality cleanup (2026-10-01)

- `multi_judge.run` now determines completion from unique latest successful rows
  for the entire requested population, including on resume. Failed judgments and
  limited partial runs report INCOMPLETE; aborts remain ABORTED. Both return a
  nonzero CLI exit status. Planned, attempted, completed-this-run and cumulative
  requested-population counts are distinct. Saved rows are flushed and fsynced.
- Synthetic regression tests first reproduced the old failure, then passed after
  the implementation change. No provider requests or fresh fitting were needed.
- The 554-row script label is now dynamic and explicitly a diagnostic subset.
  No case was removed from the original benchmark and no historical prediction changed.
- The error-cause audit now says zero-score secondary criteria are unknown and
  cause assignments are provisional AI-assisted judgments. Unsupported ceiling
  claims based on cross-dataset leaderboard comparisons were removed.
- Next deliverable is [the data-quality pilot specification](ROUTEBENCH_DATA_QUALITY_PILOT.md).
  It is ready for a quote request after authorization, not a purchased dataset,
  completed human review or proof of model improvement.
