# Proposal: error-type-targeted second-pass audit of "supported" verdicts (2026-09-30; NOT approved, NOT run)

Corrections to earlier wording: FactCG is closed as a negative result (dev κ 0.549 vs v18 0.675). v18's 0.675 is the *best measured result*, not a ceiling.
Dev has 46 documents (only the MeetB part has 3). The extra fresh-document evaluation has not been run; all numbers are dev only.
No relabelling or removal of benchmark cases is part of this proposal.

## What the remaining errors are (cached v18 + FactCG predictions; TofuEval expert error types joined on exact summary text, all 87 TofuEval dev rows matched)
- v18 dev errors: 19 = 14 false passes (unsupported called supported) + 5 false rejections. 8 of 19 sit in the 3 worst documents.
- Expert error type of the unsupported TofuEval rows v18 PASSED (missed): Reasoning Error 3/5, Nuanced Meaning Shift 2/6, Tense/Modality 2/2, Mis-Referencing 1/1,
  Extrinsic Information only 1/20. FactCG misses the same classes (4/5, 2/6, 2/2, 1/1) and also 2/20 extrinsic.
  => detecting invented content is mostly solved; the residual is *inferential/semantic-shift* errors that read as plausible paraphrases.
- AggreFact-CNN: 5 of 11 unsupported multi-sentence summaries passed (summary-level label; one bad sentence flips it); 0 false rejections.
- Sample size: 14 false passes. Fixing 3-4 errors would move dev κ to ~0.74, but dev κ has a standard error ≳0.07, so any dev-only win is not evidence.

## Why earlier approaches missed these
- Single-pass rubric judges (v1-v7, v16-v18, v22-v23) ask "is it supported" with generic criteria; a plausible paraphrase passes.
- Contradiction/decompose variants (v8-v15) used non-reasoning models and no error-type definitions; v20 (decompose+o4-mini) was not combined with an evidence check.
- Entailment classifiers (MiniCheck, HHEM, FactCG) and the RAGTruth LoRA are trained on baseless/contradiction-style errors, so they share v18's blind spot (corrected+introduced errors overlap heavily: 5 of FactCG's 7 misses in these classes are also v18's).
- Combiner on dev/train scores failed because MiniCheck is *more* permissive than v18 (adds false passes).

## Experiment (one, materially different)
Asymmetric audit: leave v18 unchanged; run a **second, adversarial pass only on cases v18 passed** (~88 dev, ~64 train positives). The auditor (o4-mini) gets the
TofuEval error-type definitions (reasoning step not in source, meaning/scope shift, tense/modality, mis-referencing, opinion-as-fact) and must output (a) the claim part at risk
and (b) a **verbatim source quote** that shows the discrepancy. A flip to "unsupported" is accepted only if the quote is found verbatim in the source (mechanical check) — this targets false passes without
re-creating the combiner's permissiveness, and bounds new false rejections.
- Data: existing 128 dev + TRAIN split only for prompt writing; the prompt/decision rule is frozen on TRAIN first (v18 TRAIN: 31 FP / 4 FN — the same failure mode). Definitions come from the published
  TofuEval taxonomy; no TofuEval CSV rows are used as training data (evaluation-only licence). No weights, no GPU.
- Cost/compute: ≈ (64 + 88) × ~$0.01 ≈ $1.5 plus retries; **requires a small API budget (currently $0): proposed cap $2.00**; no GPU hours.
- Success criteria, fixed before running (dev, once): corrected ≥ 4 and introduced ≤ 1 relative to v18 (net ≥ +3 errors), no per-subset kappa drop > 0.1, 100% coverage.
  Only if that passes: the same frozen audit on the planned fresh-document sample (≈320 rows, separate approval, ≤$3.50) with weighted κ and document-bootstrap CI vs frozen v18; target κ improvement CI lower bound > 0.
  0.74 is reported as measured, never as a pass criterion on dev alone. Failure → close; do not iterate the prompt on dev.
- Honest expectation: modest. Best case recovers the 5-7 inferential misses; it cannot touch the 5 false rejections, and 0.74 on fresh documents is unlikely to be reached by this alone.

## Revision after review (implemented offline, frozen, NOT run; $0 spent)
- Population corrected: **94 train + 88 dev v18 passes = 182 audits** (empirical v18 cost on these: ≈$0.0076 / $0.0070 per call, ~1.1-1.2k completion tokens incl. reasoning).
- Cost incl. reasoning tokens and retries: expected ≈ **$2.16**; worst case if every call hit max_tokens=4000 and retried ≈ $7.57, so a **hard total cap of $3.00** is enforced by a reservation meter
  (a call is admitted only if spent + in-flight + its worst case fits; unknown cost is charged at its bound; reaching the cap ends the stage INCOMPLETE). Train stage first (~$1.1), dev only after a COMPLETE train stage.
- Auditor: sees source + candidate only (no label, no v18 verdict). Must give disputed assertion (checked to be inside the candidate), error type, claim-says / source-says (must differ), evidence quote, explanation.
  Quote matching = authenticity only; a flip also needs lexical overlap between quote and assertion (an irrelevant-but-real quote is rejected, though a relevant-looking but misread quote can still pass — residual risk, reported not hidden).
  Unsupported additions flip only if the listed details are in the candidate and share no content word with the source.
- Synthetic tests (18, `routebench/tests/test_audit2.py`): paraphrase, tense change, speaker attribution, irrelevant real quote, fabricated quote, additions, short explanation, parse failures, cap enforcement, failed calls, κ arithmetic.
- Gate vs target kept separate. Gate (dev, once): corrected ≥ 4 and introduced ≤ 1. Actual κ is reported: on dev 4 corrections + 0 new = 0.7471; 4 + 1 new = 0.7313 (0.74 NOT reached by the gate alone). Incomplete runs stay INCOMPLETE and authorize nothing. All cases preserved.
- Frozen: prompt sha `f52f7800…`, rule sha `4c41342e…`, settings `1675a980…`; train/dev population hashes in `.local/audit2-v1/protocol.json`.
- Exact invocation (requires your approval; refuses without the confirm token):
  `PYTHONPATH="packages/core;routebench" .venv/Scripts/python.exe -m evalops.audit2 run --stage train --cap 1.5 --confirm RUN-AUDIT2-PAID`, then, only if train is COMPLETE, `--stage dev --cap 1.5 ...`.

## v2 execution layer (2026-09-30; supersedes v1, which was never executed; `.local/audit2-v1` preserved)
Fixes for the three reproduced defects (`routebench/evalops/audit2_v2.py`, 21 regression tests in `test_audit2_v2.py`):
1. Spend: fsynced journal (reserve before each provider attempt, settle after); crashed reservations count at their bound; total cap spans stages and restarts; every attempt (incl. failed/unparseable) is billed;
   input bound = 1 token per 2 chars (+200) plus the full 4,000 output tokens; cost above its reservation, NaN/negative/non-numeric cost → ABORTED (never COMPLETE); missing cost charged at the bound.
2. Binding: protocol stores full-content digests (source, candidate, label, v18 verdict per case), sha256 of audit2.py, audit2_v2.py, cases/annotations/judgment_cache/splits and settings. Re-derived and verified before any provider call; changed/missing/duplicate/malformed cases stop the run.
3. Failures: field types validated (bad explanation etc. → recorded failed audit); rows durable; κ comparison WITHHELD unless every audited case has an ok audit; κ is `None` when chance agreement is total; resume supported without repeating audited cases.
Frozen: audit2_v2.py `5a7ffe30…`, audit2.py `06ddfae0…`, train digest `d98dd62d…`, dev digest `c2bc3a50…`; total cap $3.00; expected ≈ $2.16; per-call reservation ≈ $0.019. Because the cap is a real total, a run costing more than expected ends INCOMPLETE rather than overspending.
Invocation (needs your approval): `PYTHONPATH="packages/core;routebench" .venv/Scripts/python.exe -m evalops.audit2_v2 run --stage train --confirm RUN-AUDIT2-PAID`, then `--stage dev` only after train is COMPLETE. Dev writes `final_dev.json` with full-population κ.

## v3 execution layer (2026-09-30; supersedes v2; v1/v2 preserved, none executed with paid calls)
Defect found by testing the real client path: v2 called `llm.chat()`, which retries internally, so one journal reservation could cover several billed HTTP requests; the shared client also turns missing usage into 0 tokens / $0.
`routebench/evalops/audit2_v3.py` (20 tests in `test_audit2_v3.py`, mocked HTTP through `aisys.llm._call`, `llm.chat` forbidden):
- one reservation = one HTTP request (calls `llm._call` directly); every retry is an outer, journaled attempt (503→success test: 2 reserve/settle pairs, 2 POSTs).
- input bound = UTF-8 bytes + 64 (a byte-level BPE tokenizer cannot emit more tokens than bytes; replaces len//2+200, which dense/non-ASCII text can exceed — tested); output bounded by the API's max_completion_tokens=4000 (reasoning included). Reservation per call $0.021-$0.026 (mean $0.024).
- raw usage validated; missing/malformed → charged the full reservation (never 0), flagged `usage_unverified`; usage above the reserved token ceilings, a served-model mismatch, or a quota error → run ABORTED.
- v2 population binding / durable journal / type-checked parsing / withheld-κ reused and re-hashed with all three code files.
Experiment, prompt, rule, populations (94 + 88) and the $3.00 total cap are unchanged. Frozen in `.local/audit2-v3/protocol.json` (train digest `d98dd62d…`, dev `c2bc3a50…`).
Invocation (needs approval): `PYTHONPATH="packages/core;routebench" .venv/Scripts/python.exe -m evalops.audit2_v3 run --stage train --confirm RUN-AUDIT2-PAID`, then `--stage dev` only after train is COMPLETE.

## Result of the audit2_v3 run (executed once, 2026-09-30; one process, train then dev)
Both stages COMPLETE, no aborts, usage verified on every call. **Actual total API spend $0.596** (journal, both stages, all attempts; cap $3.00).
- Train (94 v18 passes): auditor flipped 1 (corrected 1, introduced 0); 92 judged supported; 1 blocked (quote not in source).
- Dev (88 v18 passes): **0 flips**; 85 judged supported; 3 auditor "unsupported" verdicts blocked by the rules (2 additions with details present in source, 1 quote not in source).
- Full-population dev κ: v18 0.6745 → audited 0.6745 (corrected 0, new errors 0). **0.74 not reached; the improvement gate (≥4 corrected, ≤1 introduced) failed.**
Outcome: negative result; the experiment is closed. No prompt/rule tuning, held-out evaluation, or further runs were done. Files: `.local/audit2-v3/` (rows, journal, terminals, final_dev.json).

## Diagnostic: raw auditor verdicts vs filtered verdicts (cached outputs only; nothing re-run; recorded 2026-09-30)
**Official result is unchanged: dev κ 0.674518, 0 flips, gate failed.** This section is post-hoc and exploratory; it did not change any prediction.
Computed by `.local/diag_audit2.py` from `.local/audit2-v3/rows_{train,dev}.jsonl` ("raw" = auditor verdict "unsupported" before the application filters; full population = all cases of the split, v18 rejections unchanged):

| Split (N) | v18 κ | Raw flips → corrected / new errors / κ | Filtered flips → corrected / new errors / κ |
|---|---|---|---|
| TRAIN (129) | 0.447571 | 2 → 1 / 1 / 0.448246 | 1 → 1 / 0 / 0.463683 |
| DEV (128) | 0.674518 | 3 → 3 / 0 / **0.729172** | 0 → 0 / 0 / 0.674518 |

All three raw dev rejections matched gold (all three gold "unsupported"; 0.729172 is still below 0.74).
Which filter blocked what:
- TRAIN: the raw flip that was WRONG (gold supported; "A CBS employee tested positive for anthrax exposure", nuanced_meaning_shift) was blocked by `quote_not_in_source` — the filter helped there. The other raw flip was correct and passed (authentic related quote).
- DEV: `quote_not_in_source` blocked one CORRECT flip (tense_or_modality); the lexical-overlap addition test blocked two CORRECT flips ("What is your opinion on the comparisons between Bush and Clinton?", "Rudy Giuliani admits pushing out Ambassador Yovanovitch to pursue personal interests in Ukraine policy").

Correction of an earlier statement: sharing content words between a listed "unsupported detail" and the source does NOT show the detail is present in the source *as a fact* (the dev cases share words such as the topic, the names or "Ukraine policy" while the asserted claim, an opinion question or a motive, is not in the source). The filter treated word overlap as presence, so my earlier description ("details present in source") overstated what it checked; it only showed lexical overlap.
Evidence on whether filters help: quote filter 1 helpful (TRAIN) vs 1 harmful (DEV); lexical addition filter 0 helpful vs 2 harmful. Totals: raw flips 5 → 4 correct, 1 wrong (precision 4/5, 95% Wilson ≈ 0.38-0.96); filtered flips 1 → 1 correct. The samples are tiny and the filter conclusions on DEV are seen-after-the-fact, so they are hypotheses, not validation.

### One fixed decision rule proposed for an independent evaluation (not run; needs your approval and a budget)
Flip a v18 pass to "unsupported" iff ALL hold: auditor output parses with validated types; verdict "unsupported" with a known error_type; the disputed assertion is a verbatim substring of the candidate (normalized); claim_says and source_says are present and different; explanation ≥ 8 words.
Dropped as GATES: exact source-quote match and the lexical-overlap test for additions. Quote authenticity and overlap are still recorded as diagnostics, not used to decide.
Rationale: the two dropped filters are the only ones that blocked anything; their record is 1 help / 3 harms, and the overlap test is logically unsound. This rule is chosen AFTER seeing dev, so its dev/TRAIN numbers (raw: 4 corrected, 1 new error over 182 audits (94 TRAIN + 88 dev)) are not evidence of generalization.
Independent evaluation requirements: fresh documents not in TRAIN/dev/test (e.g. the TofuEval upstream-dev sample already proposed), v18 verdicts and the auditor outputs frozen before labels are joined, weighted κ with a document-bootstrap interval; success = net corrections > 0 with CI lower bound of the κ gain > 0 and new errors < corrections; a result of 0.74 is reported as measured, never a criterion. Cost ≈ v18 + auditor on ≈320 rows (≈$3.5 + ≈$1.1); requires separate budget approval. Held-out test stays sealed.
