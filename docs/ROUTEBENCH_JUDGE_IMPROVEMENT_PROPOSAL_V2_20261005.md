# PROPOSAL v2 (offline, NOT APPROVED, nothing executed): judge-improvement screening work order

Prepared 2026-10-05. Supersedes `ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_20261005.md` (kept; a banner now points here) after `ROUTEBENCH_JUDGE_IMPROVEMENT_INDEPENDENT_REVIEW_20261005.md`. **No vendor contact, purchase, download, training, paid call, budget increase or sealed-test access has happened or is requested.** Offline work done for this revision: a deterministic sample draw, two CPU simulations, 13 new tests (full set for the three new modules: 20 tests, all pass, Ruff clean), reading repo files. The candidate judge is **not implemented**; this is a plan for one bounded screening experiment, not a result.

## 0. The four findings and what changed
| Finding | Verdict after checking | Resolution (criteria not weakened) |
|---|---|---|
| 1. Power overstated for the full gate | **Correct.** I reproduced it: at a true 3-point gain the full gate passes about 47-51% of the time, not 90-99% | Section 6: the complete gate (c1-c5) and the exact label-cap branch are simulated at N = 1,500 directly; **detection power** and **full-gate success probability** are reported separately; unsimulated assumptions are listed. Every criterion is unchanged |
| 2. Training-source policy inconsistent | **Correct.** The original TRAIN split holds 63 TofuEval-MediaS + 31 TofuEval-MeetB + 35 AggreFact-CNN cases and dev holds 63 + 24 + 41; excluding only the extra 70 documents did not remove the recorded no-training concern | Section 2: one rule for every fitting, calibration and selection input. **No TofuEval, AggreFact or FRANK-derived row (including all of original TRAIN and dev) is used to fit, select, calibrate or threshold anything.** They are used only to score a frozen model (evaluation-only use) |
| 3. Training-feature calls and scope unbudgeted | **Correct.** The v1 budget priced 1,028 calls and no RAGTruth features | Section 3-4: exact fitting population (1,050 RAGTruth sentences, 160 document groups), feature inventory, **6,278 total provider calls**, separate Phase 1 hard cap **$56.00** (estimate $50.22), cost estimate and cap reservation distinguished |
| 4. Unsupported ceiling claim; procedure not frozen | **Correct.** Shared errors do not prove a mathematical ceiling; repeated calls may share systematic errors | Section 1 removes the claim and states the benefit as the hypothesis under test; Section 5 freezes the nested-CV procedure, scaling, grid, seeds, thresholds, feasibility handling and the scoring of uncertain human labels |

## 1. Evidence statements, corrected
- Several judges fail the same cases. That is consistent with correlated model error, upstream rater disagreement, or policy mismatch; it **does not establish a ceiling** and I withdraw the v1 wording. Nothing here assumes 0.74 is or is not reachable.
- v18's 0.6745 (cached) and 0.6028 (repaired-input rerun) are single draws of a noisy instrument; its true run-to-run distribution is unmeasured. Phase 1a measures it (four more runs on the same 257 inputs).
- Repeated calls may reduce variance but are **not** shown to be independent; correlated systematic errors remain possible. Whether k = 5 helps is a hypothesis, tested by A1 (voting alone) and A2.
- **Prior evidence is unfavourable and I should say so.** A close relative of this candidate (frozen MiniCheck + v18 regularized combiner, `ROUTEBENCH_COMBINER_PROTOCOL.md`, fitted on TofuEval-derived TRAIN rows) scored dev κ 0.593 vs v18 0.6745, paired interval [-0.214, +0.015], and was not adopted. A2 differs in what it fits on (RAGTruth only), in the repeated v18 calls, and in the two class-conditional bands, but a prior that it works should be modest. That is the reason for a cheap screen before any purchase.

## 2. One data-eligibility rule (every input)
| Population | Exact content | Allowed use | Basis |
|---|---|---|---|
| **FIT (the only fitting, selection and calibration data)** | 1,050 RAGTruth CNN/DM sentences: 160 document groups, 338 unsupported + 712 supported, drawn by rule `phase1-v1` from the TRAIN partition of the frozen training manifest (`manifest_sha256` 1645dd3f...; 45 boundary-crossing sentences dropped before the draw). Sample file `.local/judge-improvement/ragtruth_fit_sample_phase1-v1.json`, `sample_sha256` `7edc80e3ead8e5f6bc308807078e6a3a7a40d9f0c6afcff9d3f7445452d66012` | Features, nested CV, hyperparameters, thresholds, final fit | Labels MIT; CNN/DM text is your recorded private-research risk acceptance (2026-09-30), **not rights clearance**; "Recent News" excluded; the manifest itself says `training_use_cleared: false`. Its sentence labels are derived from span annotations, not independent expert sentence annotation: a label-quality and policy-transfer risk |
| **SCREEN (evaluation-only)** | RouteBench original TRAIN 129 + dev 128 cases, original unrepaired inputs, original labels and splits | Scoring the **frozen** A0/A1/A2 once. No fitting, no hyperparameter or threshold choice, no feature selection, no model choice | Dataset terms for TofuEval/AggreFact forbid training; evaluation is the permitted use. Already heavily inspected, so this is a futility screen, never confirmation |
| Not touched | 70 TofuEval upstream documents (your 2026-10-01 private-training decision stays recorded but is **not used** here); RAGTruth select/validation/test partitions and the pilot's samples; the sealed test (263 items, 50 documents); all frozen v1-v3 artifacts | none | The primary recipe then contains no restricted-source fitting, so a later sealed-test run is not tainted by it |

Honest consequence: all fitting is on a **different distribution** (LLM summaries of CNN/DM articles, 93% supported) than the screen (TofuEval meetings/media, AggreFact CNN, about 60-70% supported). The screen tests exactly that transfer. If it fails, that is the answer for this recipe.

## 3. Feature inventory and exact call counts (complete feature generation)
Feature schema S (frozen, nine features, identical for every row): (1) mean over five v18 calls of the raw rubric score, (2) fraction of the five calls that pass all criteria, (3) standard deviation of the raw score across the five calls, (4) MiniCheck-Flan-T5-Large full-source logit, (5) MiniCheck weakest-sentence logit (same definition and clipping as `stack_minicheck._features`), (6) log sentence length, (7) log source length, (8) count of digits in the sentence, (9) negation-or-modal flag. No label-derived and no source-type feature. A row missing any feature is an incomplete run, not a different schema.

| Feature | RouteBench 257 (screen) | RAGTruth 1,050 (fit) |
|---|---|---|
| v18 judgments (config `12fc8b55...`) | one cached per case (cache keys hold exactly one judgment); **need 4 more: 1,028 calls** | none; **need 5: 5,250 calls** |
| MiniCheck full + weakest sentence | cached for all 257 (`minicheck_flan_t5_large_{train,dev}_scores.json`, 129 + 128) | not cached for these 1,050 (the earlier pilot scored select/validation rows only); compute locally, inference only; weights are already in `.local/ckpts` and no download is allowed (absent weights = stop) |
| Structural features | free | free |
| **Total provider calls** | 1,028 | 5,250 → **6,278** |

## 4. Phase 1 work order (the one item for approval)
Approving this order authorizes **only W0-W2 (offline build, tests, freeze, no spend)**. I return the protocol hash and the exact command, and execution (W3 onward) needs a second approval naming that hash, as for srcint-v3.

| Step | What | Spend |
|---|---|---|
| W0 | Offline: write the frozen runner for the declared feature jobs (reusing the srcint-v3 safeguards: manifest binding input/code/config hashes, fsynced reserve-before-launch journal with one HTTP request per reservation, raw-JSON model-identity check, fail-fast on the first error, numeric-domain journal validation, explicit `--resume`), a case builder that turns a RAGTruth sentence plus its article into the same case record v18 sees for RouteBench, and the fitting/analysis script | none |
| W1 | Offline: tests (synthetic provider only), inventory check of every cache with content hash, dry-run call count and cost projection | none |
| W2 | Offline: freeze manifest `judge-v19-phase1` listing: sample sha `7edc80e3...`, RouteBench case ids and input digests, v18 config/template/rubric hashes, MiniCheck revision, schema S, nested-CV constants (Section 5), gate constants (Section 5), caps, code hashes | none |
| W3 = **Stage 1a** | 1,028 v18 calls on the 257 original inputs. **Cap $10.00** (estimate $8.22 = 1,028 x $0.0080, where $0.0080 is the observed $2.0094 / 257 settled in srcint-v3). Report: five single-run κ values for v18 (its run-to-run distribution), A1 (majority vote over the five calls' original all-criteria-pass verdicts) vs A0 | $8.22 est |
| Checkpoint | **Stop and report. Stage 1b starts only on your explicit go.** If A1 and the five single runs show no usable signal you may stop here at $8.22 | |
| W4 = **Stage 1b** | Local MiniCheck features for 1,050 rows (inference only, <= 1.5 GPU-h; ledger now 4.47 of 40 h). Then 5,250 v18 calls on the fitting sample. **Cap $46.00** (estimate $42.00 = 5,250 x $0.0080; plausible range $31.5-$57.8 at $0.006-$0.011 per call because RAGTruth sources average 3,711 characters). **Projection stop:** after the first 500 calls, if projected 1b spend exceeds $46.00 the run stops with about $4 spent | $42.00 est |
| W5 | Offline fit: nested CV, final fit, frozen weights and thresholds (hashed) before any screen label is read | none |
| W6 | Score frozen A0/A1/A2 on the 257 RouteBench cases and apply the screening gate once | none |
| W7 | Report whatever happens; update BLOCKERS; no re-fit, no second candidate, no sweep | none |

**Phase 1 hard cap: $56.00 total (1a $10.00 + 1b $46.00); estimate $50.22.** The estimate is not the cap: the cap bounds committed spend (settled charges plus every unsettled reservation at its bound of about $0.047 per outstanding call) and enforcement is by the journal, not by this arithmetic. Recorded settled campaign spend since 2026-10-01 is about $3.66 (multi-judge $1.5681, claim-relation ablation $0.0347, srcint-v3 $2.0557; my reading of which journals count against the 2026-10-01 $60 ceiling). $3.66 + $56.00 = $59.66, **inside the $60 ceiling; no increase is requested.** If a cap or projection stop triggers, the stage ends INCOMPLETE, nothing is analysed, and any continuation is a new approval. Wall-clock is unmeasured: the runner is serial (no concurrency in srcint-v3); I will report measured latency after 1a rather than guess.

## 5. Frozen learning and analysis procedure (all constants fixed in the W2 manifest)
- **Splits:** document group = `group_id`. Outer fold of a group = `int(sha256("phase1-v1|outer|20261105|" + group_id)) mod 5`; inner folds within each outer-training set = `... "inner|<outer>|20261105" ... mod 4`. A group is never split. Folds are stratified by nothing and may be unbalanced; fold sizes are reported.
- **Model:** logistic regression, L2, class-balanced weights (so p = 0.5 is the equal-prior point and does not depend on RAGTruth's 93% supported share). StandardScaler is fitted on each training fold only. Grid C in {0.01, 0.1, 1, 10}, chosen by inner-CV class-balanced log-loss; ties go to the smaller C. No other hyperparameter, feature subset or model class.
- **Out-of-fold predictions:** the outer loop yields one prediction per FIT row from a model that never saw its group. Pooled out-of-fold predictions fix the thresholds.
- **Bands (class-conditional, so they do not depend on prevalence):** t_hi = the smallest t >= 0.5 with P(p >= t | unsupported) <= 0.10; t_lo = the largest t <= 0.5 with P(p <= t | supported) <= 0.10. Both always exist (t = 1 / t = 0 qualify). Pass if p >= t_hi, reject if p <= t_lo, otherwise uncertain. **Feasibility rule:** if the confident band covers under 30% of out-of-fold rows, the abstain output is dropped from any claim and A2's binary result is judged alone. No threshold is touched afterwards.
- **Final model:** refit on all 1,050 rows with C from the same inner procedure over all groups (5 inner folds, same seed rule); weights, scaler, C, t_lo, t_hi hashed before step W6.
- **Binary rule (used for kappa):** p >= 0.5 passes, else rejects; uncertain items take that verdict and stay in every denominator; **uncertain is never auto-rejected**. The uncertain flag is reported separately (coverage, error rate in each band, kappa on the confident band with its coverage).
- **Human-uncertain or unresolved labels (Phase 3 only; none exist in Phase 1 because the original labels are binary):** the primary binary kappa excludes them and reports how many; two sensitivities map them to supported and to unsupported; the claim stands only if the direction holds in all three, and they are always counted in the report.
- **Arms:** A0 = the cached frozen v18 run (the single call that produced 0.6745, an optimistic draw, so the comparison is conservative); A1 = majority vote (at least 3 of 5) over each v18 call's ORIGINAL all-criteria-pass verdict at the threshold of 1.0 (no fitting; it isolates voting and does not loosen the rubric: five calls that each score 0.8 are five rejections, and A1 rejects them; a mean-score threshold is not used anywhere). A2 keeps the raw scores as features; A2 = the primary candidate. A3 (veto-flag feature) is dropped from Phase 1: no new candidate, no sweep.
- **Screening gate (pooled TRAIN + dev, unweighted binary kappa, document-grouped intervals reported but not gating):** s1 κ(A2) - κ(A0) >= +0.04; s2 the TRAIN difference and the dev difference are each > 0; s3 neither false-pass nor false-rejection count rises by more than 1% of the 257 cases and at least one falls; s4 flip precision > 0.5. All four required. Also reported (not gating): κ(A2) vs the mean of the five single-run κ values, A1, the uncertain band, TRAIN and dev separately, and the RAGTruth out-of-fold diagnostics.
- **A pass is not a result.** It means only that a purchase decision may be considered.

## 6. What the simulations show (and do not)
`.local/judge-improvement/gate_simulation.json`; `routebench/evalops/judge_improvement_gate_sim.py`; 1,000 replications per cell, 300 bootstrap draws in simulation (5,000 in the real analysis), seed 20261105. Assumptions: v18 accuracy 0.85, prevalence 0.70, document-level heterogeneity in prevalence and disagreement, a candidate whose wins are spread over both error classes in proportion to base rates, uncertain flags with P(uncertain | wrong) 0.30 vs 0.10 when right, human labels treated as truth.

**Evaluation gate, N = 1,500 directly (150 documents x 10; labels: all disagreements plus a 15% sample of agreements; the 250-label cap never engaged at N = 1,500):**
| Disagreement rate (mean labels) | True gain | Detection power (c1 only) | **Full-gate success (c1-c5)** |
|---|---|---|---|
| 6% (301) | 0.00 | 0.03 | 0.00 |
| | 0.03 | 0.997 | **0.47** |
| | 0.04 | 1.00 | 0.91 |
| | 0.05 | 1.00 | 0.99 |
| 9% (340) | 0.00 | 0.02 | 0.00 |
| | 0.03 | 0.97 | **0.51** |
| | 0.04 | 1.00 | 0.89 |
| | 0.05 | 1.00 | 0.98 |
| 12% (378) | 0.00 | 0.03 | 0.00 |
| | 0.03 | 0.91 | **0.51** |
| | 0.04 | 0.99 | 0.85 |
| | 0.05 | 1.00 | 0.98 |
Reading: detection power is high at a true 3-point gain but the **full gate succeeds only about half the time at exactly +0.03**, because criterion c2 is a point-estimate threshold; it succeeds about 85-91% at a true +0.04 and about 98% at +0.05. A useless candidate essentially never passes (0.00). More data does not fix the c2 coin flip (N = 3,000 gives 0.48-0.50 at +0.03 and 0.91-0.99 at +0.04, with the label cap engaged for 85-100% of runs at 9-12% disagreement). A forced asymmetric scenario (disagreements 40% supported instead of 70%) changes every probability by under 0.03. c3, c4 and c5 rarely bind in this simulation, which is a statement about the assumed process, not evidence they are easy. **Unsimulated:** label noise in the human labels; candidates that trade one error class for the other (c3 could then bind); between-document correlation beyond the two heterogeneity terms; non-stationary v18 behaviour; the real (unknown) distribution of the candidate's gain; sentence-level dependence inside a document.

**Screening gate (257 fully labelled cases, TRAIN 129 / dev 128):** a useless candidate passes about 10-14%; a true +0.02 gain passes about 42-47%; +0.03 about 58-72%; +0.05 about 86-98% (second setting, v18 accuracy 0.78 / prevalence 0.55: within 0.04 of these). The screen is deliberately permissive about noise: 1 in 8 useless candidates would continue to a purchase decision, which is why the purchase decision stays separate.

## 7. Cost, itemised (estimates distinguished from caps)
| Item | Phase 1 (now) | Later, conditional on a screen pass and your separate approvals |
|---|---|---|
| OpenAI v18 calls | 6,278 calls; estimate **$50.22**; **hard cap $56.00** (1a $10.00, 1b $46.00); $3.66 already counted against the $60 ceiling | Independent evaluation at k = 5: 1,500 x 5 = 7,500 calls, about $60 (range $45-75). **Would exceed the $60 campaign ceiling; no raise is requested or implied** |
| Local GPU | MiniCheck inference on 1,050 rows, <= 1.5 h (ledger 4.47 -> <= 5.97 of 40 h) | Same on evaluation rows, <= 1.5 h |
| CPU | nested-CV logistic fits, minutes | bootstrap, minutes |
| Data rights | none | about 160 documents x S1-S7 per the v2 quote draft; **unpriced, I will not guess** |
| Human annotation | none | about 740 annotations (cap 940), 90-115 adjudications, 20 QA, 10 calibration documents; unpriced. Quote draft needs the labelled-case amendment (about 300-380, cap 450) before any sending |
| Excluded | vendor fees, taxes, your time, legal review beyond a provenance record | same |
Not purchased and not sent: the vendor quote request stays a draft.

## 8. What success would and would not support
- **Screen pass** supports one thing: continuing is not obviously futile on already-inspected data; it supports no benchmark claim.
- **Independent-evaluation success (later, new human-labelled documents, full gate c1-c5)** supports an improvement on a **new benchmark** only, not the original target: different documents, annotators, written policy, sampling and source mix.
- **The original claim** (κ >= 0.74 against the original labels with a frozen judge) can be supported only by one frozen run on the sealed test under the readiness gate, with its interval (50 documents make it wide), after separate authorization. Because FIT contains no restricted source, such a run would not rest on training data whose terms forbid it. Reaching it is not implied by any of the above.
- A failed screen or failed evaluation is a negative result for this recipe; the original target stays unmet and nothing is relabelled, removed or re-split.

## 9. Requested decision
Approve W0-W2 (offline build, tests and freeze of `judge-v19-phase1`, no spend), after which I return the hash and exact command and stop. Execution of Stage 1a, then separately Stage 1b, each needs its own approval. Nothing else is requested.

## 10. Corrections after the second independent review (2026-10-05)
- The fitting-sample hash in Section 2 had a transcription typo (63 characters). The verified value is `7edc80e3ead8e5f6bc308807078e6a3a7a40d9f0c6afcff9d3f7445452d66012` (rederived by re-running the draw and by the independent check).
- A1 is now defined as majority voting over each call's original pass/fail verdict at 1.0, not an average of raw scores: averaging five 0.8 scores and testing at 0.5 would pass five rejections. Regression tests prove the two definitions differ (`test_judge_v19_analysis.py`).
- Build status: see `docs/ROUTEBENCH_JUDGE_V19_PHASE1_FREEZE_20261005.md` for the frozen protocol hash, exact commands, call counts, caps and the full-suite result.
