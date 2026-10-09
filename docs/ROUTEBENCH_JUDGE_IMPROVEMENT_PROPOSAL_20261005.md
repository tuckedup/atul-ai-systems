> **SUPERSEDED 2026-10-05 by `ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_V2_20261005.md`** after the independent review. Known defects in this v1 text: the 90-99% power figures cover only two of the gate's conditions (the full gate succeeds about half the time at a true +0.03 gain); excluding only the 70 extra TofuEval documents left restricted-source rows in the fitting data; the Phase 1 budget omitted RAGTruth feature calls; the 'ceiling from the labels' wording is unsupported. Kept unchanged below for the record.

# PROPOSAL (offline, NOT APPROVED, nothing executed): improving the RouteBench groundedness judge

Prepared 2026-10-05. **Status: proposal only.** No vendor has been contacted, nothing purchased or downloaded, no model trained, no paid call made, no held-out access. Every phase below needs your separate written approval. The only things run for this document were a CPU simulation (`routebench/evalops/judge_improvement_power.py`, 7 tests, no data, no model) and reading existing repo files.

## 0. Closure of srcint-eval-v3
srcint-eval-v3 is **closed as a completed negative result**: COMPLETE 257/257, committed $2.0557 of the $3.00 cap, repaired-input dev kappa 0.6028 against the cached 0.6745 (document-bootstrap interval of the difference [-0.201, 0.000]), four dev flips, all error-creating; 0.74 not reached. The repair effect is not measurable against v18 run-to-run noise. All artifacts are preserved and untouched: `.local/srcint-eval-v1..v3/` (protocol sha256 `5cbfa028...`, results, spend journal, terminal, comparison), `.local/source-integrity-v2-20261004/`, and the v1/v2/v3 code and tests. The unspent $0.94 of that cap is not used by anything here.

## 1. What the evidence says the judge needs (why this is not just "validate the veto")
| Finding (existing artifacts) | Consequence for the design |
|---|---|
| v18's best dev kappa 0.6745 is likely optimistic: a repaired-input rerun gave 0.6028 and no tested change beat the noise | One v18 call per sentence is a noisy instrument. Measure and reduce its variance before attributing anything to a method. Always compare against a *pre-declared single v18 run*, not the best of several |
| Dev errors are FP-heavy (14 false passes / 5 false rejections); fresh documents are FN-heavy (37 false rejections / 5 false passes) | A single global strictness knob (veto-only, or threshold-only) fixes one profile and damages the other. The method must model both directions and be judged on both |
| 14 of 128 dev cases are wrong for every judge tried; many trace to upstream rater disagreement or missing context | A ceiling exists from the labels themselves. Uncertainty must be a first-class output and the human labels must carry it too |
| Every tested rescue failed (LoRA on RAGTruth/TofuEval, FactCG, second-pass auditor, Opus judge, stackers, claim-relation 0.58 post-hoc, srcint 0.60); the typed veto reaches 0.7134 only in-sample on 3 documents | Do not add another hard rule. Estimate a calibrated probability from several noisy signals, with document-grouped validation |
| All inspected TofuEval/AggreFact documents are exposed; only the sealed test (263 headline items, 50 documents) is unexposed | New evaluation documents with new human labels are the only way to get independent evidence beyond a single sealed run |

## 2. The candidate (one primary hypothesis)
**judge-v19-calibrated-abstain ("A2")**: a small, calibrated, three-outcome judge built on top of the frozen v18 recipe. v18 itself (model, rubric, template; hashes `12fc8b55...`, `d72496fb...`, `ef895d79...`) is not edited.

Signals (all declared before fitting; no signal added after seeing evaluation labels):
1. **v18 repeated judgments**, k = 5 independent calls per sentence on the *original, unrepaired* inputs: mean all-criteria-pass indicator, per-criterion fail rates, and between-call disagreement. This attacks the measured run-to-run noise directly.
2. **Local verifier scores** already used in earlier screens (MiniCheck, FactCG, HHEM) as inference only.
3. **Structural features** of the sentence/source pair: sentence length, count of numbers, named entities, negations and modals, source length, source type. No label-derived features.
4. Optional secondary arm A3 only: the typed-veto-v1.1 audit flags as one more feature (not a hard veto).

Model: L2-regularised logistic regression (CPU, minutes) producing p = P(source supports the sentence). No LoRA or other large-model training: every prior attempt failed and it is the most expensive, least interpretable lever.

Decision rule (predeclared form, thresholds fitted in calibration then frozen):
- **pass** if p >= t_hi, **reject** if p <= t_lo, otherwise **uncertain**. t_hi is chosen so the out-of-fold false-pass rate among passes is at or below a target; t_lo so the false-rejection rate among rejections is at or below a target. The targets are fixed in the pre-registration, equal for both directions, so neither error class is traded for the other silently.
- **Uncertain is never auto-rejected.** For the binary kappa required by the original benchmark definition, an uncertain item takes its best-guess verdict (p >= 0.5 passes, otherwise rejects) and stays in every denominator. The uncertain flag is reported separately: coverage, error rate in the uncertain band versus the confident band (it must be higher, otherwise the flag carries no information), and kappa on confident items only with its coverage.
- Arms: **A0** frozen v18, one call per sentence (the control; its run is made once and never re-selected); **A1** mean of the 5 v18 calls at the 0.5 rule (noise reduction only); **A2** primary; **A3** A2 plus veto flags. Only A2 vs A0 is the confirmatory test; A1 and A3 are explanatory and carry no separate success claim.

## 3. Data: training-permitted versus independent evaluation (disjoint by document)
| Role | Contents | Rights status (not legal advice, not clearance) | Used for |
|---|---|---|---|
| **T: training** | RAGTruth rows (labels MIT; CNN/DM text recorded as your private-research risk acceptance, 2026-09-30; "Recent News" excluded) + original RouteBench TRAIN split | As in `ROUTEBENCH_TRAINING_PERMISSION_STATUS.md` | Fitting the logistic model |
| **C: calibration** | Original dev split (128 cases, exposed, development role only) + out-of-fold predictions from T | Existing roles | Choosing regularisation, t_hi/t_lo, and the **futility gate** (Phase 1). Document-grouped nested cross-validation: no document in both fit and score folds |
| **Tier B (sensitivity only)** | The 70 TofuEval documents your 2026-10-01 instruction allowed for private training (annotations MIT-0, README says do not train; transcripts CC-BY-NC-SA) | Contested; recorded as your private risk decision | A *sensitivity arm* only. **Excluded from the primary candidate** so a later sealed-test result is not tainted by a data source whose terms forbid training |
| **E: independent evaluation** | ~150 new documents x 10 sentences, human-annotated, permissioned (Section 6) | To be obtained in writing; "unresolved" = not usable | The confirmatory comparison. Labels are never used for fitting, threshold choice or any design change |
| **S: sealed original test** | 263 headline items, 50 documents | Existing | Untouched. One frozen run only on separate authorization after the readiness gate |

Fingerprint checks (exact and near-duplicate, against the protected inventory) run on every E document before upload and before labelling. These do not prove semantic or pretraining independence; the proposal states that limit instead of relying on it. A document's sentences never cross a role boundary.

## 4. Procedure (document-separated, in order, with gates)
| Phase | What | Gate to continue |
|---|---|---|
| 0 | Send the amended quote request (Section 6). Preregister this protocol with code/config hashes (same freeze discipline as srcint-v3: manifest, reverify before first request, fsynced spend journal, fail-fast) | You approve sending |
| 1 | **Futility gate on existing data, no purchase.** Run v18 four more times on the 257 TRAIN+dev original inputs; compute features; document-grouped nested CV; estimate A2 vs A0 on out-of-fold predictions | Continue only if out-of-fold kappa(A2) - kappa(A0) >= +0.04 on dev-document folds **and** neither error class worsens. Exposed data cannot confirm anything; it can only stop spending early |
| 2 | Freeze A2/A1/A3 (weights, t_lo, t_hi, mapping, code hashes) and write the evaluation analysis script before any E document exists | Hashes recorded |
| 3 | Obtain E documents and a 10-document / 20-case calibration set (policy sanity and annotator training only; not part of E). Run frozen v18 once and frozen A2/A1/A3 on all E sentences **before** any human sees them. Select what humans label by a label-independent rule: every sentence where A2 and A0 disagree at probability 1 (subsampled at fixed probability if more than 250, weights kept) and a uniform 15% of the rest. Annotators are blind to every model output and to the hypothesis | Rights and quality checks pass |
| 4 | **Single confirmatory analysis** on E (Section 5). No re-fitting, no second candidate on E, no threshold change | Result recorded whether positive or negative |
| 5 | Only if Section 5 improvement criteria are met, ask for a separate authorization for **one** frozen run on S under the existing readiness gate | Your approval |

## 5. Predefined criteria (all fixed before E labels exist)
**Metrics:** weighted (inverse-probability) accuracy and Cohen's kappa for A0 and A2, false-pass rate (judge pass, humans unsupported) and false-rejection rate (judge reject, humans supported) reported separately, flip precision (fraction of changed verdicts that fix an error; >0.5 means net errors fixed), uncertain-band coverage and error rate. Document-grouped paired bootstrap, 5,000 draws, 95% intervals. Accuracy and kappa improvements are reported separately; kappa can rise with zero accuracy change.

**Improvement (A2 over A0) requires ALL of:**
1. lower 95% bound of the accuracy difference > 0 **and** lower bound of the kappa difference > 0;
2. point-estimate accuracy gain >= +0.03 (about +0.07 kappa at 85% accuracy and 70% prevalence; reaching 0.74 needs roughly +4 points from 0.85);
3. **both error classes non-inferior:** neither the false-pass rate nor the false-rejection rate rises by more than 1 percentage point of all sentences, and at least one falls;
4. flip precision > 0.5 with its lower bound reported;
5. the uncertain band's error rate exceeds the confident band's (otherwise the abstain output is dropped from the claim, though A2's binary result still stands on its own).

**Label quality guard:** annotators' pre-adjudication agreement is reported. If their Cohen's kappa is below 0.60, absolute kappa claims are withheld ("label-noise-limited"); only the paired comparison is reported, and its noise assumption is stated.

**Stopping rules:**
- Stop and close as a negative result if the Phase 1 gate fails (nothing is bought).
- Stop at Phase 3 if written rights for E documents are unresolved, if fingerprint overlap is found, or if any cost cap below would be exceeded.
- Stop after Phase 4 whatever the result. A failure is final for this candidate: E is consumed, no alternative candidate is evaluated on it, and no "one more tweak" loop starts.
- Stop any phase on the first request error, unsettled spend, or integrity failure, as in srcint-v3.
- The sealed test is never touched in Phases 0-4.

**Power (simulation, not a guarantee; assumptions in `.local/judge-improvement/power_analysis.json`):** 150 documents / 1,500 sentences, v18 accuracy 0.85, prevalence 0.70, document-level heterogeneity, 15% sampling of agreements, disagreements fully labelled. The chance that both lower bounds clear zero:
| True accuracy gain | disagreement rate 6% | 9% | 12% |
|---|---|---|---|
| 0.00 (useless) | about 2% | about 3% | about 2% |
| 0.02 | about 0.85 | about 0.70 | about 0.60 |
| 0.03 | about 0.99 | about 0.97 | about 0.90 |
| 0.05 | 1.00 | 1.00 | 1.00 |
(1,500 values interpolate the simulated N = 1,000 and 2,000 rows; N = 2,000 gives 0.97 to 1.00 at a 0.03 gain and 0.70 to 0.94 at a 0.02 gain.) A true gain of 2 points would often be missed; the proposal does not claim to detect it.

## 6. Itemised cost (unit counts are mine; **prices for rights and annotation are not known and I will not invent them**)
**Data rights (vendor-quoted, per document, 160 docs = 150 + 10 calibration):** S1 annotate, S2 train/evaluate, S3 private retention, S4 optional redistribution, S5 provenance and legal review, S6 sensitive-content screening, S7 acquisition if the vendor supplies sources. Training permission must cover text and annotations. The v2 quote draft's quantities need one amendment for this study: labelled cases **about 300-380 (cap 450)** instead of about 265 (cap 320), because A2 vs A0 will disagree on more sentences than a veto flags. Planned human work: about 740 annotations (cap 940, two per case incl. calibration), about 90-115 adjudications (every disagreement and every "uncertain"), 20 blind QA reviews. Total: **unquoted**; send amended draft only on your approval. **Your own risk acceptances** are free in money but are not rights clearance.

**Compute (local):** Phase 1 and Phase 3 feature extraction is inference only, <= 3 GPU-hours (ledger now about 4.47 of 40 h; projected <= 7.5 h). Logistic fitting and bootstrap: CPU, minutes. No LoRA/large-model training. Power simulation: done (about 77 s CPU).

**API (OpenAI, o4-mini medium; observed $2.0557 for 257 calls, about $0.008 per call):** Phase 1: 4 extra runs x 257 = 1,028 calls, about **$8.2**. Phase 3: E at k = 5 for A1/A2/A3 (A0 is call one of the five): 1,500 x 5 = 7,500 calls, about **$60** (plausible range $45-75 depending on source length) plus calibration cases about $1. **Requested hard cap $90 total across all phases including retries**, enforced by the fsynced reserve-before-launch journal; this is above the $60 campaign ceiling recorded on 2026-10-01 and needs your explicit raise. Claude-subscription runs for arm A3 draw on subscription allowance (notional cost only, no cash).

**Not included:** vendor project fees, taxes, your time, any legal review beyond the vendor's provenance record. The proposal gives no total because the dominant lines are unpriced.

## 7. What a success would and would not support
- **New-data success (Section 5 met on E)** supports exactly this: *on these new, independently human-annotated, permissioned documents under the new written policy, the frozen A2 judge is more accurate and has a higher kappa than frozen v18, without trading one error class for the other.* It is evidence for a **new benchmark** (RouteBench-E, a separately named version). It is **not** a pass of the original benchmark target: the documents, annotators, policy, sampling and source mix differ from the original, and an absolute kappa >= 0.74 on E would not substitute for it.
- **Original-claim support** exists only through the sealed test S: one frozen run of the *unchanged* A2 under the original labels, reported with its interval (50 documents make that interval wide), after separate authorization. Passing E is a precondition for requesting that run, not a substitute for it. Reaching 0.74 on S would support the original claim with the stated uncertainty; missing it closes the claim with a negative result. Using the primary candidate without Tier B training keeps that run free of a data source whose terms forbid training.
- Failure on E (or at the Phase 1 gate) means this line of improvement is a negative result: the original target would remain unmet and the judge unimproved.
- Nothing here changes, removes or relabels any original case, label or split; the original results, caches and frozen protocols stay intact.

## 8. Risks stated plainly
1. Variance reduction may be most of any gain; the design isolates it (A1). That is still a real improvement if it passes, but it multiplies API cost by 5.
2. Exposed development data are weak evidence; Phase 1 is a futility test only.
3. The 0.74 target may exceed what the labels allow; label-agreement reporting guards against over-reading.
4. A reproduction of 1,500 sentences from a different source mix may transfer poorly to S; E success does not guarantee S success.
5. All cost figures except API and GPU are unknown until the vendor replies.

## 9. Approval requested (each separately; nothing runs without them)
1. Phase 0: approve preparing the preregistration manifest (offline) and sending the amended quote request (no purchase).
2. Phase 1: approve about $8.2 of OpenAI calls and about 0-1 GPU-hours (futility gate only).
3. Phase 3-4: approve vendor purchase, the $90 total API cap, and the single confirmatory analysis, only after the Phase 1 gate and the rights check.
4. Phase 5: a separate sealed-test authorization.
