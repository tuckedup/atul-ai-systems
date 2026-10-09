# Independent review: judge-improvement proposal

Date: 2026-10-05. Scope: read-only assessment of Claude's proposal and simulation, with local synthetic computations. No provider calls, vendor contact, purchases, downloads, GPU work, training, or acceptance-test evaluation. Production code and frozen experiments were not edited.

## Decision

Revise before execution or vendor contact. The candidate is a reasonable **hypothesis**, not a demonstrated improvement: repeated v18 judgments plus existing verifier signals feed a regularized classifier, with both rejection directions evaluated. Preserving uncertain cases, original labels/splits, and separate independent evaluation is appropriate. The only new executable code reviewed is a power simulation, not a trained judge or ready paid experiment.

I independently reran all 7 simulation tests and scoped Ruff checks; both pass. I did not rerun the reported 1,288-test full suite. Passing these tests establishes basic simulation behavior, not the scientific validity or achievable quality of the proposed judge.

## Findings to resolve before approval

### 1. Power does not cover the complete success rule

`judge_improvement_power.simulate_once()` returns `both_ok = acc_ok * kap_ok`: only the two lower-confidence-bound conditions. It does not apply the additional required point-estimate accuracy gain of at least 0.03, the error-class non-inferiority conditions, or uncertain-band informativeness. The simulated design also does not implement the proposed disagreement-subsampling branch above 250 items. The stored table uses N=1,000/2,000/3,000/4,000, while the proposal interpolates to N=1,500.

Independent direct N=1,500 check: 300 replications per scenario, 300 bootstrap draws per replication, using the existing simulator, v18 accuracy 0.85, true candidate accuracy gain 0.03, prevalence 0.70, and seeds `20261005 + int(disagreement * 100)`:

| Disagreement rate | Both confidence bounds positive | Also meets point accuracy gain >=0.03 | Mean labels |
| --- | ---: | ---: | ---: |
| 0.06 | 1.000 | 0.537 | 301.3 |
| 0.09 | 0.953 | 0.503 | 341.2 |
| 0.12 | 0.903 | 0.520 | 378.7 |

These are Monte Carlo results under the simulator's assumptions, not estimates of real-world judge performance. They show that 90-99% detection power for the two interval conditions is not 90-99% probability of satisfying the complete proposed gate. Do not weaken a criterion merely to restore a headline power number. Either implement the full gate and exact sampling rule in the simulation, or explicitly label unsimulated conditions and make only the narrower power claim.

### 2. Primary fitting data contradict the stated training-source exclusion

The proposal excludes the 70 upstream TofuEval documents to avoid a source whose terms forbid training, but includes the original RouteBench TRAIN split for fitting. The bounded loader gives 63 TofuEval-MediaS + 31 TofuEval-MeetB + 35 AggreFact-CNN cases in TRAIN. The original dev calibration set also contains 63 MediaS + 24 MeetB + 41 CNN cases.

The repository's own `ROUTEBENCH_TRAINING_PERMISSION_STATUS.md` records the no-training concern for TofuEval/LLM-AggreFact/FRANK-derived rows, not merely the 70 additional documents. Fitting a logistic model on their labels, or selecting its thresholds/regularization with them, does not make that concern disappear. This is an internal policy inconsistency, not a new legal determination. Apply a consistent eligibility rule to every fitting and calibration input; disclose any private-risk acceptance separately. Do not claim that excluding only the 70 documents clears the primary recipe or resolves release rights.

### 3. Training-feature calls and scope are not fully budgeted

Phase 1 prices only four additional v18 passes on 257 RouteBench items. Yet training also includes RAGTruth, and the declared feature vector requires five v18 predictions per training sentence. The proposal does not specify the eligible RAGTruth sample size, five-call feature availability, or the cost of generating missing features. The local verifier inference estimate also depends on that missing sample size.

Before approval: inventory usable feature caches with content/config bindings, choose the exact training population, specify how every training row receives the declared features, and price/reserve every missing provider attempt. For illustration only, 1,050 RAGTruth sentences with five missing judgments each would mean 5,250 additional calls; this is not included in the current 1,028-call Phase 1 calculation. Do not assume zero missing calls or silently use a different feature schema for RAGTruth. Give Phase 1 its own hard cap and retain the distinction between cost estimates and cap reservations.

### 4. Freeze the learning/analysis procedure and remove unsupported ceiling claims

Document-separated nested CV is the right direction, but the proposal must specify exact eligible documents, feature normalization within folds, regularization grid, outer/inner split seeds, calibration fitting, threshold targets and feasibility handling. Evaluation labels must not feed any of these. Predeclare the binary scoring rule for human-uncertain or unresolved adjudications and how all such cases remain reported.

The proposal states that a label ceiling exists because multiple judges miss the same cases. That observation can also reflect shared model errors or policy mismatch; it does not establish a mathematical ceiling. Likewise, 0.6745 on original inputs and 0.6028 on repaired inputs do not establish v18's true performance distribution. Repeated calls may reduce variance, but they are not proven statistically independent and may share systematic errors. Treat benefit as the hypothesis under test.

## Recommended next instruction to Claude

Revise the proposal offline only. Preserve the negative results and every existing frozen artifact. Correct the four findings above, with a complete phase-specific budget and a consistent data-use policy. Simulate N=1,500 directly and distinguish detection power from full-gate success probability; document any remaining unsimulated assumptions. Return the exact small Phase 1 work order, its manifest and its hard cap for separate approval. Do not send the vendor quote, raise the $60 campaign ceiling to $90, spend, download, train, or open the sealed test during this revision. Do not add candidates or a sweep.

Once the revised Phase 1 is coherent, it can be considered as one bounded screening experiment. Vendor acquisition and independent evaluation should remain conditional on a useful screening result. Neither a simulation nor a new-population result establishes the original kappa >=0.74 claim.

API-equivalent cost receipt: unavailable; parent token usage is not exposed. Evaluation-provider spend during this review: none.
