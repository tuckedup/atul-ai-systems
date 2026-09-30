# Frozen protocol: MiniCheck + v18 regularized combiner (Option 1)

Recorded 2026-09-29 BEFORE any TRAIN v18 judgment was purchased and before any fit or dev evaluation.
Authorised: Option 1 only, hard total API cap $1.50 including retries. Forbidden: dev re-tuning, test evaluation,
new downloads, fine-tuning. Labels (including AI-review-disputed cases) unchanged.

protocol_hash: 42474034f80cf6fb

The hash covers `routebench/evalops/stack_minicheck.py` (all code) plus every constant below; each stage refuses
to run if it does not match this line.

## Exact settings
- Cases: headline-provenance TRAIN (129 cases, 40 documents) for fitting; headline dev (128) for the single check.
- Features (3): logit(clip(MiniCheck full-claim probability)), logit(clip(min over claim sentences of MiniCheck
  probability)), v18 raw score. MiniCheck scores are the precomputed, already-saved
  `minicheck_flan_t5_large_{train,dev}_scores.json` (model MIT, minicheck commit b58b9fa).
- Probability clipping: [0.001, 0.999] before the logit.
- Preprocessing: StandardScaler fitted ONLY on the training rows of each CV fold (and on all TRAIN for the final
  refit); the same fitted scaler is then applied to held-out rows and to dev.
- Model: sklearn LogisticRegression, L2, lbfgs, max_iter 1000, no class weights, no interactions, no subset terms.
- Folds: 5, document-grouped (a document is never split). Assignment: sorted document ids, numpy
  `default_rng(20260929).permutation`, round-robin over folds.
- Grid: C in {0.01, 0.1, 1, 10}; decision threshold on predicted probability in 0.05..0.95 step 0.05.
- Selection: maximise pooled out-of-fold binary Cohen kappa; ties -> smaller C, then threshold closest to 0.5.
  (Threshold is chosen on the same pooled OOF predictions, so the OOF kappa is mildly optimistic.)
- v18: config hash 12fc8b550d954359; 129/129 usable TRAIN judgments required, otherwise no fit is run.
- Dev (once): frozen scaler/coef/threshold; baselines = v18 at its frozen all-criteria-pass cut, MiniCheck at 0.5.
  Paired document-bootstrap interval (2000 resamples, seed 20260930) for the kappa difference vs each baseline;
  coverage, confusion counts and corrected/new errors versus v18.
- Dev is NOT an independent validation set (inspected and tuned on repeatedly in earlier rounds); the result is
  reported as such. No re-tuning after seeing it. No test evaluation.

## Results (2026-09-29)
- Spend: $1.0195 of the $1.50 cap (129 calls, no errors, no retries needed, no cap or reservation breach).
  Coverage 129/129 TRAIN v18 judgments; dev coverage 128/128.
- Selected by grouped CV in TRAIN: C=1, threshold 0.25, pooled out-of-fold kappa 0.495 (mildly optimistic).
  TRAIN baselines: v18 alone 0.448, MiniCheck alone 0.496. Coefs on standardized features: MiniCheck full 0.40,
  MiniCheck weakest-sentence 0.63, v18 0.83.
- Dev (single evaluation, non-independent): combiner kappa 0.593 (23 errors) vs v18 0.6745 (19) vs MiniCheck 0.526.
  Corrected 4 v18 errors, introduced 8 new. Paired document-bootstrap 95% CI of kappa difference: vs v18
  [-0.214, +0.015]; vs MiniCheck [-0.029, +0.213].
- Conclusion: the frozen combiner did not beat v18. Not frozen as a judge; no test run. Stopped as instructed.
  Note v18's TRAIN kappa (0.448) is far below its dev kappa (0.675): the dev-tuned figures are optimistic and
  judge quality is document-dependent with only 40-46 documents per split.
