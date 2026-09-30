# Frozen reporting rule: v18 on the TofuEval upstream-dev primary sample

Recorded 2026-09-29 BEFORE any judgment was purchased. Authorised: frozen v18 (config hash 12fc8b550d954359, decision
rule raw_score >= 1.0) on the 320 PRIMARY manifest rows only; hard cap $3.50 including retries; no diagnostic rows, no
training, no threshold change, no RouteBench test evaluation. Kept separate from the original RouteBench result.

protocol_hash: 1734d6f6894e8186

The hash covers `evalops/upstream_validation.py`, the constants and the sampling manifest (sha256 4906220a...). Each stage
refuses to run on a mismatch.

## Reporting rule
- PRIMARY (headline) = sentence-population kappa from inverse-inclusion-probability-weighted confusion counts
  (w = 1 / inclusion_prob from the manifest; inclusion_prob = (20/35) x (8/N_doc)), pooled over both sources. It
  targets "a random summary sentence from the 70 upstream-dev documents", NOT an equal-weight-per-document average, and
  it does not pretend to represent the RouteBench mix or prevalence. Uncertainty: document bootstrap, stratified by
  source (resample each source's 20 sampled documents with replacement, seed 20261004, 2000 draws, weights kept),
  95% percentile interval.
- SECONDARY, reported separately: unweighted 320-row kappa (with its own interval), weighted and unweighted
  sensitivity / specificity / balanced accuracy, per-source results, confusion counts (false passes vs false
  rejections), effective sample size of the weights.
- Coverage must be 320/320 usable judgments; otherwise the report is INCOMPLETE and no kappa is published.
- Interpretation rule: a strong result is independent evidence about v18's behaviour on fresh documents (which are
  about 80% supported), not completion of the original RouteBench target. A weak result stops small prompt experiments.

## Result (2026-09-29) -- frozen v18, fresh upstream-dev documents, evaluation only
Coverage 320/320, no errors; actual spend $2.4693 of the $3.50 cap (320 calls, no cap or reservation breach). No diagnostic
rows, no training, no threshold change, no RouteBench test evaluation. Separate from the original RouteBench result.
- PRIMARY sentence-population weighted kappa **0.404**, 95% document-bootstrap interval [0.282, 0.525].
  Unweighted sample kappa 0.417 [0.298, 0.536]. Effective sample size of weights 310 of 320.
- Sensitivity 0.848, specificity 0.596, balanced accuracy 0.722 (weighted). Unweighted confusion TP 218, FP 24, FN 40, TN 38.
- Per source (weighted / unweighted kappa): MediaSum 0.375 / 0.367 (spec 0.52); MeetingBank 0.441 / 0.471 (spec 0.72).
- Labels on these documents are ~81% supported (weighted), so kappa is depressed relative to balanced accuracy.
- Error profile differs from RouteBench dev: 40 false rejections vs 24 false passes here (dev: 5 vs 14). The judge's failure
  mode is not stably "too lenient"; it is document/annotation dependent.
Interpretation under the pre-stated rule: WEAK. v18's dev kappa of 0.675 does not generalise to fresh documents; consistent
with its TRAIN kappa of 0.448 and with the dev-selection optimism noted earlier. Stop small prompt experiments. Any further
work needs a genuinely different approach and appropriately licensed training data (these documents are evaluation-only).
The 30 reserve upstream-dev documents remain untouched.
