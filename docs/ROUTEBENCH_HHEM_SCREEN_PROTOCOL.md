# Frozen protocol: HHEM-2.1-Open screening (one bounded run)

Recorded 2026-09-29 BEFORE any weights were downloaded or any pair was scored. Authorised: one pretrained-verifier screening
run; maximum 1 GPU-hour; $0 API spend; no training, no threshold search, no ensemble. Not touched: the 30 reserve upstream
documents and the RouteBench test split. Cached v18 predictions are reused (no new judging).

protocol_hash: e6ea6a433b369756

The hash covers `evalops/hhem_report.py`, `evalops/hhem_score.py` and the constants below; `export` and `report` refuse to run on
a mismatch.

## Pins and code review
- Model: `vectara/hallucination_evaluation_model` (HHEM-2.1-Open), revision `8e4a2e6e96c708cc76c2344f7e4757df2515292c`,
  Apache-2.0, about 0.1B parameters, fine-tuned from flan-t5-base. Weights are safetensors.
- Its remote code (`configuration_hhem_v2.py`, `modeling_hhem_v2.py`, ~3 KB) was read before execution: torch/transformers
  only; no subprocess, network, file-write, eval/exec or pickle. It loads config and tokenizer of `google/flan-t5-base` from the
  hub at an UNPINNED revision, so those files are pinned to revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` and loaded from
  local disk (the only local edit: `foundation` in the local copy of `configuration_hhem_v2.py` points at that local directory).
- The code applies no truncation to inputs (documented; T5 uses relative positions, so long prompts run, quality at length is
  the model's own).
- Isolated environment `.local/hhem-venv`; nothing installed into the project environment.

## Frozen procedure
- Premise = the case's full source document, unmodified. Hypothesis = the candidate text, stripped; multi-sentence CNN
  summaries are scored as one hypothesis (not split). The model's own prompt template. fp32, batch size 4.
- Decision rule: "supported" iff P(consistent) >= 0.5 (argmax of the model's two output neurons). The model card gives no
  calibrated threshold, and none is searched here.
- Sets: the 128 headline dev cases, and the 320 already-inspected fresh-document rows (same weights and reporting rule as
  `ROUTEBENCH_UPSTREAM_REPORTING_RULE.md`). v18 baseline = cached judgments at the all-criteria-pass rule.
- Report: coverage, kappa, balanced accuracy, paired document-bootstrap intervals for the kappa difference vs v18
  (2000 draws, seed 20261005; fresh set stratified by source with inclusion weights), corrected/new errors, runtime, token
  lengths and any prompt over 512 tokens.
- Contamination: the training data of HHEM-2.1-Open are not documented, so overlap with AggreFact/TofuEval/RAGTruth cannot be
  excluded. This is a screening result only; even reaching 0.74 here would NOT certify the original target.
- Budget: 1 GPU-hour hard limit for the whole run; stop afterwards regardless of outcome.

## Result (2026-09-29) -- HHEM-2.1-Open screening, frozen rule P(consistent) >= 0.5
Coverage 448/448, runtime 272.6 s on the RTX 4060 (limit 1 GPU-hour), $0 API spend, no training, no threshold search, no ensemble.
Reserve documents and RouteBench test untouched. 436 of 448 prompts exceed 512 tokens (max 1,717); none truncated by this
procedure, so quality on long inputs is the model's own. Load sanity check on the model card toy pairs: 0.011 / 0.647 / 0.129.
- DEV (128, not independent): HHEM kappa 0.586 vs v18 0.6745; balanced accuracy 0.769 vs 0.825; errors 23 vs 19 (HHEM fixed 6 v18
  errors, introduced 10); paired document-bootstrap 95% CI of the kappa difference [-0.310, +0.089].
- FRESH 320 (already inspected): HHEM weighted kappa 0.437 [0.285, 0.585] vs v18 0.404 [0.277, 0.523]; paired difference
  [-0.101, +0.171] (includes zero). Unweighted 0.445 vs 0.417. Errors 47 vs 64 (fixed 37, introduced 20).
  HHEM weighted sensitivity 0.960, specificity 0.405 (balanced accuracy 0.683); v18 0.848 / 0.596 (0.722).
  Per source (weighted kappa): MediaSum 0.453 vs 0.375; MeetingBank 0.409 vs 0.441.
- Reading: HHEM is a very lenient verifier (mostly false passes) and v18 is more balanced (more false rejections); neither is
  convincingly better, and neither is near 0.74. This is NOT a convincing improvement, so pretrained-verifier trials stop here.
  Training-data overlap of HHEM with AggreFact/TofuEval/RAGTruth is undocumented, so even its small fresh-set edge is uncertified.
  Original target not certified; nothing frozen; no ensemble evaluated (the complementary error types are noted, not exploited).
