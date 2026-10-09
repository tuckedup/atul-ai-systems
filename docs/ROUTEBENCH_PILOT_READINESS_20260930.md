> **SUPERSEDED IN PART** by `ROUTEBENCH_PILOT_READINESS_CHECKLIST_20260930.md`: the manifest (now v5), configuration and freeze hashes, the tied-score metric, the direction check and the entry point changed in the second pass. Kept as the first-pass record.

# Pilot-readiness report (2026-09-30)

One bounded readiness pass, as requested. Existing work kept; v3 preserved (its files are untouched: the v3 manifest file hash was
`5e1132c1cb2aac30` before and after). No paid calls, GPU training, model scoring or original-test evaluation. AI-assisted engineering;
not human adjudication and not legal clearance. Permission basis: `ROUTEBENCH_TRAINING_PERMISSION_STATUS.md`.

## 1. Text-only input construction (annotation independence)
Segmentation and scope now depend on the response text alone. Wrapper detection no longer looks at annotations, and a span on a wrapper
is reported (`Wrapper.unsupported`) and still makes the **response** unsupported, so annotated content is never silently discarded.
List prefaces are removed only when they carry no digits or names, so a heading with specific content stays as content.
Real-data check (17,790 responses): running the adapter once with the real labels and once with every label stripped gives
**identical segments (21,700), identical wrapper decisions (994) and identical rejections**; only the label values differ.
The one annotated wrapper seen in v3 is now classified from its text alone (0 annotated wrappers in v4).
The non-default option `implicit_true_policy="exclude"` is annotation-dependent; it is off and must not be used for validation or inference.

## 2. Boundary error
`He said "I live in the U.S." She left.` now splits after the closing quotation mark, while `The show "A.D. The Bible Continues"
premiered ...` stays protected. The period rule takes the closing quote into account on the same line and across a newline. Offsets are
still those of the original text. Regression tests cover both. Zero heuristic flags in a sample does not mean the segmentation is
correct: the earlier heuristic flags (short fragments, missing terminal punctuation) are proxies, and abbreviation edge cases remain.

## 3. v4 manifest and measured effect of the changes
Manifest `ragtruth-training-manifest-v4`: content hash `2f3325ad73ebd0a18037a0bf0ea354d664408b3fb66593e37a3ca10712f12de7`
(file hash `2bc152280446ab8a...`), policy `ragtruth-sentence-v4`, splitter `regex-v5`, same downloaded files, all 6 external and 1
internal exclusions unchanged. Measured against v3 (`ingest_audit compare`): **5 rows added (4 train, 1 test), 0 removed, 0
relabelled, 0 repartitioned**, all supported sentences. The change is tiny in count; its value is that inputs no longer depend on the annotations.

| Partition | Documents (independent) | Responses | Sentences | Unsupported |
| --- | ---: | ---: | ---: | ---: |
| Train | 418 (416) | 2,507 | 14,743 | 1,008 (6.84%) |
| Validation | 104 (104) | 629 | 3,466 | 240 (6.92%) |
| RAGTruth test (not used) | 98 (98) | 588 | 3,270 | 197 (6.02%) |

## 4. The frozen pilot (configuration hash `b93ab024cb03de19...`; freeze file hash `4556a18ef516ef6a...`)
Code: `evalops/pilot/config.py` (single source of truth) and `evalops/pilot/harness.py` (guards).
- **Labels and direction**: label 0 = unsupported, 1 = supported. Score = P(supported) (higher = more supported). Predicted
  unsupported iff score < threshold. Unsupported is the positive class for every detection metric. Inverted scores or labels raise `ScoreDirectionError`.
- **Model and scoring**: MiniCheck-Flan-T5-Large revision `96eafd01cee2d16cf81aaa2fb226b14f422a37b3`; input `"predict: " + document + EOS + sentence`;
  first decoder step, logits of token ids 3 (unsupported) and 209 (supported), softmax index 1. Identical scoring path for the untuned baseline and the tuned model, so the difference isolates training.
- **Baseline**: the same model, untuned, scored on the same samples, with its threshold chosen by the same rule (its native 0.5 is also reported).
- **Tuned model**: LoRA (r 8, alpha 16, dropout 0.05, modules q and v), bf16, gradient checkpointing, lr 2e-4, 1 epoch, micro-batch 2 x accumulation 8,
  seed 20261102. The final checkpoint is used; there is no checkpoint selection.
- **Class weighting**: none in the loss (both weights 1.0). Imbalance is handled by a fixed sampling ratio: every unsupported example up to 500
  plus 2 supported per unsupported, by seeded shuffle from `train_fit`. The threshold is chosen on the natural-prevalence `train_select` sample, so the sampling ratio does not set the operating point.
- **Threshold selection**: grid 0.05..0.95, maximise balanced accuracy on `train_select` only, ties to 0.5, separately for each model. Validation is scored once per model
  and never used to tune anything.
- **Source length**: the whole document, no truncation; examples over 2,048 tokens are excluded from training and from every evaluation, identically for both models, and counted.
- **Aggregation**: response score = minimum over its in-scope sentence scores; supported iff that minimum >= the sentence threshold. Wrappers are out of scope.
- **Stop rule** (continue only if all hold): evaluation coverage complete; validation balanced accuracy gain >= 0.03; the 95% document-group bootstrap
  lower bound of the paired gain > 0 (2000 draws, seed 20261103); unsupported recall not more than 0.02 below the baseline's. Anything else stops, and an incomplete run is reported as incomplete.
- **Selection stays within training data**: `train_fit` (fit) and `train_select` (threshold) are group-disjoint parts of TRAIN; the RAGTruth test partition is never loaded.

Frozen samples (built from the real manifest and MiniCheck tokenizer lengths; group overlaps between fit, select and validation: 0):
| Sample | Sentences | Unsupported | Notes |
| --- | ---: | ---: | --- |
| Training sample | 1,500 | 500 (33%, by design) | 702 of 12,001 `train_fit` sentences (5.9%) excluded for length |
| Selection sample (`train_select`) | 823 | 61 (7.4%) | 24 document groups; label-blind group sampling |
| Validation sample | 1,446 | 103 (7.1%) | 45 of 104 groups; label-blind |

## 5. Feasibility finding (important)
The full training set does not fit the 2-hour cap: median input is 1,017 tokens, 5.4% of all sentences exceed 2,048 tokens, and at the inference rate measured earlier (about 2.3 pairs/s)
training on the 12,001 `train_fit` sentences for two epochs would take many hours. The frozen sample is therefore small. Projected times use measured inference throughput and a **guessed** training rate:
scoring both samples (2,269 pairs) per model is about 990 s against a 1,100 s cap; training 1,500 examples fits the 3,300 s cap only if throughput is at least about 0.46 examples/s (unmeasured).
Hence the first phase of the pilot is a timed feasibility probe (300 s cap, inside the 7,200 s total): if any projected phase exceeds its remaining cap, the run stops and reports "not feasible".
**Statistical limit**: the validation sample has about 103 unsupported sentences from 45 documents. That can only detect a large gain; a stop or inconclusive result is more likely than a clear pass.

## 6. Guards verified offline (18 harness tests, synthetic data)
Rejects: a changed or missing manifest hash, a wrong manifest version, a changed configuration after freezing, training inputs from `train_select`, validation, the RAGTruth test partition or unknown ids,
groups shared across partitions, missing/extra/NaN/infinite/out-of-range scores (incomplete evaluation), inverted score direction, and unknown phases. Resource caps: per-phase and total caps raise `BudgetExceeded`
(fake-clock tests), and `run_with_timeout` kills a real subprocess at its wall-clock limit (real test). Total: 644 tests pass in the repository, Ruff clean, `evalops.run_calibration verify` unchanged.

## 7. What is not ready
1. **Permission**: unresolved (see the permission-status document). The GPU pilot must not run until the decision record there is completed.
2. **Training and scoring script**: not written. `peft` is not installed in the isolated environment; installing it is a new dependency download and needs approval. The scoring path above is specified but not yet coded in the harness.
3. **Feasibility**: unmeasured training throughput and memory at micro-batch 2 with 2,048-token inputs on the 8 GB card.
4. **Domain transfer**: the pilot answers "does this recipe help on a held-out RAGTruth validation sample". Gains on news summaries need not transfer to RouteBench meeting and interview material with different annotation conventions, and a pass is not evidence for kappa 0.74.

## 8. Approval requested (one bounded pilot, not a sweep)
When item 1 is settled: approval to (a) install `peft` (and nothing else) into the isolated environment, (b) write and unit-test the training and scoring script against the frozen configuration, and (c) run the single pilot within
the 7,200 s total GPU cap at $0 API, starting with the feasibility probe and stopping per the rule above. A negative or inconclusive result will be reported as such.
