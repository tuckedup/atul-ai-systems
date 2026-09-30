# RouteBench judge calibration: phase close-out (2026-09-29)

**Target (binary Cohen kappa >= 0.74): NOT ACHIEVED.** No further paid calls, training, downloads or RouteBench test
evaluation are planned. No judge inference was run on the original RouteBench test split; its aggregate
human-label prevalence was inspected. No qualifying groundedness calibration bundle was frozen.

## Documented baseline
v18 = `v18-reasoning-o4mini` (o4-mini, medium reasoning effort, task rubric, config hash `12fc8b550d954359`,
"supported" iff every rubric criterion passes). It is the selected comparison baseline, not a qualifying judge.
The highest single-judge dev score was v21 (o4-mini high), 0.6771, versus v18's 0.6745; neither reached 0.74.
The earlier gpt-4.1 baseline v16 (`51695bd659270b7f`, dev kappa 0.6532) is preserved.

Later HHEM screening: dev kappa 0.586; separate 320-row weighted kappa 0.437, versus v18's 0.404.
The paired difference interval includes zero; this is not a convincing improvement. See
`ROUTEBENCH_HHEM_SCREEN_PROTOCOL.md` and the current `ROUTEBENCH_PROJECT_BRIEF.md`.

## Evidence, most independent last (binary unweighted Cohen kappa unless stated)
| Measurement | Data | kappa |
| --- | --- | ---: |
| v18, dev, threshold/variant selected on dev (optimistic) | 128 cases / 46 docs, 62% supported | 0.6745 |
| v18, TRAIN, frozen rule | 129 cases / 40 docs, 52% supported | 0.448 |
| v18, fresh upstream-dev documents, weighted (primary) | 320 rows / 40 docs, 81% supported | **0.404** [0.282, 0.525] |
| same, unweighted | 320 rows | 0.417 [0.298, 0.536] |

On the fresh documents: sensitivity 0.848, specificity 0.596, balanced accuracy 0.722. The error direction flipped between
sets (dev 14 false passes / 5 false rejections; fresh 24 / 40), so errors are document- and annotation-dependent.

## Everything tried (dev unless stated; each run pre-declared, results kept)
- gpt-4.1: rubric 0.653, decomposition 0.615, addressed 0.609, contradiction 0.587.
- o4-mini: medium 0.675, high 0.677, decomposition 0.603. o3: 0.575.
- Restored summary context (v22): 0.656 (paired CI vs v18 [-0.071, +0.058]).
- Diverse expert few-shot (v23): 0.634 (CI [-0.140, +0.021]).
- MiniCheck-Flan-T5-Large alone: 0.549. Hand-tuned blends ~0.70-0.71 (tuned on dev; discarded as optimistic).
- Frozen train-fitted MiniCheck + v18 combiner: 0.593 (CI vs v18 [-0.214, +0.015]).
- Unmeasured: v17 (gpt-4.1 expert few-shot).

## Spend (token-priced at configured list rates, not invoices)
This session about $10.88: o4-mini dev 0.99; o3 0.89 + 0.34 retry; o4-mini decomposition + high effort 2.36; v22 1.03;
v23 1.77; v18 on TRAIN 1.02; fresh-document validation 2.47. Earlier rounds (four gpt-4.1 variants) about $2.50 plus a
health probe. Every capped run ended under its cap with no reservation breaches. Local MiniCheck compute had no API cost.

## Provenance and preserved artifacts
Labels and splits are unchanged; new judgments were appended to the existing local cache. The latest cache
additions are not part of this publication. Local records: `data/dev_experiments.json` (original four variants; `verify`
passes), `dev_experiments.round3_reasoning.json`, `round4_v22.json`, `round5_v23*.json`, `data/stack_minicheck/`
(combiner protocol and results), `data/upstream_validation/` (fresh-document judgments and report),
`minicheck_*_scores.json`. Protocols and hashes: `ROUTEBENCH_COMBINER_PROTOCOL.md`,
`ROUTEBENCH_UPSTREAM_REPORTING_RULE.md`, `ROUTEBENCH_UPSTREAM_VALIDATION_PLAN.md`.
Code added: reasoning-model client support (hash-neutral), `tofu_context.py`, diverse exemplar builder,
`stack_minicheck.py`, `fetch_tofueval_docs.py`, `tofu_upstream_sample.py`, `upstream_validation.py`, and an
incomplete-result safeguard in `cmd_test`. Licensed third-party transcripts are local and git-ignored.

## UNRESOLVED dataset-use restrictions (do not treat as release-ready)
TofuEval and LLM-AggreFact state that the data are an evaluation benchmark and should not be used in training NLP models
(LLM-AggreFact: not for pretraining or fine-tuning). The following used TofuEval-derived TRAIN-split data and are NOT cleared:
- `gedd_train_exemplars.json` (v17) and `gedd_diverse_exemplars.json` (v23): few-shot demonstrations with expert memos.
- The MiniCheck + v18 logistic combiner (`data/stack_minicheck/fit_result.json`), fitted on 129 TofuEval/CNN TRAIN rows.
- `exemplars.json` (earlier v6/v7 few-shot), if it draws on the same sources.
The 70 upstream-dev documents were used only for evaluation, as intended. Other caveats: the local LLM-AggreFact mirror is
a filtered copy of unknown provenance (MediaSum texts carry encoding damage in 9 of 15 documents); MeetingBank and MediaSum
transcripts are CC-BY-NC-SA / research-only, so no redistribution; the 0.433 test-label prevalence was inspected once as an
aggregate (no decision used it) and the 0.627 "projection" is arithmetic, not a measurement.

## Findings worth keeping
1. The metric was over-trusted. With only 30 Tofu documents in the whole benchmark, per-split kappa swings by document
   (dev 0.675 vs fresh 0.404). Future claims need document-level sampling and intervals.
2. Prompt, model size, reasoning effort, context and few-shot changes all sit in one band; errors are shared and partly
   annotation-dependent (FRANK's antecedent rule, rater disagreement on CNN cases).
3. Reusable infrastructure: frozen-protocol scripts with hashed procedures, a capped spend meter, document-grouped bootstrap.

# Proposed new project scope (NOT STARTED, needs approval): task-trained groundedness verifier
Goal: a verifier for "is this summary sentence supported by the dialogue document", trained only on data whose terms permit
training, then measured once on untouched evaluation sets.
- Training-data candidates (licences to be re-verified before use): MiniCheck C2D/D2C synthetic set (14,395 rows, MIT per its
  dataset card); RAGTruth (2,965 instances, MIT per its repo, includes summarization; audit overlap with AggreFact-CNN, and
  note LLM-AggreFact folded it into its dev split); FaithBench (licence terms not confirmed; check); NLI sets used by
  MiniCheck (check each licence). Excluded: TofuEval, LLM-AggreFact, and the 70 fetched upstream documents.
- Overlap audit before any training: document-ID and normalised-text-fingerprint matching against every RouteBench and
  upstream evaluation document and the few-shot documents; report zero overlap or drop the rows.
- Model: start from MiniCheck-Flan-T5-Large (MIT, 770M) with LoRA or a comparable open verifier, on the local RTX 4060
  (8 GB); no API spend except an optional capped comparison against v18.
- Fixed budget: API $0 for training; optional up to $4.00 for one v18-vs-verifier comparison on the fresh sample; compute up
  to 40 GPU-hours; time-box two working weeks; one training configuration chosen by document-grouped CV on training data only.
- Evaluation, each used once and frozen before running: (a) the 320-row fresh sample already judged (paired vs v18);
  (b) the 30 reserve upstream-dev documents sampled by the same manifest rule; (c) only if (a) and (b) are strong and
  separately approved, the sealed RouteBench test split. Metric: weighted sentence-population kappa with a document
  bootstrap, plus sensitivity, specificity and balanced accuracy.
- Success criteria: weighted kappa >= 0.74 with 95% lower bound >= 0.60 on (a), and >= 0.70 point estimate on (b), with
  complete coverage. Stop criteria: point estimate < 0.55 on (a), or no complementary-error gain over v18.
- Risks: small domain-specific signal; synthetic data may not match TofuEval's annotation policy (antecedent errors,
  question sentences); high prevalence depresses kappa; labels may cap achievable agreement, so a human-adjudicated
  sample would be needed to know the ceiling.
