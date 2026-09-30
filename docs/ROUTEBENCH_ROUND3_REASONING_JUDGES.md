# RouteBench round 3: reasoning judges (2026-09-29)

**Target kappa >= 0.74: NOT achieved.** Dev only; the held-out test split was not touched and no bundle was frozen.

## Findings
- "gpt-5 unavailable" was really "gpt-5 / gpt-5-mini need OpenAI org verification". `o3`, `o4-mini`,
  `gpt-5.4`, `gpt-5.5` work on this key. Earlier rounds used only gpt-4.1. `llm.py` sent `max_tokens`
  and `temperature`, which reasoning models reject; fixed (`max_completion_tokens`, `reasoning_effort`).
  Existing measured config hashes are unchanged (`run_calibration verify` passes).
- Same 128 dev cases, threshold tuned on dev (optimistic):

| Variant | Dev kappa |
| --- | ---: |
| v16 gpt-4.1 holistic (previous best) | 0.6532 |
| v18 o4-mini medium | 0.6745 |
| v21 o4-mini high | 0.6771 |
| v19 o3 medium | 0.5751 |
| v20 o4-mini per-fact decomposition | 0.6028 |
| best mean ensembles (v8+v18, v8+v18+v21) | 0.711 / 0.713 (selection-optimistic, not a result) |

- A stronger model did not help (o3 is worse). **14/128 dev cases (11%) are misjudged by ALL six judges**
  (11 human-fail, 3 human-pass). Inspected examples (e.g. Tevez, Ueda) are extractive sentences the source
  visibly supports yet are labelled unsupported. With 11% irreducible disagreement, 0.74 needs near-perfect
  judging of everything else. Consistent with label noise (AggreFact-CNN kappa is 0.45), not judge weakness.
- These 14 are in `docs/LABEL_ADJUDICATION_QUEUE.json` for HUMAN adjudication. Nothing was relabelled.
  Bound only (not a result, not valid to report): excluding them gives kappa 0.878 on the remaining 114.
- Spend this round: ~$4.6 (o4-mini 0.99, o3 0.89+0.34, v20+v21 2.36). Priced from configured list rates.
- Records: `data/dev_experiments.round3_reasoning.json`; original 4-variant record restored as
  `data/dev_experiments.json` (CLI overwrites it per invocation; earlier copies are `*.previous.*`).

## Recommendation
Do not spend on a test run yet: no candidate is expected to reach 0.74 and dev-tuned scores are optimistic.
Next step with real leverage: have a human adjudicate the queue (and audit CNN labels), then re-measure the
frozen best config once on test. v17 (expert few-shot, gpt-4.1) remains unrun; expected value is low given
shared errors.

## Follow-up: recovered summary context (offline, no paid calls)
Acting on `LABEL_ADJUDICATION_AUDIT.md` (the audit corrected my "label noise" reading: judges share a
representation gap, and CNN cases have upstream rater disagreement). `evalops/tofu_context.py` rebuilds the
full original summary + topic for 401/404 TofuEval cases from the pinned published annotation files
(unique-join, label-agreement and contiguity checks; 3 ambiguous cases excluded) into
`data/tofu_context.json`. `v22-context-o4mini` = v18 plus that context with the graded sentence marked and
neighbours declared non-evidence. UNMEASURED. No labels, cases, splits or existing hashes changed
(`verify` passes; 577 tests, ruff clean). 85/128 dev prompts change; max prompt 12.6k chars.
Estimated dev cost ~$1 (v18 cost $0.99); awaiting authorization. The test split stays untouched.

## v22 result (recovered summary context) -- dev, 2026-09-29
Authorized one run, cap $1.50. Actual spend $1.034 (128 calls, 0 errors, no cap breach). Coverage 128/128,
same cases/scoring/threshold grid as v18. Labels, splits, test split untouched (`verify` passes).
- v22 dev kappa **0.6560** (t=0.65) vs v18 **0.6745**; at t=0.5: 0.6314 vs 0.5612. Not an improvement:
  paired group-bootstrap 95% CI of the kappa difference [-0.071, +0.058].
- Errors at each variant's own threshold: v18 19, v22 20. 3 corrected (2 MeetB, 1 MediaS), 4 newly introduced
  (3 MediaS, 1 MeetB). The Giuliani "This compromised..." case, which motivated the change, still scores 1.0 under v22.
- Consistent with TofuEval's own report that adding previous sentences did not change model performance.
Conclusion: context restoration is not the lever. Nothing is frozen; no test run.
Also added: `cmd_test` now refuses to publish a terminal result when any judgment errored or a budget stop left
cases unjudged (2 offline tests). Cumulative round-3+4 paid spend ~= $5.6.

## v23 prepared (diverse few-shot on o4-mini) -- offline only, UNMEASURED
v17 (gpt-4.1) is preserved; its packet regenerates byte-identically. `v23-diverse-fewshot-o4mini` = v18 plus
`data/gedd_diverse_exemplars.json` and nothing else (a test enforces one differing field vs v18).
Packet: 6 TRAIN examples from 6 distinct documents (v17: 6 of 8 shared one), 4 FAIL with published expert memos
(Opinion-as-Fact, Nuanced Meaning Shift, Contradiction, Extrinsic Information) + 2 PASS (one per TofuEval subset),
26,191 source chars (v17: 31,305), full sources, memos verbatim, no relabelling. Selection rule is general and
recorded in the file. FAIL memos matching a completeness lexicon are excluded (3 of the eligible pool, including
`gnd-33aed12bc35c7bcd` "not a staff report" which teaches a completeness rule our rubric contradicts).
Verified: train membership, 0 document-group or content-hash collisions with dev/test, 0 of the 18 AI-reviewed
cases used, provenance validator passes, all 128 dev prompts build (mean 38.4k chars vs 9.6k, max 40.2k).
Cost estimate from v18's measured 0.215 tokens/char and 1,239 completion tokens/call: ~$1.86 for one pass.

## v23 result (diverse expert few-shot, o4-mini) -- dev, 2026-09-29
Config hash eb977b35e5aa738d; packet `gedd_diverse_exemplars.json` sha256 753e4cf4...2b670 (rule recorded in the
file). Pre-checks: recorded hashes for v16, v8, v12, v14, v18-v22 all equal the current configs (verify passes).
Run: cap $2.50 authorised. First pass 127/128 (one provider_error, no cap involvement); the one case was
re-dispatched under the same frozen config for $0.013. Final coverage 128/128, actual spend $1.7686, no cap
breach or reservation breach. Test split untouched; labels/splits unchanged; prior records preserved
(`dev_experiments.round5_v23*.json`).
- Dev kappa 0.6344 (all-criteria-pass threshold, also v23's dev-selected one; v18's frozen threshold is the same
  1.0) vs v18 0.6745. At t=0.5: 0.5735 vs 0.5612.
- Errors 21 vs 19: 1 corrected (MediaS), 3 new (MeetB, MediaS, CNN). Per subset errors v18/v23: MediaS 7/7,
  MeetB 7/8, CNN 5/6.
- Paired group-bootstrap 95% CI of kappa difference [-0.140, +0.021]: no evidence of improvement, leaning worse.
Conclusion: expert demonstrations did not help. Nothing frozen, no test run. Cumulative paid spend ~$7.4.

## MiniCheck-Flan-T5-Large baseline (local, no API cost) -- dev, 2026-09-29
Weights MIT-licensed (checked on the model card), 770M params, run on the local RTX 4060 in an isolated venv
(`.local/mc-venv`, minicheck pinned at b58b9fa, torch 2.6+cu124). Nothing installed into the project env.
All 128 dev cases scored (~2.5 min GPU); raw scores in `data/minicheck_flan_t5_large_dev_scores.json`.
- Dev kappa (threshold tuned on dev, optimistic): full-claim 0.549 (t=0.6; 0.526 at MiniCheck's own 0.5),
  min-over-sentences 0.550, mean-over-sentences 0.534. Balanced accuracy 0.76. Per subset kappa (full-claim):
  MediaS 0.648, MeetB 0.276, CNN 0.399.
- Versus o4-mini v18 (19 errors): MiniCheck 26 errors; 11 shared, 15 MiniCheck-only, 8 o4-mini-only.
- Errors are only partly shared, but blends do not reach the target: 0.3*MiniCheck + 0.7*o4-mini 0.698
  (weight AND threshold tuned on the same 128 dev cases, so selection-optimistic; no held-out support),
  MiniCheck+o4-mini+decompose mean 0.672. Not a valid basis for freezing.
Conclusion: a specialised verifier alone is worse than the LLM judge and the best blend (~0.70) is still below
0.74 with only 128 dev cases to tune on. Nothing frozen; test untouched; no paid calls.
