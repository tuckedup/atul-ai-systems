# RouteBench: where the above-target score came from

Read-only analysis, 2026-10-04. No provider calls, threshold searches, prediction changes, label edits or original held-out evaluation were performed.

## Independently reproduced historical peak

The cached Reveal source probe genuinely gives ordinary binary Cohen's kappa **0.838835456199636**:

- Variant: `v16-holistic-4.1`, whole-response rubric mode.
- Stored config hash: `51695bd659270b7f`; current named config reports that same hash.
- Served model in all 124 usable judgments: `gpt-4.1-2025-04-14`.
- Threshold: **0.30**, selected on the probe itself from 0.05 to 1.00 in steps of 0.05.
- Population: 128 sampled Reveal claims, 64 per class before failures; 124 usable judgments, 4 missing.
- Confusion: TP 56, TN 58, FP 3, FN 7. Agreement: 114/124 = 91.94%.
- The source builder used seed 20260927 and truncated documents to 6,000 characters.
- Judge settings: rubric, temperature 0, max output tokens 2,000, one sample, no few-shot packet, boundary examples enabled.

Evidence: `routebench/evalops/data/probe/probe_results_recomputed.json`, `probe_judgment_cache.jsonl`, `annotations_reveal.jsonl`, and `routebench/evalops/probe_sources.py`. The kappa above was recomputed directly from cached scores and published labels at the recorded threshold, not merely copied from the report.

This was a diagnostic, same-sample threshold-selected result with 96.875% completion. It was not the original summarization acceptance result. Matching a config hash alone does not establish complete historical prompt provenance; verify prompt/rubric identity before claiming an exact replay.

## What can and cannot be copied

The useful recipe is source-only entailment, a simple task-aligned rubric, whole-response evaluation, a strong judge, and development-only calibration. The rubric permits supported paraphrase and omission while rejecting unsupported assertions, incorrect specifics and invented relations.

The peak does **not** isolate a universal technique: the documents, claims, annotation policy and selected threshold differed from the summarization experiment. It is therefore not evidence that Reveal labels are better, that all other labels are wrong, or that threshold 0.30 transfers to every task.

Historical `docs/ATTRIBUTION_EXPERIMENT_RESULT.md` also reports Reveal kappa 0.8000 on 40 completed attribution-test items at the pooled threshold 0.80. However, the mixed-source result was only 0.5431 on 195/206 completed items. The report's audit correction discloses prior-probe exposure, missing bindings and source truncation; this is not clean independent confirmation. Do not reuse that spent test as new confirmation.

## Why the latest result collapsed

The frozen Claude claim-relation-v2 development result is reproduced at kappa **0.36089177891314445**, 43 errors on all 128 cases.

All **26** aggregated `uncertain` verdicts had model-level overall verdict `supported`. No assertion-level uncertain verdict caused those 26 demotions. In 25, the response listed declared uncovered spans; 23 of those 25 contained no substantive words under the existing word policy. Another case triggered the broad-span heuristic.

The prompt asks the model to list **any** uncovered candidate text. But `claim_relation_v2.aggregate()` demotes whenever the declared list contains a dictionary, including punctuation and whitespace. A synthetic entirely supported assertion plus an uncovered final period reproduces the demotion. This is a prompt/aggregation mismatch, not evidence of uncertain model reasoning.

Of 31 new false rejections versus v18, 23 came from code-generated uncertainty and 8 from explicit unsupported verdicts. Thus the phrase that uncertainty caused essentially all 31 is too strong.

Offline, post-hoc diagnostic only: filtering declared spans using their verified candidate offsets/text and the existing substantive-word rule changes kappa to **0.5975855130784709** (25 errors; 20 errors corrected and 2 introduced relative to the original candidate). Other guards remain unchanged. Using the model overall verdict directly gives **0.6241062308478039** (23 errors). Neither reaches v18's 0.674518 or the target. No saved result was overwritten, and neither diagnostic is an accepted new experiment.

## Next bounded work order

1. Preserve all measured results, frozen protocols, cases and labels.
2. Implement a separately versioned aggregation repair; do not edit frozen v2. Test punctuation, whitespace, ordinary function words, authentic and invalid offsets, uncovered negation, modals, numbers and substantive assertions. Never use model-provided reasons alone to exempt a span.
3. Reprocess all 128 cached raw responses without calls, with every case retained. Mark the result post-hoc development diagnosis, not held-out acceptance. Audit remaining semantic errors rather than attributing all failures to the code.
4. Recover the historical holistic rubric recipe and record exact model/prompt/rubric/scoring identities. Reproduce stored metrics first. Do not silently treat a different provider as an identical judge.
5. Prepare one matched development comparison across all original task subsets using task-aligned criteria. Select settings only on permitted development data, fix the population before calls, and publish both pooled and per-task results. A future frozen router must select task class without using gold labels or observed correctness; its pooled evaluation keeps the entire original denominator.
6. Propose any new calls/downloads with a bounded budget before execution. Keep the original held-out test sealed. Any Reveal-only confirmation is a separately scoped claim on genuinely unexposed documents, not a replacement for failed summarization results.

There is evidence of above-target performance on a narrow Reveal probe. There is no evidence yet of above-target performance across the original RouteBench groundedness population.

---
## Follow-up work (2026-10-04, later): v3 aggregation repair, post-hoc reprocessing, v16 provenance, matched comparison (frozen, NOT run)

### 1. `claim-relation-v3` (separately versioned aggregation repair; v1/v2 and their frozen protocols untouched)
- A declared uncovered span now demotes a case only if the candidate text it points at contains a substantive word (content words, digit-bearing tokens, negation incl. n't, modals, quantifiers). Whitespace, punctuation and function words are not factual uncertainty. The span is resolved against the stored candidate by verified offsets, else a unique text match, else its declared wording; every reading is checked, and a model-supplied reason never exempts a span. Uncovered substantive words computed from the assertion spans themselves (omitted "not", modals, numbers) still demote. All other guards unchanged.
- 27 synthetic regression tests (`test_claim_relation_v3.py`): punctuation/whitespace/function words, undeclared and declared negation/modal/digit, lying offsets, unresolvable spans, other guards unchanged, identity and no-mutation of inputs.

### 2. POST-HOC DIAGNOSIS: offline reprocessing of the 128 cached frozen responses (no calls; original run, attempts log, comparison untouched and re-hashed)
Outputs in `.local/claim-relation-claude-v1/post-hoc-v3/` (judgments.jsonl, comparison_posthoc.json, audit.json). Produced after seeing the frozen result and its failure analysis: development diagnosis only, not a pre-registered experiment, not held-out, not an accepted candidate.
- Verdict transitions (original -> v3): supported 50 -> 50, **uncertain -> supported 23**, uncertain stays uncertain 3, unsupported 52 -> 52. v3 verdicts: 73 supported / 52 unsupported / 3 uncertain.
- Dev κ: v18 0.6745; frozen candidate 0.3609; **v3 reprocessed 0.5799** (4 false passes fixed, 0 false rejections fixed, 11 new false rejections, 0 new false passes; interval of the gain vs v18 [-0.234, +0.066]). Improvement gate: no; 0.74: no. (The earlier analysis's own diagnostic gave 0.5976; this implementation also keeps demotions for declared spans containing words like "summary"/"though", one case more; the difference was not investigated further.)
- Remaining 11 new errors vs v18: 8 explicit "unsupported" verdicts by the model (semantic, not an aggregation artefact), 2 code-uncertain (substantive words "summary", "though" left unreviewed) and 1 broad-span heuristic. So the aggregation defect explained most of the collapse, but the repaired pipeline is still below v18.

### 3. Provenance of the historical v16-holistic-4.1 Reveal recipe (checked tonight, offline)
- Recipe: `v16-holistic-4.1`, config hash `51695bd659270b7f`, gpt-4.1, mode rubric (whole response), max_tokens 2,000, temperature 0, 1 sample, boundary examples on, no few-shot, rubric `summarize@1.0.0+d72496fbb2991071`, prompt-template hash `ef895d791c0bd860`.
- All 128 Reveal cases are found under the CURRENT-style cache key (which covers case content, config hash including the prompt template, and the rubric hash); all 276 stored probe judgments carry that config hash and rubric version and `gpt-4.1-2025-04-14`. This verifies identity against the present tree; it cannot prove what the prompt was on the day of the run beyond that the stored hash equals today's (the run predates this audit; no earlier snapshot exists to compare).
- Metrics reproduced from cached scores and published labels: n=124 usable of 128 (4 provider errors), TP 56 / FP 3 / FN 7 / TN 58, κ **0.838835456199636**.
- **The 0.30 threshold does no work**: predictions at 0.30 and at the strict all-criteria rule (score >= 1.0, the rule v18 uses) are identical on Reveal. So no threshold needs to transfer; the comparison below uses the strict rule for both judges.

### 4. The matched comparison (FROZEN, not run) - `routebench/evalops/matched_v16_v18.py`, `.local/matched-v16-v18-v1/protocol.json`
- sha256 **`281b169d64519cce2c4b03bad21ddf39ebcbf241dc527f363879ab79f2c4b77f`**. Binds populations, label/case/population files, cached-judgment digest, call plan, judge configs (both pinned), rubric, prices, code and decision rule.
- Populations (all cases kept): dev 128, TRAIN 129, Reveal 128, WiCE 128. Subsets: TofuEval-MediaS 126, AggreFact-CNN 76, TofuEval-MeetB 55 (dev+TRAIN), Reveal 128, WiCE 128.
- Judges: v16 = gpt-4.1 (recipe above), v18 = o4-mini medium. Never substituted across providers. Same rule for both on every subset: pass iff every criterion passes (score >= 1.0). No threshold search; v16 at 0.30 is reported only as a labelled sensitivity analysis.
- Reuse: 635 of 1,026 (judge, case) pairs are verified cached judgments (current cache key, config hash, rubric hash, served model). **New calls: 391** = v16 on TRAIN (129) + the 6 probe cases that failed with provider errors, and v18 on Reveal (128) + WiCE (128). Dev needs no calls (both cached).
- Reported: per-population and per-subset κ for each judge, false passes/rejections, paired difference with a group-bootstrap interval (seed 20261104, 2,000 draws); dev+TRAIN pooled is supplementary; probe populations use different label sources and are never pooled into an original-target claim. Completeness (every pair valid) is checked before labels are parsed; failed pairs are retained and the comparison withheld.
- **Usage ESTIMATE** (not bounds): about $0.88 for v16 (135 calls) + $1.98 for v18 (256 calls) = **~$2.87**; ~$5.73 if every call needed two attempts; hard cap **$6.00** enforced by the fsynced spend journal (one HTTP request per reservation; unsettled counts at its bound). Wall time roughly 40-60 minutes sequentially.
- **Blocker:** these are OpenAI calls and the account returned `credit_balance_exhausted`; the Claude subscription cannot stand in for gpt-4.1/o4-mini, and mixing providers would make the judges non-identical. About $6 of OpenAI credit must be added before this can run.
- Tests: 20 in `test_matched_v16_v18.py` (peak reproduction, recipe pins, tampered cache rejection, freeze binding, altered inputs, per-judge request settings = registered recipe incl. max_tokens 2000, restart without repeating pairs, quota/model-mismatch stop, failed pairs retained with labels unread, labels never read in `run`). Full suite 1089 passed, ruff clean.
```
export PYTHONPATH="packages/core;routebench"; H=281b169d64519cce2c4b03bad21ddf39ebcbf241dc527f363879ab79f2c4b77f
.venv/Scripts/python.exe -m evalops.matched_v16_v18 run --protocol-sha256 $H --confirm RUN-MATCHED-OPENAI
.venv/Scripts/python.exe -m evalops.matched_v16_v18 compare --protocol-sha256 $H
```
The original held-out test was not touched.
