# RouteBench: staged path after HHEM

Date: 2026-09-30. Plan only; no training budget or data download is authorized by this document.
The existing binary Cohen's kappa >= 0.74 target remains unmet. No guaranteed score or completion date.

## Current checkpoint

- Published code checkpoint: `22b01e1`, branch `codex/routebench-finish-audit`.
- HHEM screening completed, not crashed: 448/448 results, dev kappa 0.586 and separate-set weighted
  kappa 0.437. Its paired improvement interval over v18 includes zero.
- Highest single-judge dev result: v21, 0.6771. Selected baseline v18: dev 0.6745 and separate-set
  weighted 0.404. These different populations are not interchangeable acceptance tests.
- Prompt/context/few-shot changes and the train-fitted combiner have not established a qualifying judge.
- The original groundedness test has had no judge inference; aggregate label prevalence was inspected.
  The existing dev set and 320-row comparison set are repeatedly inspected, not fresh validation.
- Preserve the remaining 30 upstream documents for separately approved evaluation. Do not inspect their
  case labels, predictions or errors to develop the next model.

## Working rule: one repository, one declared experiment at a time

Continue in `wt-routebench-finish`; do not restart from main or create another copy of the corpus.
Keep existing experiment scripts, data, labels and hashes intact. New code goes in a distinct
`routebench/evalops/pilot/` package if the implementation stage is approved. Use local, ignored storage
for data and checkpoints, and versioned manifests/reports for provenance. Never commit credentials,
raw restricted sources or uncleared model artifacts. The no-Docker amendment still applies.

## 1. Define exactly what the next model must learn (offline first)

Deliverable: `docs/ROUTEBENCH_TARGET_ALIGNMENT.md`, an evidence-backed policy comparison, not new labels.

Compare the current `rubrics/summarize.yaml` with the existing annotation audit and original dataset
instructions. Record supported/unsupported meaning, sentence versus full-summary unit, pronoun antecedents,
questions, tense/modality, omissions and noisy transcripts. Distinguish verified differences from hypotheses.

Known starting points from the repository:

- The current rubric accepts ordinary pronoun resolution from the source; the FRANK research audit records
  a separate requirement for antecedents within the summary. Pure source support and dataset-native
  factuality must not be silently conflated.
- Tofu annotators had candidate-summary context. Restoring it was already tested as v22 and did not improve
  aggregate dev kappa; do not rerun that hypothesis as if it were new.
- Shared judge errors and an AI label review do not establish wrong human labels or an agreement ceiling.

Gate: retain the original metric, labels and denominator. Identify a defensible training target compatible
with that contract, or explicitly report unresolved ambiguity. Any alternative product definition is a
separate evaluation, not a way to pass the original target. No rubric edits or test inspection in this step.

## 2. Audit a candidate training source before downloading or training

Deliverable: a manifest of permitted use, upstream revision, source-document identity, label definition,
known prior model exposure and intended split. Begin with metadata/already-available local material.
Any live lookup or download must follow the applicable permissions and usage terms.

Do not use TofuEval/LLM-AggreFact examples for a new fit while their training-use restrictions remain
unresolved. A repository license alone does not establish permission for all underlying data.
Do not recycle the uncleared few-shot packets or fitted combiner as approved training inputs.

Gate: sufficient eligible independent documents with compatible labels; exact and near-duplicate
document checks against all existing evaluation/probe/example sets; all exclusions recorded. Use a
pre-existing text/ID exclusion manifest for held-out sets, without examining their labels. An automated
overlap check cannot prove absence of all semantic overlap or unknown pretrained-model contamination.

## 3. Build the smallest reproducible training/evaluation harness

Deliverable: one dataset adapter, grouped train/validation split, one model interface, training entry point,
evaluation entry point and synthetic tests. Choose a base model only after its terms/provenance and local
feasibility are checked. No download or training just because a script exists.

Required controls: seed/config/input hashes, no source truncation without a declared policy, duplicate and
split checks, invalid/NaN score rejection, full-coverage checks, real wall-clock timeout, checkpoint resume,
and isolated output paths. Mock/synthetic tests must demonstrate these controls without provider calls.

Gate: offline tests pass; report a measured or explicitly estimated pilot resource requirement. Freeze one
training recipe and a success/stop rule BEFORE training. Do not reuse the old scripts' unenforced time caps.

## 4. Ask for a small pilot budget, not 40 hours upfront

Proposed first budget: at most 2 local GPU-hours including the baseline, training and evaluation, API $0;
this is a proposal requiring approval and feasibility confirmation, not an instruction to launch now.
One untuned baseline and one training configuration. Select settings using only a designated validation
partition of training-permitted data, grouped by source document. Report incomplete if the cap prevents
completion; do not silently increase it or treat partial coverage as a valid comparison.

Gate: examine full-coverage results, paired document-level uncertainty, false passes/rejections, and evidence
of generalization within the permitted development data. Predeclare a practically useful improvement before
the run. A small pilot can also be inconclusive; neither a tiny gain nor training loss alone justifies scale-up.

## 5. Decide whether to stop or approve one larger run

Deliverable: a pilot report including negative results, actual compute and the next decision.
Stop if the recipe is broken, has incompatible labels, or fails its predeclared continuation gate. Do not
cycle through the benchmark until a favorable number appears. If evidence supports continuing, separately
approve a bounded extension up to the overall 40-hour ceiling. That ceiling is not a promise of kappa 0.74.

The old dev and 320-row comparison sets may be used only as disclosed, non-independent comparisons under a
frozen procedure. Do not count improvements there as independent confirmation or use them for an open sweep.

## 6. Freeze and independently evaluate, only when justified and authorized

Freeze model weights, tokenizer, prompt, preprocessing, aggregation, threshold, code and data identities.
Predeclare what the reserve-document evaluation measures; it does not substitute for the original benchmark.
If evidence justifies it, request separate approval for the original held-out test, including any provider
transmission/cost. Require full declared coverage, original binary kappa, confusion counts and document-level
uncertainty. Report the achieved value whether above or below 0.74; never revise against test errors.

If the target fails, preserve the result and close that experiment. A human-reviewed product can be useful,
but its automated coverage and reviewed performance are a different claim from full-coverage automated kappa.

## Next task to hand to Claude

Execute Step 1 only, offline using the existing repo and audit material. Write the target-alignment table with
file/section evidence and unresolved questions. Inventory locally available training candidates without
loading model weights or changing data. Preserve all labels, splits, cached predictions and frozen scripts.
No downloads, paid calls, GPU training or held-out evaluation. Report the completed audit and the smallest
remaining permission/data requirement before proceeding to Step 2. The eventual goal is 0.74, not authorization
to fabricate a passing score or alter the benchmark to obtain one.
