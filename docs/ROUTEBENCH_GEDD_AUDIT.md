# RouteBench: GEDD comparison and offline integrity audit

Date: 2026-09-29. No paid judge calls were made during this audit. The target remains
binary, unweighted Cohen's kappa >= 0.74; no labels, splits, cached judgments, old
bundles or raw reports were rewritten to improve the number.

## Bottom line

The original 128-case dev experiment remains reproducible: best single judge
0.6532, inspected two-judge mean 0.6719. Its original 263-case test is unmeasured.
The separate attribution experiment reports 0.5431 on 195/206 test items. Neither
achieves the target. More data alone does not make a judge agree with its labels.

The actionable judge-quality hypothesis is to teach the published annotators'
specific failure distinctions using complete-source training examples and their
expert explanations. This is a hypothesis to measure, not a guaranteed fix. The
older few-shot builder omitted source context, and selected by response length,
so it did not actually demonstrate why a groundedness verdict was correct.

## What the independent audit found

Reproduce read-only with `python -m evalops.audit_attribution`; exit 2 means blockers.
The machine-readable snapshot is `docs/ATTRIBUTION_INTEGRITY_AUDIT.json`.

- 96/600 attribution sources were cut to 6,000 characters; 55 have positive human
  labels. Truncation can remove supporting evidence. This does NOT establish that
  all 96 labels are wrong, and no labels were changed.
- 14 attribution test document/claim pairs already appeared in the Wice/Reveal
  probes (Wice 10, Reveal 4). Different ID prefixes hid content reuse. Within-corpus
  group separation is not proof of independence from earlier experiments.
- The stored attribution bundle has empty annotation and split-membership hashes.
  The historical result document's claim that these were bound is false.
- Coverage is 195/206 (94.7%). With TP=64, FP=18, FN=26, TN=87, even if every
  missing item were correct, maximizing over its possible label mix gives kappa
  <= 0.5720086883. This is a mathematical optimistic bound, not a measured score
  or confidence interval. Retrying 11 failures cannot deliver 0.74.

Treat this attribution result as an incomplete diagnostic, not a clean independent
confirmation. The dev/test drop is consistent with several explanations; one
comparison does not prove it is entirely selection optimism. Source differences
do not prove labels alone dominate, nor establish a human-agreement ceiling.
Equal sampling of the initial corpus also does not make the grouped test split
equally weighted, and pooled kappa is not a mean of subset kappas.

An operational retry of missing provider calls against an unchanged frozen judge
is not model reselection. However, the old result lacks sufficient bindings for
silent migration. It is preserved; new incomplete test attempts retain successful
cache entries without publishing a terminal report, permitting safe same-config
resumption.

## What AWS GEDD actually provides

Inspected aws-samples/sample-GEDD at commit
`68f96b77e696fc43d3ab44e370abef2607b3bb41`, including the kappa guide,
`judge_builder/calibrate.py`, few-shot selection and methodology.

- The guide's 0.74 is an illustrative multi-criterion table, not a measured result
  on RouteBench. Its displayed weighted arithmetic is 0.728, approximately 0.73,
  not 0.74. Do not treat it as a reproducible benchmark recipe.
- Its ordinal, per-criterion setting differs from this project's binary whole-item
  labels. The implementation computes quadratic-weighted kappa on scores 1-5;
  the guide describes linear weighting. Neither changes our acceptance metric.
- The useful method is expert error codes and explanations, targeted criteria,
  and carefully selected demonstrations, followed by measurement. Our binary
  annotations are not per-criterion human scores; fabricating those would be invalid.
- Do not copy the metric implementation blindly: it silently zips unequal-length
  inputs and returns perfect agreement for a degenerate expected-agreement case.
  Retain RouteBench's undefined-kappa checks and document-group bootstrap.

Reference: https://github.com/aws-samples/sample-GEDD/blob/68f96b77e696fc43d3ab44e370abef2607b3bb41/grounded-evals/docs/cohens-kappa-for-llm-judges.md

## Code changes and prepared candidate

1. Few-shot examples now carry complete source/task/candidate evidence. Oversized
   examples are excluded, not cropped. File-content keyed loading prevents stale
   path-only cache reuse. Old few-shot hashes are invalidated; all four measured
   no-exemplar judge hashes remain unchanged.
2. Calibration validates each example's train membership, complete content,
   verdict, corpus, annotations and split bindings before provider execution.
3. `gedd_exemplars.py` joins only original TRAIN cases to unique published TofuEval
   expert records, requires matching labels and full raw sources, and rejects
   ambiguous or conflicting joins. It prepares eight demonstrations covering six
   expert failure types and two passes, from 93 eligible train cases. It never
   selects examples from dev/test or by judge performance.
4. `v17-expert-fewshot-4.1` is UNMEASURED. Its matched control is cached v16; the
   experimental change is the source-complete training demonstrations. Its packet
   contains 31,305 source characters, so it costs more per call than the control.
5. Attribution commands protect existing artifacts, bind future runs to full
   configs/data/protocol, recompute selection from cache, stop on quota failures,
   honor concurrency, and avoid publishing partial terminal test results.
   Future builds retain full sources, exclude probe documents by content, and
   fail if declared class quotas cannot be filled. Wice has only 47 eligible
   positives against a quota of 50, so a fresh build currently refuses.

These fixes do not rehabilitate the historical attribution test. A successor
confirmation experiment needs a new protocol, separate storage, and an exposure
registry including ALL previously examined experiments, not just the two probes.
Full-source request sizes must also be checked against budget/context limits.
Do not delete existing artifacts to bypass the guards.

## Next decision: one bounded dev experiment, not another broad grid

No new inference is authorized by the old four-variant approval. Ask explicitly
before sending the eight labelled training examples with the original 128 dev
cases for v17; propose a new $4 cap, which is a ceiling, not a promise that all
calls will complete. Before execution, verify token accounting and reservation
sizing for the longer prompts: the largest dev prompt is 46,566 characters,
the existing runner assumes a 32,000-input-token ceiling, and no local tokenizer
was available to prove that bound. The candidate is prepared, not cleared for
paid execution until this preflight is resolved. Reuse cached v16, use serial execution and stop on quota or
budget. Compare coverage, paired dev kappa, confusion and grouped uncertainty.
If it fails, report the result rather than tune against the original test.

Only after a candidate qualifies on dev should a separately authorized, frozen
original held-out evaluation be considered. A dev score alone does not close the
project. Before that future test, harden the generic `run_calibration.cmd_test`
path too: it stops on quota failures but can still publish an incomplete terminal
result on other provider errors or budget stop. The new attribution path already
guards those cases; the generic path must receive equivalent protection before
any later original held-out execution. Selecting Reveal because its already-seen test score is 0.80 would change
the task after seeing results, not achieve the original goal. There is no honest
guarantee or reliable completion ETA for 0.74 before this measurement.

## Offline verification

- RouteBench tests: 519 passed. Shared core tests: 52 passed.
- Ruff passed for all changed Python modules and new tests; tracked diff whitespace
  check passed.
- `evalops.run_calibration verify`: all four measured variants reconstruct with
  their exact recorded hashes.
- The eight-example packet passes live corpus/split/label provenance validation;
  all 128 original headline dev prompts construct without a provider call.
- The attribution audit reproduces the four integrity blockers above. This is an
  expected failing data-quality check, not an implementation test failure.
- Fresh independent read-only review: ship for the offline changes. Paid execution
  and successor confirmation remain subject to the explicit prerequisites above.
