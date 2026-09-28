# RouteBench: implementation handoff for held-out Cohen's kappa >= 0.74

Prepared 2026-09-27. Scope: RouteBench M5 judge calibration, with the minimum shared-core and promotion changes needed to make the result trustworthy. This is an implementation plan, not a report that the target has been achieved.

## 1. Objective and acceptance contract

Assumption: the requested target is RouteBench's existing **unweighted binary Cohen's kappa between automated judgments and independent human pass/fail labels**. This follows the current implementation. Do not substitute accuracy, correlation, weighted ordinal kappa, or a human-assisted score for this target.

Primary success: finite held-out kappa >= 0.74 on a frozen, representative test set, with an automated decision for every test item, no unresolved inference/parse errors, and no test labels used in prompts, fitting, threshold selection, model selection, or ensemble selection. Report the 95% confidence interval, confusion matrix, raw agreement, class prevalence, item/group counts, per-task results, model versions, and cost alongside the point estimate.

This is a point-estimate target. A stronger claim that the population kappa is at least 0.74 requires the lower confidence bound to exceed 0.74; do not silently conflate these claims. No method or timeline can guarantee either result before measurement.

For intuition only: kappa = (observed agreement - chance agreement) / (1 - chance agreement). If the chance term is 0.5, kappa 0.74 requires 87% observed agreement. This equivalence changes with the label marginals. Preserve the intended task distribution instead of rebalancing the test after seeing scores. [S1]

## 2. Verified starting state and defects

Inspected source in `C:/Users/atulp/Downloads/files/wt-forge`, branch `feat/forgecode`, HEAD `46b0d1c`.

1. `routebench/evalops/generate_data.py` constructs 50 alternating values and assigns each value to BOTH human_score and judge_score. `human_labels/labels.csv` contains those values. The current kappa 1.0 is synthetic fixture agreement, not human calibration.
2. The same generator makes 200 placeholder prompts, with expected answer `ok` and exact grading. These cannot establish representative task quality.
3. `packages/core/aisys/evals.py::calibrate` uses the same threshold for judge scores and human scores. Tuning it would redefine the target labels as well as the predictions.
4. The core function hardcodes a 0.6 trust threshold; the RouteBench wrapper separately hardcodes 0.6. A single shared policy must govern trust and release.
5. Undefined kappa is possible. A local check of 50 all-positive pairs returned NaN. In the wrapper, `NaN < 0.6` is false, so the rejection condition does not reject it. The core trusted flag is false, but the wrapper ignores that flag.
6. A local call to `EvalSuite.load('routebench/evalops/suites/generated.yaml')` loaded ZERO cases: the loader only searches directories. This must fail visibly or load the file correctly.
7. `g_llm_judge` returns an unvalidated float from JSON and a generic rubric. It has no persistent structured evidence or rubric/version binding. The main eval runner collapses grader errors into a score of zero; calibration must preserve errors distinctly.
8. Suite tags (`coding`, `extraction`, `summarization`, `reasoning`, `tool-use`) do not match router keys (`code`, `extract`, `summarize`, `reason`, `tool_use`). Quality may fall back to the router's default rather than the measured value.
9. `offline.py` returns tag means and can write JSON; `online.py` is an in-memory deque. The full persisted, sampled evaluation loop is still missing. `bench.py` prints illustrative constants.
10. `promote.py` checks quality drops but does not require a calibration artifact. The CI workflow runs tests and the synthetic-label check rather than proving an independent calibrated judge.

The previously passing 10 tests establish local behaviors. They do not establish human agreement, a valid eval dataset, or completed M5. The earlier 1-3 hour estimate for remaining M5 was too optimistic given these findings.

## 3. Research and what to implement

| Source | Technique studied | RouteBench use and limitation |
|---|---|---|
| G-Eval [S3] | Explicit evaluation criteria and steps, structured scoring, probability-weighted scores | Use fixed task checklists and structured criterion results. Test token-probability weighting only if the chosen API actually exposes usable probabilities; self-reported confidence is not that method. Its reported correlations are not kappa. |
| Prometheus / Prometheus 2 [S4, S5] | Fine-grained rubrics, reference answers, evaluator-specific training; direct and pairwise judgments | Use verified references and anchored rubrics first. Consider a dedicated evaluator as a later model candidate if an authorized endpoint exists. The papers do not prove a prompt-only reproduction will reach 0.74. |
| MT-Bench judge study [S6] | Reference-guided grading, order swaps, few-shot judging; analysis of verbosity and other biases | Hide generator identity; judge correctness against references. Existing `pairwise()` already swaps order: test and strengthen it rather than reimplementing it. Pairwise agreement is supplementary to the binary target. |
| Rulers [S7] | Locked rubric specification, structured evidence, calibration to human scoring scales | Version task rubrics and validate cited evidence. Fit a simple calibration model only if it beats a threshold baseline in development validation. Adapt the architecture; do not claim a reproduction of its ordinal scoring protocol. |
| Trust or Escalate [S8] | Confidence-based selective evaluation and cascades, with agreement/coverage tradeoffs | Test a second automated judge for uncertain cases. Report human escalation and selective coverage separately. Its selective agreement guarantee is not a full-coverage kappa guarantee. |
| scikit-learn [S1, S2] | Kappa definition and threshold tuning with held-out data | Keep human labels fixed; tune judge thresholds on development data only. Undefined values must fail the release gate. |

Recommended order: better data and references -> task rubrics -> a stronger judge candidate -> train-only examples -> development-only threshold -> optional ensemble. Fine-tuning and elaborate calibration should wait until the data volume and error analysis justify them. These are proposed experiments, not guaranteed improvements.

## 4. Data and annotation protocol

### Pilot, then frozen evaluation

- First collect a 60-item DEVELOPMENT pilot, ten examples per task family. Use real candidate outputs, including clearly correct, clearly wrong, and boundary cases. Fix annotation ambiguities and rubric errors here. Never reuse the pilot as final test data.
- Planning target after the pilot: 600 additional labeled response items, initially 200 train / 200 development / 200 locked test. These counts are a starting design, not a statistical power guarantee. Use pilot prevalence, errors, and group structure to determine whether the final test needs more independent examples BEFORE unblinding it.
- Define the intended task mixture before sampling. If production data are unavailable, call the result a specified six-task benchmark result, not production agreement. Maintain both label classes where naturally supported without editing labels to force balance.
- Split by source document, prompt family, incident/task identity, and near-duplicate group. All responses to the same prompt and all paraphrase/template variants belong to one split. Adjust approximate counts to respect groups.
- Have two humans independently label the items using the same rubric, blind to model identity and judge outputs. Use a third adjudicator for disagreements and preserve original votes, reasons, timestamps, and adjudication. Measure human-human agreement before adjudication as a diagnostic. Do not call it a strict ceiling on judge agreement with an adjudicated reference.
- AI may propose drafts or identify review candidates, but cannot be recorded as a human rater. Public human labels may supplement development only when rubric, provenance, and task semantics fit; do not casually mix preference labels with pass/fail correctness labels.
- Fit examples or calibrators on train, choose variants/thresholds on development, freeze the winning bundle, and evaluate the test once. If it fails, report the failure. Subsequent development requires a genuinely new confirmation set before making a new held-out claim.

### Data artifacts

Use separate versioned records, joined by stable IDs:

- `cases.jsonl`: case_id, group_id, canonical task_class, difficulty, input, reference, reference_provenance, rubric_id/version, source/license, candidate_output, generator_id/version, content_hash, origin (`real` or `synthetic_fixture`).
- `annotations.jsonl`: case_id, annotator_id, binary label, rubric_version, brief evidence/reason, timestamp, adjudication metadata. Keep labels out of judge inputs and logs.
- `splits.json`: case/group IDs per split, split seed, sampling policy, manifest hashes. Make overlap/duplicate checks mandatory.
- `judgments.jsonl`: case_id, model requested AND served, judge/rubric/prompt/config hashes, raw response, validated criteria/evidence, raw score, predicted label, status/error, token/cost/latency data, trace_id.
- `calibration_bundle.json`: fitted rule, chosen threshold, model/version binding, rubric hashes, train/dev hashes, selection protocol and software version.
- `calibration_report.json` plus Markdown: complete held-out metrics and provenance, including failures and the release decision.

The judge runner should not receive test labels. Join predictions to labels in a separate reporting step after freezing the bundle. Manifests document provenance; a hash alone does not prove independent human annotation.

## 5. Concrete code changes, in dependency order

### P0: correct the measurement and document the fixtures

1. `routebench/evalops/generate_data.py`, `human_labels/labels.csv`, `suites/generated.yaml`: retain generated examples as explicitly named test fixtures under a fixtures directory. Stop generating anything advertised as human labels. The production loader must reject fixture provenance and missing real annotation provenance.
2. `packages/core/aisys/evals.py::calibrate`: separate immutable binary human labels from continuous judge scores; accept keyword-only `judge_threshold` and `min_kappa`. Validate equal nonempty lengths, numeric finite scores in [0,1], and binary human labels. Report a fixed 2x2 confusion matrix. Fail closed on undefined metrics. Introduce a RouteBench policy default of 0.74 without silently changing unrelated projects' behavior; migrate all RouteBench callers to that policy.
3. `EvalSuite.load`: support a YAML file and a directory; reject missing paths, zero cases, duplicate IDs, invalid schemas and unknown graders. Preserve existing directory callers.
4. `routebench/evalops/calibrate.py`: production validation, finite-value checks, configurable policy, machine-readable output, and nonzero exit on unmet acceptance. Save failed reports as well as successful ones. `trusted` and the CLI exit status must use the same gate.
5. Add `evalops/metrics.py`: agreement, kappa, confusion, prevalence, per-task results and reproducible 95% group-bootstrap intervals (initially 2,000 replicates). Resample independent groups, retain class prevalence variation, report undefined-replicate counts and refuse an unsupported interval. Small/degenerate strata get an explicit insufficient-data result.
6. `README.md` and `STATUS.md`: state that the current 1.0 is fixture agreement; real baseline is not measured. Do not mark 0.74 achieved until a valid report exists.

### P1: implement a reproducible judge and real data workflow

7. Add `evalops/dataset.py`, `evalops/splits.py`, `evalops/annotation.py`: validated records, stable group splits, annotation export/import, original-vote preservation, provenance checks and no-overlap verification. Export blank human labels, never prefill them with model predictions.
8. Add `evalops/rubrics/{code,extract,sql,summarize,reason,tool_use}.yaml`: task-specific observable pass/fail criteria, critical-failure rules and boundary examples. Human acceptance semantics remain fixed; a learned judge threshold must not change them.
9. Add `evalops/judge.py` and extend the core grader through a compatible adapter: typed output with criterion verdicts, short evidence spans/reasons, critical_errors, and bounded score. Validate evidence references against the supplied material. Treat candidate text as untrusted content. Reject malformed, nonfinite, out-of-range, unknown-criterion and truncated outputs; record bounded retries explicitly.
10. Reuse deterministic evidence where appropriate: unit test outcomes for code, schema AND semantic checks for extraction, result-set comparisons for SQL, executable checks for applicable reasoning/tool tasks, and claim support for summaries. Schema validity alone is not correctness. If deterministic checks override the judge, report the primary score as an automated hybrid evaluator and separately report the LLM-only score.
11. Pin judge configuration and actual served model. The shared `llm.chat` uses `fallback or settings.fallback_models`, so passing an empty list does NOT reliably disable fallback: add an explicit compatibility-preserving way to distinguish unspecified from disabled fallback. Never silently mix models in one calibration bundle. Reuse existing token/cost tracing.
12. Add `evalops/experiments.py`: cached candidate judgments, resumable bounded runs, explicit spend cap, and a predeclared small candidate grid. Cache keys must include content, rubric, prompt, model and decoding hashes. An unavailable price is not zero cost; require a configured estimate before paid runs.

### P2: calibrate without leaking test labels

13. Use a score -> fixed binary label rule. Optimize the threshold on development predictions with an explicitly bounded grid and deterministic tie-breaking. Never threshold/relabel the human targets during tuning.
14. Baseline: existing generic judge on REAL pilot/development records at 0.5. Then compare task rubric + references, that configuration plus a small train-only set of pass/fail/boundary examples, and a second stronger configured judge. Add threshold tuning as a separately reported ablation.
15. Only if needed, compare a two-judge automated cascade/ensemble. Choose its trigger, combination rule and budget on development data, record coverage and total cost, and require a final automated decision on every test item for the primary claim. Human-reviewed or abstention-only results are separate metrics.
16. If there is enough training data, compare logistic calibration on structured criterion features fitted on train and selected on development. Start with a global threshold; allow per-task thresholds only with adequate prespecified support and validation. Isotonic regression and high-dimensional calibrators are optional, not defaults for small samples.
17. Freeze the entire bundle, including any examples, calibrator, ensemble and escalation rule, before the locked test. Do not run every candidate on test and pick the best.

### P3: connect the accepted evaluator to RouteBench

18. Define one canonical task taxonomy and an explicit legacy alias mapping. Use it in datasets, `gateway/classifier.py`, `gateway/router.py` and quality storage; add reasoning/tool-use classification coverage. Test that a measured task score reaches the router rather than its default 0.5.
19. `evalops/offline.py`: evaluate real suites per configured backend; persist observations and measured quality with sample counts and calibration bundle IDs. Keep provider/infrastructure failures separate from quality failures. Reject empty/incomplete runs before publication.
20. `evalops/online.py`: implement bounded sampling and async grading with versioned judgments, explicit error counts, a minimum sample count and defined rolling window. Keep monitoring samples out of the sealed evaluation.
21. `gateway/state.py`, `app.py`: persist and load accepted matrices using the single configured DSN and a backend-neutral database layer. The present direct sqlite3 adapter violates the one-line backend-swap constraint; replace it behind the shared storage interface when adding calibration records. Bind state to dataset, rubric and judge versions.
22. `evalops/promote.py`: require a valid calibration artifact plus complete offline evidence before existing quality-drop gates. Fail for fixture-only, missing, stale or incompatible bundles, undefined kappa, and kappa < 0.74. Artifact validity should be tied to model/rubric/config changes and a documented expiry policy.
23. `evalops/bench.py`: print measured artifacts with sample counts and version IDs, or explicitly report not measured. Remove illustrative constants from the measured reporting path.
24. `.github/workflows/routebench-gate.yml`: run deterministic unit/integration tests and validate frozen artifacts, including negative gate cases. Live measurements are an explicit budgeted job; default CI needs neither provider keys nor Docker. Make CI invoke modules in the correct workspace environment.

## 6. Required tests and experiment record

Unit and integration tests must cover:

- Known confusion matrices with hand-computed kappa; all-correct, all-wrong, one-class, empty, length mismatch, NaN/Inf and invalid-range inputs. A balanced synthetic 200-item case with TP=87, TN=87, FP=13, FN=13 yields 0.74: use ONLY as an arithmetic fixture.
- Moving the judge threshold cannot change human labels. Nonfinite kappa, 0.7399, insufficient evidence and synthetic provenance fail release. Exactly 0.74 passes the numeric condition when all other conditions pass. Do not gate rounded displays.
- File/directory suites load the expected count; empty or duplicate suites fail. Group/source/near-duplicate overlap fails; holdout IDs cannot be retrieved as examples.
- Structured-output errors remain errors and block incomplete release reports. Prompt injection in candidate text cannot bypass rubric/output validation. Pairwise reversal and malformed responses are handled explicitly.
- Calibration/model/rubric hash mismatches invalidate published quality and promotion. Unknown/stale task labels cannot silently map to a default quality.
- A below-target artifact blocks promotion despite a high model quality score; a compatible passing artifact permits normal staged rollout; regression still causes rollback with audit evidence.
- Judge retries, fallbacks, costs and trace IDs are recorded. Unknown price cannot be logged as a measured zero.
- All default tests run without live network services. SQL/code fixture execution must obey the subprocess constraints below.

Experiment table fields: variant, split, dataset hash, judge served, rubric/example hashes, threshold, n/groups, kappa, CI, agreement, confusion, prevalence, completion/coverage, errors, cost and latency. Preserve unsuccessful variants and their development results.

## 7. Infrastructure amendment remains binding

- Do not run, check for or wait on Docker, Docker Compose, Postgres:5432, Redis:6379, Phoenix:6006, Prometheus:9090 or Grafana:3000.
- Database defaults to `./.local/aisys.db` through the single settings DSN. No new hardcoded SQLite selection outside the settings default; use a backend-neutral abstraction so a future DSN change switches the backend.
- Use the shared in-process cache implementation. Keep existing OpenTelemetry decorators and span attributes, with ConsoleSpanExporter and `./.local/traces.jsonl`.
- Executable task checks use a fixture copy as cwd, a hard timeout and a memory cap, including a supported Windows mechanism. Never execute a task shell command in the repository root. Subprocess limits do not provide container-grade isolation; use trusted, curated fixtures under this amendment.
- Commit dashboard JSON only. Any criterion that specifically requires Docker or Phoenix UI/live service proof is BLOCKED with the exact reason `requires docker` in BLOCKERS.md.
- Record these decisions and actual deviations in DECISIONS.md. Historical service runs do not count as present verification.

## 8. Delivery sequence and realistic dependencies

1. Deliver P0 plus tests and annotation tooling first. All of this can proceed without human labels or paid inference.
2. Use the 60-item pilot to settle the rubric, discover error categories, estimate annotation effort and choose the final sampling plan.
3. Collect and adjudicate the independent annotations while implementing P1/P2. The agent cannot complete this human step by inventing labels.
4. Run the bounded development experiments; freeze one winner; execute the locked test; publish its actual result.
5. Connect the accepted evaluator to routing/promotion and deliver P3. If the target fails, keep the release gate closed and provide error analysis and the next data/model experiment.

Planning estimate: several focused engineering days for this scope, plus annotation availability and provider runtime. At 1-3 minutes per annotation, 600 items x 2 raters is 20-60 person-hours before adjudication, plus the pilot; complex coding/SQL examples can take longer. Parallel annotation changes elapsed time, not the amount of work. Measure pilot effort before committing to dates. No ETA can promise the metric itself.

## 9. Copy/paste assignment for the implementing agent

Implement docs/ROUTEBENCH_KAPPA_074_PLAN.md in C:/Users/atulp/Downloads/files/wt-forge. Target genuine held-out unweighted binary Cohen's kappa >= 0.74 for RouteBench's automated evaluator against independent human labels. Start with P0 and deterministic tests, then complete the annotation, judge, experiment, calibration and integration tooling. Preserve unrelated working-tree changes. Use current repository evidence rather than assuming the historical milestone checkboxes prove M5.

Keep the infrastructure amendment in section 7. Do not fabricate human annotations, provider runs, raw measurements or a passing score. Generated examples are fixtures only. Freeze groups/splits and the human-label semantics; develop only on train/development; freeze the complete judge bundle before the final test. Report hybrid, LLM-only, selective and human-assisted results separately. Continue all independent implementation when real annotations or authorized provider access/budget are missing; leave only the dependent measurement blocked and provide the concrete annotation export and bounded run instructions.

Deliver code, meaningful regression tests, verified local commands, versioned rubric/data schemas, experiment runner, full failed/passed reports, provenance-bound calibration artifact, routing/promotion gates and truthful README/STATUS/BLOCKERS/DECISIONS updates. Provide a report showing the exact data/split, threshold, kappa/CI/confusion, completion/errors and measured cost. Do not claim success until the acceptance contract in section 1 is satisfied. Any new post-test optimization requires a fresh confirmation set.

## Sources

Sources were selected for direct applicability, not as an exhaustive survey of all work. Research outcomes on other benchmarks are not RouteBench guarantees.

- [S1] scikit-learn, Cohen's kappa API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
- [S2] scikit-learn, decision threshold tuning and separation of fitting/tuning data: https://scikit-learn.org/stable/modules/classification_threshold.html
- [S3] Liu et al., G-Eval: https://arxiv.org/abs/2303.16634 ; author implementation: https://github.com/nlpyang/geval
- [S4] Kim et al., Prometheus: https://arxiv.org/abs/2310.08491
- [S5] Kim et al., Prometheus 2: https://arxiv.org/abs/2405.01535 ; author implementation: https://github.com/prometheus-eval/prometheus-eval
- [S6] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena: https://arxiv.org/abs/2306.05685
- [S7] From Rubrics to Reliable Scores: Evidence-Grounded Text Evaluation with LLM Judges (Rulers): https://arxiv.org/abs/2601.08654 ; author implementation: https://github.com/LabRAI/Rulers
- [S8] Jung et al., Trust or Escalate: https://arxiv.org/abs/2407.18370

Primary paper metadata and relevant body passages were inspected for G-Eval, Prometheus, MT-Bench, Rulers and Trust or Escalate; Prometheus 2 is included as a model-family alternative based on its paper metadata/abstract. Linked author repositories are follow-up implementation references, not repositories audited in this planning pass.
