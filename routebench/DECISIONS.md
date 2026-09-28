# RouteBench decisions

- Use the shared in-process cache interface for classifier and response caching; no gateway code imports a cache vendor.
- Keep routing data-driven: backend metadata and per-task quality matrices are inputs, while policy scores quality, cost, latency, availability, SLA, context, and tenant budget.
- Represent providers through one `BackendClient` protocol; OpenAI-wire provider endpoints share the tested implementation.
- Use completed-response streaming chunks for protocol compatibility in this local control-plane build; backend-native token streaming can replace the client implementation.
- Commit the Grafana dashboard JSON without attempting to render it on this host.


## Judge calibration (2026-09-27)

- The judge does not produce a score. It answers observable yes/no questions from a versioned
  task rubric, and `evalops/rubrics.py::score` computes the number in Python. A free-floating
  model-emitted 0-1 score has no stable meaning across runs and leaves no clean threshold, which
  is the usual reason judge-vs-human kappa stalls in the 0.4s.
- Exactly one criterion per rubric is `critical`. It forces score 0.0 on failure, giving a hard
  floor; the remaining weighted criteria give the score a gradient so dev threshold tuning has
  real leverage. A rubric with no critical criterion is rejected at load time.
- Each rubric is written to predict *its own oracle*, and explicitly declares out of scope
  everything the oracle ignores (complexity, style, units, SQL text, argument order). A judge
  penalising something the label ignores disagrees with the label by construction.
- Human labels are binary on arrival and are never re-thresholded. Only judge scores are
  binarized. The previous `calibrate()` thresholded both, so tuning moved the target.
- Cohen's kappa is computed in closed form and returns `None` when expected agreement is 1.0.
  `sklearn` returns `nan` there, and `nan < threshold` is `False`, so the previous release gate
  passed an undefined kappa. `Agreement.passes()` requires `defined is True`.
- Label provenance is an enum and results are reported per track. `human_expert_annotation` and
  `human_local_annotation` back the headline claim; `human_gold_reference_oracle` (hidden tests,
  gold SQL, gold final answer) is a strong but mechanically-applied label and is reported
  separately; `synthetic_fixture` is refused by `pair()`.
- Splits are assigned by hashing `(seed, group_id)`, not by shuffling, so adding corpus rows does
  not reshuffle existing assignments and the frozen test set stays frozen. Groups are source
  documents / prompt families, so sibling claims about one document cannot straddle a split.
- The bootstrap confidence interval resamples groups rather than items, and refuses to produce
  an interval (`usable=False`) below 10 groups or above 5% undefined replicates.
- Judge errors are recorded with a status and excluded from the metric. `aisys.evals.run`
  collapses grader exceptions into `score=0.0`; for calibration that turns a provider timeout
  into "the judge said fail" and biases kappa. The release gate additionally requires >=98%
  completion, so coverage cannot be traded for agreement.
- Models with no entry in `aisys/pricing.yaml` are refused before a paid run starts.
  `estimate_cost` returns `None` for an unknown model, and `None` accumulating as zero is how a
  spend cap silently stops binding.
- The judge pins fallback off and verifies the served model against the requested one.
  `aisys.llm.chat` used `fallback or settings.fallback_models`, so `fallback=[]` was falsy and
  fell back anyway; a silent mid-run model switch would blend two judges under one label.
- `evalops/fixtures/` holds the previous synthetic suite and the self-agreeing label file. They
  are retained only as negative fixtures; the gate must be observed rejecting them.
- Spider databases are vendored under `evalops/data/raw/spider_db/` so the SQL oracle can compare
  result sets. A textual SQL comparison was considered and rejected: two correct queries rarely
  look alike, so a judge trained to predict text similarity would be predicting noise.
- `AggreFact-XSum` and the attribution/fact-verification subsets of LLM-AggreFact are excluded
  from the groundedness corpus, on documented annotation-quality and task-semantics grounds, as a
  pre-registered sampling decision made before any judge was run. See the policy docstring in
  `evalops/build.py`.
- Judgment cache entries are only treated as hits when `status == "ok"`. Errored judgments are
  still written (so the error rate is auditable) but are re-attempted on a re-run. Re-running
  `run_calibration dev` is therefore the intended way to raise a variant's completion rate
  toward the >=98% the release gate requires: it re-judges only the failures and costs nothing
  for work already done.
- Known wart, deliberately not changed mid-measurement: `JudgeConfig.concurrency` is an
  operational knob that cannot change a verdict, but it is included in `config_hash` and so
  invalidates cached judgments when tuned. Excluding it would be correct, and doing so while a
  dev run is in flight would silently invalidate that run's cache, so it is left until after the
  current dev -> freeze -> test cycle completes. It affects cache reuse only, never a reported
  number.

## Judge calibration, second round (2026-09-28)

- Evidence quotes are addressed to the material they came from, not matched against one
  concatenated haystack. The previous check searched `context + candidate_output + task_input +
  reference` and reported a single `evidence_verbatim` count, which inverts the groundedness
  question: a quote the judge lifted out of the candidate's own unsupported sentence counted as
  successfully cited evidence. `EvidenceLocation` counts `in_source` and `in_candidate`
  independently (not as a partition — a faithful summary quotes text that is in both), and in
  decompose mode a support verdict whose span is absent from the source increments
  `support_unverified`. Rubric mode still credits a candidate-side quote, because several criteria
  ask about the response itself and quoting it is the honest way to answer; the fix is to address
  quotes, not to forbid one side.
- `decompose_addressed` makes the judge cite numbered `[S<n>]` source sentences, and the quote is
  checked against the sentences it named rather than the whole document. Plain `decompose` can only
  ask whether a string appears somewhere in the source, which a topically-similar sentence satisfies
  while supporting nothing. A cited id outside the numbering is a hard parse error, not a soft
  signal: the judge was shown the numbering, so citing S99 of a 3-sentence document means its output
  does not describe the material it was given.
- `contradict` is in the grid as an ensemble member with a different failure mode, not as a better
  judge. The combiner probe found eight variants of one rubric mechanism ensembling to a margin of
  exactly +0.0000 over the strongest single one, which is what correlated errors look like. Adding a
  ninth rubric variant would have been more of the same; the complement question (does the source
  *contradict* anything?) can be wrong in different places.
- Combiners are fitted per task class, not pooled. Criterion ids belong to a task's rubric, so
  pooling puts a mostly-absent column in the matrix for every criterion of every other task: on dev
  a pooled criterion-feature combiner scored κ = −0.13, worse than chance, while the same features
  fitted within the groundedness track scored 0.62.
- `combiners.py` hand-rolls its logistic fit rather than importing sklearn, for the reason
  `metrics.py` hand-rolls κ: the production path stays dependency-light and exactly reproducible,
  and the tests cross-check against `sklearn.linear_model.LogisticRegression` at three
  regularisation strengths so the hand-rolled version cannot drift.
- An ensemble must earn its components. Coverage is multiplicative — the eight dev variants cover
  229/371 cases jointly (61.7%) because two lost a fifth of their judgments to rate limits — and the
  release gate requires ≥98% on the primary track. `require_coverage` raises rather than reporting a
  κ computed over the cases where everything happened to succeed, because that subset is biased by
  whatever made the rest fail: long inputs are both what times a judge out and what is hard to grade.
- A tie never counts as a combiner win. `Comparison` breaks ties toward the simpler contender, so
  "as good as the single judge for k times the cost and k times the failure surface" is recorded as
  a loss.
- `bundle_id` is recomputed by the release gate. It was written at freeze and never checked again,
  so hand-editing `threshold`, `variant_id`, `dataset_hash`, `split_seed` or `judge_config` in the
  JSON left a stale id that nothing compared against anything — the artifact's integrity identifier
  was ornamental. The honest limit is stated in a test: a hash proves the fields were not edited, not
  that the reported measurement used them.
- `validate_bundle` can bind an artifact to the corpus, not just to the rubrics. A κ measured on a
  different (smaller, easier, older) corpus is not evidence about this one. The CLI supplies the
  corpus hash by default so CI checks it; `--no-dataset-check` exists for validating an artifact on
  a machine that does not carry the corpus, and prints that the binding was not verified.
- `Paired` carries its split and the `(variant_id, config_hash)` of every judgment it joined.
  `select_threshold` refuses the test split and `evaluate` refuses anything that is not test, or
  judgments from a variant the bundle does not name, or a `config_hash` that drifted after the
  freeze. The module docstring claimed this ordering was "enforced by the CLI, not by
  documentation"; it was in fact enforced by neither, only by the order in which the CLI happened to
  call things.
- `freeze` rebuilds the judge from the experiment record, not from `GRID` — the same rule `cmd_test`
  already applied to a frozen bundle. Keying the cache off `GRID` meant that any change to a
  `config_hash` input renamed every judge in the record, every lookup missed, and freeze failed with
  "0 paired items" while blaming the dev set. `dev` now stores the full `judge_config` and
  `prompt_template_hash` in each row so the record describes its own judge, and `make
  calibrate-verify` answers "is this record still usable?" without attempting a freeze.
- `prompt_template_hash` is per mode rather than one hash over every template in the module. The
  principled reason: a variant's hash should depend on the prompt it actually renders. The practical
  one: `generic`, `rubric` and `decompose` keep the historical five-blob scaffold set as a
  deliberate compatibility anchor, because re-hashing them would discard the only paid judgments in
  the repository without one of their prompts having changed. The wart this leaves — those three
  remain sensitive to each other's templates — is over-invalidation, which can cost a needless
  re-judge but can never attribute an old verdict to a new prompt, so it is the safe direction.
- The combiner probe is committed as an artifact that labels itself. It fits and selects inside dev
  by grouped cross-validation, which is NOT the acceptance protocol, because the cache holds no
  train judgments to fit on. `COMBINER_PROBE.md` says so in its first line, the JSON carries a
  `protocol_deviation` field, and a test asserts both are present — a deviation that is only
  described in a commit message is one that will be read as a result later.
