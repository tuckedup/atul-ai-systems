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
