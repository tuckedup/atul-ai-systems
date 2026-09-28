# RouteBench

RouteBench is an OpenAI-compatible gateway whose routing and rollout inputs are measured quality, price, latency, availability, SLA, context length, and tenant budget. SDK-wire and streaming behavior are covered by local integration tests.

## Current control-plane checks

- 1,240 eval cases built from public benchmark sources with recorded provenance, across
  summarization/groundedness, coding, SQL, reasoning and tool use. See
  `evalops/data/raw/MANIFEST.json` for every source, row count and licence, and the sampling
  policy docstring in `evalops/build.py` for what was included and excluded, and why.
- A structured rubric judge, a calibration protocol and a release gate — but **no calibrated
  judge yet, and no κ result**. The held-out split has never been scored and no bundle has been
  frozen, so there is no `calibration_bundle.json` and `make calibrate-check` correctly exits
  non-zero. What exists is the machinery plus a dev-only variant grid in
  `evalops/data/dev_experiments.json`, whose numbers carry **unverified prompt provenance** —
  `make calibrate-verify` shows all 8 variants were recorded under an older `config_hash` formula
  that did not cover the prompt scaffolding. Their rubrics check out against this tree; the rest of
  the prompt cannot be checked either way. They are admissible for development probes and not
  behind a frozen bundle, so the dev split needs re-judging under pinned prompts before anything is
  frozen, which needs a provider key. See `BLOCKERS.md`.
- The ensemble and learned-combiner question is answered on the cached judgments at zero cost
  (`make probe-combiners`, `evalops/data/COMBINER_PROBE.md`): **no combination of the eight existing
  judges beats the strongest single one** on either track backing the headline claim. The artifact
  labels itself a dev-internal cross-validated probe rather than a held-out measurement, because
  the cache holds no train judgments to fit on.
- A 100-request backend-termination chaos test reroutes with 0 failed requests.
- Promotion tests exercise offline, shadow, 5%, 25%, full and automatic rollback with an audit
  event, **plus** the calibration gate in front of the ladder: an artifact that is unmeasured,
  rubric-drifted, incomplete, or below the κ policy blocks promotion regardless of how good the
  candidate's quality score looks.

### Correction to earlier versions of this README

A previous revision claimed "50 committed calibration labels; calibration pipeline reports
κ = 1.0 on this seed set". That number was an artefact, not a measurement: the generator wrote
one variable into both the `human_score` and `judge_score` columns, so κ = 1.0 was arithmetically
guaranteed and no model was ever called. The file is retained at
`evalops/fixtures/synthetic_selfagreement_labels.csv` purely as the negative fixture that the
gate is tested against. `docs/KAPPA_DESIGN.md` documents that defect and the three others found
alongside it, and §10 documents five more found in a second review — including the reason the
recorded dev numbers no longer reproduce. **Nothing in this repository has demonstrated κ ≥ 0.74.**
The best agreement measured anywhere here is a dev-internal cross-validated estimate: κ ≈ 0.64 on
the groundedness track, ≈ 0.50 pooled.

The self-hosted backend and concurrency/context/cache/quantization measurements are Docker-blocked tonight. The matrix printed by `make bench` is configuration data used to exercise the router, not a claim of freshly measured inference performance.

## Harness & Evals

Classifier results are cached by prompt hash. The router consumes a `(model, task_class) → quality`
matrix rather than model-name conditionals, then normalizes cost and p95 latency and applies hard
penalties for open circuits, SLA misses, context overflow, quality floors, and exhausted budgets.
Promotion advances only when each measured stage remains within the configured regression limit.

One thing that had to be fixed for the matrix to reach the router at all: the eval suites tagged
cases `coding`/`extraction`/`summarization` while `gateway/router.py` looked up
`code`/`extract`/`summarize`. Every lookup missed and the router silently substituted its `0.5`
default, so the policy was data-driven in shape but constant in practice. `evalops/taxonomy.py`
now owns one canonical set of names, maps legacy spellings explicitly, and raises on anything
unknown rather than defaulting — a tag the router cannot look up is a bug, not a 0.5.

### Judge design

The judge is never asked for a score. It answers a small set of observable yes/no questions from
a versioned, task-specific rubric, and `evalops/rubrics.py::score` — ordinary Python — turns those
verdicts into a number. Each rubric has exactly one `critical` criterion which forces 0.0 on
failure, giving a hard floor, while the remaining weighted criteria leave the score a gradient for
threshold tuning. Rubrics are written to predict their own oracle and explicitly declare out of
scope everything the oracle ignores, because a judge that penalises something the label ignores
disagrees with the label by construction.

Judge outputs are validated, not parsed optimistically: a missing criterion, an invented criterion
id, an unreadable verdict or a truncated response is an *error* with a status, excluded from the
metric. The release gate requires ≥98% completion so coverage cannot be traded for agreement. The
candidate response is delimited and declared untrusted data, and the generator's identity is
withheld from the judge.

### Calibration protocol

Threshold and variant are selected on the dev split only; the winning bundle is frozen with its
model, rubric hashes, dataset hash and split seed; the test split is then scored exactly once.
`run_calibration test` refuses to run before `freeze` and refuses to re-select anything.
Splits are assigned by hashing `(seed, group_id)` — groups being source documents or prompt
families — so sibling items cannot straddle a split and adding corpus rows does not reshuffle the
frozen test set. The confidence interval bootstraps over groups rather than items, and declines to
produce an interval at all when the sample cannot support one.

Label provenance is explicit and reported per track: published expert annotations and local hand
labels back the headline claim, deterministic gold-oracle labels (hidden tests, gold SQL, gold
final answer) are a strong but mechanically-applied label reported separately, and synthetic
fixture labels are refused outright.

## Governance & Lineage

Each route and backend call shares a trace ID with hash-chained audit events. Per-tenant budgets fail closed, queue saturation returns 429 with `Retry-After`, and backend failures open a breaker and reroute. Model traffic weights and quality are persisted through the single DSN-backed state adapter, making decisions reproducible from request metadata and the recorded matrix.

