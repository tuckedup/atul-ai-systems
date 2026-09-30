# RouteBench: verified debugging handoff

Audit date: 2026-09-28. Cloud baseline: `ea262db` on
`claude/optimistic-shannon-qmnv1j`. Audit and quota-handling fixes are maintained on
`codex/routebench-finish-audit`. This is a historical handoff; see
`ROUTEBENCH_PROJECT_BRIEF.md` for the current publication scope and results.

## Latest result (2026-09-28)

Credits were restored and the owner explicitly authorized the four-variant,
128-case development evaluation within $12. It is now complete: 512/512 usable
current-config judgments, recorded token-priced cost $2.502474 including retries
and the health probe. Best single dev kappa is 0.6532; best inspected simple
combination is 0.6719. The 0.74 target is not achieved. No held-out inference or
freeze occurred. See `ROUTEBENCH_DEV_RESULTS.md` and
`../routebench/evalops/data/dev_four_variant_analysis.json`.

The sections below document the earlier audit and execution blocker, not the
current API/account state. No label or prompt was changed during the paid run.

## Earlier blocked execution attempt (2026-09-28)

The user requested execution. A credential is now configured locally, pointing at
`api.openai.com`. One actual tiny GPT-4.1 request returned HTTP 429 with
`error.type=insufficient_quota` and `error.code=credit_balance_exhausted`. The
512-result dev run was therefore not started. No new agreement measurement,
held-out result, or successful inference was produced. Credentials were not logged.

The shared client now distinguishes billing/quota failures from retryable rate
limits. Quota failures stop fallback, judge repair/sample retries, pending work,
and later dev variants. Calibration returns exit status 3 on quota exhaustion;
the test command preserves its original bundle and report. Cached successes remain
resumable. Mocked regression tests exercise these behaviors without paid inference.
Verification for this update: 540 core/RouteBench tests passed (five existing
warnings); changed-core and whole-RouteBench Ruff checks and `git diff --check`
passed. The four-variant offline preflight still reports 512 missing dev results.
The fresh quota-path review returned `ship`, with no findings. Requests already in
flight when quota exhaustion is observed cannot be recalled; pending/new work stops.

Restoring credits for the configured API project is an account-owner action; no
purchase or billing-limit change was made. No more labels/files are needed to begin
the planned dev comparison. An installed MiniCheck/AlignScore runtime or their
checkpoint weights were not found in the active environment/local model cache.

Official references checked for this attempt:
[quota error codes](https://developers.openai.com/api/docs/guides/error-codes) and
[GPT-4.1 pricing](https://developers.openai.com/api/docs/models/gpt-4.1).
The configured $2 input/$8 output per million token rates match the latter.
Retrying `credit_balance_exhausted` does not restore credits. The initial intended
dev cap was $12; this was not a credit purchase or a claim that the target can be
achieved for $12.

## Pre-experiment assessment (historical)

The 0.74 target is NOT achieved or disproved. The latest work mostly improves
measurement infrastructure. It contains no measured result for the proposed new
groundedness judges. Code copied from a successful paper does not transfer the
paper's accuracy to this corpus, labeling convention, split, or metric.

An offline `plan --select-on headline` on the actual corpus found 128 dev cases
and 128 missing current-provenance judgments for EACH of v16, v8, v12 and v14.
That is 512 missing case/variant results, not evidence that those methods failed.
Legacy cached judgments remain useful for explicitly unverified development
analysis; they are not silently relabelled as current-prompt evidence.

The committed combiner probe's groundedness result (0.6440 on 87 common cases)
is dev-internal grouped CV using legacy components. It is neither the new
variants' score nor an independent held-out result. Do not compare this subset
directly to a different variant's partial-coverage best score.

## What Claude got right

- Corrected the legacy-hash diagnosis: formula changes do not prove prompts were lost.
- Added explicit evidence signals, bundle/data bindings, strict release checks,
  model-price reservations, and optional checker adapters.
- Did not claim the target was achieved or manufacture a held-out score.
- Left the no-Docker amendment intact.

Baseline reproduced here: 505 tests passed across `packages/core/tests` and
`routebench/tests`. These are code checks, not agreement measurements. The entire
ForgeCode/incident-commander suite was not reverified in this audit.

## Concrete defects corrected locally

1. **Wrong winner for the requested metric.** `freeze --select-on headline`
   selected the variant using pooled dev kappa, THEN tuned only its threshold on
   headline labels. It now recomputes each candidate's threshold AND score on the
   requested track, with >=98% coverage of that track's complete dev population.
   Stale precomputed summary scores are not trusted for winner selection.
2. **Data bindings weren't checked before test inference.** The test command now
   rejects altered bundle identity, labels, split membership, split seed, corpus,
   rubrics, or judge configuration before making provider calls. Freeze also
   refuses dev records whose corpus/labels/split bindings are absent or changed.
3. **Overwrite guard was late.** A protected measured bundle is now checked
   before corpus loading and selection, not after those operations.
4. **Budget reservations caused premature stopping.** A worker previously marked
   the entire variant stopped if other workers temporarily held the available
   budget. It now waits for those reservations to settle, while still refusing
   work that cannot fit the remaining budget. Unusable `status=ok` cache rows are
   no longer treated as successful cache hits.
5. **AlignScore wrapper changed the upstream baseline.** The declared `nli_sp`
   baseline now passes the complete response to AlignScore, which performs its
   own sentence/chunk aggregation. It no longer unnecessarily calls MiniCheck's
   splitter first. CPU is passed as `"cpu"`, not torch's invalid device `-1`.
   Upstream contract checked against the local pinned AlignScore source, not by
   downloading or executing its weights. No-splitting metadata now says `none`.
6. **Strict groundedness threshold was unavailable.** The grid ended at 0.95;
   1.0 is now included so fraction-based scores can express "all claims supported."
   This is a dev candidate, not an asserted accuracy improvement.
7. **No focused execution/preflight.** Added `plan` and `--variants`/
   `--select-on` for dev, preserving explicit variant order. Headline-only runs
   avoid spending on unrelated oracle tasks. Earlier dev records are archived
   before replacement. Existing cache entries remain append-only.

Regression coverage uses synthetic, isolated fixtures only. No real test-split
judgments were generated or inspected, no provider money spent, no weights
downloaded, and no Docker services probed.

Post-fix verification: 524 core/RouteBench tests passed, with five existing
deprecation/undefined-kappa warnings; whole-RouteBench Ruff checks are clean.
The fresh read-only review returned `ship`, with no remaining patch findings.
It corrected this document's distinction between dev eligibility and release
coverage. Its separate test attempt was blocked by dependency download; the
passing test evidence above is from the primary agent's existing local environment.

## Reproducible next experiment

From this worktree in PowerShell (existing local Python environment):

```powershell
Set-Location C:\Users\atulp\Downloads\files\wt-routebench-finish
$env:PYTHONPATH = 'packages/core;routebench'
$rbPython = 'C:\Users\atulp\Downloads\files\wt-forge\.venv\Scripts\python.exe'
$rbVariants = @('v16-holistic-4.1', 'v8-decompose-4.1',
                'v12-addressed-4.1', 'v14-contradict-4.1')
& $rbPython -m evalops.run_calibration plan --select-on headline --variants $rbVariants
```

This command is offline. The first three variants have a configured reservation
envelope of $30.72 each for 128 entirely uncached cases; v14 has $28.2624.
The four sum to $120.4224. This is deliberately conservative reservation math
(32,000 prompt tokens and all configured retries), NOT a price quote, expected
bill, or request to spend that amount. Actual prices/settings need verification
before a paid run. The existing prompt-ceiling estimate is an assumption, not a
provider-side billing limit. A smaller authorized cap can run/resume the same
cache; incomplete coverage is not grounds to publish a favorable score.

Only after a provider key is configured securely and a spend cap is authorized:

```powershell
# Set $rbApprovedCap to the owner's explicitly approved USD cap first.
& $rbPython -m evalops.run_calibration dev --select-on headline --variants $rbVariants --budget $rbApprovedCap
```

Use the same command to resume cache misses. Preserve every run record. Do not
run the whole 17-variant grid by default. v16 is the matched holistic control;
v8 tests decomposition; v12 tests addressed evidence; v14 tests contradiction.
This small declared comparison answers whether mechanisms improve agreement.

Inspect dev coverage, confusion matrices, model actually served, parser failures,
false positives/negatives, and per-document disagreements. Record claim omission,
unsupported details, entities/numbers, evidence retrieval failure, and label
ambiguity separately. Do not edit human labels to match the model. Do not treat
literal quote presence as proof of entailment: the current evidence signals are
diagnostics, not semantic verification and not automatically learned penalties.

When a sufficiently covered candidate justifies the independent test:

```powershell
& $rbPython -m evalops.run_calibration freeze --select-on headline
# Separate approved test budget required; execute the frozen candidate once.
& $rbPython -m evalops.run_calibration test --budget $rbApprovedTestCap
```

Freeze alone is NOT a pass. Acceptance requires the measured human-response
track's unweighted binary Cohen kappa >=0.74, 100% coverage under the default
release gate (distinct from >=98% dev freeze eligibility), valid bindings,
and the uncertainty/confusion report. Do not rerun/reselect against a disappointing
test result; further development requires genuinely new confirmation evidence.

## Remaining engineering work, not claimed complete

- `checkers.py` is NOT integrated with `run_calibration`'s execution, cache,
  freeze, or test dispatch. Fake adapter tests do not establish real MiniCheck
  or AlignScore inference. Add a typed backend dispatch and result-to-Judgment
  conversion, retaining per-sentence diagnostics and failure statuses. Bind
  actual upstream installation and weight content/revision, not just constant
  commit strings/checkpoint paths; verify licenses and execute real smoke cases
  in an isolated dependency environment before comparison.
- Learned combiners need train-split predictions and fitting, followed by dev
  selection. The committed dev-internal CV probe is not that training workflow.
  Never feed test labels into fitting, thresholds, prompt design or selection.
- Check raw dataset provenance/subset identity before claiming reproduction of
  an upstream paper's results. A different dataset or metric is an adaptation.
- Deployment/offline-router integration and other projects' remaining milestones
  are separate acceptance work. This evaluator audit does not close all projects.
- Docker-only proofs remain BLOCKED, reason `requires docker`; no simulated proof.

## Cloud handoff boundary

The pushed cloud commit was enough for this inspection; no additional screenshots
or pasted code are needed. Unpushed files/private caches in Claude's temporary
cloud filesystem would need exporting only if they exist and matter. A cloud coding
subscription is not the provider credential used by these judge calls. Never paste
keys into chat or commit `.env`. The local fixes are not available to Claude Cloud
until explicitly pushed.

Do not promise an ETA for reaching 0.74. Measure a small real inference batch's
latency/error rate first, extrapolate run time, and keep execution ETA separate
from the uncertain number of research iterations required to meet the target.
