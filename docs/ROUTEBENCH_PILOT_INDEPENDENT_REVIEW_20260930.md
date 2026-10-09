# Independent review of Claude's third pilot-readiness pass

Date: 2026-09-30. Verdict: **fix-first, then consider one bounded pilot**.

Scope: local code and the supplied Claude transcript. AI-assisted review, not human label adjudication, legal clearance, GPU certification or a measured model improvement. No production code, labels, manifests, permissions or frozen recipes changed in this review. No GPU work, provider calls, installations or downloads.

## Confirmed progress

- `cli.execute` now validates the frozen inputs and starts a supervised worker; it is no longer an intentional stub.
- Input text is regenerated from hash-checked raw files and compared against selected manifest rows.
- The configured feasibility probe receives 32 training examples, sufficient for two optimizer updates. The torch-free schedule steps and rescales the final partial accumulation group.
- The real backend seeds before adapter initialization and synchronizes CUDA around timed sections. Stub tests check the ordering; they do not establish actual GPU behavior.
- The watchdog runs outside the worker. Existing tests exercise real synthetic subprocess hangs at startup, phases and total cap. Progress writes retry Windows file-sharing failures.
- The documented 7,200-second total has a 15-second external-watchdog grace, plus polling/termination latency. It is not a strict 7,200-second wall-clock cutoff.

## P1: terminal success can survive a worker/artifact failure

Locations: `routebench/evalops/pilot/cli.py:111` (result written before adapter save), `:143` (parent checks only result-file existence), `:183` (CLI success depends only on whether the watchdog killed the process). `run.py:198` writes the result directly, not via atomic publication.

Independent reproduction, using synthetic inputs and no model:

1. Call the actual `child_main` with a successful synthetic backend whose `save_artifacts` raises `OSError`. It raises the error but leaves `pilot_result_v1.json` with `status: COMPLETE`.
2. Launch a synthetic worker through the actual `execute` supervisor. The worker writes a COMPLETE result and exits 7 without an adapter. The parent retains COMPLETE. Feeding that outcome through the actual `main` gives exit code 0.

Observed output: worker exit 7, persisted COMPLETE, missing adapter, CLI exit 0. This contradicts the checklist's unconditional claim that a killed or crashed worker yields INCOMPLETE. The issue can also affect a kill during artifact saving; partial/invalid JSON is not validated by the parent either (the latter cases are inspection findings, not separate executed reproductions).

Reproduction script: `.local/review_completion_20260930.py`. It creates uniquely named synthetic directories under `.local`; it does not edit real experiment artifacts or bypass their permission records. Run from the repository root with `PYTHONPATH=packages/core;routebench` and the existing test interpreter.

Finite correction required:

- Stage a provisional result; save and validate the artifacts required for the terminal status before publishing final success atomically.
- Parent must reconcile worker exit/killed state, valid result JSON, run/config/input identity and required artifacts. A nonzero worker exit or failed/missing artifact cannot leave a successful terminal record or successful command exit.
- Keep provisional diagnostics separately; do not destroy or silently overwrite previous runs. COMPLETE with a negative research decision remains a valid completed experiment; INCOMPLETE, NOT_FEASIBLE and failure need explicitly documented command-exit semantics.
- Add regression tests for adapter-save exception, nonzero exit after result creation, kill during finalization, malformed/truncated result and a clean successful run.

## Reproducibility addition

`run_pilot` currently persists aggregate baseline/tuned metrics but not the per-example selection and validation scores used to compute them. Preserve IDs, split assignment, scores, thresholds and relevant hashes in a separate prediction artifact, without raw source text. Verify that the final report can be recomputed from it offline. This does not require a new model or experimental sweep.

## Independently executed checks

- `pytest packages/core/tests routebench/tests -q --basetemp <unique .local directory>`: **688 passed**, 5 warnings, 142.33 seconds. This is not the entire three-project repository suite.
- Ruff on `routebench/evalops/pilot` and `routebench/tests/test_pilot_*.py`: passed.
- `python -m evalops.run_calibration verify`: passed, 1,232 corpus cases, split and recorded configuration identities verified. Four recorded variants verified; this is not a fresh inference or universal verification of every later experiment.
- Both synthetic finalization reproductions above: confirmed.

Passing existing tests does not negate the additional uncovered failure path.

## What remains before and after the pilot

Before execution: fix finalization, preserve recomputable predictions, record the user's actual source-use decision, and obtain approval for the isolated dependency installation and single GPU pilot. Do not fill the acceptance line on the user's behalf. Hardware feasibility is intentionally unknown until the bounded probe; NOT_FEASIBLE is a valid stop.

The current pilot trains on 1,500 selected sentences and evaluates 1,446 length-eligible sentences from 45 RAGTruth validation document groups. Its primary continuation criterion is balanced-accuracy improvement >= 0.03 with the predeclared uncertainty and unsupported-recall controls. It is **not** the original RouteBench kappa acceptance test. Passing it cannot establish 0.74 or transfer from news articles to meeting/interview material.

After a passing pilot: first confirm the saved adapter reloads and reproduces its recorded scores within a declared numerical tolerance. Then propose one frozen, separately approved RouteBench development comparison with identical labels, denominator, coverage and declared calibration rules. Existing dev is repeatedly inspected, so any improvement there is exploratory. If sufficiently promising, freeze the full system and request a distinct held-out evaluation; preserve unused reserve/test data until that decision. A failed pilot is a stop, not permission for an automatic 40-hour extension.

Existing report context, not re-measured here: highest single-judge dev kappa 0.6771; frozen v18's separate 320-row population-weighted kappa 0.404. Different populations, not interchangeable acceptance numbers. Kappa >= 0.74 remains unmet and no runtime or prompt guarantees it.

## Fresh-session prompt

See `ROUTEBENCH_FRESH_SESSION_PROMPT.md`. It uses an independent first-pass diagnosis followed by a mandatory history check. This avoids early anchoring without hiding failed experiments, source restrictions or exhausted validation data. Use a new session, but not a second concurrent writer on this worktree.
