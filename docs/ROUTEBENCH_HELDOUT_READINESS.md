# RouteBench: integrated held-out readiness policy

Date: 2026-09-30. This document supersedes the conclusions and integration instructions in the supplied cloud patch. The supplied attachment remains unchanged. No new judge result was produced by this implementation.

## Decision

**Kappa >= 0.74 is unmet, not proved impossible.** The readiness command is an offline evidence/sufficiency check, not a model improvement. It does not replace the separate, bounded GPU pilot or authorize it. No new data, model calls, GPU work, dependency installation, label changes, commits or pushes are part of this integration.

The latest local pilot finalization work already saves predictions and the adapter before publishing a result, then records an independent parent terminal verdict. Consumers must use `pilot_terminal_v1.json` and the command's exit code, not just the worker result: a worker that publishes COMPLETE and later crashes leaves a forensic COMPLETE worker record but a FAILED parent verdict. The parent verdict is authoritative. Hardware behavior and reloading a real trained adapter remain unverified.

## Acceptance remains unchanged

- The release target is binary, unweighted Cohen's kappa >= 0.74 on the original human-reference headline test population, not a weighted score from a different dataset.
- The code uses a point-estimate gate and reports uncertainty separately; this change does not silently replace it with a lower-confidence-bound criterion.
- The existing provenance-based population contains 263 items across 50 documents. Labels are published reference annotations, not 100 newly hand-labeled local items.
- Full required coverage and an immutable frozen judge are still necessary. Unresolved headline label conflicts now cause `test` to refuse before inference instead of silently shrinking its denominator.
- Existing dev and previously inspected comparison sets are not fresh independent confirmation. No acceptance labels or split memberships were edited.

## What was integrated and corrected

1. Added `evalops.readiness`, `run_calibration readiness`, and `make calibrate-readiness` from the supplied patch, then corrected their interpretation and trust boundary.
2. New freezes bind the selected track, ordered eligible candidate configurations, variant override and selection-rule version into the bundle identity. Existing bundle IDs with no selection record retain their historical identity, but missing/non-headline lineage cannot automatically clear this new gate.
3. Readiness replays only the bound candidates. Missing candidate predictions/identity changes make the diagnostic incomplete, not silently easier. This is a replay of the recorded selection procedure, not removal of all prior adaptive development bias.
4. `test` re-computes a nominally READY report from current cached development evidence before inference. Editing the verdict, estimates and document counts and rehashing the JSON does not authorize test. A content hash by itself is not an attestation. This cannot protect against an actor who changes code or the underlying evidence.
5. Readiness uses the release-policy floor even if a bundle requested a lower minimum. NOT_READY can be overridden only by the explicit recorded command option described below; the override cannot waive bundle identities, full coverage, label-conflict checks or the release metric.
6. CLI and Makefile freeze now default explicitly to `headline`. Pooled/oracle experiments remain explicit alternatives but cannot masquerade as a replay of headline selection.
7. Annotation export defaults to dev. CLI test review requires `--allow-test-review`; interactive review defaults to `.local/label-review/annotations.jsonl`. Queue export and label entry reject the canonical acceptance-label path. `annotate status --annotations PATH` reports the separate review file without merging it into acceptance data.

## Statistical interpretation: important corrections to the cloud write-up

- The bootstrap pass fraction is a **conditional empirical simulation**, not a calibrated probability of the real test passing. In particular, the supplied 1.8% figure is not independently established here as the true chance of success.
- Reweighting confusion cells is appropriate for a declared mixture under transport assumptions. It assumes development prevalence and error behavior transfer within each subset; document mix, domain difficulty and reference policy may differ.
- Cross-validation replays candidate/threshold selection on fixed cached predictions. It does not undo prior prompt design, variant invention or repeated inspection of dev. It assesses the selection procedure, not necessarily the frozen winner alone.
- Omitted previously tried variants do not make the resulting estimate a mathematically proven upper bound. New bound candidate omissions now prevent automatic readiness.
- The five-document sufficiency cutoff and the bootstrap fraction cutoff are conservative project policies, not statistical laws. With only three MeetB development documents and a large target weight, the current data cannot satisfy the five-document policy even if a candidate's point estimate improves. Do not quietly lower that cutoff to get READY. Consider additional independently collected development evidence only if permitted, or a separately authorized diagnostic test with the limitation recorded.
- Published balanced-accuracy rankings alone cannot determine pooled kappa on this benchmark: class prevalence and the two types of errors matter. The identity `kappa = 2 * balanced_accuracy - 1` needs additional marginal conditions; it is not a universal conversion or an impossibility proof.
- Human-human or new-human-versus-published disagreement does not establish an absolute ceiling for model agreement against a particular adjudicated reference. A label-agreement study should report uncertainty, each rater and adjudicated comparisons, and remain separate from the acceptance data. There is no automatic retire/fund decision at human point estimates 0.74 or 0.85.
- The cloud attachment's external literature claims were not re-verified in this local integration and are not adopted as evidence that the objective is impossible.

Machine-readable legacy keys containing `honest` and `p_pass` remain for compatibility. The report's `interpretation` field and command output explicitly qualify them. Protocol is `heldout-readiness-v2`; older reports must be recomputed, not hand-edited.

## Offline commands (from this worktree)

```powershell
$env:PYTHONPATH = 'packages/core;routebench'
& ../wt-forge/.venv/Scripts/python.exe -m evalops.run_calibration verify
# Only when intentionally preparing a new frozen candidate; does not call a provider:
& ../wt-forge/.venv/Scripts/python.exe -m evalops.run_calibration freeze --select-on headline
& ../wt-forge/.venv/Scripts/python.exe -m evalops.run_calibration readiness
```

`freeze` and `readiness` write artifacts. They were not run against the real calibration bundle during this integration. Synthetic tests exercise the full paths.

The `test --override-readiness "<reason of at least 20 characters>"` option records the reason, OS user, host, report ID and failed policy conditions. An OS username is not proof of human authorship or authorization. An agent must not invoke this option without the user's explicit test-run authorization and budget. No override was used here.

## Next step

Finish offline verification, then decide the private-training source-use question and whether to approve the isolated dependency install plus **one** bounded pilot. A run may stop NOT_FEASIBLE, INCOMPLETE or FAILED, or finish with no gain. A pilot pass measures improvement on its filtered RAGTruth validation population; it does not establish RouteBench kappa 0.74. Only a separately approved frozen RouteBench evaluation can measure the original target. Do not purchase a 40-hour run or consume held-out data merely to force a favorable result.

Optional: omit the human label-agreement study if it is not feasible, but keep the uncertainty explicit. It is not a prerequisite for fixing software and must not be impersonated by an AI. Do not omit the metric, coverage, provenance, held-out separation or failure-status checks.

## Local verification of this integration

- Shared-core plus RouteBench suite: **780 passed**, five warnings, 192.49 seconds, using the existing project interpreter and a unique `.local` pytest directory. Includes the 12 pilot-finalization tests; does not certify actual GPU behavior or the other two projects.
- Ruff: clean on changed Python files and the pilot implementation/finalization tests.
- Strict mypy: clean on `readiness.py`, `run_calibration.py` and `calibrate.py`. Including `annotate.py` reports its pre-existing unused `type: ignore` at the Windows `msvcrt` import (line 60), unchanged in this patch.
- `evalops.run_calibration verify`: passed; 1,232 corpus cases, split/dataset identity checks and the four recorded variants unchanged. This does not rerun models or establish a new kappa score.
- New regressions cover a fully self-consistent forged readiness JSON, incomplete/incorrect/malformed selection lineage, historical bundle identity compatibility, an explicit variant override, the headline freeze default, protected annotation output, and unresolved test-label conflicts even under a readiness override.
- No real freeze/readiness output, acceptance-label file, split manifest or model artifact was regenerated by this integration. Code changes are local and uncommitted; existing unrelated work is preserved.
- Fresh read-only review of the scoped integration: **ship**, no substantive findings. It relied on the parent's 780-test verification rather than repeating the suite. Residual risks: mutable source evidence/code are outside the report's trust boundary; override authorization is procedural; statistical transport assumptions and actual GPU/adapter behavior remain unverified.
