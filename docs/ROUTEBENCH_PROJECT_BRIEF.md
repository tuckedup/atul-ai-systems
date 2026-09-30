# RouteBench: project and current bottleneck

Status: 2026-09-29. The groundedness judge target, binary Cohen's kappa >= 0.74, is NOT achieved.

## In plain English

RouteBench is an evaluation-driven gateway for choosing language models based on quality, cost and latency.
Its evaluation layer checks whether an AI answer is actually supported by the supplied source text, then
compares that judgment with published human annotations. Cohen's kappa measures agreement beyond chance;
0.74 is an agreement target, not 74% accuracy.

It belongs to a shared AI-systems repository alongside ForgeCode (a bounded coding-agent workflow) and
AI Incident Commander (an audited incident-response workflow). Shared infrastructure provides model calls,
tracing, budget controls and evaluation tools. Claude and Codex have implemented and audited the work;
automated implementation does not establish model quality on its own.

## What is working and what is blocked

The repo includes routing/evaluation code, cached and cost-capped judging, provenance checks, incomplete-run
guards, document-grouped comparisons and experiment reports. These are engineering deliverables, not proof
that the judge meets the quality gate. Docker-only verification remains blocked under the existing amendment.

The current bottleneck is judge generalization: agreement with human labels varies substantially across
documents and annotation policies. Many sentences share a source document, so hundreds of cases provide
fewer independent observations than their count suggests. Repeated dev-set inspection also makes selected
dev scores optimistic. More prompt changes and combining correlated judges have not reliably solved this.

| Evidence | Result | Interpretation |
| --- | ---: | --- |
| Highest single-judge dev result, v21 o4-mini high | kappa 0.6771 | Selected on repeatedly inspected dev; below target |
| Selected comparison baseline, v18 o4-mini medium | dev kappa 0.6745 | Not a certified judge |
| Frozen v18 on separate upstream documents, 320 rows / 40 docs | weighted kappa 0.404, 95% CI [0.282, 0.525] | Generalization concern; separate population, not the original test |
| Frozen train-fitted MiniCheck + v18 combiner | dev kappa 0.593 | Worse point estimate than v18 |
| Latest HHEM screening | dev 0.586; separate-set weighted 0.437 | No convincing paired improvement; training overlap unknown |

The original groundedness test split has not received judge inference. Its aggregate human-label prevalence
was inspected, so saying it was entirely unseen would be inaccurate. The separate 320-row set has now been
inspected and must not be presented as untouched validation for a newly developed method. The separate,
earlier attribution experiment was compromised and is not evidence that the groundedness target passed.

## Publication and reproducibility

This branch publishes source code, regression tests, aggregate experiment records and reports. It is not a
production-release certification and does not merge into main. Frozen protocol scripts are preserved as run.

The HHEM scoring script trusts code in a local model directory. The historical run's manual revision/code
review is documented, but the script does not itself verify that directory's pinned identity or enforce the
one-hour wall-clock cap. Do not run it on an unreviewed download; any future execution requires a separate
review of those controls. Preserving this script is not authorization to rerun it.

The transcript fetcher uses upstream `main` URLs without pinned download checksums, a timeout or a response-size
cap. It is a historical best-effort retrieval helper, not a byte-reproducible or hardened downloader. A future
retrieval needs those controls and a fresh usage review; do not assume it recreates the measured input bytes.

New raw datasets/transcripts, quote-bearing judgment-cache additions, source-containing label-review JSON,
summary-context artifacts, model weights, expert few-shot packets and fitted combiner parameters remain local.
Some dataset uses have unresolved restrictions; this publication does not clear them. Existing tracked
historical data are not removed or retrospectively relicensed by this commit.

Consequently, a fresh checkout cannot reproduce every empirical score from the new aggregate records alone.
Historical replay needs the corresponding local caches and permitted source artifacts. Do not launch paid
inference to replace absent caches without explicit approval. `calibrate-verify` checks config identity, not
that a checkout contains all judgments or that a quality target passed.

The default `dev` command selects the whole variant grid, including configs whose local-only exemplar
packets are omitted. It will fail preflight in a fresh checkout. Use explicit artifact-free variant IDs with
`--variants` for a separately authorized experiment; do not expect the entire grid to run out of the box.
For a read-only inventory, `python -m evalops.run_calibration plan --select-on headline --variants
v18-reasoning-o4mini` makes no provider calls. Missing cache rows in that inventory are expected.

The archived legacy experiment record retains its original Windows exemplar path because the historical
hash formula included the literal path. It is regression evidence, not a portable runnable configuration.
Historical plans and reports retain superseded proposals; the current brief and final reports take precedence
for project status. In particular, shared model errors are not proof of bad labels or an irreducible ceiling.

The real-data integration tests for the diverse packet and restored summary context explicitly skip if their
local-only artifacts are absent. Synthetic regression tests remain runnable. Missing files are not a reason
to silently fabricate examples or alter labels. Audit JSON referenced by historical reports may also be local-only.

## Next decision, not work authorized by this publication

Either ship a clearly scoped human-reviewed evaluation workflow, without claiming the full-coverage 0.74
gate, or approve a separate research effort using training-permitted, overlap-audited data and independent
evaluation. Neither path guarantees kappa 0.74. No further experiments or spending are authorized here.

Details: `ROUTEBENCH_FINAL_PHASE_REPORT.md`, `ROUTEBENCH_HHEM_SCREEN_PROTOCOL.md`, and `BLOCKERS.md`.

## Publication checks

RouteBench plus shared-core tests: 582 passed with local research artifacts present; the staged-only snapshot
passed 577 tests and explicitly skipped 5 local-artifact integration tests. Ruff passed for the RouteBench
modules/tests and shared model client. Staged diff whitespace checks and credential-pattern scan passed.
These checks did not run Docker, new model inference, downloads or paid judging. They are not verification
of every other system in the monorepo or a comprehensive security/licensing audit.
