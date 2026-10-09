# RouteBench: completed Astra adjudication and implementation handoff

Date: 2026-10-01 (America/New_York).

## Outcome

All three AI reviews are complete. The separate Astra adjudicator read all 60
full sources and both frozen first-pass reviews, then saved 28 supported,
28 unsupported and 4 uncertain decisions. It did not receive reference labels or
historical model verdicts. Sol and Luna agreed on 47 cases and disagreed on 13.
The adjudicator reviewed agreements too; the outcome is not a majority vote.

This is AI-only diagnostic work, not human annotation or a measured judge improvement.
The sample deliberately contains all 42 baseline errors from the already inspected
240-case upstream evaluation plus 18 correct controls. Do not compute or advertise
a new population kappa from these 60 cases. No original labels, cases, predictions,
splits or acceptance-test artifacts were changed. The 0.74 target remains unmet.

## What the evidence supports

The useful distinction is faithful meaning versus unsupported attributes and
relationships, not simply strict versus lenient judging. Parent spot-checks of the
complete sources and Astra rationales for C002, C029, C032 and C019 support the
following diagnostic examples:

- C002: the source's spoken song title uses You; the candidate uses U and different
  punctuation. Astra accepts this faithful orthographic normalization even though
  both first-pass reviewers rejected it. This does not require outside knowledge
  of the official title.
- C029: a letter sent to ensure answers describes a purpose, not proof that every
  answer has already been delivered. Astra accepts the contextual paraphrase while
  both first-pass reviewers had rejected it.
- C032: the source gives one initiative a gun-safety subject but only names another.
  Calling both gun-control measures adds an unsupported attribute. Two authentic
  initiative-number quotes do not establish that shared classification.
- C019: the source gives some funding details but no itemized allocation. What
  counts as "specific details" is unresolved; Astra retains uncertainty.

The four uncertain cases are C015, C019, C025 and C047. No case is dropped. These are
fallible AI interpretations under the stated policy, not authoritative corrections
to published human labels. Shared model errors remain possible.

## One next intervention

Implement ONE opt-in source-linked claim-relation audit, covering every assertion
before the final verdict. Capture actor, time, modality, numeric relationships,
source spans and causal/purpose/evidential links; track uncovered candidate spans.
Distinguish harmless paraphrase from added facts. Apply the audit to both would-be
passes and rejections; the previous pass-only auditor could not repair false rejections.

Implement offline with synthetic minimal-pair regression fixtures first, preserving
v18's identity. Plan one subsequent frozen comparison; do not run it implicitly.
This is a hypothesis supported by examples, not proof it will improve agreement.
Do not relabel this panel, train on it, or choose thresholds to make its result pass.

The complete work order is `docs/ROUTEBENCH_CLAUDE_EXECUTION_HANDOFF.md`. It also
covers a small permission/provenance-aware data workflow, an end-to-end mocked CLI,
coverage/error reporting, and one separately approved experiment proposal. No new
paid run, training, download, vendor contact, commit or push is authorized here.

## Artifacts and checks

Private evidence is in `.local/ai-panel-20261001-v1/`:

- `astra_review.md`: complete source-backed synthesis and initial technical findings.
- `astra.jsonl`: 60 final AI adjudications, SHA256
  `525e20f56326200575373e6dcf41456f38865a96765cf3c3a2e6d235430cc428`.
- `sol.jsonl`: SHA256
  `beb9c41dec74283e52e97d236dc2e3e20d408981212c8bd917da480f4c72f8e3`.
- `luna.jsonl`: SHA256
  `249c4d098b13f7c174cedb652546d291918279084be1a4fb8ba7eaf760e661fb`.
- `blind/`, `manifest.json`, `parent_only_mapping.json`: frozen diagnostic inputs.

Both first-pass outputs and the Astra output were rehashed unchanged after the
technical fixes. The original fresh-evaluation freeze still verifies all 240 cases.
The parent reran 31 targeted tests (20 panel tests and 11 completion-reporting tests)
and Ruff successfully. This is not a claim of a newly rerun full repository suite.

The initial Astra review identified a frozen-population validation gap. The parent
repaired actual filenames/count/internal IDs and mapping identity with five tests.
A fresh Sol technical review found a second-read gap; the parent changed comparison
to consume the exact bytes/records already validated and added a mutation regression.
The protocol now explicitly discloses that substring validation does not establish
complete claim coverage and that the legacy validator does not enforce issue_type.
Frozen reviewer rows were not retroactively rewritten to a different schema.

Fresh read-only acceptance review `/root/panel_acceptance_review` returned **ship**,
with no blocking findings in the stated structural scope. The parent then ran the
one-time comparison. All three frozen output hashes matched; original-input and
mapping hashes passed. The comparison is saved separately as `comparison.json`.

| Selected group | N | Astra agrees with reference | Disagrees | Uncertain |
| --- | ---: | ---: | ---: | ---: |
| Previously known baseline errors | 42 | 15 | 23 | 4 |
| Sampled baseline-correct controls | 18 | 16 | 2 | 0 |

These counts were joined only after adjudication was final. No judgment or
intervention was revised in response. They show that this panel is NOT a reliable
replacement answer key: reference disagreement can reflect panel error, policy
interpretation, missing context, or a reference problem, and this exercise does not
settle which. Its 15 reference-aligned decisions on baseline-error cases are also
not measured improvements to the deployed judge. We have not implemented or tested
the proposed candidate on a representative, independently judged population.

Do not train a candidate merely to imitate this panel, revise benchmark labels to
match it, or claim that majority AI agreement establishes a human-label defect.

## Provenance and cost

Requested reviewer settings were GPT-6 Sol/high, GPT-6 Luna/high and GPT-6 Astra/high.
Native child runtime settings are not independently exposed; requested settings
are not runtime proof. The root session was observed as GPT-6 Astra/high in current
turn-context metadata at 2026-10-02T01:28:07.899Z. Blinding is instruction-level on
a shared filesystem, not a security boundary. Root has prior benchmark exposure
and did not substitute itself for the separate blinded adjudicator.

The Astra orchestration skill required independent review and repair before
acceptance; this caught two integrity defects in the diagnostic helper. No
provider-key calls or external purchases were made. Native agents consume account
allowance. API-EQUIVALENT COST RECEIPT: unavailable because complete observed token
usage for parent and all delegates was not exposed. Routed cost, Astra repricing
and savings are unknown, not zero; no subscription-charge comparison is claimed.
