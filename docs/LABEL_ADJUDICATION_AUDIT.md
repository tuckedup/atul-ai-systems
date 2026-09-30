# Flagged-label audit: findings and next actions

Date: 2026-09-29. **AI-assisted audit, not independent human adjudication.**

> Research follow-up: `LABEL_REVIEW_RESEARCHED.json` and
> `LABEL_REVIEW_RESEARCH_NOTES.md` contain the completed, revised AI review.
> FRANK's published protocol explicitly penalizes missing in-summary antecedents;
> the Dorset inn finding below must not be read as proof of a wrong human label.
> TofuEval's initial model study found no gain from previous-sentence context, so
> this audit's context diagnosis is not evidence of an aggregate kappa improvement.
> Earlier findings below are retained for audit history, not replacement labels.

## Outcome

The 14 cases do **not** establish 14 wrong human labels or an irreducible ceiling.
The most actionable finding is a mismatch between the material the original annotators
judged and the sentence-only material supplied to RouteBench judges. The original
human records also reveal disagreements that the imported binary labels hide.

No labels, cases, splits, prompts, caches, measured results or bundles were changed.
No paid judge calls and no held-out inference were made. Only audit/review documents
were created. No new kappa was computed. This does not achieve kappa >= 0.74.

## Verified provenance

- The pasted attachment is structurally identical to the local 14-case queue.
- Every queued candidate/document exactly matches its stored case; all 14 are in
  RouteBench **dev**. Every queue label matches its stored annotation and raw mirror.
- All 14 indexed raw joins are unique. No source was cropped in these 14; the largest
  is 5,013 characters. The earlier 96 cropped sources were in the separate attribution
  experiment, not these cases.
- The nine TofuEval sentence joins to local original-format annotation files are unique
  after normalizing CRLF/LF. Those CSVs lack source document text, so uniqueness plus
  exact sentence matching is evidence, not a durable upstream key. Preserve upstream
  document/annotation/model/topic/sentence identifiers in future data.
- The five CNN cases are FRANK/BART records. Local AggreFact final/sota copies agree
  on document, summary and label. I additionally recovered original FRANK per-rater
  sentence annotations and matched all five **article and summary strings exactly**.
- Raw `source_row_index` is zero-based mirror enumeration, not an original dataset ID.
  The structured audit records original Tofu file/record/physical-line locators.
- Original source files named “test” refer to their upstream dataset splits; these
  audited items belong to the existing RouteBench dev split. No RouteBench test was run.

Protected input hashes and 37 checked source-quotation spans are in
`LABEL_ADJUDICATION_AUDIT.json`.

## Case-by-case findings

Categories describe this audit's reasoning, not replacement benchmark labels.

| Case ID | Subject | Finding | Evidence and action |
| --- | --- | --- | --- |
| `gnd-c956780fd0b574bb` | SB 145 / Proposition 6 | Transcript ambiguity | Repeated “SB one” conflicts with a garbled “SB 145 19”; the original requested topic itself says SB 145. Needs transcript/topic adjudication. |
| `gnd-54a1abd246af1ba1` | Haitian migrants processed | Possible positive-label inconsistency | Source says awaiting processing, not completed processing; hearing language is qualified. Positive label needs review. |
| `gnd-72c8a872b5c969ea` | $8m annually | Time-scope ambiguity | Source specifies FY19 and projects $11m for FY20. Decide whether the summary expresses a current annual baseline or a fixed future amount. |
| `gnd-65b135f17ee0e207` | Election certification vote | Missing annotation context | Literal vote is supported; original summary implies it answers the election-postponement proposal. The original topic and preceding sentences were omitted. |
| `gnd-6d0a6a0bc1619eb9` | Dorset inn | Original rater disagreement | Source supports the facts; FRANK raters flag CorefE on sentences 1/3. One rater finds no errors anywhere. |
| `gnd-4ecf23068a339688` | Ueda / selfie stick | Original rater disagreement | Source supports the claims; sentence 2 has NoE / CorefE / RelE. Not a unanimous human verdict. |
| `gnd-53ef38e16bd512a5` | Cubans welcomed | Missed semantic distinction | Original expert distinguishes positive welcome from permission to remain. Models missed a plausible strengthening. |
| `gnd-71ce0e9a4754c173` | Opinion question | Task-boundary mismatch | Question follows invented poll results in the original summary. Define question handling and restore context; factual-assertion-only grading is insufficient. |
| `gnd-a13c7ba2cd8cc1cd` | Bush/Clinton compared | Missed tense distinction | Original expert contrasts a coming segment with a comparison described as already completed. |
| `gnd-70012ed4168ec7b5` | Police-supported cost | Missed relation distinction | Police support the program, not explicitly its cost; the original expert flags that grammatical attachment. |
| `gnd-406e875c9ec93511` | This compromised security | Missing annotation context | “This” originally refers to the preceding Giuliani sentence; the source attributes the effect to presidential pressure on Ukraine. The judge never saw that antecedent. |
| `gnd-468f88d2d69779b2` | Tevez transfer | Original rater disagreement | Source supports the facts; FRANK tags sentence 2 NoE / OutE / LinkE, and one rater tags CircE on sentence 4. |
| `gnd-f9b9bed541b87d55` | Paid volunteering leave | Original rater disagreement | Raters disagree with CircE/CorefE tags. Omission of the volunteering condition deserves review; no automatic label correction. |
| `gnd-b23a3beb6bfa2187` | Adeli arrest | Original rater disagreement | Source supports the reported details; FRANK includes OtherE on sentence 2 and RelE/OtherE on sentence 3. |

Counts: **2 missing-context cases, 5 upstream rater-disagreement cases, 3 plausible
missed annotation distinctions, 1 question/task boundary, 2 transcript/time-scope
ambiguities, and 1 suspected positive-label inconsistency.** These are selected
disagreements, not an estimate of the dataset's error rate. Context can matter in
additional cases too; categories are a practical primary classification.

### The clearest representation failure

For `gnd-406e875c9ec93511`, the original summary contains:

1. A sentence attributing a shadow foreign policy to Giuliani.
2. “Giuliani pushed out the ambassador and took control of Ukraine policy with others.”
3. “This compromised national security and elections.”

The expert marked sentence 3 as mis-referencing: the source attributes that effect
to pressure on Ukraine to investigate political rivals, not the preceding ambassador
claim. RouteBench supplied only sentence 3, making the critical antecedent unavailable.
Simply buying a stronger judge cannot reconstruct omitted candidate context reliably.

For `gnd-65b135f17ee0e207`, the expert explains that the certification vote did not
concern the postponement proposal discussed immediately before it in the summary.
The isolated certification sentence is supported, but it is not the same annotation task.

This loss begins in the reduced mirror representation. `sources/fetch_aggrefact.py`
persists only source/subset/doc/claim/label/index; `build_groundedness` then uses
`candidate_output=claim` with a generic task, and `judge.build_prompt` supplies that
single candidate. The raw source text is present; the missing material is **candidate
summary context and annotation topic**, not more source evidence.

### What the recovered FRANK votes actually say

All five cases have one rater marking every sentence `NoE`, while two other raters
flag at least one sentence. Error types can differ. This is direct evidence of
disagreement, not proof that any particular rater or the aggregate label is wrong.

`FRANK_FLAGGED_ANNOTATIONS.json` preserves the five matched records and tags, pinned
to repository commit `80a88fb12cc0bfc17ff6ee2c5ebb0c4f4dd8a5f4`.
The original binary-label aggregation rule was not independently reconstructed in
this audit; do not substitute a new majority-vote or averaging rule silently.

[Original FRANK documentation](https://github.com/artidoro/frank#data) describes
sentence-level per-rater annotations. [Pinned annotation source](https://raw.githubusercontent.com/artidoro/frank/80a88fb12cc0bfc17ff6ee2c5ebb0c4f4dd8a5f4/data/human_annotations_sentence.json)
is the source of the recovered metadata. Existing article/candidate quotations in
this report come from the user's supplied cases.

## Small non-flagged control check

Selection was fixed before inspecting control labels or votes: sort non-flagged dev
cases by SHA256 of `label-audit-20260929-control:` + case ID, exclude every flagged
document group, then take two distinct groups per subset. Only four were available
(two CNN and two MediaS); there were no remaining eligible MeetB document groups.
The initial source-only assessment was “supported” for all four; subsequently
revealed stored labels were all positive and raw sources were complete.

- `gnd-f4d746560fbdda88`: McCarthy report.
- `gnd-10eb9e350b01e1ac`: Nauru guards report.
- `gnd-0e65033ff59a12c2`: plane approaching a carrier.
- `gnd-53fad466179b5713`: spokesperson statement.

This tiny, group-excluded check is not a population estimate or human-human agreement
study. The parent is an AI and has already seen the flagged labels/model consensus;
the flagged review is not blind.

## Action plan: no new paid grid yet

1. **Fix annotation-input fidelity in a separately versioned experiment.** Recover
   original Tofu document/annotation/model/topic/target-sentence identity and the full
   generated summary. Give the judge that summary as untrusted candidate context,
   clearly mark the target sentence, and grade only the target. Neighboring candidate
   sentences resolve reference; they are NOT admissible evidence for factual support.
   Bind new fields into case fingerprints/cache identities. Do not regenerate or
   silently rewrite the existing benchmark.
2. **Add offline regression examples for the mechanism.** Test wrong versus correct
   antecedents, a vote on another proposal, tense, “welcomed” versus allowed to stay,
   and program support versus funding. Use train-only or synthetic unit-test fixtures;
   do not turn these selected dev labels/explanations into few-shot answer keys.
3. **Resolve the annotation policy separately from copying labels.** Document question
   handling, time scope, noisy transcripts, attribution and material omissions.
   Original expert memos explain the disagreement; they are not unquestionable truth.
4. **Human review is still required for final adjudication.** The separate
   `LABEL_REVIEW_FORM.json` contains all 14 targets plus four controls, with sources
   and recovered Tofu candidate context but no labels, model votes, error tags or AI
   assessments. Ask a reviewer to use it before the audit, record a reason and source
   evidence, and mark ambiguity rather than guess. If possible, use two reviewers and
   reconcile disagreements. This AI report does not count as a human vote.
5. Only after the offline representation checks and a declared protocol should a
   bounded, separately authorized dev run be considered. Preserve original results;
   report a versioned task/input correction as such. A test run, arbitrary relabelling,
   removing these 14, or changing the kappa definition does not close the original goal.

## Limitations and cost

No broad CNN label-error rate, inter-annotator kappa, quality ceiling or guaranteed
path to 0.74 is inferred from this selection. Original FRANK error tags were recovered,
but free-text rationales and a full reconstruction of downstream label aggregation
remain unavailable here. Tofu snapshots were already present locally; this turn did
not independently re-download their original repository revision.

No provider-judge spend in this audit. Native parent/subagent token telemetry is
unavailable, so API-equivalent agent cost and savings cannot be estimated.
