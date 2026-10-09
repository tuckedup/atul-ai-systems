# Annotation policy addendum: DRAFT wording for the owner (2026-10-07)

**Status: DRAFT `policy-addendum-v1-DRAFT`. Not adopted, not versioned for use, and every owner-acceptance field below is deliberately BLANK.** Nothing here has been sent to reviewers, and no human review has taken place. This is the "Track A" output of the 2026-10-07 handoff: it only proposes wording, with rationale, for the questions Q1-Q8 in `ROUTEBENCH_HUMAN_REVIEW_POLICY_PROPOSAL_20261006.md`. I have not verified that the benchmark's published annotation guidelines settle any of them; checking that is part of each decision. The classifier line stays CLOSED.

## 0. Provenance and status facts (re-verified from the saved files)
- Reviewer-A, Reviewer-B, the 5 adjudications and the 88 consolidated records in `.local/judge-v19-phase1/blind-review-v1/review-session-v1/` are **all `ai_assisted`** (176 initial + 5 adjudication + 88 consolidated; validated: 0 format problems, `reviewer_kind` = `ai_assisted` throughout). They are an AI-assisted diagnostic, **not** 176 human reviews and not independent human annotation. I have not renamed, relabelled or counted them as human, and I have not filled in any human response.
- **Zero independent human reviews exist.** The reviewer-handoff packets (`reviewer-handoff/Reviewer-{A,B}-packet.zip`) are prepared but there are no returned human files.
- Original benchmark labels, cases and splits are unchanged. Nothing in this addendum may alter them.

## 1. Two policies, kept distinct
| | Original benchmark policy | Newly scoped review policy (this addendum, if adopted) |
|---|---|---|
| Source | The published TofuEval / AggreFact annotation guidelines and expert annotations that produced the stored labels | A plain-language written policy for the 88-item review, adopted only by the owner |
| Used for | The benchmark's labels and every reported kappa against them | Describing why residual disagreements occur; never to relabel, filter or re-score the benchmark |
| Outcomes | binary (supported = 1 / unsupported = 0) | `supported`, `unsupported`, `uncertain` (and `not_assessable` only if Q2 option A is adopted), **kept separate: nothing is mapped silently to a binary label, and `uncertain` is not rounded to either side** |
| Change control | frozen | versioned before any annotation; a later change is a new version and invalidates earlier responses |
A reviewer outcome that differs from a stored label is a **disagreement between two policies or readings**, not evidence the stored label is wrong.

## 2. Draft wording and rationale (owner decides; blank fields below)
| Q | Draft rule (proposed wording) | Rationale and risk |
|---|---|---|
| Q1 plurals | "A plural noun ('boycotts', 'protests') is supported only if the document describes more than one instance. A single documented instance supports the singular only. If the document describes a recurring practice without counting it, mark `uncertain` and say so." | Keeps the check literal and mechanical; risk: stricter than a generic reading and may mark more `unsupported` than the original annotators did |
| Q2 non-declarative sentences | "A question or command addressed to the reader makes no factual assertion. Record `not_assessable` and do not force it into supported/unsupported." | Avoids inventing a verdict; adds a fourth outcome the original binary labels cannot represent, so these items cannot be scored against them |
| Q3 inference | "Support means a careful reader accepts the claim from the document alone, without outside knowledge or an unstated premise. Any `supported` verdict that relies on inference must name the premise from the document." | Stated as direct entailment (option A); risk: rejects ordinary inferences that the original experts may have accepted |
| Q4/Q5 metadata | "Speaker role and article byline are not inferred. If the original source has them, the same labelled addendum is shown to every reviewer for every item that has it; otherwise the item stays `uncertain` on that point." | Prevents silently supplying facts to some items only |
| Q6 opinion as fact | "If the document presents a claim as one speaker's opinion, analysis, evaluation or prediction, the summary must keep that attribution; restating it as plain fact is `unsupported`. A plainly stated fact may be restated plainly, whoever the speaker is." | Matches the largest cluster in the expert-explanation audit (one document, 8 items); evidence base is narrow |
| Q7 document-genre claims | "A claim about what the document is ('this is a report') is judged against what the text actually is, not against the topic it discusses." | **Pulls opposite to the earlier fresh-document audit**; do not adopt without checking the published guidelines |
| Q8 hedges and strength | "A dropped hedge, changed tense, or stronger verb ('encouraged' to 'organized') makes the sentence `unsupported`." | Already required by the existing rubric and by the reviewers' instructions; risk: tolerance for 'minor' strengthening is the contested part |

## 3. Process requirements (unchanged and binding)
- Version the addendum (text hash) **before** any annotation; responses record the policy hash; changing the text starts a new version.
- At least **two independent human reviewers per item**, then a later adjudication of every disagreement and every `uncertain`; reviewers receive only the permitted packet, never the reconciliation file, model verdicts, selection groups or this analysis.
- Analysis is **descriptive only**: counts by group, reviewer agreement, `uncertain` share. No 50% / 75% trigger, no "kappa if relabelled", no label ceiling, no spending decision follows from it.
- In-house reviewers avoid a vendor rights question for the CC-BY-NC-SA transcripts; a paid vendor would need a rights check first. No rights acceptance is inferred here.

## 4. Owner acceptance (BLANK on purpose)
| Item | Chosen option | Date | By |
|---|---|---|---|
| Adopt addendum version | | | |
| Q1 | | | |
| Q2 | | | |
| Q3 | | | |
| Q4/Q5 | | | |
| Q6 | | | |
| Q7 | | | |
| Q8 | | | |
| Reviewers named (two humans + adjudicator) | | | |
| Rights path for the source text confirmed | | | |

## 5. Blocker record
No independent human reviewers and no approved rights path are recorded, so the human review **cannot start**. I will not substitute another AI review or a paid sweep. The remaining owner decisions are exactly the blank rows above.
