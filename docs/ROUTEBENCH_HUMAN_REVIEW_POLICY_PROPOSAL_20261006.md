# Human review of the residual errors: proposed policy decisions (2026-10-06)

**Status: PROPOSAL ONLY. Nothing here is decided, and nothing has been sent, purchased or run.** These are the policy questions that the AI-assisted review and the expert-explanation audit left open. They must be settled **in writing, once, before any human reviews the 88-item packet, and independently of what any answer would do to a score.** I propose defaults, but the choices are yours. I have not verified that the benchmark's published annotation guidelines (`aggrefact-annotation-guidelines`) settle any of them; checking that is part of making each decision.

## A. The five items that stayed uncertain
| # | Item(s) | Question | Options | My proposed default (not binding) |
|---|---|---|---|---|
| 1 | R046 | **Generic vs literal plurals.** A document describes one boycott; the sentence says boycotts happened | A. literal: a plural needs more than one documented instance. B. generic reading allowed when the document describes a pattern or a single instance of a practice. C. always record `uncertain` | **A**, with `uncertain` allowed when the document describes a recurring practice without counting it |
| 2 | R054 | **Non-declarative candidates** (a question or an imperative addressed to the reader) | A. not assessable: reported as a separate outcome, never forced into supported/unsupported. B. judge the implied assertion. C. judge the question's presupposition | **A** |
| 3 | R065, R081 | **Strength of inference.** May a conclusion that needs a premise the document does not state count as supported? | A. direct entailment only (a careful reader accepts it without outside knowledge or an unstated premise). B. A plus an inference whose premise is itself stated in the document. C. B plus ordinary world knowledge | **A**, with a written explanation naming the premise whenever a reviewer picks `supported` on an inference |
| 4 | R013 | **Speaker identity or role** when the transcript names an office, not the person | A. supply the speaker/role metadata from the original source if it exists (shown identically to every reviewer as a labelled addendum). B. record permanent uncertainty. | **A where the original source has it, otherwise B** |
| 5 | R022 | **Byline** missing from the supplied article text | same two options | same default |
Both metadata options keep the original packet untouched: any addendum is a separate, versioned field, applied to every item that has it, never only to disputed ones.

## B. Other policy questions surfaced by the expert-explanation audit
These drove most of the persistent disagreements, and the packet instructions did not settle them:
| # | Question | Options |
|---|---|---|
| 6 | **Opinion or analysis restated as fact** (8 audited items, all one interview document): must a summary keep the speaker's attribution for an analyst's view? | A. yes, unattributed opinion is `unsupported`. B. only when the document marks it as opinion. C. no |
| 7 | **Document-genre claims** ("the document is a report ...") when the text is a meeting transcript in which the report is read | A. judged on what the document actually is. B. judged on whether the document contains or presents what is described |
| 8 | **Hedges, tense and verb strength**: the existing rubric already requires them; confirm the same standard for human reviewers | A. strict (a dropped hedge or a stronger verb is `unsupported`). B. tolerant for minor strengthening |
Question 7 pulls in the opposite direction from the earlier fresh-document audit (experts there accepted statements about the document), so I would not choose it without checking the published guidelines.

## C. How the study would run (unchanged from the prepared packet)
Blind packet of 88 shuffled items (44 persistent errors + 44 controls; reviewers see only an id, the sentence and its document); at least two independent **human** reviewers per item who do not see one another's work, a third adjudicator for disagreements and every `uncertain`; each response needs the outcome (`supported` / `unsupported` / `uncertain`, plus `not_assessable` if you pick option 2A), verbatim evidence and an explanation; `reviewer_kind` = `human` (AI-assisted responses are kept separate and never counted as independent); the validator checks format and quotations. Results are **descriptive only**: counts by group, reviewer agreement, the uncertain share. No 50% or 75% threshold, no label ceiling and no spending decision follows automatically; "uncertain" does not mean a published label is wrong.

## D. Decision record (to be filled in by the owner)
| Q | Chosen option | Date | Notes |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4/5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |

## E. What this does not do
It does not change any original label, case or split; it does not add an AI adjudicator as if it were human; and it does not authorise recruiting reviewers, sending documents anywhere (the TofuEval transcripts are CC-BY-NC-SA, so a paid vendor needs a rights check; in-house review does not), or any further model run.
