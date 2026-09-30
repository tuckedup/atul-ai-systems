# Completed researched label review

Date: 2026-09-29. Reviewer: Codex (AI), non-blind. This is a completed review,
not a request for the user to fill the remaining decisions.

`LABEL_REVIEW_RESEARCHED.json` contains **18 completed decisions: 12 supported,
6 unsupported**, under the unchanged RouteBench source-support rubric. It also
records confidence, exact source quotes and offsets, candidate context,
counterarguments, research references, and separate original-protocol findings.
Seven primary decisions changed from the previous AI review. Binary decisions are
best judgments, not certainty; the question-only case has low confidence.

The earlier AI JSON and blank human-review form are preserved. No benchmark label,
case, split, rubric, cache or measured result was changed. No new paid judge calls,
held-out evaluation, or kappa calculation was performed. Web-research connector
credits may apply. The review does **not** demonstrate kappa >= 0.74.

## What the original sources clarified

FRANK explicitly counts references lacking an antecedent within the summary,
even when readers can resolve them from the article. My earlier interpretation
of the Dorset inn case missed this: its facts are article-supported, but that does
not make it error-free under FRANK. The same rule plausibly explains part of the
paid-leave disagreement. This is a real protocol mismatch, not proof of bad human
labels. The JSON separates article support from protocol errors rather than
switching standards silently. [FRANK, sections 2.2 and A.5](https://arxiv.org/html/2104.13346v2).

TofuEval evaluates factuality separately from relevance and completeness. Human
annotators saw full summaries. Importantly, the paper also reports that adding
previous sentences did not change model performance in an initial study. Restoring
context is therefore a reasonable hypothesis for specific reference errors, not
a demonstrated route to the desired aggregate score. [TofuEval, F.1, F.4 and C.3](https://arxiv.org/html/2402.13249v2).

A contemporaneous Long Beach city memo identifies SB 1 and the $8 million FY19
amount. This corroborates the suspected transcript issue but does not certify the
meeting transcription. External material was not appended to benchmark documents
or used as replacement grounding evidence. [City budget memo, Road Repairs](https://www.longbeach.gov/globalassets/city-manager/media-library/documents/memos-to-the-mayor-tabbed-file-list-folders/2018/october-23--2018---california-state-fy-19-budget-update).

## Decisions I resolved or corrected

These are my applications of the current local rubric, not decisions endorsed by
the original dataset authors.

| Case suffix | Previous | Current source-support decision | Reason |
| --- | --- | --- | --- |
| `65b135f17ee0e207` | Ambiguous | Supported | Certification vote is explicit; topic placement alone does not assert a different vote. |
| `70012ed4168ec7b5` | Ambiguous | Unsupported | Police support attaches to cost in the target, but to the program in the source. |
| `71ce0e9a4754c173` | Ambiguous | Supported, low confidence | Question adds no unsupported comparison result; this factuality-only pass is not summary acceptance. |
| `72c8a872b5c969ea` | Ambiguous | Unsupported | Source corrects annual $8m wording to FY19 and gives a different FY20 amount. |
| `a13c7ba2cd8cc1cd` | Unsupported | Supported | My prior reading added an already-aired claim; the target does not explicitly say that. |
| `c956780fd0b574bb` | Ambiguous | Unsupported | Repeated coherent SB one references outweigh an isolated garbled numeric fragment. |
| `f9b9bed541b87d55` | Ambiguous | Supported | Omitted volunteering purpose does not explicitly assert unrestricted leave under this rubric. |

The Dorset inn decision remains source-supported but gains a separate FRANK
CorefE finding. The paid-leave case also gets a medium-confidence protocol-error
finding. Original rater disagreement for Ueda, Tevez and Adeli is preserved;
I did not manufacture a rationale just to agree with the imported binary labels.
No FRANK binary aggregation rule is claimed to have been reproduced.

## How the implementation agent should use this

1. Use this artifact as error analysis, not replacement human ground truth or new
   few-shot examples copied from dev into training.
2. Establish whether the evaluation contract means pure source support or original
   dataset-native factuality. The present rubric claims alignment but demonstrably
   differs on FRANK coreference. Any policy change must be versioned and disclosed.
3. Treat non-summary questions as a separate response-validity diagnostic. Do not
   retroactively add a factuality failure merely to match a difficult label.
4. Preserve original corpus labels, splits and denominators. Test general policy
   changes on dev under a declared experiment and approved remaining budget;
   do not hardcode case IDs or use this review as a held-out success claim.

This completes the requested review. It does not establish that any specific
prompt change will reach the target or that every disputed published label is wrong.
