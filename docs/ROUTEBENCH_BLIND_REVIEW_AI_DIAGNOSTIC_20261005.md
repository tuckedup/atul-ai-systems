# AI-assisted blind-review diagnostic of the 44 persistent errors (2026-10-05): descriptive, NOT human gold

**Provenance.** Two AI reviewers (requested gpt-5.6-sol, high effort; realised model and effort not observable) each reviewed all 88 packet items in separate contexts without labels, model verdicts, selection groups or the other's answers. A parent adjudicated five items and is **not fully blind** (it had project history). Every record declares `ai_assisted`; **there are 0 independent human reviews**, so the human-only study remains pending. Nothing here changes an original case, label or split, improves the benchmark score, or shows kappa 0.74. No spend, call, relabelling, exclusion, training or held-out access is authorised or occurred. I did not compute any "kappa if relabelled" figure and will not: the review summary rightly forbids resolving items by whichever answer improves kappa.

**Checks I re-ran (offline).** All three response sets validate with my packet validator: 0 problems, all 88 IDs once, all `ai_assisted`, quotations verbatim; packet hashes intact. The reconciliation file was opened by me only after the reviewers finished, to produce the table below. Quotation authenticity is not entailment.

## What the unblinded cross-tab shows (consolidated outcomes, n = 88)
| Selection group (published label; v18 five verdicts) | n | reviewers: supported | unsupported | uncertain |
|---|---|---|---|---|
| Persistent false pass (published unsupported; v18 passed five times) | 38 | **35** | 1 | 2 |
| Persistent false rejection (published supported; v18 rejected five times) | 6 | 2 | 2 | 2 |
| Control, published supported (v18 and label agree) | 31 | 31 | 0 | 0 |
| Control, published unsupported (v18 and label agree) | 13 | 0 | 12 | 1 |
Excluding uncertain, the consolidated outcomes match the published label on 43 of 43 controls but on only 3 of 40 persistent errors. Initial agreement between the two reviewers was 86 of 88; their two disagreements were both persistent errors (R013, R046).

## How far this can be read (and what it cannot show)
- **It is consistent with, not proof of, label or policy problems on the persistent false passes.** On 35 of 38 the AI reviewers side with v18 against the published label. That is exactly what a lenient published-label policy, upstream rater disagreement, or a shared model interpretation would produce; this diagnostic cannot tell them apart.
- **Shared model bias is the main confound.** v18 is an OpenAI reasoning model and the reviewers are model-based; agreement between them is not independent evidence, and the earlier finding that several judges fail the same cases points the same way. The 35 "supported" outcomes are therefore the strongest reason to get independent humans on exactly these items, not a verdict on them.
- **The controls are not a policy calibration.** They were drawn from cases where v18 and the published label already agree, so 43 of 43 agreement partly reflects selection. It shows the reviewers do not disagree with the labels wholesale, not that their policy matches the original annotators' guideline (it is a plain-language policy written for this packet).
- **Uncertain does not mean wrong.** Five items stayed uncertain and each reflects a missing fact or an unsettled policy rather than a defect of the label.
- **Small, error-enriched sample;** no population rate or kappa follows from it. The 6 false rejections split 2 / 2 / 2.

## The five unresolved items: what is missing (documented, not resolved)
| Item | Group, subset | Published label; v18 | Reviewer A / B -> consolidated | What is missing (kept distinct from any model defect) |
|---|---|---|---|---|
| R013 | persistent false rejection, MeetingBank | supported; reject x5 | unsupported / uncertain -> uncertain | **Source metadata:** speaker identity or role for the supportive speaker; the transcript names an office ("city manager"), not the person |
| R022 | control (published unsupported), AggreFact-CNN | unsupported; reject x5 | supported / supported -> uncertain | **Source metadata:** the article byline; the supplied text does not name the author the sentence attributes |
| R046 | persistent false pass, MediaSum | unsupported; pass x5 | unsupported / supported -> uncertain | **Policy decision:** generic versus literal plurals (one documented boycott vs "boycotts" in the past) |
| R054 | persistent false pass, MediaSum | unsupported; pass x5 | uncertain / uncertain -> uncertain | **Policy decision:** how to judge a non-declarative candidate (a reader-directed opinion question) |
| R065 | persistent false rejection, MediaSum | supported; reject x5 | unsupported / unsupported -> uncertain | **Policy decision:** permitted strength of inference (autonomy inferred from jurisdiction and distributed control) |
R022 and R065 were shared gaps among the reviewers, which is why agreement alone cannot establish correctness.

## What this does and does not change
- Stage 1b stays closed; no paid judge experiment is supported by this. If anything the cross-tab strengthens the case that the 38 false passes are not clearly "judge leniency": under a plain-language policy, model reviewers mostly accept them. That raises, but does not settle, the possibility that the original-label target is limited by label or policy, and it does not show 0.74 is unreachable.
- The informative next step is unchanged and still unpurchased: the same packet reviewed by **independent humans** (two reviewers plus adjudication), with the policy questions above settled in writing first and the source-metadata gaps (speaker role, byline) either supplied or recorded as permanent uncertainty. Their outcomes would be descriptive, with no 50%/75% rule.
- Policy decisions on generic plurals, non-declarative candidates and inference strength should be made once, in writing, before any human review and independently of any score effect.

Artifacts preserved unchanged: `.local/judge-v19-phase1/blind-review-v1/{packet,reconciliation,review-session-v1}`; my cross-tab is saved separately as `reconciliation/ai_review_crosstab.json`.
