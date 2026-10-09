# Offline error-cause audit of v18 on the 240-case fresh evaluation (2026-10-01)

AI-assisted diagnosis by one reader (me), not independent human relabelling. No paid calls, training, prediction changes or relabelling; every original label is preserved. Source of the 240 cases/predictions: `.local/fresh-eval-v1/`. These cases are now examined and cannot serve as fresh evidence again.

## Established facts (from saved data / arithmetic)
- v18 errors on the 240: 37 false rejections (gold supported, v18 rejected), 5 false passes. Both numbers come from `final_report.json`.
- The fresh run saved only score/pass, not criterion reasoning. Positive scores identify the criterion verdicts under the fixed weights (1/4/2/1 plus the critical rule). Zero identifies a failed critical `no_contradiction` criterion, but the other three verdicts are UNKNOWN, not passing. `claims_entailed` is therefore known to have failed in at least 31 of the 37 false rejections (84%); the three critical-zero cases may also have failed it. There is one known `no_invented_relations`-only failure and two known `specifics_grounded`-only failures. Criterion verdicts are not explanations of the model's reasoning. In the first 40-document sample (criteria saved), claims_entailed failed in 33 of 40.
- The suspected 'overlapping criteria' mechanism is NOT supported: the decision is any-criterion-fails, so `no_invented_relations` co-failing with `claims_entailed` (19 of 37 here, 20 of 40 before) adds no rejections; removing it would recover at most 1 case.
- Errors cluster by document: 5 of 30 documents hold 17 of the 37 false rejections; 9 documents have none.
- Descriptive associations (rates among gold-supported cases): pronoun-initial claims were rejected LESS (9% vs 20%), meta 'The document...' claims 22% vs 18%, MeetingBank 19% vs 18%, later summary sentences 19% vs 19%; claims of 25+ words were rejected more (30% vs 16%). These coarse proxies do not establish or rule out missing context as a cause; actual reasoning was not saved.

## Counts by primary cause (37 false rejections; confidence medium, mechanism of v18's reasoning unknown)
| Cause | n | Meaning |
|---|---|---|
| R rubric stricter than annotators | 14 | claim has a mild inference, generalisation, dropped hedge or cross-speaker synthesis that annotators accepted but v18's rubric text fails |
| J judge misses a supported claim | 10 | source supports the claim (often near-verbatim); v18 rejected it; includes 1 critical-contradiction false alarm |
| S surface-form / ASR variant | 5 | names or titles differ from the transcript spelling (Do What U Want, Lazar/Lady Fish, Good Barn/Bar, Delridge) |
| L label permissive or doubtful | 4 | claim states something the source does not (e.g. a 40% 'increase'); rejection defensible |
| M meta statement about the document | 2 | claim is about the summary/document, not a source fact |
| U unclear | 2 | could not verify either way |

False passes (5): 3 subtle errors v18 missed (tense 'will vote', overstated 'specifically Homeland Security', unsupported 'defiant'), 1 world-knowledge item (Initiative 1433 content), 1 label-strict/ambiguous (one song vs all music). Published expert memos back the first four.

## What the counts imply
- The AI-assisted categorization suggests R+S+M = 21 (57%) policy/precision conflicts, J = 10 (27%) judge errors, and L+U = 6 (16%) ambiguous cases. These are provisional case-reading judgments, not independently established causes or replacement gold labels.
- The R cases sit right next to the false passes: e.g. dropping 'might have been strung out on spice' is accepted by annotators, while 'will vote' for a vote in progress is an error. A blanket loosening of the rubric would trade false rejections for false passes. This is a hypothesis from reading, not a tested result.
- Arithmetic bounds (not predictions), holding false passes at 5 on the 240 unweighted: fixing S+M (7 cases) -> κ 0.59; S+M+R (21) -> 0.73; J only (10) -> 0.62; all 37 -> 0.92. Reaching 0.74 needs most of R as well as some of J.

## Recommended single intervention (hypothesis; nothing implemented or run)
Test one rubric-policy revision whose wording comes from the published TofuEval/AggreFact annotation guidelines, not from these 240 cases, covering three rules: surface-form variants of names/titles/ASR spellings count as the same entity; statements about the document itself are judged only on what they claim about its content; mild inference tolerance exactly as the guidelines define it. Evaluate it on permitted development material (TRAIN/dev, which also show the false-pass failure mode), report false rejections AND false passes against v18, with success criteria frozen first. A claim on fresh documents needs a new unexamined set; the original held-out test needs separate authorization. The J cases (10) are a model-reasoning problem and are not addressed by this change.

## Published context on 'other judges' (web search 2026-10-01; untrusted secondary sources)
- LLM-AggreFact leaderboard (balanced accuracy, MediaS / MeetB): Bespoke-MiniCheck-7B 76.0 / 78.3; Claude-3.5 Sonnet 73.4 / 84.6; Granite Guardian 3.3 (8B) 74.0 / 78.6; Mistral-Large 2 69.6 / 84.2; gpt-4o 71.4 / 79.8; FactCG-DeBERTa-L 72.3 / 74.3; MiniCheck-Flan-T5-L 73.6 / 77.3. No listed system exceeds 85 on either subset.
- Our v18 on the fresh reserved documents had weighted balanced accuracy 0.84. These documents, sampling weights and metrics differ from the cited leaderboard; that comparison does not establish a state-of-the-art ceiling or predict another judge's kappa. Kappa depends jointly on prevalence, sensitivity and specificity, so a balanced-accuracy number alone cannot determine whether 0.74 is attainable.
- Candidates seen: Bespoke-MiniCheck-7B (CC-BY-NC, 7B), Granite Guardian 3.3 8B, Paladin-mini 3.8B (CC BY-NC-ND, average 73.1 on eight subsets, no MediaS/MeetB figures), FactCG (already screened, worse than v18 on dev), and a 2026 ACL paper on test-taking-strategy prompting (code not yet released). None has evidence of beating v18 on these subsets.

Case-level evidence table: `.local/error_cause_table.json` (42 rows: case id, source, document, claim, failed criteria inferred from score, primary cause, evidence note, confidence).
