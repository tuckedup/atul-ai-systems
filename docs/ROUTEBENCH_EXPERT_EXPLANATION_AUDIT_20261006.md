# Expert-explanation audit of the 35 disputed "supported" items (2026-10-06)

**Scope and provenance.** Existing files only: no provider call, training, exclusion, relabelling or held-out access; Stage 1b stays closed; every case and label is unchanged. The classification below is my provisional reading, AI-assisted, **not** independent human adjudication. The published explanations are the `exp` and `type` columns of the pinned TofuEval annotation files. The full audit (candidate, summary context and topic, expert objection, the evidence the reviewers relied on, closest source passages, the distinction missed) is in `.local/judge-v19-phase1/blind-review-v1/reconciliation/expert_explanation_audit.md`. It contains gold explanations and **must never enter an evaluation prompt**.

**Correction to my earlier wording.** I described the published labels as "lenient". That was backwards: where a published label rejects a claim that models accept, the benchmark's experts applied a stricter or different standard. AI agreement with v18 does not show the labels are bad.

## What was checked
35 consolidated-"supported" items among the 44 persistent errors. **27** match exactly one published TofuEval row with a non-empty explanation (MediaS and MeetB); **8** are AggreFact-CNN items with no published rationale, which remain unexplained (R012, R018, R025, R029, R044, R071, R084, R088). Expert error types among the 27: Stating Opinion as Fact 12, Nuanced Meaning Shift 8, Reasoning Error 3, Mis-Referencing 2, Contradiction 1, Tense/Modality 1. Judge-visible inputs for both v18 and the reviewers: the sentence and its source only (v18's `context_path` is empty); the experts also saw the full summary and its topic.

## Result (27 items)
| Category | n | Sub-failure (n) |
|---|---|---|
| Policy mismatch (we saw what we needed but applied a different standard) | 14 | analysis or opinion stated as fact, i.e. attribution lost (8); document-genre claims such as "the document is a report" (3); simile or connotation stated more strongly (3) |
| Genuine reasoning error (our own written instructions plus the source suffice to reject) | 8 | hedge or unconfirmed status dropped (4); tense/modality (1); verb strength or attribution, "organized" for "encouraged" (1); explicit counter-evidence not reported (1); two separate facts merged (1) |
| Input-context gap (needs the topic, summary position or antecedent the judge never saw) | 3 | R002 topic, R024 summary order, R043 antecedent of "This" |
| Questionable annotation | 1 | R033: the expert's stated reason does not match the source |
| Unresolved | 1 | R081: how much inference is allowed (same open question as R065) |

Covered by the instructions the reviewers had: yes 8, partly 11, no 8.
**Concentration caveat:** 11 of the 27 come from one document (an interview with an industry analyst), including all 8 attribution cases; the 27 come from 13 documents. They are not 27 independent observations.

## Dominant actionable failure modes
1. **Analysis, evaluation or prediction restated as the document's fact** (8, all one document; the benchmark expects the speaker's view to stay attributed). The reviewers' instructions mention attribution but not opinion versus fact, and v18's rubric only checks for *invented* attributions.
2. **Hedge, tense, modality or verb strength not preserved** (6, five documents; the "genuine reasoning" cluster). **Correction (2026-10-06):** an earlier draft said no criterion asks about hedging or strength. That was wrong. v18's `claims_entailed` criterion already fails "a generalisation stronger than the document supports" and the rubric's boundary example pins "will expand" versus "is considering expanding" as a fail. So these are cases where the judge did not apply an instruction it already has, not a missing rule.
3. **Document-genre claims** (3): our instruction on statements about the document pushed toward accepting them, while the earlier fresh-document audit found the opposite pressure on false rejections, so this one is not a safe single-direction change.
4. **Context the judge never sees** (3 of 27): real, but restoring it was already tested (v22: dev kappa 0.656 vs 0.6745, interval [-0.071, +0.058]) and the "This compromised ..." case (R043) still scored 1.0.

## Prepared variant (offline; not run; paid evaluation remains closed)
`claim_fidelity` is an **explicit attention mechanism**, not a previously missing rule: it gives the distinctions the existing rubric already prohibits (modal strengthening, stronger generalisations) their own yes/no verdict, so they are checked on every response, and it extends them to opinion attribution, predictions, disputed reports and tense. It is defined by the assertion's **evidential status in the document, not by the speaker's authority**: opinion stays attributed; predictions, plans, possibilities and concerns stay qualified; disputed or unconfirmed statements keep that qualification; tense and material qualifications are kept; an official's prediction or disputed report is held to the same standard as anyone else's; an ordinary unqualified factual report that the document states plainly may be restated plainly; faithful paraphrase passes. It is a new rubric directory and an opt-in variant (`v24-claim-fidelity-o4mini`); v18 and its identity are unchanged. Details, exact diff, tests and the bounded evaluation proposal: `docs/ROUTEBENCH_CLAIM_FIDELITY_VARIANT_20261006.md`. The criterion text is generic and quotes no benchmark sentence, source or explanation.

**Synthetic regression examples** (`routebench/evalops/fixtures/claim_fidelity_regression_v2.json`, which supersedes the v1 file kept for the record; tests check structure, paired controls, mode coverage, authority and non-authority speakers, and that no 6-token run is copied from any reviewed sentence, source passage or expert explanation). Expected values follow the criterion text and have **not** been checked against any model:

| id | source | sentence | expected |
|---|---|---|---|
| attribution-opinion-violation | ANALYST DANA REYES: In my view the steelmaker's board has grown complacent, and I'd expect it to keep ignoring the unions. | The steelmaker's board has grown complacent. | fails |
| attribution-opinion-control | ANALYST DANA REYES: In my view the steelmaker's board has grown complacent, and I'd expect it to keep ignoring the unions. | An analyst says the steelmaker's board has grown complacent. | passes |
| attribution-prediction-violation | REYES: I predict the board will not negotiate before spring. | The board will not negotiate before spring. | fails |
| attribution-prediction-control | REYES: I predict the board will not negotiate before spring. | Reyes predicts the board will not negotiate before spring. | passes |
| hedge-concern-violation | There is a concern that the outbreak may have reached the northern villages. | The outbreak has reached the northern villages. | fails |
| hedge-concern-control | There is a concern that the outbreak may have reached the northern villages. | There is concern that the outbreak may have reached the northern villages. | passes |
| hedge-unconfirmed-violation | By our count that makes more than 40 sites, though we have not confirmed the newest one. | More than 40 sites are affected. | fails |
| hedge-unconfirmed-control | By our count that makes more than 40 sites, though we have not confirmed the newest one. | A count puts the number of affected sites above 40, though the newest is unconfirmed. | passes |
| tense-preview-violation | Coming up after the break: we compare the two budgets. | The programme compared the two budgets. | fails |
| tense-preview-control | Coming up after the break: we compare the two budgets. | The programme will compare the two budgets after the break. | passes |
| strength-verb-violation | The guild urged its members to avoid the new store. | The guild organized a boycott of the new store. | fails |
| strength-verb-control | The guild urged its members to avoid the new store. | The guild urged its members to avoid the new store. | passes |
| strength-simile-violation | He called the reviewer 'something like a paid publicist'. | He accused the reviewer of being a paid publicist. | fails |
| strength-simile-control | He called the reviewer 'something like a paid publicist'. | He likened the reviewer to a paid publicist. | passes |
| strength-connotation-violation | Residents of the island may stay for ten years; other arrivals are usually sent back. | Island residents are welcomed while other arrivals are sent back. | fails |
| strength-connotation-control | Residents of the island may stay for ten years; other arrivals are usually sent back. | Island residents may stay while other arrivals are usually sent back. | passes |
| merged-claims-violation | The clinic supports the registry. Its running cost is low. | The registry's low running cost is supported by the clinic. | fails |
| merged-claims-control | The clinic supports the registry. Its running cost is low. | The clinic supports the registry, and its running cost is low. | passes |
| prediction-official-violation | The finance minister said she expects inflation to fall below three percent by autumn. | Inflation will fall below three percent by autumn. | fails |
| prediction-official-control | The finance minister said she expects inflation to fall below three percent by autumn. | The finance minister expects inflation to fall below three percent by autumn. | passes |
| prediction-nonofficial-violation | A local baker predicted that prices would rise next month. | Prices will rise next month. | fails |
| prediction-nonofficial-control | A local baker predicted that prices would rise next month. | A local baker predicted prices would rise next month. | passes |
| disputed-official-report-violation | The ministry reported 12 cases; the opposition disputed the figure. | There were 12 cases. | fails |
| disputed-official-report-control | The ministry reported 12 cases; the opposition disputed the figure. | The ministry reported 12 cases, which the opposition disputed. | passes |
| disputed-official-claim-violation | The mayor insisted the road was safe, though inspectors said otherwise. | The road was safe. | fails |
| disputed-official-claim-control | The mayor insisted the road was safe, though inspectors said otherwise. | The mayor insisted the road was safe, though inspectors disagreed. | passes |
| disputed-nonofficial-report-violation | A passer-by said the fire started at nine, but the fire brigade's log gives a different time. | The fire started at nine. | fails |
| disputed-nonofficial-report-control | A passer-by said the fire started at nine, but the fire brigade's log gives a different time. | A passer-by said the fire started at nine, but the fire brigade's log gives a different time. | passes |
| narrator-fact-guard | The council voted 7-2 on Tuesday to approve the budget. | The council approved the budget on Tuesday. | passes |
| speaker-tag-kept-guard | MAYOR LEE: The new bridge will cut commute times by ten minutes. | Mayor Lee said the new bridge will cut commute times by ten minutes. | passes |
| faithful-paraphrase-guard | The report says unemployment fell from 6.1 to 5.4 percent over the year. | The report says the jobless rate dropped from 6.1% to 5.4% during the year. | passes |
| hedge-reworded-guard | It is possible the line will reopen in March. | The line might reopen in March. | passes |
| authority-plain-fact-guard | The health ministry said 12 new cases were recorded on Monday. | Twelve new cases were recorded on Monday. | passes |
| nonauthority-plain-fact-guard | A shopkeeper on the street said the bridge closed at noon. | The bridge closed at noon. | passes |
| official-qualified-prediction-kept-guard | The central bank governor said rates are likely to stay flat until June. | Rates are likely to stay flat until June, the central bank governor said. | passes |

## Uncertainties and limits
- The wording was written after reading these explanations, so these 27 items (and the 35) are consumed: any measurement on them is descriptive and contaminated. A fair test needs new documents (the human-annotated evaluation set, still unpurchased) or, much later, the sealed test under its separate authorization.
- Stricter checking should lower false passes but may raise false rejections: v18 was false-rejection heavy on fresh documents (37 FR / 5 FP), where the earlier audit found experts accepting statements about the document and mild inference. The change targets specific fidelity checks, not global strictness, but nothing here shows the net effect on kappa, and the guards now follow your decision that fidelity depends on evidential status, not the speaker's authority.
- Opinion-attribution evidence is one document. The classification is mine; a second reader (ideally human) could move items between "policy mismatch", "genuine error" and "questionable".
- 8 items have no published rationale; R033 and R024 are arguable annotations; R081 and R065 hinge on an unmade inference-strength decision.
- This does not demonstrate kappa 0.74 and changes no benchmark score.

## Appendix: per-item classification (distinction missed, abbreviated)
| item | category | sub-failure | covered by instructions | distinction missed |
|---|---|---|---|---|
| R002 | input-context gap | topic / summary position | no | The sentence is supported by the source taken alone (an item certifying the primary results was voted). |
| R006 | policy mismatch | analysis stated as fact (attribution) | partly | 'The gun industry obeys the NRA' restates a guest analyst's characterisation without attributing it; the benchmark expects the opinion to stay attributed. |
| R008 | policy mismatch | document-genre claim | no | 'The document is a report from the committee' is rejected because the transcript is a meeting in which the report is read and discussed, not the report itself. |
| R010 | policy mismatch | connotation / paraphrase tolerance | partly | 'Cubans are welcomed' adds a positive attitude the source does not state (Cubans are allowed to stay, Haitians often turned back). |
| R011 | policy mismatch | document-genre claim | no | 'The provided document is a report from the city manager': the source is the city manager presenting a report plus other speakers' comments. |
| R015 | policy mismatch | analysis stated as fact (attribution) | no | 'Pugnacious' is the analyst's evaluative description; the summary presents it as the document's own fact. |
| R016 | genuine reasoning error | verb strength / attribution | yes | The source says the NRA encouraged buyers to boycott one manufacturer; the sentence says the NRA organized boycotts against manufacturers that backed control. |
| R019 | genuine reasoning error | hedge dropped | yes | The source says there is a 'concern that' Iran has forces well placed in Gulf Shiite communities; the sentence states the presence as fact. |
| R024 | input-context gap | summary position (arguable annotation) | no | The source contains BOTH convening announcements: Gateway Village at the start of the item and a different district after the vote. |
| R033 | questionable annotation | rationale does not match the source | no | The expert says the 'uneasy alliance' framing is only the analyst's opinion. |
| R036 | policy mismatch | document-genre claim | no | 'The provided document is a staff report': the document contains the staff presentation, the vote and more. |
| R037 | genuine reasoning error | hedge dropped | yes | The source says the two 'may not always have the same opinions'; the sentence says they 'have different opinions'. |
| R040 | genuine reasoning error | counter-evidence missed | yes | The source says the Quds Force 'ultimately do not report to' the Supreme Leader, directly against 'including the Quds Force'. |
| R042 | genuine reasoning error | hedge dropped (unconfirmed figure) | yes | The anchor says 'over 40' by her own count and adds that CNN has not confirmed the new case; the sentence states it as established ('one of over 40 cases nationwide'). |
| R043 | input-context gap | reference resolved without summary context | partly | 'This compromised national security' follows a sentence about Giuliani and the ambassador, so 'This' reads as that event; in the source the compromise is attributed to the president pressing Ukraine to investigate rivals. |
| R045 | genuine reasoning error | hedge dropped | yes | Same hedge as R019: 'concern that ... |
| R047 | policy mismatch | analysis stated as fact (attribution) | partly | 'The industry stays silent to avoid fighting the NRA publicly' is the analyst's explanation of motive, stated without attribution. |
| R048 | genuine reasoning error | conjoined claim conflated | yes | The source says the police department supports developing the program and that the cost is low; the sentence says the cost is 'supported by the police department'. |
| R049 | genuine reasoning error | tense / modality | yes | The source is a programme preview ('coming up ... |
| R050 | policy mismatch | analysis stated as fact (attribution) | partly | The motive claim is the analyst's opinion. |
| R060 | policy mismatch | simile stated as identity (paraphrase tolerance) | partly | The president wrote that the author is 'a liar who is like a Dem operative'; 'accused of being a Dem operative' turns a comparison into an accusation of identity. |
| R069 | policy mismatch | analysis stated as fact (attribution) | partly | 'Pugnacious' appears in the source as 'what I would describe as ... |
| R075 | policy mismatch | analysis stated as fact (attribution) | partly | 'The NRA sets the terms of the debate' is the analyst's judgement ('if anything, it is the NRA that ...'), stated as fact. |
| R076 | policy mismatch | simile stated as identity (paraphrase tolerance) | partly | Same as R060 in a different summary of the same article. |
| R077 | policy mismatch | analysis stated as fact (attribution) | partly | 'Unlikely to publicly oppose' is the analyst's prediction, stated without attribution. |
| R081 | unresolved | inference strength | no | The source calls it a '2017 tax repeal initiative' and urges opposing 'this proposition'; it never states in so many words that Proposition 6 would repeal the gas and vehicle taxes. |
| R082 | policy mismatch | analysis stated as fact (attribution) | partly | The two reasons for the industry's inaction are the analyst's reading of motives, stated as the document's facts. |
