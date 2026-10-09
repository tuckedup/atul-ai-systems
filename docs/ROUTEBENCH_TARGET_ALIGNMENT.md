# RouteBench target alignment and local training-candidate inventory (Step 1, offline)

Date: 2026-09-30. Scope: Step 1 of `ROUTEBENCH_NEXT_PHASE_PLAN.md` only. Built from existing local files. No labels, splits, rubric,
caches, frozen scripts or experiment records were changed. No downloads, paid calls, GPU work, held-out evaluation, or model
loading. The 30 reserve upstream documents' labels, predictions and errors were not read: error-type counts below come from the
Tofu **test-half** annotation CSVs, whose 30 documents are already RouteBench corpus documents.

Verification levels used below: **[local]** checked in a file in this repository or on disk in this step; **[secondhand]** stated in a
repository audit document or AI research note but not re-verified against the primary source in this step; **[absent]** no local
evidence either way.

## 1. What the current rubric says (`routebench/evalops/rubrics/summarize.yaml`)
- Pass = every factual assertion is stated in the document or follows by direct entailment (line 14-16). Support comes from the
  document only; real-world truth is not evidence (line 85). Omission/terseness is a PASS (line 80). Modal strengthening
  ("considering expanding" -> "will expand") is a FAIL (line 88-93). Ordinary paraphrase, **pronoun resolution** and combining
  adjacent sentences count as entailed (line 40-41). Attribution to a speaker or source the document does not make is a FAIL
  (line 71). Fluency is out of scope.
- `docs/KAPPA_DESIGN.md` section 4.3 says these choices copy the annotation question "is every claim supported by the source
  document?" of the AggreFact/TofuEval family. **[absent]** No primary-source annotation-guideline text is stored in the repo; that
  statement is the rubric author's reading, not a checked quotation.

## 2. Policy comparison
| # | Dimension | RouteBench rubric | Documented annotation policy | Status |
| --- | --- | --- | --- | --- |
| 1 | Meaning of supported (label 1) | Supported by the supplied document; world knowledge does not count | LLM-AggreFact: 1 = claim supported by the document (`data/raw/MANIFEST.json`, cross-checked to the MiniCheck README) [local]. Tofu README: `sent_label` yes = "factual consistent"; `Extrinsic Information` is the most common error type (165 of 334 error tags in the test half) [local] | Aligned in direction; the exact upstream guideline wording is [absent] |
| 2 | Unit judged | The whole "response" (every assertion) | Tofu: one label per summary **sentence**; README says a full summary is the sentences joined by `sent_idx` [local]; annotators saw the full summary [secondhand: `LABEL_REVIEW_RESEARCH_NOTES.md`]. CNN/FRANK: per-sentence, per-annotator tags (NoE, CorefE, ...) [local, `FRANK_FLAGGED_ANNOTATIONS.json`], but the rule that produced the single binary AggreFact-CNN label was not reconstructed [secondhand: `LABEL_ADJUDICATION_AUDIT.md`] | **Verified mismatch in unit**: RouteBench corpus cases are single sentences for Tofu but whole 3-4 sentence summaries for CNN. Label-derivation for CNN unresolved |
| 3 | Pronoun and reference resolution | Resolved from the source counts as entailed (yaml line 40-41) | Tofu has a `Mis-Referencing` error type (30 tags in the test half) [local]. FRANK tags coreference errors (`CorefE` is tagged 6 times across the 5 audited records) [local] and its protocol reportedly penalises references with no antecedent inside the summary [secondhand: `LABEL_REVIEW_RESEARCH_NOTES.md`, citing FRANK sec 2.2/A.5] | **Verified rubric/dataset difference in text**; exact upstream rule unverified. Worked example `gnd-406e875c9ec93511` [local audit] |
| 4 | Questions / non-assertions | Rubric speaks of "factual assertion"; silent on questions | Tofu README defines no rule [absent]. Example `gnd-71ce0e9a4754c173` (a question after invented poll results) [local audit] | **Unresolved** |
| 5 | Tense / modality | Modal strengthening = FAIL (explicit). Tense (future vs completed) not mentioned | Tofu `Tense/Modality Error` (7 tags) and `Nuanced Meaning Shift` (41 tags) [local]; example `gnd-a13c7ba2cd8cc1cd` (coming segment vs completed comparison) [local audit] | Modality aligned; tense has no explicit mention in the rubric text, though the entailment and contradiction criteria may already cover it (unknown; hypothesis only) |
| 6 | Omission / completeness | Omission is not an error (yaml line 80) | Tofu scores completeness as a separate annotation (`completeness/` files, README) [local]; but some factual-consistency memos hinge on incomplete characterisation, e.g. `gnd-33aed12bc35c7bcd` "not entirely accurate ... staff report" [local exemplar file] | **Unresolved**: annotators' factual labels sometimes absorb completeness. Exemplar selection already excludes 3 such memos |
| 7 | Opinion presented as fact | Only implicit, via attribution wording (yaml line 71) | Tofu `Stating Opinion as Fact` (24 tags, mostly MediaSum) [local]; train example `gnd-bc43473f55c71d46` [local] | **Possible rubric gap** (hypothesis, not tested in isolation) |
| 8 | Unsupported inference | "inference that needs an unstated premise" fails (yaml `claims_entailed`) | Tofu `Reasoning Error` (55 tags) [local] | Aligned |
| 9 | Contradiction | Critical criterion `no_contradiction` | Tofu `Contradiction` (12 tags) [local] | Aligned |
| 10 | Noisy or garbled transcripts | Silent | No policy [absent]. Example `gnd-c956780fd0b574bb` ("SB one" vs "SB 145") [local audit] | **Unresolved** |
| 11 | Multi-annotator disagreement | Single binary label used as truth | FRANK: 3 annotators per sentence, one often finds no error (`FRANK_FLAGGED_ANNOTATIONS.json`) [local]; number of Tofu annotators per sentence [absent] | Rater disagreement is recorded for the audited CNN records; no ceiling is estimated |

Error-type totals above are from `tofueval_{mediasum,meetingbank}_factual_eval_test.csv` (322 flagged sentences carrying 334 tags; multi-type sentences are counted once per tag).

## 2b. Update from primary sources (Step 2 lookups, 2026-09-30; public papers read directly)
- **FRANK definition of CorefE [verified, FRANK paper, error-typology section]:** "pronouns and other types of references to previously mentioned
  entities either are incorrect or have no clear antecedents, making them ambiguous". So row 3 is now a verified rule difference:
  the RouteBench rubric accepts pronoun resolution from the source; FRANK counts a reference lacking a clear antecedent in the summary
  as an error.
- **FRANK's own label rule [verified, FRANK paper, results section]:** a sentence has a factual error if a majority of its 3 annotators marked one; a
  summary is incorrect if any sentence has one.
- **AggreFact-CNN binary label [verified chain]:** AggreFact says FRANK was included "directly" from SummaC (AggreFact paper, dataset section);
  SummaC labels a FRANK summary consistent "if a majority of annotators labeled the summary as containing no error" (SummaC paper,
  benchmark-dataset section). So the CNN label is a summary-level majority vote over 3 crowd annotators. This matches the audited records (2 of 3
  annotators flagging CorefE gives label 0 for `gnd-6d0a6a0bc1619eb9`). Not verified: how SummaC turns an annotator's
  sentence tags into that annotator's summary verdict (in SummaC code, not text). Row 2 is therefore: **CNN = summary-level
  majority vote; Tofu = sentence-level label**.
- Remaining unresolved: the exact TofuEval annotator instructions (the NAACL paper appendix was not extracted in this step), question
  handling (row 4), noisy transcripts (row 10).

## 3. Reading of the table (verified vs hypothesis)
- **Verified**: (a) the rubric text and the datasets' error taxonomies overlap on unsupported content, contradiction, reasoning
  and modality; (b) the rubric explicitly accepts pronoun resolution from the source while FRANK and Tofu carry reference-error
  categories; (c) the corpus mixes a sentence unit (Tofu) with a whole-summary unit (CNN).
- **Hypotheses (untested)**: that in-summary antecedent rules, tense, and opinion-as-fact explain a material share of the
  shared judge errors. Restoring the summary context did not raise dev kappa (v22), which weakens "visibility alone" but does not
  test the policy difference itself.
- **Not established**: that any label is wrong, or that an agreement ceiling exists (`LABEL_ADJUDICATION_AUDIT.md`).

## 4. Defensible training target compatible with the original contract
Contract retained: binary label 1 = supported by the supplied document, original labels, splits and denominator.
Candidate target: *given a source document and one candidate claim (sentence), predict source support.* This matches the rubric and
the Tofu unit; it does **not** encode the within-summary-antecedent rule (row 3), a question rule (row 4) or a completeness rule
(row 6). Consequences to disclose: (1) CNN cases are whole summaries with an unverified label derivation, so a sentence-level model
does not automatically cover them; any sentence-to-summary aggregation (for example "every sentence supported") would have to be
declared in advance and validated on training-permitted data, without altering the original cases or labels; (2) if the
dataset-native definition (rows 3 and 6) governs the labels, agreement between a source-support-only judge and those labels may
be reduced. That is a possibility, not an established numerical ceiling, and it cannot be estimated offline. **The ambiguity is unresolved and is reported, not resolved by
editing the rubric or the labels.** Any dataset-native product definition would be a separate evaluation.

## 5. Local inventory of training candidates
None of the on-disk labelled summarisation material is cleared for training.
| Local material | What it is | Training permission | Verdict |
| --- | --- | --- | --- |
| `data/raw/aggrefact.jsonl` (12,949 rows; CNN 558, XSum 558, MediaS 726, MeetB 772, Wice 358, Reveal 1,710, ClaimVerify 1,088, FactCheck-GPT 1,566, ExpertQA 3,702, Lfqa 1,911) | Filtered copy of the LLM-AggreFact test split | Evaluation benchmark only ("not for pretraining or fine-tuning") | Not eligible |
| `data/raw/authoritative/*.csv` (Tofu dev/test annotations, AggreFact final/sota CSVs) | Expert labels and memos | Tofu: "should not be used in training NLP models" | Not eligible |
| `data/probe/*`, `data/attribution/*` | Wice/Reveal-derived probes and cases | Derived from LLM-AggreFact | Not eligible |
| `data/raw/tofueval_upstream/docs.jsonl` (100 transcripts) | Source documents | Evaluation-only; CC-BY-NC-SA / research-only; reserve set must stay unread | Not eligible |
| `gedd_*exemplars.json`, `exemplars.json`, `stack_minicheck/fit_result.json` | Few-shot packets and fitted combiner from Tofu/CNN train rows | Restriction unresolved | Not cleared |
| `routebench-reference-sources/MiniCheck` (`synthetic_data_gen/`) and `AlignScore` (`train.py`, `generate_training_data.py`) | Code only; no data files | Apache-2.0 / MIT code | Reusable as code; no data |
| `.local/ckpts/...MiniCheck-Flan-T5-Large` (5.9 GB), `.local/hhem/model` (0.4 GB) | Pretrained verifiers | MIT / Apache-2.0 | Candidate base models. The MiniCheck README states its 14K C2D/D2C data was already used to train the MiniCheck models [local], so re-training MiniCheck on that set again would repeat data it has already seen (not new evidence; whether further training helps at all is unproven) |
| `gsm8k`, `humaneval`, `mbpp`, `spider`, `bfcl` | Other tasks | Task-mismatched | Irrelevant to groundedness |
Result: **zero locally available, training-permitted, task-matched labelled examples.**

## 6. Ready vs missing before Step 2
- Ready: the comparison table with evidence; the sentence-level target statement; the inventory; the list of known unit and policy
  differences; frozen protocols and hashes for all earlier experiments (unchanged).
- Missing (smallest remaining requirements): (1) a **training-permitted, task-matched source** identified and license-checked
  (metadata-level lookup first; candidates named in `ROUTEBENCH_FINAL_PHASE_REPORT.md`: RAGTruth summarisation, FaithBench, plus
  synthetic C2D/D2C which MiniCheck already consumed) with an exact and near-duplicate check against every evaluation, probe and
  exemplar document; (2) **primary-source guideline text** for AggreFact/TofuEval/FRANK to settle rows 2-4 and 6 (needs a live
  lookup, not done here); (3) a base-model decision that accounts for MiniCheck's prior exposure to C2D/D2C; (4) the enforced
  timeout and resume controls the plan requires (not built). Step 2 needs approval for lookups; a data download needs separate
  approval after the license and overlap gate.
