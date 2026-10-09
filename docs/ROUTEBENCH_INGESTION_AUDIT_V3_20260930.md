# RAGTruth ingestion: preprocessing fixes, v3 audit and frozen training manifest (2026-09-30)

Follows `ROUTEBENCH_INGESTION_AUDIT_20260930.md` (kept unchanged as the historical v1 audit; its report
`.local/ragtruth-ingest-20260930/ingest_report.json` is byte-identical, checksum verified before and after). No download was repeated,
and there were no paid calls, no GPU training, no threshold search and no held-out evaluation. No labels in any existing RouteBench
or upstream benchmark were touched. No new kappa is claimed. This is an AI-assisted technical result: not human adjudication and not
legal clearance. The rules below are my own general heuristics; I did not copy another project's splitter or wrapper list.

## What changed (policy `ragtruth-sentence-v3`, splitter `regex-v4`; every rule is general, none is case-specific)
Rules were derived from frequency patterns in the TRAIN partition only. Validation and test text were not inspected.
1. **Internal overlap safeguard** (`internal_overlap`): every TRAIN or VALIDATION document that is an exact or near-duplicate of a
   document in a later partition (train vs validation/test, validation vs test) is excluded; the RAGTruth test partition is never
   excluded by this rule. It is evaluated on the original partition contents, so it does not depend on iteration order, and it never
   uses model output. Full fingerprint exclusion manifests are exported. Kept documents that are near-duplicates of each other inside a
   partition share one group id, so grouped cross-validation cannot separate them.
2. **HTML entities**: whitespace-like entities (for example `&nbsp;`) act as spaces for boundary detection through a same-length
   shadow string, so all offsets still index the original undecoded text. Other entities (`&amp;`, `&rsquo;`) are left in place.
3. **Abbreviations**: any dotted token ("A.D", "R.J", "Ph.D") is abbreviation-like; such a period never ends a sentence inside an
   unclosed quotation, and elsewhere only when a common sentence starter follows.
4. **List handling**: a numbered marker at the start of a line ("1.") is not a sentence end; a newline after a colon-terminated line
   separates.
5. **Non-content wrappers**: a leading interjection ("Sure!") or a short colon-terminated "Here is/are ..." preface (summary or list) is
   removed from the training examples and listed separately, only as an unbroken prefix of at most two segments and never if a
   hallucination span touches it (one such case was kept). A response with no content left is rejected.
6. **Versioned outputs**: `audit` writes to a new `v3/` folder and refuses to overwrite it.
Regression tests (all failed before the fix, or guard the rule): entities, quoted abbreviation titles, wrappers as prefix only, span-
touched wrappers, list prefaces, numbered lists, internal overlap in both directions, group ids, and a manifest with no source text.
Suite: 621 tests pass (RouteBench 568 + shared core), Ruff clean, `evalops.run_calibration verify` still reproduces the recorded variants.

## Measured availability (v3)
Same downloaded files (RAGTruth commit `c103204b9ce28d6bbad859304bf30de72b8ed8fe`; response.jsonl sha256 `e4c2e4ac...8073`,
source_info.jsonl sha256 `0dffc26e...578b`). Parsed 17,790 responses and 2,965 sources, zero malformed lines; offsets exclusive
(1,796 exclusive-only, 0 inclusive-only, 276 ambiguous-both, 0 neither). Recent News stays excluded (1,890 responses).

| Partition | Documents | Independent groups | Responses | Sentences | Unsupported sentences | Share | All-supported accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 418 | 416 | 2,507 | 14,739 | 1,008 | 6.84% | 93.16% |
| Validation | 104 | 104 | 629 | 3,466 | 240 | 6.92% | 93.08% |
| RAGTruth test | 98 | 98 | 588 | 3,269 | 197 | 6.03% | 93.97% |

- Exclusions: 6 documents matched the protected evaluation inventory (3 exact, 3 near-duplicate; inventory of 2,039 entries,
  1,115 unique documents, including the 30 reserve documents as fingerprints only); 1 TRAIN document matched a VALIDATION document
  (Jaccard 0.582) and was excluded; rejections 12,132 non-summary, 1,890 disallowed source, 2 truncated.
- Wrapper segments removed (all partitions): 999 = 694 summary prefaces, 298 interjections, 7 list prefaces. Effect on train: 15,500
  sentences in the v1 audit fell to 14,739 (wrappers plus the one excluded document).
- Near-duplicate groups: 618 groups, 2 with two documents.
- Splitter check (train only, heuristics not error rates): the fixed 40-response sample has 0 flags (v1: 9). Over all train
  sentences before exclusions: 61 without terminal punctuation, 34 under 15 characters (was 63), 14 containing raw entities, 1 ending
  in a title or initial, out of about 14.8k. Known residual issues: colon-terminated introductions later than the second segment
  ("Best moments include:") are still counted as supported; raw entities remain in the text (any decoding for model input must be done
  on a copy with an offset map, never before applying the original span offsets); abbreviation edge cases remain and their magnitude is unmeasured.
- Class imbalance is severe. Accuracy is not a valid success measure: an all-supported predictor scores about 93%. Derived sentence
  labels are not independent expert sentence annotations; sentence labels come from response spans.

## Frozen training manifest
`.local/ragtruth-ingest-20260930/v3/training_manifest_v3.json` (gitignored: it derives from third-party data; it holds identifiers,
partitions, group ids, labels and original offsets, and no source or response text): 21,474 rows, `manifest_sha256`
`beff83bc88e488a70a1211a6c4a361ad33e6255ec89ba8318389dfe8478006ea` (a hash of the manifest content excluding that field; the file's own SHA-256 begins `5e1132c1cb2aac30`). Companion files: `ingest_report_v3.json`,
`splitter_review_sample_train_v3.json`. Manifest permissions field: `training_use_cleared: false`. All existing RouteBench and upstream
evaluation documents remain excluded, including the reserve set.

## Unresolved (blocks the GPU pilot; technical eligibility is not permission)
- Rights to the underlying CNN/DM article text are not established. The dataset-card licence is a research-use question you must
  decide, and your authorization to download does not grant third-party rights. Recent News stays excluded pending written terms.
- Whether the derived sentence-label rule and the "every sentence supported" response rule reproduce any human policy is unproven.
- A lexical overlap check cannot show absence of semantic overlap or of pretrained-model exposure (the MiniCheck base model's
  Flan-T5 pretraining is undocumented for this purpose).
- The remaining splitter and wrapper residuals above; audit them against a larger train-only sample before relying on sentence counts.

## Proposed bounded pilot (proposal only; needs separate approval, and rights resolved first)
- Model: MiniCheck-Flan-T5-Large (already on disk, MIT) with a small LoRA; baseline = the same model untuned, evaluated identically.
- Data: TRAIN partition only, sampling or loss weights that handle the 6.8% positives; hyper-parameters chosen by document-grouped
  folds using the manifest group ids; VALIDATION only for the single stop/continue decision; the RAGTruth test partition untouched.
- Metrics that expose false passes: recall and precision on the unsupported class, PR-AUC for unsupported, balanced accuracy and
  Cohen kappa on validation, plus sentence-to-response aggregation checked on validation responses. Never accuracy alone.
- Resource limits: at most 2 local GPU-hours in total including the baseline, $0 API, an enforced wall-clock timeout with checkpoint
  resume, seeds and input hashes recorded.
- Predeclared continuation rule (proposal): continue only if validation balanced accuracy beats the untuned baseline by at least
  0.03 with a paired document-bootstrap lower bound above zero and the unsupported-class recall does not fall; otherwise stop.
- Afterwards, and only with separate approval: a frozen evaluation on the RouteBench dev and fresh-document sets as disclosed,
  non-independent comparisons; nothing here promises kappa 0.74.

## Close
Data preparation for this ingestion path is complete and frozen at manifest `beff83bc...`. The next gate is a decision on the article-
rights question, not more code. If it is not resolved, the report is: no permitted training recipe is established.
