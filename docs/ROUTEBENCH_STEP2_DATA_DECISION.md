# Step 2: training-data and base-model decision table (public read-only lookups)

Date: 2026-09-30. Sources: papers read directly (FRANK arXiv 2104.13346, AggreFact 2205.12854, SummaC 2111.09525, MiniCheck 2404.10774,
RAGTruth 2401.00396, AlignScore 2305.16739), and public repo/model/dataset pages and licence files. No downloads of data or model
weights, no paid services, no training or evaluation. Status tags: **[V]** verified in a primary source read in this step;
**[D]** derived by arithmetic from a verified figure; **[U]** unresolved; **[H]** hypothesis.

## 1. Original annotation instructions and the AggreFact-CNN label
| Question | Finding | Status |
| --- | --- | --- |
| How is the CNN binary label produced? | FRANK: 3 MTurk annotators tag each summary sentence (NoE, CorefE, ...); AggreFact took FRANK "directly" from SummaC (AggreFact paper, dataset section); SummaC labels a summary consistent iff a majority of annotators marked it as having no error | [V] chain of 3 papers |
| What counts as an error? | FRANK CorefE = a reference "incorrect or with no clear antecedent"; OutE = information not present in the source article; LinkE = wrong discourse link | [V] FRANK sec 2, Table 1 |
| How does an annotator's sentence tags become a summary verdict? | Not stated in the SummaC text | [U] (SummaC code) |
| TofuEval instructions and how many annotators per sentence | README gives columns only; the paper appendix was not read | [U] |
| Consequence | CNN labels are summary-level majority votes including antecedent and discourse errors; Tofu labels are sentence-level. A source-support-only sentence model differs from the CNN definition by construction (rows 2-3 of the alignment doc) | [V] rule; effect on kappa [H] |

## 2. Candidate training data
| Candidate | Explicit terms | Task / label compatibility | Training splits | Known evaluation overlap | Verdict |
| --- | --- | --- | --- | --- | --- |
| **RAGTruth** (Niu et al. 2024) | Repo LICENSE = MIT (c) 2023 Particle Media [V]. The authors themselves fine-tune Llama-2-13B on its training portion [V]. Underlying article text: CNN/DM training-set articles (Apache-2.0 status on HF **[U]**) and "recent news articles" from a news platform (**rights [U]**) | Summarisation: LLM summaries with span labels: Evident/Subtle Conflict, Evident/Subtle Baseless Information [V]. Baseless information = unsupported by the source, which matches the target's source-support meaning. Labels are response spans, not sentences: a derived sentence label (unsupported iff it overlaps a hallucination span) must be declared. English news; **no dialogue or meeting transcripts** | 943 summarisation instances (628 CNN/DM + 315 Recent News, 6 responses each) [V]; test = 450 instances, 150 per task type [V], so summarisation train ~ 793 instances (~4,760 responses) [D; assumes the 150 summarisation test instances were drawn from the pooled CNN/DM + Recent News set, which the text does not state]. Hallucination present in ~29.8% of summarisation responses (1,686 of 5,658) [D from Table 2] | LLM-AggreFact includes RAGTruth rows (folded into its dev split; this repo's manifest note). RAGTruth CNN/DM documents come from the CNN/DM **training** set [V]; whether AggreFact-CNN/FRANK articles coincide is **[U]** and needs a document-level check after access | **Best available** but domain-mismatched to TofuEval dialogue |
| **FaithBench** (Vectara) | LICENSE = CC BY-NC-SA 4.0 [V]. NC-SA restricts commercial use and requires ShareAlike on adaptations; it does not by itself make the data evaluation-only or forbid training. No separate training statement was found [U], so it is NOT cleared (and not excluded on licence grounds alone) | Consistent / Questionable / Hallucinated (Benign, Unwanted) taxonomy; different label definition | ~16 annotation batches, small [V from repo text, exact count **[U]**] | Unknown | Not cleared; keep out of training until a training statement or permission is found |
| **MiniCheck C2D/D2C** synthetic data | HF dataset card MIT, 14,395 rows [V, previous step]. Claims/summaries generated with GPT-4 / GPT-3.5 [V]; OpenAI output-use terms **[U]** | Binary supported/unsupported; documents synthetic (C2D) or human-written (D2C) | 7,076 C2D + 7,320 D2C [V] | Not an evaluation set | **Already used to train MiniCheck-FT5** [V], so fine-tuning MiniCheck-FT5 on it repeats data it has seen (not new evidence) |
| **ANLI** | CC-BY-NC-4.0 [V dataset card] | 3-class NLI, not grounding | r1 16,946 / r2 45,460 / r3 100,459 [V] | Not an evaluation set | Already part of MiniCheck-FT5 training (21K subset) [V]; non-commercial only |
| **TofuEval / LLM-AggreFact / FRANK-derived rows** | "Should not be used in training NLP models" (TofuEval, LLM-AggreFact) [V]; FRANK annotations' terms **[U]** | Exactly the target task | n/a | These ARE the evaluation data | **Excluded** |

## 3. Did each candidate base model already train on those data?
| Base model | Licence | Trained on (verified) | Notes |
| --- | --- | --- | --- |
| MiniCheck-Flan-T5-Large | MIT (model card) | ANLI 21K subset + 14K C2D/D2C synthetic (35K total) [V MiniCheck paper, model section 3.3]. The paper lists only that ANLI subset and its synthetic data as fine-tuning data [V]; no statement about other exposure of the underlying Flan-T5 [U]; LLM-AggreFact is its evaluation benchmark | Best-documented fine-tuning provenance of the options, but NOT proven disjoint from the evaluation data: the exposure of the underlying Flan-T5-Large pretraining/instruction data is unknown [U]. Reaches balanced accuracy ~0.76 on dev here (kappa 0.55) |
| HHEM-2.1-Open | Apache-2.0 | **Undocumented** [V: card does not list it] | Overlap with evaluation data cannot be excluded; poor base for a clean claim |
| AlignScore | MIT code; checkpoints separate | 4.7M examples from 7 tasks incl. summarisation (WikiHow, etc.); evaluated on FRANK, SummEval, QAGS (Table in AlignScore sec 4) [V] | Exposure to CNN/DM-family data through its NLI/summarisation sets is possible; not audited |
| flan-t5-large/base (untuned) | Apache-2.0 | Flan collection (generic) [U as to summarisation-faithfulness sets] | Would need far more task data than RAGTruth alone provides |

## 4. Recommendation (one pairing, or none established)
**Established as viable in principle, not yet cleared:** continue fine-tuning **MiniCheck-Flan-T5-Large** (MIT; provenance documented and disjoint
from the evaluation sets) with a small LoRA on the **RAGTruth summarisation training portion**, using a declared derived sentence label.
Why this pairing: it is the only candidate with a permissive repository licence, a source-support label definition, human span
annotations, published training use, and no dependency on the excluded benchmarks; the base model's documented fine-tuning data are known, but its underlying pretraining exposure is not.
Why it may fail: domain mismatch (news vs dialogue transcripts), derived sentence labels, one LLM-generation era, and unverified
article-rights and possible CNN/DM document overlap with AggreFact-CNN. Whether it changes kappa at all is **[H]**, not evidence.
Sentence-to-summary aggregation for CNN cases must be declared in advance ("summary supported iff every sentence supported", matching
the FRANK/SummaC any-error rule) and validated on RAGTruth held-out responses; it must not alter the benchmark's cases or labels.

**Not established:** any pairing that includes TofuEval, LLM-AggreFact, FaithBench-for-training, HHEM (opaque provenance) or that
claims a TofuEval-domain in-distribution training source. None of the surveyed sources contains labelled dialogue-summary
groundedness data cleared for training.

## 5. Proposed download, sizes, audit and permissions (STOPPED; nothing downloaded)
- **Exact proposal:** clone/download `ParticleMedia/RAGTruth` at commit `c103204b9ce28d6bbad859304bf30de72b8ed8fe` (2024-12-02) and use
  only `dataset/response.jsonl` (21,458,735 bytes) and `dataset/source_info.jsonl` (15,117,971 bytes), about 36.6 MB in total, into
  an ignored local directory. No model weights: MiniCheck-Flan-T5-Large (5.9 GB) is already on disk.
- **Order of checks (terms first, as instructed):** (1) record the licence text and the article-source terms for CNN/DM and the
  "recent news" platform, and get your decision on the unresolved rights; (2) only then download; (3) freeze the split (RAGTruth's own
  train/test flag, plus a document-grouped validation partition of TRAIN, seeded); (4) exact and normalised-text-fingerprint overlap
  of every RAGTruth source document against all RouteBench corpus, probe, attribution, exemplar and upstream-dev/reserve documents
  (fingerprints only for the reserve set; no labels, predictions or errors read); (5) drop overlapping documents and record them.
  Automated matching cannot prove semantic overlap or base-model contamination.
- **Permissions needed from you:** (a) approval to download those two files after the terms review; (b) a decision on the
  news-article rights question; (c) a decision that the derived sentence-label rule and the summary aggregation rule are acceptable
  as declared; (d) later, separate approval for any GPU pilot (proposed ceiling 2 local GPU-hours, $0 API).
- **Stop rule for this step:** if overlap with AggreFact-CNN or the reserve set is material, or the rights terms are not acceptable,
  the pairing is dropped and the report says no training recipe is established.

## 6. Corrections and source-rights check (2026-09-30, offline implementation round)
- Corrected: MiniCheck's fine-tuning data are documented, its full exposure is not; CC BY-NC-SA is not evidence of "evaluation-only";
  "every sentence supported => summary supported" is a proposed aggregation rule that has not been shown to reproduce FRANK's human
  policy (FRANK: any sentence with a majority-flagged error makes a summary incorrect, which is why the rule is plausible, not proven).
- RAGTruth schema read from the pinned README: `response.jsonl` = id, source_id, model, temperature, labels[], split (train/test),
  quality (good / incorrect_refusal / truncated), response; label objects = start, end, text, label_type, meta, implicit_true,
  due_to_null; `source_info.jsonl` = source_id, task_type (QA / Data2txt / Summary), source (e.g. CNN/DM, Recent News), source_info
  (string or dict), prompt. Whether offsets are half-open or inclusive is not stated (the adapter refuses to guess; see spec).
- Source rights (unresolved items are excluded provisionally, and nobody's approval of this document counts as legal clearance):
| Source | Evidence | Status | Missing permission |
| --- | --- | --- | --- |
| Repository text/labels | MIT (c) 2023 Particle Media [V] | Usable subject to MIT terms | none for the labels themselves |
| CNN/DM articles | HF dataset card: "released under the Apache-2.0 License"; articles by CNN/Daily Mail journalists [V]. Whether that licence covers the underlying article copyright is not addressed [U] | Provisionally allowed (default `allowed_sources={"CNN/DM"}`) | Your acceptance of the dataset-card licence for research use |
| "Recent News" articles | Paper says only "recent news articles from a prestigious news platform" [V]; no permission or licence located [U] | **Excluded provisionally** (rejected as `source_not_allowed`) | Written terms from the platform, or the authors' confirmation |
- Effect on the estimate: the ~793-instance figure included Recent News; eligible CNN/DM-only train instances are fewer and unknown until
  the data are inspected. The real checkpoint is the count of eligible, non-overlapping training documents after the overlap audit.

## 7. Offline implementation status (no downloads; synthetic fixtures only)
- `routebench/evalops/pilot/ragtruth.py`: adapter with a declared, versioned label policy (`ragtruth-sentence-v1`) in its module docstring.
- `routebench/evalops/pilot/audit.py`: exact + near-duplicate document audit; indexes store hashes only (usable for the reserve set).
- Tests `test_pilot_ragtruth.py` (14) and `test_pilot_audit.py` (6) pass; three mutations (missing labels read as supported,
  `implicit_true` read as supported, partitioning by example instead of document) each make a test fail.
- Smallest proposed download unchanged: the two RAGTruth files at commit `c103204b...` (36.6 MB). Not downloaded. Run
  `probe_offset_convention` on it first, before any conversion, to fix the offset convention once.

## 8. Adapter defects found by independent review, fixed (2026-09-30)
Each was reproduced and given a regression test that FAILED on the previous implementation and passes now (synthetic fixtures only):
1. Train/test overlap could be missed when the test copy of a document was later rejected (truncated, missing labels, bad offsets).
   Document split membership is now recorded from every resolvable Summary response before any quality/label filtering.
2. Duplicate response ids produced conflicting example ids and a silently overwritten response label. Every record sharing an id is
   now rejected (`duplicate_response_id`).
3. The sentence splitter cut "Dr. Smith ..." into fragments. Splitter regex-v2 handles titles, initials, dotted and common
   abbreviations, quotations, line wraps, blank lines and list markers, keeping original offsets. Known limitation documented in the
   module (an abbreviation followed by a capitalised non-starter word is not split; the resulting label bias is unmeasured, not small).
Suite: 607 tests pass, lint clean, `verify` unchanged. No downloads, training, paid calls or evaluation. Rights and the download
approval remain open, as before.

## 9. Follow-up fix: newline handling (2026-09-30)
Independent review found that single-newline handling bypassed the abbreviation rules ("Dr.
Smith", "The U.S.
economy grew.") and
treated a bare closing quotation mark as a sentence end. The newline branch now applies the same title/initial/abbreviation test as
the single-line path, using the first word of the next line, and requires actual terminal punctuation (optionally followed by
closers). Regression tests added; blank-line, list-marker and punctuation-plus-newline behaviour is unchanged. The limitation above
remains unmeasured; auditing the splitter against the approved data is a required step before deriving training labels.

## 10. Ingestion audit prepared (2026-09-30; not run on real data)
`routebench/evalops/pilot/ingest_audit.py` implements the bounded audit as two commands: `download DIR` (the two pinned files at
commit c103204b..., to be run ONLY after explicit approval) and `audit DIR` (offline). The audit records file hashes; fixes the offset
convention from a probe and refuses to guess when the evidence is not decisive; applies the declared label policy; excludes Recent News
(downloaded but never eligible); checks exact and near-duplicate overlap against a fingerprint index of every local evaluation, probe,
attribution, exemplar and upstream document (no labels, predictions or errors read); reports documents / responses / sentences per
partition, class balance, every rejection reason and malformed-line count; and writes a fixed-seed splitter audit sample for human
review (heuristic flags, not an error rate). Verified on synthetic files only. Deliverable after approval: eligible, non-overlapping
counts, not a model score. Third-party article rights remain unresolved and are not settled by running the audit.

## 11. Authorized real-data audit completed (2026-09-30)

The later user request authorized the bounded download and local audit, now completed. Sections 7-10 above describe earlier
checkpoints, not the current download status. See [the ingestion audit and next-step decision](ROUTEBENCH_INGESTION_AUDIT_20260930.md).
The initial external-overlap-filtered counts are 419 training documents and 104 validation documents, but an additional internal
near-duplicate pair and training-only splitter findings require cleanup before freezing a training manifest. This is AI-assisted
review, not independent human adjudication. No training or provider judging occurred, and third-party article permissions remain
unresolved; user acceptance alone does not grant those rights. The RouteBench/core suites now pass 613 tests.
