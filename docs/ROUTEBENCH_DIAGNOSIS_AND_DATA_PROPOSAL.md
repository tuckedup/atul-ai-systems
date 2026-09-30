# Offline diagnosis of the failed combiner, and a bounded data proposal (2026-09-29)

No spend, no test evaluation, no new thresholds or fits. Every classifier below is a frozen one (v18 at its
all-criteria-pass cut, MiniCheck at 0.5, the combiner exactly as recorded); TRAIN combiner predictions are
out-of-fold under the recorded folds. Diagnosis only. The unsuccessful combiner result is preserved unchanged.

## Findings
1. False passes dominate. v18 TRAIN: 31 FP / 4 FN (specificity 0.50); DEV: 14 FP / 5 FN (0.71). On DEV the combiner
   fixed 3 false rejections and 1 false pass but introduced 7 new false passes and 1 new false rejection: MiniCheck is
   more permissive than v18 and the TRAIN-selected threshold (0.25) predicts "supported" for 72-75% of cases against
   label prevalences of 0.52 (TRAIN) and 0.62 (DEV). It traded false rejections (rare) for false passes (common).
2. A handful of documents dominates the results. TofuEval MediaS is 5 documents per split (63 cases each in TRAIN and
   DEV); MeetB is 3 documents in TRAIN and DEV. On TRAIN one MediaS document (17 cases, 2 positive) holds 13 of v18's 35
   errors and its 3 worst documents hold 24. Excluding those 3 documents v18's TRAIN kappa is 0.668 (descriptive only,
   not a claim). On DEV the 3 worst documents hold 8 of 19 errors and excluding them gives 0.752 (descriptive only).
   The TRAIN 0.448 vs DEV 0.675 gap is largely a few documents, not a stable difference in judge quality.
3. Subset kappa for v18 is inconsistent between splits: CNN 0.77 (TRAIN) / 0.64 (DEV); MediaS 0.24 / 0.78;
   MeetB 0.58 / 0.32. There is no subset in which the judge is reliably good or bad at this sample size.
4. The effective sample is 30 documents. All 15 MediaS and 15 MeetB documents in the local LLM-AggreFact mirror are
   already in the corpus (MediaS 5/5/5, MeetB 3/3/9 across train/dev/test). The TEST split is 54% MeetB
   (143/263 cases, 9 documents), the subset where v18 is weakest and least measured (DEV: 3 documents, kappa 0.32).
   With v18's DEV sensitivity/specificity (0.94 / 0.71) the implied kappa falls from 0.675 (prevalence 0.62) to
   0.627 (prevalence 0.43), before any subset-mix effect. So the dev figure is probably optimistic for TEST. This uses
   only case counts, not TEST labels or judgments.
5. Label vs judge: consistent with the AI review, disputed labels are a minority of errors; nothing was relabelled.

## Proposal: a fresh, larger validation set from the SAME annotators (needs approval; not started)
The limiting factor is the number of independent documents. The upstream TofuEval annotation files already in
`data/raw/authoritative/` contain 35 additional MediaS and 35 additional MeetB documents in the *dev* CSVs
(1,813 and 1,624 expert-labelled summary sentences; 80-82% supported) that are NOT in the LLM-AggreFact mirror. The CSVs
have labels and expert error memos but no source text.
- Data needed: the 70 source transcripts (public MediaSum interviews and MeetingBank meetings, keyed by `doc_id`).
  This is a NEW DOWNLOAD and needs (a) your approval, (b) a licence/terms check for each corpus, (c) exact-key alignment
  (`doc_id` -> transcript) with an assertion that summaries mention only that transcript's content.
- Overlap checks before any judging: `doc_id` disjoint from the 30 corpus documents (dev vs test files are disjoint
  upstream; verify), normalized-text fingerprint disjoint from every corpus and exemplar document, and no use of these
  documents in the 4/6-example few-shot packets or MiniCheck training data audit.
- Use: as a SEPARATE, versioned validation benchmark ("TofuEval upstream-dev documents"), never merged into the
  original splits and never relabelled. Freeze v18 and its all-criteria-pass decision rule FIRST (no fitting), then
  measure once on a stratified sample so that documents, not rows, are the unit.
- Sample and cost: e.g. 20 MediaS + 20 MeetB documents, up to 8 sentences per document stratified to include all
  negatives (about 320 rows, ~50% negative to limit prevalence distortion; report prevalence and reweighted kappa). v18
  cost is ~$0.0079/case, so ~ $2.5-3.0 (cap $3.50 including retries). MiniCheck scoring is local and free.
- Decision gates: coverage must be complete; report kappa with a document-bootstrap interval, per-subset kappa,
  confusion, and whether false passes still dominate. If v18 is roughly 0.65 with a narrow interval the target needs a
  different mechanism or better labels; if it is much lower, dev was optimistic; either way the test stays sealed.
- What it will NOT do: reach 0.74 by itself, change the original benchmark, or justify picking the subset that scores best.

Alternatives considered: unused CNN documents (279 docs, 94% supported: little information about false passes);
AggreFact-XSum (558 rows, 483 documents, 51% supported, a different annotation guideline -> guideline-mismatch risk);
fine-tuning MiniCheck on 129 rows (rejected earlier: overfitting risk, not the bottleneck).
