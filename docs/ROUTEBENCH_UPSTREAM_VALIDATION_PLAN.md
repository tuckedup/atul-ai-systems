# TofuEval upstream-dev documents: verified inventory and revised validation plan (2026-09-29)

Status: PREPARATION ONLY. No judging, training, tuning or split changes. Existing labels, splits, caches and results are
untouched (tests 582 pass, lint clean, `verify` reproduces all recorded variants). Stopped for approval.

## 1. What the local data is vs the full upstream dataset
- Local mirror (`raw/aggrefact.jsonl`, 12,949 rows) is a FILTERED, undocumented-provenance subset of the official
  LLM-AggreFact test split (29,320 rows). Its TofuEval part is 15 MediaSum + 15 MeetingBank documents = exactly the
  upstream TofuEval TEST half. All 30 are already in the RouteBench corpus (MediaS 5/5/5, MeetB 3/3/9 across
  train/dev/test).
- Full upstream TofuEval (amazon-science/tofueval) = 50 MediaSum + 50 MeetingBank documents: 15+15 test (the 30
  above) and 35+35 DEV. `document_ids_dev_test_split.json` (saved under `raw/`) matches our annotation CSVs exactly
  (doc-id sets equal; dev and test disjoint). The dev half is NOT in the local mirror or RouteBench.
- Our earlier phrase "30-document mirror" therefore means the upstream test half only; the 70 dev documents are new.

## 2. Terms of use (checked)
- TofuEval repo: MIT-0 licence, but its README states: "Data in the benchmark should not be used in training NLP
  models." LLM-AggreFact card: usable "as an evaluation benchmark... should not be used in pretraining or
  fine-tuning any NLP models." (CC-BY-ND-4.0; the local mirror was fetched from a non-gated copy.)
- MeetingBank (lytang/MeetingBank-transcript and huuuyeah/meetingbank): CC-BY-NC-SA-4.0. MediaSum: authors ask for
  research-only use; the retrieval mirror (nbroad/mediasum) is tagged CC-BY-NC-SA-4.0. Non-commercial/ShareAlike:
  the extracted transcripts stay local (git-ignored), are not redistributed, cite MediaSum and MeetingBank.
- Consequence for USE: the 70 new documents are EVALUATION-ONLY. They must not be used for fitting, exemplars,
  threshold selection or fine-tuning. Flagging honestly: earlier internal work used the existing TofuEval TRAIN
  documents as few-shot demonstrations (v17/v23 packets) and to fit the MiniCheck+v18 logistic combiner. Those are
  arguably "training"-adjacent uses of benchmark data under this term. They were internal experiments, not released
  models; I recommend treating them as ineligible for any released artifact and not extending them. Your call.
- Intended use is compliant: measuring a frozen judge. Research, non-commercial.

## 3. Verified retrieval (no paid service; no full-corpus training download)
- Retrieved from public non-gated files: MeetingBank test.csv (14 MB) and MediaSum test.json (97 MB) -> extracted the
  100 named transcripts (50 MediaSum, 50 MeetingBank), then deleted the bulk files. Output
  `raw/tofueval_upstream/docs.jsonl` (sha256 0b243763a84d07de..., git-ignored). Reproduce: `evalops/fetch_tofueval_docs.py`.
- Identity check on the 30 documents we already hold: MeetingBank 15/15 byte-identical to the mirror; MediaSum 15/15
  identical after normalising non-alphanumerics; 6/15 byte-identical. The 9 MediaSum differences are encoding damage in
  the local mirror (curly quotes stored as U+FFFD) plus a trailing newline. So the retrieval/reconstruction is correct,
  and the existing benchmark's MediaSum texts carry minor mojibake (not corrected; noted).
- Identity was NOT established by checking that summaries are source-consistent (unsupported summaries are legitimate
  negatives); it rests on doc-id equality with the official split file and the 30-document text identity above.
- Overlap of the 70 dev documents: 0 by doc_id and 0 by normalised text fingerprint with any RouteBench corpus
  document, with the test-part documents, and with every few-shot exemplar document.
- Inventory (dev CSVs): MediaSum 1,813 sentences / 35 docs (40-63 per doc, 20% unsupported); MeetingBank 1,624 / 35
  docs (21-60 per doc, 18% unsupported). Documents average ~4.6-4.8k chars; the longest (CNN-340132, 6,042 chars, in the sampled set) is
  above the corpus BUILDER's 6,000-char cap, but the judge prompt never truncates, so it is shown in full. No cropping.

## 4. Reservation status of every new document
All 70 upstream-dev documents are RESERVED for independent validation (evaluation-only). None is permitted for
training, development or tuning. A first validation sample (40 docs) is drawn now; the other 30 documents stay
untouched as a reserve validation set for a later, separately approved frozen candidate. After a document set has been
used to compare candidates it is no longer independent for further tuning; the reserve exists for that reason. The 30
upstream-test documents (RouteBench train/dev/test) and the RouteBench test split are not touched by this plan.

## 5. Revised sampling plan (frozen manifest: `data/tofueval_upstream_dev_sample_manifest.json`,
##    sha256 4906220a26656c279db93831ab5a750c5bed01321c12cbfb900d75150a02d598)
- Primary evaluation (label-blind): simple random sample of 20 of 35 documents per source (seed 20261001), then a
  simple random sample of 8 sentences per sampled document (seed 20261002). No label is read by either stage. Inclusion
  probability = (20/35) x (8/N_doc). 320 rows; realised unsupported share 22% (MediaSum) / 17% (MeetingBank), i.e.
  ~62 negatives. Natural prevalence is kept; report kappa AND sensitivity, specificity, balanced accuracy, because
  kappa depends on prevalence (these documents are ~80% supported vs 43-62% in RouteBench splits).
- Diagnostic stratum (separate, never pooled with the primary): from the same sampled documents up to 6 additional
  unsupported sentences per document (seed 20261003), 200 rows (114 + 86). Within-document inclusion probability
  min(6, M)/M is recorded per row in the manifest before any judging, so weighted estimates are possible. Its results are
  labelled "negative-enriched, diagnostic". Any combined weighted estimate would use Horvitz-Thompson weights
  1/pi with pi = 1-(1-pi_primary)(1-pi_diag), declared here in advance.
- Uncertainty: document bootstrap (documents are the resampling unit).

## 6. Provenance of the earlier test-prevalence projection (correcting my earlier wording)
- The 0.433 test positive rate came from the aggregate of the HUMAN LABELS of the 263 headline test cases, computed
  in my split inventory. I said the projection used "only case counts, not TEST labels"; that was wrong. No judge
  output was ever produced or seen for test, and no decision was based on that rate, but an aggregate of test labels
  was inspected. Counts by subset (MeetB 143/263) come from the split plan.
- The projected 0.627 is arithmetic, not a measurement: v18's DEV sensitivity (0.937) and specificity (0.714) fed into
  the kappa formula at prevalence 0.433, assuming rates transfer unchanged. It ignores subset mix, document variance and
  the test documents' actual difficulty. Do not cite it as a test estimate.

## 7. Proposed run (awaiting your approval; not started)
- Judge v18 (config hash 12fc8b550d954359, unchanged, all-criteria-pass decision rule frozen, no fitting) on the 320
  primary rows. v18 costs ~$0.0079/case, so ~$2.5; requested cap $3.50 including retries, serial-safe, coverage must be
  complete or the result is reported as incomplete. Optional diagnostic stratum (+200 rows, ~$1.6) only if separately
  approved. MiniCheck scoring for comparison is local and free.
- Report: coverage, actual cost, kappa with a document-bootstrap interval, sensitivity/specificity/balanced accuracy,
  per-source results, confusion, and whether false passes still dominate. No re-tuning; RouteBench test stays sealed.
- What it will not do: reach 0.74 by itself, alter the original benchmark, or select a favourable subset.
