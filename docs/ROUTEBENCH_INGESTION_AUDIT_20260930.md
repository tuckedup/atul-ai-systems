# RAGTruth ingestion audit and decision — 2026-09-30

## Decision

The user's latest request authorized the bounded local download and audit. These are complete. No further download approval is needed for these two files. **Data preparation: proceed. GPU training: not ready.** This is an AI-assisted technical review, not independent human adjudication or legal clearance.

No provider judging, GPU training, threshold search, relabelling of the existing benchmarks, or model evaluation was performed. This audit produces no new kappa score and cannot guarantee 0.74.

## Reproducibility and scope

- Repository revision of RAGTruth: `c103204b9ce28d6bbad859304bf30de72b8ed8fe`.
- Local files: `.local/ragtruth-ingest-20260930/response.jsonl` and `source_info.jsonl` (gitignored; do not redistribute).
- Download sizes: 21,458,735 and 15,117,971 bytes respectively; 36,576,706 bytes total.
- SHA-256, response: `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`.
- SHA-256, source: `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b`.
- Bounded HTTPS downloads used curl with connection/total timeouts and per-file size limits. The module's unbounded `download` helper was not used.
- Offline command: `python -m evalops.pilot.ingest_audit audit .local/ragtruth-ingest-20260930`, with `PYTHONPATH=packages/core;routebench`.
- Original output preserved at `.local/ragtruth-ingest-20260930/ingest_report.json`; policy `ragtruth-sentence-v1`, splitter `regex-v2`.

The splitter-review sampler was corrected to use training responses only, with a synthetic regression test. Validation and test response text must not be manually inspected to improve preprocessing. Aggregate ingestion counts are not model evaluation. The test partition below is RAGTruth's, not RouteBench's.

## Measured availability

Parsed 17,790 responses and 2,965 sources, with zero malformed lines. Summary sources comprise 628 CNN/DM records and 315 Recent News records. Recent News remains excluded; it was present only because both sources share the downloaded files.

Rejections: 12,132 non-summary responses, 1,890 disallowed-source responses, and two truncated responses. Offset evidence: 1,796 spans match exclusive offsets only, zero inclusive only, 276 both, and zero neither. Exclusive offsets were selected; the 276 ambiguous spans do not individually distinguish conventions.

The protected inventory contains 2,039 entries representing 1,115 unique normalized documents. It includes the existing evaluation corpus, probes, attribution documents, exemplars and all 100 upstream Tofu documents, including the reserve. Only source text/identity was used to build overlap fingerprints, not protected labels or predictions. Six candidate documents were flagged against this inventory (three exact and three near-duplicate) and excluded.

Counts **after external-overlap exclusions, before resolving the internal near-duplicate below**:

| Partition | Documents | Responses | Derived sentences | Unsupported sentences |
| --- | ---: | ---: | ---: | ---: |
| Train | 419 | 2,513 | 15,500 | 1,010 |
| Validation | 104 | 629 | 3,644 | 240 |
| RAGTruth test | 98 | 588 | 3,427 | 197 |

These are technically eligible under the current adapter, not permission-cleared or final training counts. Unsupported training sentences are only 6.52%; an all-supported predictor obtains 93.48% sentence accuracy. Accuracy alone is therefore unsuitable for deciding pilot success. Derived sentence labels are not independent expert sentence annotations.

## Additional problems found

### Internal overlap

A supplemental fingerprint-only check, using the same `ReferenceIndex` and `audit` functions after external-overlap exclusions, found one train/validation near-duplicate pair (five-word-shingle Jaccard 0.5815602836879432):

- Train fingerprint: `c88328381daad1de783d2666401831db17e7bfce96bffcc9d854d2ac9910adce`.
- Validation fingerprint: `7312a8da89d8c746eee9d28b5a44b8fec6aa14e70631e345b8f42bd091493383`.

No train/test or validation/test matches were flagged by this check. This is a lexical-overlap result, not proof of no semantic overlap or pretraining contamination. The main ingestion command currently does not include this internal near-duplicate check; its `cross_split_documents` field is not a substitute.

### AI-assisted training-only splitter inspection

Inspected the fixed-seed sample of 40 training responses (232 derived segments). Heuristic flags were seven segments without terminal punctuation and two shorter than 15 characters; these are not measured error rates. Concrete issues include:

- Response 1004: a quoted title beginning `A.D. The Bible Continues` was incorrectly split after `A.D.`.
- Response 3258: `II.&nbsp;The women` did not split at the apparent sentence boundary because of the HTML entity.
- Responses 1503, 1228, 3646, 2020, 2680 and 3237 contain summary-preface wrappers treated as supported training segments; 1228 also contains a separate `Sure!` segment. These are not substantive source claims.

This review did not adjudicate factuality against source documents and is not a human label review. Finding two concrete boundary failures does not establish an exhaustive splitter error rate.

## Next bounded implementation step for Claude

1. Preserve the current audit as historical evidence. Reuse these local files; do not download again.
2. Integrate internal cross-partition near-duplicate checking into the reproducible audit, with synthetic tests. Use a declared document-level exclusion rule (for example, exclude every training document matching validation or test), not hand-picked exclusions based on model errors. Export full fingerprint exclusion manifests and resulting counts.
3. Specify a general, versioned policy for abbreviation/title boundaries, HTML entities and non-content wrappers. Preserve original character offsets and provenance; do not decode entities in place before using original annotation offsets. Avoid case-ID-specific fixes. Add regression tests, then generate a new versioned audit output and training-only review sample. Do not inspect held-out text to tune these rules.
4. Freeze a training manifest only after the above checks. Keep all existing RouteBench/upstream evaluation documents excluded and all original labels unchanged. Report class imbalance and independent-document counts, not just sentence counts.
5. Record the unresolved underlying-news-article permissions separately from technical eligibility. The prior dataset-card discussion and user authorization are not grants of third-party rights. Recent News stays excluded; do not claim this report clears CNN/DM article training rights. Resolve that question or select a clearly permitted alternative before the GPU pilot.
6. Only then propose the bounded pilot: model/revision, training-only sampling or weights, document-grouped selection, metrics that expose false passes (including kappa and class-specific errors), resource limits, frozen evaluation procedure, and a stop rule. GPU/pilot approval is separate from this ingestion request. No promise of 0.74.

## Resource accounting

Provider judging spend this step: $0. GPU training: none. Agent API-equivalent cost is unavailable because observed token telemetry was not exposed.

## Verification

Independently ran the RouteBench and shared-core test suites: **613 passed**, with five warnings. Ruff passed on the pilot modules and ingestion-audit test. `evalops.run_calibration verify` passed for all four variants present in its current experiment record (v16, v8, v12 and v14); this does not assert verification of every historical experiment file. No Docker-dependent suite or service was invoked.
