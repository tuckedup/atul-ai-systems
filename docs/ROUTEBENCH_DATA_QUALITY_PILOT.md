# RouteBench: data-quality pilot specification

Date: 2026-10-01. Status: PREPARED, NOT PURCHASED OR EXECUTED.

This is an AI-authored specification, not an independent human annotation study.
No new labels have been supplied, no benchmark labels have been changed, and no
vendor, purchase, upload, model call or training run is authorized by this document.

## Decision

Commission a small custom source-groundedness annotation pilot before ordering
bulk training data. The purpose is to establish whether people can apply a clear,
evidence-backed policy consistently on relevant documents. It is NOT a model
evaluation, a promise of kappa 0.74, or a way to replace difficult benchmark labels.

The current evidence does not establish an accuracy ceiling. It shows failure of
the particular configurations tested. Do not buy thousands of generic preference
labels or launch another training run before checking task alignment and label quality.

## Scope for a vendor quote

Deliver 100 cases from 50 NEW independent English-language source documents,
two cases per document, spanning news-style material and meeting/discussion
transcripts. Use only sources with documented permission for the intended use.
Prefer commissioned/owned material or clearly permissioned sources. Do not treat
an accessible URL, public availability or the annotation vendor's contract as
automatic permission to reuse third-party source text.

Two independent human annotators label each case. A third, qualified human
adjudicates disagreements and reviews a random sample of agreements. Annotators
must not be the people who constructed that case's candidate or expected label.
Pay for careful work and documented uncertainty, not agreement with a model.

Suggested composition, fixed before model predictions:

- 80 natural cases from 40 documents: candidate sentences with the original
  summary context retained. Select without filtering for current judge errors.
- 20 deliberately constructed diagnostic cases from 10 OTHER documents:
  meaning-preserving paraphrases and subtle changes to numbers, attribution,
  modality, time or causal relations. Record construction provenance. Human
  annotators do not see the intended label or edit description before judging.
- Do not combine the constructed stratum with the natural stratum to claim
  population performance. Neither stratum is the original RouteBench benchmark.

Stage A: 20 calibration cases from 10 documents (16 natural, four constructed),
double-annotated and discussed. Resolve instructions using documented evidence,
not by forcing agreement. These cases are development material.

Stage B: freeze policy v1 after Stage A, then label the remaining 80 cases from
40 different documents (64 natural, 16 constructed) independently. Any substantive
policy change during Stage B requires a new version and disclosure; it must not
silently change already collected labels. Stage B is a pilot quality check, not
the final held-out model benchmark. Once inspected, it is development material.

## Rights, identity and overlap: mandatory before upload

For every source retain its stable ID, provenance URL or collection record,
full-content SHA-256, version/date, usage terms and the basis for permission to
annotate, train, retain and (separately) redistribute. Record limits explicitly.
Written permission to train must cover both the source text and the annotations;
if unresolved, BLOCK training rather than recording the user's willingness as
third-party clearance. Do not release restricted prior TofuEval-trained artifacts.

Compare document IDs, exact hashes, normalized text and near-duplicate candidates
against existing corpus, train/dev/test, prior 40/30-document evaluations,
70-document census, exemplars, probes and training sources. Use the existing
protected-document index without inspecting held-out labels or errors. Report
the normalization, near-duplicate method/threshold and all reviewed flags.
Near-duplicate checks cannot prove absence of semantic or pretraining overlap.

Group related documents, versions and all derived candidates into one split group.
Do not split sentences of one source across train, development or future evaluation.
Personal/sensitive material requires an appropriate consent and handling review
before sending anything to a vendor. Default storage is private and local.

## Annotation question and decision rules

Question: Does the source support every factual assertion in the marked candidate
sentence? This is source support, not style, usefulness or general-world truth.
The full source is evidence. Neighboring summary sentences can resolve what an
expression refers to but are not independent evidence for the claim.

Use three annotation outcomes: `supported`, `unsupported`, `uncertain`.
Uncertainty is not an automatic failure and must not be forced into a binary label
to improve agreement. Keep every raw response and every case in the audit inventory.

| Issue | Instruction |
| --- | --- |
| Ordinary paraphrase | Accept when it preserves the asserted meaning. Word overlap is not required. |
| Omission | A shorter statement is not wrong solely for omitting unrelated facts. If omission changes the asserted meaning, identify that change. |
| Names/ASR/spelling | Accept an entity variant only when context establishes the same referent; explain the linkage. If not resolvable, mark uncertain, not a guessed correction. |
| Numbers and relations | Check value AND its entity, unit, date and relationship. Matching a number somewhere is insufficient. |
| Time/modality | Do not equate planned, possible, ongoing and completed events. Identify the changed proposition. |
| Attribution/opinion | Distinguish a speaker's belief or allegation from an established fact; distinguish speakers. |
| Inference | Accept direct entailment; do not supply an unstated motive, causal premise or outside fact. Any proposed broader inference tolerance requires a written policy decision, not a leniency knob. |
| Reference resolution | Resolve from the supplied context when unambiguous. Keep a separate ambiguity flag. Do not claim this matches every upstream benchmark's coreference policy. |
| Meta-statements | Judge an assertion about the document against the document's actual content, not external events. |
| Questions | Do not invent an assertion from a pure question, but inspect factual presuppositions. Escalate unclear cases. Record `no_factual_assertion` when applicable. |
| Garbled/incomplete source | Record an input-quality issue and uncertainty if it prevents judgment; never silently delete it. |

These are proposed rules for NEW data. They do not amend the current judge, claim
exact agreement with every upstream dataset's annotation manual, or reinterpret
the original gold labels. During Stage A, compare the policy with authoritative
target-task guidelines and document unresolved differences before finalizing it.

## Evidence requirements

For each factual assertion provide its candidate span, a verdict, a concise
explanation and supporting or contradicting source spans where available.
Record zero-based, end-exclusive Unicode-code-point offsets into the exact stored
text, plus the quoted text. Preserve text before recording offsets.

- Supported: explain how the cited span(s) support the assertion; record explicit
  reference/arithmetic steps where needed.
- Contradicted: cite the conflicting source statement and identify the mismatch.
- Unsupported addition: identify the unsupported detail and describe what relevant
  parts were checked. A contradicting quotation may not exist; do not fabricate one.
- Uncertain: state what information or policy clarification is missing.

Automated quote matching tests only whether the quotation is authentic, not whether
it supports the claim. Lexical overlap or absence is not an entailment test.

## Deliverable schema (separate files; no model predictions in labeler packets)

1. `sources.jsonl`: document/group IDs, text/version hashes, provenance, permission
   evidence, allowed uses, restrictions, overlap-check records.
2. `cases.jsonl`: case/document/group IDs, source hash, candidate text, full summary,
   marked sentence boundaries, stratum, candidate provenance and policy version.
3. `annotations.jsonl`: case ID, pseudonymous annotator ID, `human_independent`
   provenance, policy version, outcome, assertion-level evidence, uncertainty and
   input-quality flags, timestamp and active time. Record any tool assistance;
   do not disguise AI-generated labels as human work.
4. `adjudications.jsonl`: both original annotation IDs, third-person reviewer ID,
   final outcome, reason and evidence. Preserve the original disagreements.
5. `quality_report.json`: counts and denominators for every stage/stratum, rights
   and overlap outcomes, completion, missing fields, evidence checks, uncertainty,
   pre-adjudication agreement and adjudication totals, active time and actual costs.

Use an isolated new-data directory such as `.local/data-quality-pilot-v1/` after
purchase approval. Never write into acceptance annotations, original cases/splits,
cached judgments or calibration bundles. A manifest must account for all 100 cases.

## Quality gates before a larger order

Proposed operational gates, to be agreed BEFORE Stage B, not established universal
standards or guarantees of model performance:

- All 50 documents have acceptable provenance, intended-use permission and
  overlap review. Zero unresolved rights issues in any proposed training export.
- All Stage B cases have two independent annotations; all disagreements and all
  uncertain responses receive adjudication. Missing work is reported as incomplete.
- Offset/quote validation passes for all supplied spans after logged repairs.
  A reviewer checks semantic relevance, not just string matching.
- Blind human quality review of 20 Stage B cases: 10 randomly sampled agreements
  and 10 disagreements/uncertainties (or all such cases if fewer). Record the draw
  and report each stratum separately. At most two reviewed final labels may have
  a material policy/evidence error before repairs; otherwise revise the workflow
  and do not automatically expand the order. Material errors and repairs remain logged.
- At least 85% exact pre-adjudication agreement across the three outcomes on
  Stage B and at most 10% adjudicated `uncertain` outcomes. Report these by stratum
  as well as overall; both gates are provisional pilot acceptance criteria.
- Report pre-adjudication Cohen's kappa as supplementary, including class counts,
  coverage and a document-level interval. Report three-way agreement and separately
  the binary-resolved subset with its denominator. Never suppress uncertainties
  to manufacture a high agreement number. Human agreement is not a model ceiling.

Failed gates trigger diagnosis/rework or stopping, not replacement of inconvenient
cases. Adjudicated uncertainty can remain excluded from a BINARY TRAINING export
only under a declared policy, while remaining in the overall inventory and quality
report. It cannot silently disappear from a final performance denominator.

## Budget and next experiment

Request a fixed-price, capped quote broken down into source rights/collection,
annotation, qualification, adjudication, QA, platform fees and taxes. Time Stage A
before accepting extrapolated throughput for long documents. A rough labor formula
is `100 cases * 2 annotations * minutes_per_annotation / 60 * hourly_rate`, plus
adjudication, source acquisition, screening and fees. No vendor pricing is pinned
by this document; no purchase is approved.

If pilot quality passes, separately propose the size and budget of a larger
document-diverse training set. Freeze document-level development and final
evaluation allocations before model selection. Pilot labels may inform training
only when their rights permit it; do not reuse inspected pilot cases as unseen
evaluation. Include learning curves and a matched baseline to test whether the
new labels actually help, rather than assuming that a larger dataset does.

The original RouteBench acceptance test remains sealed and subject to its existing
readiness gate and separate run authorization. A result on a new custom dataset is
not a pass on that original benchmark. A human-in-the-loop result is not an
automatic-judge result. No kappa target is guaranteed by this purchase.

## Vendor request (ready to send after user approval)

Please quote a two-stage, 100-case source-groundedness annotation pilot across
50 new documents under the attached protocol. Include two independent annotators,
third-person adjudication, evidence spans, source/annotation training permissions,
document-level overlap controls and a private delivery. Separate natural cases
from constructed diagnostics. Report raw disagreement and uncertainty, and disclose
any AI assistance. Do not use our benchmark predictions or target score to select
or revise labels. Provide a capped all-in price, Stage A timing estimate, proposed
annotator qualification process, rework policy and ownership/use terms. This is a
request for a quote, not authorization to collect data or begin paid work.
