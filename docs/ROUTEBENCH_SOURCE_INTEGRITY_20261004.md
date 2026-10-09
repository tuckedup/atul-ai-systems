# RouteBench source integrity: repaired development copy, not a score claim

## Decision and scope

The owner's request was to continue and remove corrupted samples. Exclusions must
be based on demonstrable text damage, not judge disagreement, disputed labels,
unsupported summaries, low scores, or a desired kappa. Prefer reproducible recovery
to throwing away a recoverable document. Preserve the original acceptance population.

The implemented offline command audits the 129 headline TRAIN and 128 headline DEV
cases only. It never loads annotation files or predictions, invokes models, downloads
anything, or runs acceptance evaluation. Original held-out case records are skipped
before inspecting their text fields. Existing upstream-test transcripts are used for
full-text alignment; this is not a claim that no file containing held-out source text
was read. Reserved upstream-dev transcripts are not used for recovery.

## Measured findings

- All **257 cases retained**; **0 quarantined** under the stated corruption rules.
- **62 cases** have source punctuation restored: 15 TRAIN, 47 DEV, across 6 MediaSum
  documents. Across those six unique documents, 70 opening-curly-quote characters
  correspond to straight double quotes in the local retrieved upstream transcripts.
- No replacement character (U+FFFD) or NUL was found in the four audited text fields
  of the current train/dev corpus. The historical upstream-validation document's
  U+FFFD description is not an accurate description of this current corpus snapshot.
- In the punctuation-only v1 copy, no words, numbers, factual assertions,
  candidates, or labels were changed. A
  punctuation mismatch does not establish an annotation error or an invalid sample.
- Source alignment requires a UNIQUE whole-transcript match with every other
  character identical, except end-of-file newlines for matching. The input's trailing
  newline bytes are preserved. Alphanumeric-only/topic similarity is not sufficient.

### Additional clipped-source finding and final v2 copy

One TRAIN CNN case, `gnd-0d4fa06b89b5aabd`, was clipped by the builder at 6,000
characters. Its full source is 9,368 characters in the existing raw mirror (row 420).
`source_integrity_v2.py` restores it only after checking unique raw row identity,
matching subset, exact candidate equality and an exact whole-source-prefix match.
The extra source text is restored verbatim, not invented. Raw label fields are
ignored; their values are neither inspected nor used in selection or recovery.

The final v2 copy retains **257 cases**, with **63 changed source inputs** (16 TRAIN,
47 DEV): 62 punctuation restorations and one full-source restoration. There are
still **0 quarantines**. V1 and its artifact remain unchanged. This does not imply
that every possible form of corruption was screened: automatic checks cover the
specified encoding markers, structural validation and the known builder's 6,000
character cap. Missing/mismatched source bindings abort rather than guess.

## Implemented safeguards

`routebench/evalops/source_integrity.py` creates a new directory and refuses to
overwrite one. Missing/duplicate case identity, malformed JSON or text fields, and
changes to input files during the audit abort instead of silently shrinking a set.
For future damaged inputs, an unrecoverable replacement character or NUL leads to
quarantine with the original record, split and field-specific reason preserved.
Replacement-character recovery is restricted to one punctuation character against
a unique source; it cannot supply a missing word or number. Empty or unsupported
candidate text is not a corruption exclusion criterion.

Every exported case has a new version-namespaced ID. The manifest records its
original ID and group/split, input/code/output hashes, and every source repair. This
prevents accidentally joining these changed inputs to historical cached judgments.
Group membership remains unchanged. No annotation file or acceptance split is
created; a future scorer needs an explicit original-ID label mapping and an isolated
cache. Do not feed these cases to training merely because their split is TRAIN;
the upstream benchmark's evaluation-only use restrictions still apply.

## Artifacts and reproduction

Final output: `.local/source-integrity-v2-20261004/`

- `cases.jsonl`: SHA-256
  `e4057d79db40f9e0785ca2693094e294d40d1004c97b3271e865ce52a7f8c738`
- `quarantine.jsonl`: empty; recoverable storage, not deletion
- `manifest.json`: per-case mapping, repair provenance, full-content digests
- `punctuation-stage/`: preserved intermediate output and manifest

Earlier punctuation-only output: `.local/source-integrity-v1-20261004/`, unchanged;
its cases SHA-256 is `71e1aa64c065d7ee878bde1a1ea04d258c8c28ad836d915e370fcf093ae79805`.

PowerShell (choose a NEW output directory to reproduce):

```powershell
$env:PYTHONPATH = 'packages/core;routebench'
.venv/Scripts/python.exe -m evalops.source_integrity_v2 --output .local/source-integrity-v2-reproduction
```

Nineteen synthetic regression tests pass, covering exact recovery, refusals to fill in
content/truncation, ambiguity, quarantine, preserving originals, ID isolation,
reserved-document exclusion, malformed inputs and refusing overwrite; v2 additionally
tests exact source-prefix binding, missing/duplicate raw rows, candidate/subset
mismatch, damaged raw sources and raw-label independence. Ruff passes for all four
new Python files. The RouteBench suite passed 1,145 tests before the eight new v2
tests were added; the final combined targeted run passed 19 tests. Do not call this
a full-suite run of the final v2 tree. `evalops.run_calibration verify` passes: original
corpus/splits and all four recorded variant identities remain reproducible.

Before/after SHA-256 checks confirmed original `cases.jsonl`, `annotations.jsonl`,
`splits.json`, and `judgment_cache.jsonl` unchanged.

Two fresh, read-only AI reviews accepted the changes: the punctuation-only v1
review returned ship, and the separate v2 review returned ship with no findings.
The latter independently checked the 257-case mapping, source recovery binding,
output hashes, original-field preservation and all 19 targeted tests. These are AI
engineering reviews, not new human gold labels or evidence of improved kappa.
Exact source-document and source/candidate hashes show no overlap across TRAIN
and DEV in the repaired copy. Neither reviewer exposed token-usage telemetry;
API-equivalent cost is unavailable, not zero.

## Claude handoff

The source-integrity-v2 copy is ready. Read this report and
`.local/source-integrity-v2-20261004/manifest.json`. Do not drop any sample for judge disagreement,
edit original labels/splits, substitute this copy for the original acceptance test,
or reuse cached verdicts for repaired input. Keep typed-veto-v1.1's original
post-hoc development result unchanged (approximately 0.7134); the source repair
has no measured kappa result. Do not send the vendor quote or make paid calls
without a separately scoped approval.
If proposing a comparison, name the new dataset, frozen judge/rule, full retained
population, fresh cache namespace and cost cap; a changed-input development result
is NOT evidence of passing the original benchmark.

The immediate request is complete without deleting any record: there were no
unrecoverably corrupted records to exclude under these rules. These limited
punctuation restorations are not a plausible guarantee of kappa 0.74.

---
## Claude follow-up (2026-10-04): independent verification and a frozen fresh-judgment comparison (NOT run)
No model call, purchase, vendor contact or held-out access. The vendor quote draft remains unsent. The original benchmark files (cases, annotations, splits, judgment cache) and the typed-veto-v1.1 post-hoc dev result (0.7133825) are unchanged.

**Independent re-check of the repaired copy.** The four recorded input hashes still equal the current files; 257 unique repaired ids map one-to-one to originals with unchanged group, split, candidate, subset and every other field; 63 sources differ, all punctuation-only except the one clipped CNN source, which is a pure extension (6,000 -> 9,368 characters); 0 removals. The 63 are 47 dev TofuEval-MediaSum, 15 TRAIN MediaSum and 1 TRAIN CNN, from 7 documents. Effects to keep in view: 47 of 128 dev cases change input, but only 4 of v18's 19 dev errors are among them, and one of the four typed-veto flips (`gnd-406e875c9ec93511`, "This compromised national security...") is among them, so under the veto's audit-to-case binding its old audit is correctly refused for the repaired text and a fresh audit would be needed. Cached v18 verdicts cannot be reused as judgments of the repaired inputs.

**Frozen comparison `srcint-eval-v1` (dataset `routebench-srcint-v2`)** - `routebench/evalops/srcint_eval.py`, `.local/srcint-eval-v1/protocol.json`, protocol sha256 **`a52d3716d5b6ba71559302a0f0b196eca50725b30bf6f78f1bd7d0416e185c29`**.
- Population: all 257 retained cases (128 dev primary, 129 TRAIN diagnostic), new version-namespaced ids, **0 removals**; cases are never dropped because the judge fails them (failed cases are retained and make the run INCOMPLETE).
- Judge and rule: frozen v18 (`12fc8b550d954359`, o4-mini medium, all-criteria-pass), no threshold search. **All 257 cases are judged fresh** in an isolated result store (`results.jsonl`, own spend journal); cached verdicts are used only as the comparison baseline.
- Why fresh judgments for the 194 unchanged cases too: they are byte-identical inputs, so any verdict that flips on them is the judge's own run-to-run noise. The report compares the verdict-flip rate on the 63 changed inputs with the 194 unchanged ones (exact Fisher test) so a repair effect can be told apart from noise.
- Reported: dev first (κ, accuracy, false passes, false rejections for cached-original vs fresh-repaired, paired document-bootstrap interval of the κ difference), TRAIN diagnostic, dev subsets, flips that fix or create errors. Completeness (257/257) is checked before labels or the cached baseline are parsed.
- Cost: one OpenAI call per case, **expected about $1.97 (an estimate, not a bound)**, hard cap **$3.00** in the fsynced spend journal (one HTTP request per reservation). **Blocker: the OpenAI account has no credit.** An optional second step (fresh claim-relation audits of the repaired v18 passes on the Claude subscription, then typed-veto-v1.1 on repaired dev) is NOT part of this protocol and would need its own approval.
- Expectation, stated plainly: the repair is punctuation-only on 62 sources, so a large change is unlikely; even 4 fixed dev errors would only reach about 0.747, and a changed-input development result is not evidence about the original benchmark or the 0.74 target.
- Tests: 14 in `test_srcint_eval.py` (population and counts, new ids, original files unchanged, fresh judging with the frozen recipe, labels/baseline never read in `run`, completeness-first, failed cases retained, restart/quota, altered inputs, Fisher). Full suite 1167 passed, ruff clean.
```
export PYTHONPATH="packages/core;routebench"; H=a52d3716d5b6ba71559302a0f0b196eca50725b30bf6f78f1bd7d0416e185c29
.venv/Scripts/python.exe -m evalops.srcint_eval run --protocol-sha256 $H --confirm RUN-SRCINT-OPENAI
.venv/Scripts/python.exe -m evalops.srcint_eval compare --protocol-sha256 $H
```

---
## srcint-eval-v2 (2026-10-05): safeguards fixed offline after the independent review; v1 preserved; NOTHING RUN
Review: `docs/ROUTEBENCH_SRCINT_EVAL_INDEPENDENT_REVIEW_20261004.md`. v1 (`.local/srcint-eval-v1`, protocol `a52d3716...`, still verifies, zero results/journal/terminal) is untouched. No provider call, purchase, download, vendor contact or original acceptance-test access. The comparison is a source-fidelity/reproducibility check, not a route to the original κ >= 0.74: only 4 of v18's 19 dev errors sit in changed inputs and the changes are punctuation-only.

**New frozen protocol `srcint-eval-v2`:** `.local/srcint-eval-v2/protocol.json`, sha256 **`ae8247561e65b22d7a57184671dec6b6c37bdbf9ff8bfabcb0d8ee00da5f54af`**; same 257-case population (digest equal to v1's), same judge (frozen v18), same $3.00 hard cap and ~$1.97 estimate (an estimate, not a bound); run confirmation token `RUN-SRCINT-OPENAI-V2`.

Fixes (module `routebench/evalops/srcint_eval_v2.py`):
1. *Result rows validated before resume and comparison:* exact expected ids; duplicates (conflicting or identical), foreign ids and unknown statuses refused; each row bound to the protocol hash and to the prompt, config and rubric hashes; finite, non-boolean score in [0, 1]; the stored RAW response must re-parse to exactly the stored score and criteria; a non-empty served model that matches o4-mini (an empty served model now makes the judgment "unverifiable", a failed row, instead of being accepted); attempt ids must exist, be settled, belong to that case and not be reused.
2. *Fail-closed financial state:* strict journal reader; unsettled reservations block resume until the operator explicitly settles them AT THEIR BOUND (charges are kept, never discarded); committed spend above the cap writes `REFUSED_OVER_CAP`, is permanent, and can never become COMPLETE; COMPLETE needs validated rows re-read from disk + a clean journal; an ABORTED/INCOMPLETE terminal needs an explicit `--resume`; a COMPLETE terminal is final; one runner at a time via an exclusive lock file; terminal written atomically and fsynced; `compare` runs every one of these gates BEFORE parsing labels or the cached baseline.
3. *Dependencies frozen:* the freeze now hashes `audit2_v3.py` (token bound and usage validator), `multi_judge.py` (loader), `dataset.py`, `run_calibration.py` and `srcint_eval.py` in addition to the matched-comparison stack. (`matched_v16_v18.py` itself, with its own unrun freeze `281b169d...`, was deliberately not edited; the stricter judge function lives in v2.)
4. *Held-out rows excluded before parsing:* `multi_judge.load_cases` is now a bounded loader (case and annotation rows are matched to a split from raw group/case ids and held-out rows are never validated or resolved); guard tests fail if a held-out case or annotation reaches typed validation; its output is identical to the old loader's for TRAIN/dev (257 cases).
5. *Statistics corrected:* the pooled case-level Fisher test and the "noise control" wording are removed. Verdict flips on changed vs unchanged inputs are DESCRIPTIVE counts only: changed cases come from 7 documents (mostly MediaSum), unchanged cases include other datasets and difficulty, sentences share documents, assignment is not randomised, provider drift is possible. The dev paired document-bootstrap interval describes the whole changed-input pipeline comparison, not the isolated effect of the punctuation repair. A causal attribution would need original and repaired versions of the same changed cases judged contemporaneously (a different, separately budgeted scope; not added).

**Disclosure about the old loader (accurate scope).** Before this fix, `multi_judge.load_cases` validated every annotation row, including held-out ones, and resolved all labels in memory before excluding held-out cases. It was used by `multi_judge`, `claim_relation.build_cases`, `matched_v16_v18.build_population` and srcint v1's freeze; `audit2._population` (frozen with the audit2 protocols, not editable) did the same. No held-out label value was printed, stored, compared or used, and no original test case was judged; but my earlier statements that the sealed test was "never loaded" were too broad for those code paths. Pre-existing project code (`run_calibration`, `stack_minicheck`, `attribution`, etc.) also calls `Corpus.resolved_labels()` over the full corpus; I did not change it.

**Checks:** 38 new tests in `test_srcint_eval_v2.py` including the review's seven regressions adapted to v2 (the review's original file still fails 6 of 7 against v1, confirming the defects are real and v1 was left unchanged); three mutation checks (no duplicate check, no over-cap refusal, no raw-response re-parse) fail the tests; full suite **1205 passed**, ruff clean, `evalops.run_calibration verify`: every recorded variant is reproducible from this tree; original `cases.jsonl`, `annotations.jsonl`, `splits.json`, `judgment_cache.jsonl` hashes unchanged.
```
export PYTHONPATH="packages/core;routebench"; H=ae8247561e65b22d7a57184671dec6b6c37bdbf9ff8bfabcb0d8ee00da5f54af
.venv/Scripts/python.exe -m evalops.srcint_eval_v2 run --protocol-sha256 $H --confirm RUN-SRCINT-OPENAI-V2
.venv/Scripts/python.exe -m evalops.srcint_eval_v2 compare --protocol-sha256 $H
```
Blocker: the OpenAI account has no credit. Waiting for separate run approval.

---
## srcint-eval-v3 (2026-10-05): the review's two remaining fixes; v1 and v2 preserved; NOTHING RUN
Review: `docs/ROUTEBENCH_SRCINT_V2_REVIEW_20261005.md`. No provider call, purchase, download, vendor contact or original acceptance-test access. v1 (`a52d3716...`) and v2 (`ae824756...`) freezes, code and work directories are unchanged (both still verify; neither has a results file or journal). Population (257, digest equal to v1/v2), judge (frozen v18), threshold, $3.00 cap and statistical scope are unchanged.

**New frozen protocol `srcint-eval-v3`:** `.local/srcint-eval-v3/protocol.json`, sha256 **`5cbfa028be3c2cdd648d99f433af1d5e76dd998843a483381c9c3bfa17747fc9`**; module `routebench/evalops/srcint_eval_v3.py`; confirmation token `RUN-SRCINT-OPENAI-V3`.

1. **Served-model identity now comes from the RAW provider JSON.** `aisys.llm._call` builds `ChatResult(model=data.get("model", model))`: a response with no `model` field is credited to the requested model, and a non-string `model` is rejected by pydantic before any caller sees the raw JSON. The new production adapter `http_request_raw` sends the same single request as the project client (a test asserts the request bodies are identical) but reads `raw["model"]` itself. Missing, blank, non-string (123, list, object) and wrong-family (`gpt-4o...`, `claude-opus...`) values produce a FAILED, unverifiable judgment whose attempt charge is retained (usage accounting preserved; a valid-usage test confirms the exact charge), and the first such failure stops the run ABORTED (a provider that omits or changes identity is a systemic problem); a valid `o4-mini-...` value passes through. Quota (429) and 5xx handling are tested to keep their meaning. Older frozen clients and experiments were not edited. The tests drive the real default adapter through the real client path with only the HTTP transport mocked (`inspect.signature(se.run)` default is asserted to be the adapter), including a full `run()` with a missing model field (one call, one failed row, charge kept, comparison withheld).
2. **Numeric-domain validation in the strict journal reader.** Reservation amounts, settlement charges and recorded actual costs must be finite, non-negative, non-boolean numbers (`None` allowed only for `actual`); non-object lines are refused. Eight bad values (NaN, ±inf, -0.01, True, False, a string, null) are rejected for reservations and settlements; the journal file is never modified (bytes compared); a malformed journal blocks a run before any call; valid journals (including a null actual and an unsettled reservation) keep every charge in the totals.

**Checks:** the review's five failing regressions (`.local/srcint-eval-review/test_v2_followup.py`) fail on the preserved v2 module as expected and pass against v3 (adapted copy `test_v3_followup_adapted.py`, the production-adapter assertion kept); 75 tests in `test_srcint_eval_v3.py` (the 38 v2 safeguard tests adapted, plus identity, adapter-equivalence and journal-domain tests); two mutation checks (falling back to the requested model; dropping the money-domain check) each fail 5-6 tests; full suite **1281 passed**, ruff clean, `evalops.run_calibration verify`: every recorded variant reproducible; original `cases.jsonl`, `annotations.jsonl`, `splits.json`, `judgment_cache.jsonl` hashes unchanged.

**Limited exposure of earlier paid runs (disclosure).** Earlier OpenAI-backed runners (`audit2_v3`, `fresh_eval`, `multi_judge`, `matched_v16_v18`, `claim_relation.provider_request_factory`) read `ChatResult.model`, so a response lacking a `model` field would have been credited to the requested model. The executed ones (audit2 v3 train/dev, fresh evaluation, the 63 GPT-5.4 rows) received real provider responses with usage, and the GPT-5.4 rows record dated served models (`gpt-5.4-2026-03-05`); the audit2 and fresh-evaluation row files did not store served-model values, so per-call identity cannot be re-verified from them. This does not suggest any anomaly; it is a verification gap, now closed for srcint-eval-v3 only. `matched_v16_v18` (frozen, unrun) still uses the old reader.

Blocker unchanged: the OpenAI account has no credit. Waiting for separate run approval. When approved:
```
export PYTHONPATH="packages/core;routebench"; H=5cbfa028be3c2cdd648d99f433af1d5e76dd998843a483381c9c3bfa17747fc9
.venv/Scripts/python.exe -m evalops.srcint_eval_v3 run --protocol-sha256 $H --confirm RUN-SRCINT-OPENAI-V3
.venv/Scripts/python.exe -m evalops.srcint_eval_v3 compare --protocol-sha256 $H
```
This stays a source-fidelity/reproducibility check, not a route to the original κ >= 0.74.

---
## RESULT of the approved srcint-eval-v3 run (2026-10-05): ABORTED on insufficient OpenAI credit - no judgments, no comparison
Protocol hash verified (`5cbfa028...`) before the single run. The first request returned HTTP 429 (quota exhausted); the frozen rules stopped the run immediately: 1 request, 0 valid judgments of 257, 0 failed rows, 256 cases never attempted. Terminal status ABORTED. The journal charges the bound for the rejected request ($0.0463 committed of the $3.00 cap; the provider almost certainly billed nothing; no reservation is unsettled). `compare` returns WITHHELD_INCOMPLETE; no label or cached-baseline file was read; no κ, accuracy, interval or flip count exists. Nothing was purchased, no retry or workaround was attempted, and no case was dropped. Original cases, labels, splits, caches and the v1/v2 freezes are unchanged. The run cannot be repeated under this protocol's rules without an explicit `--resume` after credit is added (the ABORTED terminal record, journal and the one spent bound are kept). This result says nothing about performance; the repaired-dev score remains unmeasured and, whenever measured, would not be a pass of the original target.

---
## RESULT of srcint-eval-v3, resumed after credit was added (2026-10-05): COMPLETE 257/257
Protocol hash `5cbfa028...` verified again before the explicit `--resume` (the earlier ABORTED record, journal and its $0.0463 bound were kept). Terminal COMPLETE: 257/257 valid judgments, 0 failed, 1 case needed its second attempt, served model `o4-mini-2025-04-16` on every row, no unsettled reservation, **committed spend $2.0557 of the $3.00 cap** (including the earlier bound). The comparison ran only after validated completion; one comparison, nothing else was run.

**Dev (primary; 128 cases, 46 documents), frozen v18 all-criteria-pass rule, no threshold search**
| | κ | accuracy | false passes | false rejections |
|---|---|---|---|---|
| cached judgments on the ORIGINAL inputs | 0.6745 | 0.852 | 14 | 5 |
| fresh judgments on the REPAIRED inputs | **0.6028** | 0.820 | 17 | 6 |
Paired document-bootstrap 95% interval of the κ difference (fresh - cached): **[-0.201, 0.000]** (2,000 draws, 46 documents). 4 dev verdicts flipped, all 4 creating an error (0 fixing one).
Dev subsets (descriptive): CNN 0.637 -> 0.550 (1 flip), MediaSum 0.776 -> 0.745 (1 flip), MeetingBank 0.323 -> 0.069 (2 flips; 3 documents).
**TRAIN (diagnostic; 129 cases, 40 documents):** κ 0.4476 -> 0.4798, interval [-0.040, +0.074]; 4 flips (3 fixing, 1 creating). All 257: 0.5545 -> 0.5380, 8 flips (3 fixing, 5 creating).
Descriptive flip counts only (no test, no causal reading): 2 flips among the 63 changed inputs, 6 among the 194 unchanged inputs.

**Reading, with the limits stated.** The punctuation repair did not produce a measurable improvement: dev κ fell by 0.07 and the interval touches zero. The repaired-input verdict changes are few (4 on dev) and are as common on byte-identical inputs as on repaired ones (2 of 63 vs 6 of 194; counts only, confounded by document mix), so the fall is most plausibly the judge's run-to-run variation, not the repair. The important consequence is for the baseline: the best measured dev result, v18's 0.6745, was a single run of a non-deterministic judge on inspected cases, and a second run on the same-quality inputs gives 0.60 on dev, so 0.6745 is probably an optimistic draw (regression to the mean). Whatever the true v18 level is, it is about 0.60-0.67 on this dev set and about 0.40-0.52 on fresh documents, i.e. well short of 0.74. This is a source-fidelity/reproducibility check on repaired development inputs; it is NOT a pass of, or evidence about, the original benchmark's κ >= 0.74 target, and nothing here changes any original label, split, case or cached result.
