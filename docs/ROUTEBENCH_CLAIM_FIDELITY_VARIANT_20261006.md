# claim_fidelity: opt-in judge variant prepared offline (2026-10-06)

**Status: prepared, NOT run.** No provider call, GPU work, download, training, relabelling, exclusion, Stage 1b or held-out access occurred. v18, the original labels, every frozen protocol and the sealed test are untouched. This document is the exact diff, the checks, the remaining issues and one bounded evaluation proposal. It does not show improvement, and it does not show kappa 0.74.

## Corrections to the earlier audit
- The existing summarization rubric **already** prohibits modal strengthening and stronger generalisations: `claims_entailed` fails "a generalisation stronger than the document supports ('some' becoming 'all', 'considered' becoming 'decided')" and a boundary example pins "will expand" versus "is considering expanding" as FAIL. My earlier statement that no criterion asks about hedging or strength was wrong, and `docs/ROUTEBENCH_EXPERT_EXPLANATION_AUDIT_20261006.md` is corrected.
- So the judge sometimes **fails to apply instructions it already has**. `claim_fidelity` is an explicit attention mechanism for those distinctions (plus opinion attribution, predictions, disputed reports and tense), not a previously missing rule, and it might also raise false rejections.
- Fidelity is defined by the assertion's **evidential status in the document, not by the speaker's authority**. There is no blanket exemption for officials: predictions, disputed claims and uncertainty stay qualified whoever speaks, and an ordinary unqualified factual report may be restated plainly.

## What exists
v18 (`v18-reasoning-o4mini`) is unchanged. The new variant `v24-claim-fidelity-o4mini` is `dataclasses.replace(v18_config, variant_id, rubric_dir, notes)`: same model, reasoning effort, mode, prompt scaffolding and decision rule; the only functional difference is the rubric directory.
| identity | value |
|---|---|
| v18_config_hash | `12fc8b550d954359` |
| v18_rubric_hash | `d72496fbb2991071` |
| v18_prompt_template_hash | `ef895d791c0bd860` |
| claim_fidelity_config_hash | `4c9c23cd3b42e03e` |
| claim_fidelity_rubric_hash | `f47d0ab5d07233a8` |
| claim_fidelity_rubric_version | `summarize@1.1.0-claim-fidelity-v1` |
| claim_fidelity_prompt_template_hash | `ef895d791c0bd860` |
The default rubric bundle hash (`a615d70cb0cf0c8a`) and every existing module are unchanged, so all frozen protocols still verify. No existing file under `routebench/` was edited.

## Exact diff
The repository is not a git repository, so this is the full change set: five new files (below) and the unified diff between the unchanged v18 rubric and the new one. `summarize.yaml` in `rubrics/` is byte-for-byte untouched.
| new file | sha256 (first 16) |
|---|---|
| `routebench/evalops/claim_fidelity.py` | 3c1125c7d798ad8b |
| `routebench/evalops/rubrics_claim_fidelity_v1/summarize.yaml` | f5fac35ccf447aed |
| `routebench/evalops/fixtures/claim_fidelity_regression_v2.json` | 1565ea1acf2357ee |
| `routebench/tests/test_claim_fidelity.py` | 3983cc5a7d9c8aa6 |
| `routebench/tests/test_claim_fidelity_fixture.py` | 7657b25326779890 |

```diff
--- routebench/evalops/rubrics/summarize.yaml (v18, UNCHANGED)
+++ routebench/evalops/rubrics_claim_fidelity_v1/summarize.yaml (new)
@@ -1,4 +1,8 @@
+# claim-fidelity-v1: a copy of summarize.yaml 1.0.0 with ONE added criterion (`claim_fidelity`), used only by the opt-in variant in claim_fidelity.py.
+# Everything else (pass_definition, the four original criteria, boundary examples) is byte-identical to the original; the default rubric directory and v18 are untouched.
+# The existing rubric already prohibits modal strengthening and stronger generalisations (claims_entailed, and the 'will expand' boundary example); this criterion is an explicit
+# attention mechanism for those distinctions, not a previously missing rule. No benchmark sentence, source or published expert explanation appears in it.
 id: summarize
-version: "1.0.0"
+version: "1.1.0-claim-fidelity-v1"
 task_class: summarize
 evidence_required: true
@@ -71,4 +75,28 @@
       attribution to a speaker or source that the document does not make.
 
+  - id: claim_fidelity
+    weight: 2
+    question: >
+      Does the response keep each assertion's evidential status exactly as the document
+      presents it, instead of turning a qualified statement into a plain one?
+    pass_when: >
+      For every assertion the response keeps what the document signals about its status:
+      an opinion, analysis or evaluation stays attributed to the person giving it;
+      predictions, plans, possibilities and concerns stay predictions, plans, possibilities
+      and concerns; disputed, unconfirmed or approximate statements keep that qualification;
+      tense (past, present, future) and material qualifications or conditions ("some", "may",
+      "reportedly", "if") are kept; verbs and quantifiers are no stronger than the document's.
+      Faithful paraphrase passes, and so does an ordinary unqualified factual report that the
+      document itself states plainly, whoever the speaker is.
+    fail_when: >
+      An assertion drops or upgrades its status: an attributed opinion, analysis, prediction or
+      evaluation is stated as plain fact; a hedged, possible, planned, future, disputed or
+      unconfirmed statement is stated as settled, completed or confirmed; a material
+      qualification or condition is left out; a comparison or tentative description becomes an
+      identity or an accusation; or separate statements are merged into one the document does
+      not make. Judge the evidential status of the assertion, not how authoritative the speaker
+      is: an official's prediction or disputed report must stay qualified just as anyone
+      else's must, and a plainly stated fact may be restated plainly.
+
 boundary_examples:
   - situation: >
```
Docs edited: `docs/ROUTEBENCH_EXPERT_EXPLANATION_AUDIT_20261006.md` (corrected, regenerated), `BLOCKERS.md` (appended). The earlier fixture `claim_fidelity_regression_v1.json` is kept for the record and superseded by v2 (the blanket "authority-reported-fact" guard was replaced by evidential-status guards).

## Checks (mocked end to end; scripted provider double, no network)
`routebench/tests/test_claim_fidelity.py` (20 tests) and `test_claim_fidelity_fixture.py` (4 tests):
- **Identity:** v18's config, rubric and template hashes unchanged; the variant differs from v18 only in `variant_id`, `rubric_dir`, `notes`; the new rubric is the old one plus exactly one criterion; the frozen judge-v19 protocol still verifies and records the same rubric bundle hash.
- **Prompt:** the criterion's id, question, pass and fail text reach the real prompt built by `judge.build_prompt`; the required-ids line lists all five ids; the only line of v18's prompt that changes is that ids line; v18's prompt does not mention the criterion.
- **Required output:** a reply without a `claim_fidelity` verdict is a parse error after the bounded retries (status `parse_error`, no score, no stored criteria), never a pass; a reply that supplies it on the repair attempt is accepted; unknown, duplicate or unreadable verdicts are rejected; the same five-criterion reply is rejected by v18, whose own contract is unchanged.
- **Decision:** all yes scores 1.0 and passes; `claim_fidelity` = no alone gives score 0.8 and fails under the unchanged pass-iff-1.0 rule; each original criterion still decides as before and the critical criterion still zeroes the score; the frozen analysis helper agrees.
- **Storage:** the criterion verdicts and the new rubric version survive a `JudgmentCache` write and reload; a v18 cache entry is never returned for the new variant (distinct keys).
- **Synthetic fixture:** all 35 fixture items (14 violation/control pairs across attribution, hedge, tense, strength, merged claims, predictions and disputes, with official and non-official speakers, plus 7 guards for plain facts, kept attribution and faithful paraphrase) flow through prompt, parse, storage and decision as specified when the scripted double plays a perfect judge. This tests the plumbing, **not** whether a real model makes these judgments.
- **No contamination:** the criterion text shares no 6-token run with any reviewed sentence, source passage or expert explanation and contains no case-specific names; the fixture likewise.

## Actual results (serial, this machine, 2026-10-06)
- Targeted: `test_claim_fidelity.py` + `test_claim_fidelity_fixture.py`: **24 passed** (20 + 4).
- Lint: `ruff check routebench`: **all checks passed**.
- Full suite, run once, serially, as a detached low-priority process after a memory check (2.3 GB free of 15.6 GB; no low-memory event occurred): **1395 passed, 5 warnings, 575.96 s (9 min 35 s)**. The previous full run was 1343 passed; the increase is the offline tests added since (blind-review packet, recovery, Stage 1a report, claim_fidelity). I did not skip or deselect any test, and no test reaches a network.

## Remaining issues
1. **No model has judged anything with this rubric.** The fixture's expected values follow the criterion text, not any model behaviour.
2. **A stricter all-criteria rubric can raise false rejections.** v18 was already false-rejection heavy on fresh documents (37 FR / 5 FP), and any new criterion answered "no" fails the whole response under pass-iff-1.0.
3. **The 27 audited items and the 35 reviewed items are consumed.** The criterion was written after reading them, so measuring on them is descriptive and contaminated; independent evidence needs new documents.
4. **The evidence base is thin:** 8 of the attribution cases come from one document; 8 items have no published explanation; two items depend on an unmade inference-strength decision.
5. `config_hash` includes the rubric directory path string, so the variant's hash is specific to this checkout location; the rubric hash itself is path-independent and is recorded in every judgment.

## One bounded evaluation proposal (needs your approval; nothing is authorised)
**Question:** does adding `claim_fidelity` change v18's behaviour in the intended direction without a net loss, measured where it has not been tuned?
1. **Population (no new data purchase):** the 257 original TRAIN + dev cases only, original unrepaired inputs, original labels and splits, evaluation-only. They are exposed, so this is descriptive and cannot support any benchmark claim; it estimates the direction and size of the behaviour change and the false-rejection cost.
2. **Calls:** exactly 257 calls of the new variant (one per case), single run, no sweep, no threshold search, decision rule pass iff every criterion is yes. Compare against the cached v18 run and against the four extra v18 runs already stored in Stage 1a (no new v18 calls), so the comparison sees v18's own run-to-run spread (single-run kappa 0.62-0.68 dev, 0.37-0.46 TRAIN).
3. **Cost:** about $2.1 (257 x about $0.008, from the observed $0.0078 per call); I propose a hard cap of **$3.00** including retries under the same fsynced reserve-before-launch journal and fail-fast rules as the earlier frozen runs, inside the existing campaign ceiling, with no increase.
4. **Predefined outcomes, reported, none gating spending:** (a) false passes and false rejections separately against both the cached run and the range of the five v18 runs; (b) verdict flips fixing and creating errors; (c) the share of responses failed only by `claim_fidelity`; (d) whether the dev and TRAIN movements agree in direction; (e) document-grouped intervals. Because every number is exposed data, no threshold would be treated as a pass; the result would decide only whether a further test on genuinely new, independently annotated documents is worth proposing.
5. **Freeze first:** the run would use the same protocol discipline (manifest binding input, code and config hashes; the new variant's hashes above; one runner; no autonomous transitions), prepared offline and returned with its hash for a separate approval. Not started.
6. **Stop rules:** stop on the first request error; stop at the cap; a missing `claim_fidelity` verdict is a failed row, never a pass; no second candidate, no wording edits after the freeze.
