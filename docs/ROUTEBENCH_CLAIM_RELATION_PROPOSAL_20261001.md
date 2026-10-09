# Claim-relation candidate (`claim-relation-v1`): offline implementation report and ONE evaluation proposal

Status: implemented and tested offline with mocked inference. No provider call, download, GPU work, held-out evaluation, commit or push. Score status unchanged: v18 dev κ 0.6745, the fresh evaluation did not improve, **0.74 is not reached and no improvement is claimed.**

## Targeted error and plausible failure modes (hypothesis)
Panel examples (AI-assisted, not human gold): intended purpose read as achieved result; authorised amount read as payment; share read as increase; possible read as confirmed; one entity's attribute applied to all listed entities; faithful orthography (U/You) rejected; nearby facts treated as a stated link. The candidate audits **every** assertion (so it can repair false rejections as well as false passes) in ONE structured response.
Failure modes: the model may fabricate relations, over-split a simple claim or cite an irrelevant source span; it may match v18 where annotators used a looser inference policy; abstentions may be frequent; a broad span can hide a second claim (flagged heuristically, not solved).

## What was built (changed/new files)
- `routebench/evalops/claim_relation.py`: prompt, strict structured schema ("not_applicable" required, never omitted), exact-span validation, code-side aggregation, runner (one request per case, no retry), `build-cases`, `compare`, `demo`, `identity`, and a provider path (single HTTP request, fsynced spend journal, refuses without `--confirm` and `--cap`).
- `routebench/evalops/dqdata.py`: permission/provenance-aware workflow. Separate sources, cases, annotations, adjudications, rejected rows and quality report; rights per use (permitted/prohibited/unresolved) for source and annotations; human vs AI provenance (human needs a collection-process id, AI needs model identity); blind packets from an allow-list; overlap through the existing fingerprint index; document-family split checks; training export that refuses unresolved/prohibited rights and incomplete cases; protected-folder guard.
- `routebench/tests/test_dq_claim_relation.py` (41 tests) and `routebench/tests/fixtures/claim_relation_synthetic.json` (19 newly authored synthetic toy cases, marked synthetic; minimal pairs in both directions).
- `docs/ROUTEBENCH_VENDOR_QUOTE_REQUEST_DRAFT.md` (drafted, NOT sent).
Variant id `claim-relation-v1`, content hash `91e6870ac05c0dd622036ef02ae7f7111f461e1a02218255ee9bc5cc08a85164`. v18 config hash `12fc8b550d954359` and prompt-template hash `ef895d791c0bd860` verified unchanged. The original corpus, annotation, split and cache files verified byte-identical to the audit protocol hashes.

## Verified checks
`.venv/Scripts/python.exe -m pytest routebench/tests -q` -> 904 passed. `ruff check routebench` clean. Three mutation checks (dropping the rights check, ignoring span coverage, leaking a label into packets) were each caught by the new tests.

## Runnable commands (offline)
```
PYTHONPATH="packages/core;routebench" .venv/Scripts/python.exe -m evalops.claim_relation demo --out .local/dq-dev/cr-demo-N
PYTHONPATH="packages/core;routebench" .venv/Scripts/python.exe -m evalops.claim_relation identity
PYTHONPATH="packages/core;routebench" .venv/Scripts/python.exe -m evalops.dqdata init|import|report|packets|overlap|export-training --root .local/dq-dev/ws
```
The demo ran end to end: 19/19 synthetic cases COMPLETE, 7 supported / 8 unsupported / 4 uncertain, scripted expectations met 19/19. That checks schema and aggregation only, not model accuracy.

## ONE prospective experiment (needs your approval; not started)
- **Question:** does the claim-relation audit improve on frozen v18 when both use the same model (o4-mini, medium effort) on the same cases? This isolates the method from the model.
- **Population / data:** RouteBench TRAIN + dev, 257 cases, none dropped. Already inspected, so this is a **development ablation, not confirmatory evidence** and not a pass of the original target. The sealed test and the 70-document census are not used.
- **Baseline:** frozen v18 cached verdicts (pass iff every criterion passes).
- **Procedure:** `build-cases --splits train,dev --out DIR` (label-free `cases.json`; labels, groups and baseline in separate files read only by `compare`); `run --backend provider --model o4-mini --cap 6 --confirm RUN-CLAIM-RELATION-PAID` (one request per case, no retries, no prompt or rule edits, no sweep); then `compare --uncertain-as 0`.
- **Fixed decision rule:** supported→1, unsupported→0, uncertain→0 (an abstention never counts as a pass). Abstentions are reported, and a supplementary view excludes them with its own denominator. Full coverage required: any failed or invalid case makes the run INCOMPLETE and withholds the comparison.
- **Reported:** κ for both; false rejections fixed; false passes fixed; new false rejections; new false passes; abstentions; document-grouped bootstrap interval (2,000 draws) of the κ gain; tokens and latency per case.
- **Improvement criterion (separate from the target):** errors fixed exceed new errors AND the interval lower bound is above 0. Passing 0.74 is a different question and needs a never-inspected set (the sealed test under your separate authorization, or newly permissioned data).
- **Budget:** expected about $3.2 (≈257 × $0.0125; v18 itself averaged $0.0075 per case, this prompt has a longer structured output); **hard cap $6.00** in the spend journal including failed attempts; zero GPU. Wall time roughly 40-60 minutes sequentially.
- **Stopping rule:** stop at the first INCOMPLETE or ABORTED terminal and report. If it does not improve, stop; no variants.
- **Blocker:** the OpenAI account returned `credit_balance_exhausted`; at least $6 of credit must be added first.

## Exact approval needed
"Approve one claim-relation-v1 ablation on RouteBench TRAIN+dev (257 cases) with o4-mini, hard cap $6.00 total (single attempt per case, no retries), after I add at least $6 of API credit." Nothing else is requested.

## Remaining limitations
Mocked inference proves schema, validation and aggregation only. Coverage is lexical-span based and cannot prove semantic completeness. Permission records are data fields: a person must collect and verify rights evidence and human annotations in a real process; private-research willingness is not clearance. Fingerprint overlap does not show semantic or pretraining independence. The panel's decisions are AI diagnostics; nothing here was trained or tuned on them. The provider path has been exercised only against a mocked HTTP transport.

## Repairs after review (supersedes the experiment section above where they differ)
1. **Dev is primary.** The 128-case dev result alone drives the gate and any 0.74 statement. TRAIN (129) is diagnostic only; pooled TRAIN+dev is supplementary and never gates (a test reproduces pooled improvement while dev worsens).
2. **Frozen protocol manifest** (`.local/claim-relation-ablation-v1/protocol.json`, file sha256 `8c32146ae66f0cccd5d42c429b080efbc766f23809c545f7f9a64582d40323ce`). It binds by SHA-256: cases.json `8b326a92...4535`, labels.json `29ecdf70...a81da`, groups.json `29ce59e0...acf`, baseline_v18.json `4af849e6...87d2`, splits.json `ac6ef635...a0b3`, case-id digest `ace9654c...647a`; candidate content hash `91e6870a...5164` and code sha `7d3989f6...94bb`; v18 identity `12fc8b550d954359` / `ef895d791c0bd860`; model o4-mini, effort medium; uncertain->0; bootstrap seed 20261101, 2,000 draws; price assumptions $1.10/$4.40 per 1M tokens; cap $6.00 (code refuses any cap above $6). 257 cases = 129 train + 128 dev. Verified before the first request and again before comparison, and anchored by the freeze-time manifest hash (`--protocol-sha256`). Altered, missing, duplicated, extra or label-bearing records are refused.
3. **Fail-fast spending.** The first request error, malformed response, cap exhaustion, quota failure, model mismatch, invalid usage or cost anomaly stops the run at once with no further provider call: terminal ABORTED (request/spend condition) or INCOMPLETE (malformed response), unattempted cases listed, every comparison withheld.
4. **Gate:** dev errors fixed > new dev errors AND the dev document-grouped 95% lower bound of the kappa gain > 0. Measured dev kappa and whether it reaches 0.74 are reported separately. A development result is not held-out confirmation.
5. **Tests:** 931 pass (full RouteBench suite), ruff clean, mocked demo 19/19 COMPLETE. New regression tests reproduce pooled-up/dev-down, altered files after freeze (all five), a malformed early response with zero later HTTP calls, quota/model-mismatch/invalid-usage/cost-anomaly/server-error/cap with zero later calls, and comparison refusing a changed rule, population, run records or manifest. Three mutation checks (disabling fail-fast, gating on pooled, ignoring the rule check) each fail the new tests.

## Exact commands for the single approved run (NOT executed)
```
export PYTHONPATH="packages/core;routebench"; D=.local/claim-relation-ablation-v1
.venv/Scripts/python.exe -m evalops.claim_relation run --inputs $D/inputs --protocol $D/protocol.json --out $D/run \
    --protocol-sha256 8c32146ae66f0cccd5d42c429b080efbc766f23809c545f7f9a64582d40323ce --confirm RUN-CLAIM-RELATION-PAID
.venv/Scripts/python.exe -m evalops.claim_relation compare --inputs $D/inputs --run $D/run --protocol $D/protocol.json \
    --protocol-sha256 8c32146ae66f0cccd5d42c429b080efbc766f23809c545f7f9a64582d40323ce
```
Not requested yet: no API credit, no spend, no run. The OpenAI account still returned `credit_balance_exhausted` at last use.
