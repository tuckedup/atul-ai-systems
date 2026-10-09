# RouteBench overnight result (work order 2026-10-04)

Started 2026-10-04 02:54 local; finished well inside the 8-hour limit. Status directory: `.local/overnight-20261004/` (status.json, cycles.md, commands.log, run_stdout.log, worktree_status_start.txt).

## Headline
- **Stage A (frozen claim-relation-v1, $6 cap): ABORTED after the first request.** The OpenAI account returned HTTP 429 `credit_balance_exhausted`. One run invocation only; no retry, no new output directory, no credit purchase. 256 of 257 cases were never attempted, so **no structured judgments, no comparison and no κ exist.** Dev κ remains v18's 0.6745; 0.74 not reached; no improvement measured.
- Provider spend: the request was rejected, so no usage was billed by the provider as far as can be known; the journal charges the reservation bound ($0.0346863) for it. Observed usage: none. Latency: n/a.
- **Stage B: 4 offline engineering cycles** produced a separately named, never-run candidate `claim-relation-v2` (post-processing repairs only; same prompt). Engineering fixes, not model-quality improvements.
- **Stage C: no GPU job run.** No demonstrated training defect supports a new pilot (evidence below). GPU hours tonight: 0.
- Original labels, cases, splits, cached v18 results, the 60 AI-panel files and the sealed test were not touched. No commits or pushes.

## Verified tonight (actual checks)
- Frozen protocol `8c32146a...323ce` and inputs verify; v18 identity unchanged (`12fc8b550d954359` / `ef895d791c0bd860`); claim_relation.py, audit2*.py, fresh_eval.py hashes equal the values recorded at their earlier freezes (the "changed on disk" notices were not content changes); aisys/llm.py hash equals the recorded one.
- Tests: 88 targeted (claim-relation frozen + data-quality + audit2_v3) passed before the run; full suite after the cycles: **950 passed**; ruff clean.
- Historical (not rechecked tonight): panel results, earlier pilots, fresh-evaluation result (v18 0.5216 -> pipeline 0.5121 weighted κ), campaign log.

## Commands actually run
`claim_relation run --inputs .local/claim-relation-ablation-v1/inputs --protocol .../protocol.json --out .../run --protocol-sha256 8c32146a... --confirm RUN-CLAIM-RELATION-PAID` (once; terminal.json = ABORTED, attempted 1, ok 0, stopped_on credit_balance_exhausted). `compare` was NOT run (requires COMPLETE). Then pytest/ruff, adapter inspection, ledger append.

## Stage B: cycles (details and measurements in cycles.md)
Real-data measurements on the 257 frozen source/candidate pairs: 72% of sources contain newlines, 32% curly quotes, 73% of candidates contain a conjunction.
1. Evidence authenticity was byte-exact: multi-line / curly-quote / repeated quotes were marked inauthentic and demoted supported claims to "uncertain" (counted as not supported). v2 matches after whitespace/quote/dash normalisation, maps back to original offsets, flags ambiguity; authenticity still never rescues an unsupported claim.
2. The single-broad-span demotion fired on any conjunction (73% of real candidates). v2 needs >=2 substantive clauses and fewer evidence spans than clauses.
3. One odd literal (null, "N/A") for a not-applicable field aborted a fail-fast run. v2 normalises unambiguous synonyms (flagged); missing keys and unknown vocabulary stay invalid.
4. Coverage ignored non-ASCII words; v2 uses Unicode word characters.
Each has paired synthetic tests (valid paraphrase and nearby unsupported claim), 19 tests in `test_claim_relation_v2.py` that fail for v1 where noted. Whether any of this helps accuracy is unmeasured: v2 needs its own frozen protocol and approval, and v1's frozen result must be obtained first.

## Stage C evidence (why no training)
- Pilot v2's adapter was inspected: encoder and decoder LoRA B matrices are non-zero (mean norm about 0.1), so gradients flowed and parameters changed; `enable_input_require_grads` is set with gradient checkpointing. The known "no gradient under checkpointing" bug is therefore absent.
- Found: all 24 decoder self-attention `q` adapters have B exactly zero. Cause: scoring uses a single decoder position, so self-attention weights are trivially 1 and q gets no gradient. This wastes about 1/6 of trainable adapters but does not explain the null result.
- The earlier null result (AUC 0.840 -> 0.840, scores correlated 0.9997) comes with a real update and a 2-hour budget; an unchanged retry is not defensible, and removing inert adapters is not a demonstrated fix. Fine-tuning on 1,712 same-annotator TofuEval sentences (2026-10-01) also gave no gain.
- Ledger defect fixed: the 2026-10-01 TofuEval MiniCheck scoring (1,598 s) and fine-tune (8,791 s) were never appended; added retroactively and flagged. **Cumulative ledger: 4.47 of 40 hours** (earlier dev/train MiniCheck scoring not itemised, small).

## Files changed / created
New: `routebench/evalops/claim_relation_v2.py`, `routebench/tests/test_claim_relation_v2.py`, `.local/overnight-20261004/*`, `.local/claim-relation-ablation-v1/run/` (ABORTED record), this report. Modified: `BLOCKERS.md`, compute ledger (append only). Nothing else.

## Unresolved
Whether claim-relation-v1 helps at all; the OpenAI credit blocker; AI-panel disagreement with 23 of 42 reference labels (model error vs policy/reference issue undecided); mock-only evidence for v2.

## ONE recommended next action
Add at least $6 of OpenAI credit yourself, then rerun the single frozen v1 command above after moving the aborted run directory aside (it refuses to overwrite; preserve it as evidence). That is the only step that yields new measured information. Do not start v2 or any training before that result.
