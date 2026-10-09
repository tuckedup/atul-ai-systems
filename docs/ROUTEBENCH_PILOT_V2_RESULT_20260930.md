# RouteBench LoRA pilot v2 — result (2026-09-30)

**Verdict: VALID_COMPLETE run, BELOW the predefined improvement gate. Decision: STOP. The extended stage is not authorized.**

Run: `ragtruth-lora-pilot-v2`, config `99661c99…`, freeze `e907f3a0…`, manifest `1645dd3f…`, output `.local/ragtruth-ingest-20260930/pilot_run_v2/`
(run 1, `pilot_run_v1/`, is preserved as INCOMPLETE). Watchdog exit 0, no problems, adapter saved and hashed. API spend $0.

## Scope (not the RouteBench benchmark)
MiniCheck-Flan-T5-Large + LoRA (r8) trained on 1,050 RAGTruth CNN/DM sentences; threshold chosen on 552 train-select sentences;
evaluated on 1,446 validation sentences from all 45 sampled validation document groups (label-blind, length-filtered sample;
not full RAGTruth validation coverage). 702 train candidates were over-length (excluded); none in select/validation.

## Feasibility (new rules)
Stratified probe: 0.314 s/ex scoring, 1.43 s/ex training; projected total 3,158 s vs limit 5,760 s (80% rule); all phases fit.
Measured: 4,339 s total (training 1,621; baseline 1,286; tuned 1,352) — scoring ran ~2x slower than projected (system RAM ~1 GB free, VRAM reserved above physical), but within caps.

## Results (validation sentences, unsupported = positive)
| | threshold | bal. acc. | unsup. recall | precision | PR-AUC | κ |
|---|---|---|---|---|---|---|
| Baseline (MiniCheck) | 0.95 (selected) | 0.7681 | 0.806 | 0.187 | 0.4292 | 0.212 |
| Tuned (LoRA) | 0.95 | 0.7727 | 0.835 | 0.181 | 0.4287 | 0.204 |
| Baseline @0.5 | 0.5 | 0.7215 | 0.515 | 0.356 | | 0.367 |
| Tuned @0.5 | 0.5 | 0.7230 | 0.515 | 0.366 | | 0.375 |

Paired document-bootstrap gain in balanced accuracy: +0.0045 (mean +0.0050), 95% CI [-0.0107, +0.0250]. Gate requires ≥ +0.03 with CI lower bound > 0.
Checks: coverage ✓, recall not worse ✓, gain ≥ 0.03 ✗, CI lower > 0 ✗ → `continue = False`.
Directional sanity: AUC (supported scored higher) 0.840 baseline / 0.840 tuned, not below chance.

## Interpretation
Fine-tuning on this in-domain RAGTruth sample did not measurably move the model: ranking quality (PR-AUC, AUC) is unchanged and the small bal-acc difference is within noise.
κ on this sentence-level RAGTruth sample (0.20–0.38 depending on threshold) is far below 0.74; this pilot says nothing positive about the RouteBench κ target,
which was neither run nor changed. The ceiling appears to be the judge/label definition and the data, not the amount of LoRA training in this recipe.

## Status of remaining work
- Adapter-reload verification and the extended stage are conditional on a successful pilot; neither is run.
- Compute ledger: ≈1.56 h of 40 h used (run 1 + smoke tests + run 2); pilot allowance of 2 extra hours respected. Remaining hours are NOT auto-spent.
- Original benchmark, labels, splits, and held-out protections untouched; no original test run.
