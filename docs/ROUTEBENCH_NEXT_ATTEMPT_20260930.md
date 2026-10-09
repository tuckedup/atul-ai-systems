# Next attempt after the completed LoRA pilot

Prepared 2026-09-30. Status: code and 128-case inputs prepared; new model download
and inference await the user's answer to the one-hour screening question.

## Decision

Screen the published FactCG-DeBERTa-v3-Large checkpoint once before considering
further training. This is a hypothesis about a different verifier, not evidence
that kappa 0.74 will be reached. This checkpoint has not been evaluated here.

The previous staged 40-hour authorization made extension conditional on a passing
pilot. That condition failed. A new one-hour experiment was proposed explicitly;
preparation does not itself approve GPU use. API budget remains $0.

## What the completed pilot establishes

Local source: `.local/ragtruth-ingest-20260930/pilot_run_v2/predictions_v1.json`,
the corresponding result and authoritative terminal files.

- Valid completed experiment: 1,050 training sentences, 1,446 validation sentences
  across 45 documents. There were 103 unsupported validation sentences (7.12%).
- At the preselected threshold 0.95, tuning corrected 3 decisions and introduced
  27 errors. At the native threshold 0.5, it corrected 5 and introduced 1.
- Mean absolute support-score change: 0.0043376433; baseline/tuned Pearson
  correlation: 0.9996626841. The model changed, but mostly preserved its ranking.
- Balanced-accuracy gain 0.00451, paired interval [-0.01068, 0.02502]. This failed
  the declared continuation gate. It does not establish an architectural ceiling
  or prove that longer training, another recipe, or another model cannot work.
- The selected objective was balanced accuracy, not kappa. On this imbalanced
  population a threshold can improve rare-error recall while producing many false
  alarms and low kappa. Changing the target population or selecting on validation
  labels now would not rescue the original experiment.
- These are news sentences. Original RouteBench includes dialogue sentences and
  CNN summaries under published annotation policies. Transfer remains unmeasured.

## What is being reused from published work

FactCG uses synthetic graph-based, multi-hop examples to train a grounded factuality
classifier. The released DeBERTa model is approximately 0.4B parameters. The authors
publish both tuned-threshold and fixed-0.5 results. Those results use balanced
accuracy; none certifies RouteBench kappa.

Primary sources checked:

- Paper: https://aclanthology.org/2025.naacl-long.258/
- Implementation: https://github.com/derenlei/FactCG/tree/41f185413312ed9b3904ec03963fe899693498e0
- Model: https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large/tree/0430e3509dbd28d2dff7a117c0eae25359ff3e80
- Published training recipe: https://github.com/derenlei/FactCG/blob/41f185413312ed9b3904ec03963fe899693498e0/train.sh
- GEDD calibration guide: https://github.com/aws-samples/sample-GEDD/blob/main/grounded-evals/docs/cohens-kappa-for-llm-judges.md

The model card and source repo specify MIT. The inference adaptation retains the
license notice in `FACTCG_NOTICE.txt`. The published training recipe names ANLI,
MiniCheck C2D/D2C and CG2C datasets. It is not an example-level overlap certificate;
the checkpoint is eligible for exploratory screening, not an automatic claim of
uncontaminated independent performance. GEDD is a calibration workflow, not a
transferable guarantee of a particular score.

## Frozen screening design

Code: `routebench/evalops/factcg_screen.py`.
Prepared directory: `.local/factcg-screen-v1/`.
Protocol script SHA-256:
`c68d87c15ab1a66945d529335f377b2cb255ee2769c36fb6fa3f60fdc6edce4f`.

1. All 128 existing headline dev cases; no changes to cases or annotations.
2. Model revision above; fp32 inference, one source chunk at a time; no training.
3. Upstream instruction template, NLTK sentence boundaries, chunks capped at 550
   NLTK words where possible; whole candidate in each prompt; max support score
   across source chunks. An individually long sentence stays intact.
4. Every source sentence must survive. Inputs over 2,048 model tokens fail the
   run instead of being truncated. This differs from upstream truncation and is
   explicitly an adaptation. Multi-sentence claims requiring several chunks may
   be disadvantaged; this is a known limitation of the chosen published method.
5. Supported iff score > 0.5, matching the upstream fixed-threshold comparison.
   No threshold sweep or fitting on these dev cases.
6. One-hour wall timeout includes imports/model loading; batch size effectively
   one. Parent records failed/incomplete runs and updates the existing compute
   ledger. Partial scores and log survive.
7. Full coverage, finite scores, unique exact IDs, protocol hashes and score hash
   required before a report. Report kappa, balanced accuracy, confusion matrix,
   corrections/new errors versus frozen v18 and document-bootstrap delta interval.
8. Original held-out evaluation remains a separate decision. Neither a passing
   dev score nor this model's publication clears the existing readiness gate.

Decision after the single run: if it fails to beat v18, preserve the negative result
and stop this candidate. If it improves but is below 0.74, report that shortfall; do
not silently fit a combiner. If it reaches 0.74 on dev, report an exploratory dev
result and prepare a justified independent evaluation, including provenance and
readiness requirements, before any acceptance claim.

## Execution handoff after approval

Use the existing isolated `.local/pilot-venv` for torch/transformers. First check
whether `nltk` and its English `punkt`/`punkt_tab` assets are already installed;
install missing tokenizer assets there only. Download only the model-card,
configuration, tokenizer files and `model.safetensors` at the pinned revision.
Do not download the much larger Lightning training checkpoint. Use
`huggingface_hub.snapshot_download` with the revision and `FILES` whitelist from
the script. Record the returned snapshot path and downloaded file hashes.

From the repo in PowerShell:

```powershell
$env:PYTHONPATH='packages/core;routebench'
# Inputs are ALREADY prepared; do not repeat prepare or change the script after freezing.
& '.local/pilot-venv/Scripts/python.exe' -m evalops.factcg_screen run --model-dir '<pinned snapshot directory>'
& '../wt-forge/.venv/Scripts/python.exe' -m evalops.factcg_screen report
```

The placeholder is intentional: no weights have been downloaded and no local
model path is invented. The screen must start with adequate free RAM. If the
runtime cannot support it, report that rather than starting a training detour.

Verification performed before freezing: 8 focused CPU tests pass; Ruff passes;
real-data preparation verifies the exact 128 cases and complete cached v18
predictions. Real-model inference, tokenizer asset availability, peak GPU memory
and model throughput have not yet been tested.

## Result (executed 2026-09-30, once)
Status COMPLETE, 80 s wall, $0 API, ledger updated. 128 dev cases, threshold 0.5, scores sha256 c1d6505d….
FactCG: κ 0.549, balanced acc 0.762, 26 errors (confusion [[30,19],[7,72]]).
v18: κ 0.675, balanced acc 0.825, 19 errors. FactCG corrected 5, introduced 12.
Paired κ delta (FactCG − v18) 95% CI [-0.321, -0.001].
Outcome per the frozen rule: does not beat v18 → negative result preserved, candidate stopped. No threshold fitting or combiner was tried. Dev only; held-out untouched.
