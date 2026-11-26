# RouteBench: task-label LoRA with training-linked evals

A small, real LLM post-training experiment: adapt an open 0.5B causal language
model to the eight `TaskClass` labels already declared in
[`gateway/router.py`](../gateway/router.py). The model generates a label; it does
not execute the request. This is a research component alongside RouteBench,
not an automatically enabled routing policy.

The deliverable is a frozen dataset, completion-only supervised fine-tuning,
validation-selected LoRA weights, paired base/adapter predictions, and a
recomputable quality gate. The experiment targets **evals tied to training**.
It demonstrates local fine-tuning and evaluation, not production serving or
adoption by real users.

## Recorded CUDA experiment

| Held-out metric (40 synthetic requests) | Untouched model | Selected LoRA |
|---|---:|---:|
| Exact labels correct | 2/40 (5%) | 32/40 (80%) |
| Macro F1 | 0.0425 | 0.8275 |
| Invalid outputs | 20 | 2 |

**Promotion gate: FAILED.** The adapter fixed 30
cases and regressed 0, but the declared
zero-invalid-output condition was not met. The gate was retained after seeing
the result; there was no test-driven retuning. This is a completed training
experiment with a rejected candidate, not a production-ready classifier.

Epoch 2 was selected by validation macro F1 (earliest
tie); epochs three and four did not replace it. Training updated
1,081,344 of 495,114,112 parameters.
Training plus per-epoch validation took 100.1 seconds
on an RTX 4060 Laptop GPU; peak PyTorch allocation was
1.60 GiB (not total GPU memory).
A fresh process reloaded a pristine base and then the saved adapter and
reproduced all 80 recorded predictions exactly.

Evidence: [report](evidence/seed42/report.json),
[raw base predictions](evidence/seed42/base_test.json),
[raw tuned predictions](evidence/seed42/tuned_test.json),
[configuration](evidence/seed42/config.json),
[validation history](evidence/seed42/history.json),
[fresh-process replay](evidence/seed42/reload-check.json),
[saved adapter](evidence/seed42/adapter/README.md), and
[new-request demo](evidence/seed42/demo.json).
`source/` captures the exact training implementation. The current CLI adds
integrity-only verification and fresh-process replay; the training loop,
scoring, dataset, and original recorded test results are unchanged.

All eight remaining failures are retained:

| Test case | Required label | Actual output |
|---|---|---|
| `test-classify-04-0` | `classify` | `chat` |
| `test-extract-04-0` | `extract` | `calculate` |
| `test-summarize-00-0` | `summarize` | `chat` |
| `test-code-02-0` | `code` | `sql` |
| `test-tool_use-00-0` | `tool_use` | `sql` |
| `test-tool_use-01-0` | `tool_use` | `chat` |
| `test-chat-00-0` | `chat` | `sql` |
| `test-chat-03-0` | `chat` | `rainy-station-haiku` |

The small baseline follows the strict one-label contract poorly. The large
delta includes learning that format; it is not a general capability result.

## Demonstrate the recorded result without a GPU

From this directory, with Python 3.11+:

```powershell
python -m unittest discover -s . -p 'test_*.py' -v
python experiment.py verify evidence/seed42 --integrity-only
```

These commands use only the standard library. The integrity-only command
returns 0 for an internally consistent record, including this failed candidate.
To enforce promotion quality, omit `--integrity-only`; this recorded run then
returns 1 as expected. CI checks artifact integrity and the gate's unit tests;
it does not present the rejected candidate as passing quality.

Verification checks artifact
hashes, checks that both prediction sets contain the complete frozen test set,
recomputes accuracy, macro F1, confusion matrices, fixes and regressions, and
rechecks the declared gate. This verifies the recorded evidence; it does not
rerun inference. Hashes provide integrity, not independent attestation.

## Reproduce training on an NVIDIA GPU

Prerequisites: Docker with Linux containers and NVIDIA GPU support (Docker
Desktop/WSL2 on Windows), roughly 8 GB GPU memory, and disk space for the CUDA
image and approximately 1 GB of base-model weights. Internet is needed for the
first image/model download. No API key, cloud account, or paid service is used.
The pinned CUDA image is large; it reuses this portfolio's vLLM environment
only as a training runtime. This experiment does not run a vLLM server.

From `routebench/posttraining` in PowerShell:

```powershell
docker build -t routebench-posttraining .
docker volume create routebench-posttraining-hf
$experiment = (Get-Location).Path
docker run --rm --gpus all --ipc=host `
  --mount "type=bind,source=$experiment,target=/experiment" `
  --mount type=volume,source=routebench-posttraining-hf,target=/root/.cache/huggingface `
  -e HF_HUB_DISABLE_XET=1 -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e TOKENIZERS_PARALLELISM=false `
  routebench-posttraining run --output runs/reproduction
python experiment.py verify runs/reproduction
```

Use a new output directory for each run: the CLI refuses to overwrite an
existing experiment. A failed quality gate exits with status 1 **after saving
the results**. Inspect failures instead of changing the test set to obtain a
passing result. Identical software, seed and data aid reproducibility; GPU
floating-point behavior can still vary across hardware.

## Try a new request against both models

Use the same `$experiment` binding and model-cache volume:

```powershell
docker run --rm --gpus all `
  --mount "type=bind,source=$experiment,target=/experiment" `
  --mount type=volume,source=routebench-posttraining-hf,target=/root/.cache/huggingface `
  routebench-posttraining demo --run evidence/seed42 `
  --text 'Write SQL that labels overdue invoices using CASE.'
```

The CLI reloads the saved adapter, validates its weight hash, and prints the
untouched base model's answer alongside the tuned answer. Output is generated
greedily, with no constrained-label decoder or answer-substring matching.
Nothing sends email, runs SQL, or calls external tools.

To replay all held-out predictions in a fresh process without retraining:

```powershell
docker run --rm --gpus all `
  --mount "type=bind,source=$experiment,target=/experiment" `
  --mount type=volume,source=routebench-posttraining-hf,target=/root/.cache/huggingface `
  routebench-posttraining reevaluate --run evidence/seed42 --output runs/reload-check.json
```

This loads a pristine base, evaluates it before attaching LoRA, then evaluates
the saved adapter and compares every prediction with the record. It returns 0
only when both prediction sets match. A fresh output filename is required.

## Experimental contract

- Base: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
  revision `7ae557604adf67be50417f59c2c2f167def9a775`; same base, system prompt,
  tokenizer, and greedy decoding for both conditions.
- Data: 80 human-authored synthetic training intents, expanded using three
  wrappers to 240 rows; 24 validation and 40 test intents use separate wording.
  All eight labels are balanced. No private interview text or customer data
  appears in the dataset. `dataset.py` reproduces the JSONL files; SHA-256 pins
  and split-overlap checks reject accidental changes/leakage.
- Split intent groups **before** wrapper expansion. No exact text, ID, or group
  overlaps across splits. Related tasks and common terminology remain shared:
  these are hand-built examples of the same taxonomy, not a broad benchmark.
- Train only LoRA parameters on `q_proj`, `k_proj`, `v_proj`, `o_proj`; rank 8,
  alpha 16, dropout .05, AdamW at .0003, seed 42, four epochs, microbatch 2,
  accumulation 4 (effective batch 8). Prompt and padding labels are `-100`;
  only answer tokens plus EOS contribute to loss. Overlength inputs fail
  instead of silently dropping answers.
- Select the highest validation macro F1 checkpoint (earliest epoch on ties).
  Reload that checkpoint before final test evaluation. Test examples never
  enter gradient batches or checkpoint selection.
- Scoring: greedy decoding with an eight-new-token budget for both models;
  trim/case-normalize the **entire** output. Anything other than one
  of the eight labels is invalid and wrong. Report macro F1, accuracy, invalid
  outputs, class-level confusion matrices, and paired fixes/regressions.
  A verbose or truncated baseline answer is invalid even if it mentions the
  right category. Gains therefore combine output-format compliance and task
  labeling; they do not isolate semantic reasoning improvement.
- Predeclared demo gate: accuracy >= 80%, macro F1 strictly above baseline,
  more fixed than regressed cases, and zero invalid answers. This is a
  demonstration threshold, not a deployment acceptance criterion.

The LoRA API follows the official
[PEFT quicktour](https://huggingface.co/docs/peft/en/quicktour); model loading
and chat formatting follow the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
The Docker base digest and PEFT version are pinned; `config.json` records the
actually installed versions, source/data hashes, seed, hardware, timings,
selected checkpoint and adapter hash.

## Scope and limits

This provides inspectable evidence of a complete small SFT/evaluation loop,
including a saved, reloadable adapter. It does not establish production
post-training depth, general reasoning improvement, production inference
performance, security against prompt injection, or user adoption. The test
set is only 40 synthetic English requests with five examples per class, authored
alongside training data. Class boundaries are subjective for mixed requests.
There is one seed and one predeclared hyperparameter configuration, no human
annotator agreement study, and no claim of statistical significance.

The surrounding repository has older workspace/setup claims that are not
validated by this experiment. This directory is self-contained and does not
require the portfolio's root `uv sync` or unavailable applications.
