---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- task-classification
---

# RouteBench task-label LoRA

Selected epoch 2 from one four-epoch, seed-42 SFT run.
Base revision: `7ae557604adf67be50417f59c2c2f167def9a775`. Base model license: Apache-2.0;
see the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
Only the LoRA delta is distributed here, not the original base weights.

Eight labels: classify, extract, summarize, code, sql, reason, tool_use, chat.
Use the exact system prompt and pinned base in `experiment.py`; this adapter
does not execute tools or replace the gateway's routing policy.

On 40 synthetic held-out requests: 32/40 correct, macro F1
0.827530, 2 invalid outputs. The promotion gate FAILED.
Do not automatically deploy this candidate. Results combine format compliance
and task labeling; they are not evidence of broad reasoning gains or production
performance. No customer data or private correspondence was used.

See the [experiment README](../../../README.md) for the full protocol, exact
reproduction and inference commands, and limitations. The adjacent raw
[report](../report.json), [configuration](../config.json), and
[fresh-process replay](../reload-check.json) supply the measurements.
