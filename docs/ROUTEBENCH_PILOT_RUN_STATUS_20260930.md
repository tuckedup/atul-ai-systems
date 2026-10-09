# Pilot run status and exact next step (2026-09-30)

## Authorisation as recorded from the user's answers
- Source use: **Accept, for a private research pilot only** (no release, no commercial use). This is the user's own risk decision about unresolved article rights; it is not a rights clearance. "Recent News" stays excluded.
- Installation: `peft==0.21.1` into an isolated environment, authorised and done (below).
- Run: the frozen two-hour pilot, once, at $0 API, under the watchdog.
- Scope: staged, up to 40 local GPU-hours in total. Only if the pilot passes its predefined improvement gate AND the adapter-reload check passes may ONE bounded extension use the remaining hours, and only after its recipe, evaluation population, resource allocation and success/stop criteria are recorded. If the pilot fails its gate, is inconclusive, or is infeasible: stop, diagnose, report. No automatic new configurations.
- Integrity: original labels, splits and acceptance-test cases are untouched; no test inference, no readiness override, no case removal.

## Done in this pass (no restricted data used, no paid calls)
- Isolated environment `.local/pilot-venv`: a copy of the existing MiniCheck environment's packages plus exactly one new package, `peft==0.21.1` (package-list diff: one added line). The original `.local/mc-venv` and the project environment are unchanged (package lists identical to before). Torch 2.6.0+cu124 sees the RTX 4060.
- Real-backend smoke test on the GPU stack with a tiny random T5 and synthetic text: 2 real optimizer updates in the probe, training including a partial final accumulation group, adapter saved with `adapter_config.json` and `adapter_model.safetensors`, and a fresh backend reloading the saved adapter reproduced the tuned scores exactly (max absolute difference 0.0). About 8 s. This verifies the plumbing and the reload check only; the numbers mean nothing.
- Cumulative compute ledger (`compute_ledger.jsonl`, 40 h ceiling, $0 API) built into `execute`: it refuses to start if the ledger total plus the run's own cap would exceed the ceiling or if any API spend was recorded, and appends the run's wall-clock and verdict. Ledger now: 0.008 h, $0.
- 782 tests pass (twice); lint clean; calibration identity verified; frozen identifiers unchanged (config `fe8c51a0...`, freeze v3 `b3af4674...`, manifest v5 `1645dd3f...`).

## The one thing blocking the run
`docs/ROUTEBENCH_TRAINING_PERMISSION_STATUS.md` still has the decision line blank, and the runner refuses to start without it. You asked that I not fill it in myself, so I have not. To proceed, replace the last line of that file with:

`- CNN/DM rows for a private research pilot: accept, date 2026-09-30`

(or tell me explicitly to write it for you).

## The run (once the line is in place), from the repository root
```
set PYTHONPATH=routebench
.venv\Scripts\python.exe -m evalops.pilot.cli execute .local\ragtruth-ingest-20260930 .local\token_lens.json .local\ckpts\models--lytang--MiniCheck-Flan-T5-Large\snapshots\96eafd01cee2d16cf81aaa2fb226b14f422a37b3 --python .local\pilot-venv\Scripts\python.exe --confirm RUN-GPU-PILOT
```
Outcome codes: 0 valid completed comparison, 3 incomplete, 4 not feasible, 5 failed. The GPU currently shows about 1.6 GiB in use by other programs; close GPU-heavy applications first, since the feasibility probe measures memory on a card with 8 GiB.

## After the pilot
Report pilot, development and held-out results separately, never presenting a changed benchmark as passing the original target. If the gate passes: verify adapter reload on the real adapter, then record the extension recipe before using further hours. If it does not: stop and diagnose.
