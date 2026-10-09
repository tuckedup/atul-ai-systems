# judge-v19 Phase 1: frozen protocol (W0-W2 done offline; NOTHING EXECUTED)

2026-10-05. Scope of this step, per the approval: offline implementation, synthetic tests, dry run and freeze. **No provider call, GPU inference, download, vendor contact, purchase or stage transition occurred.** Reading local files and CPU-only tests are all that ran (the one non-test read of a GPU stack was `import torch` and `torch.cuda.is_available()` in the MiniCheck virtualenv, which performs no inference).

## Verified protocol hash
`protocol.json` sha256 = **`3d6388088e4b87c0244ed7a29777dcb7a3bdbfa99fffef8ebf3af680f20ceca6`** (`.local/judge-v19-phase1/protocol.json`; `verify` passes offline).
RAGTruth fitting-sample hash (rederived, 64 characters): `7edc80e3ead8e5f6bc308807078e6a3a7a40d9f0c6afcff9d3f7445452d66012`. The 63-character typo in proposal v2 is corrected there (Section 10).
The srcint-eval-v3 protocol (`5cbfa028...`) and all earlier freezes are unchanged.

## Definitions fixed in the freeze
- **A0** cached frozen v18 verdict (score >= 1.0). **A1** majority vote (at least 3 of 5) over each call's ORIGINAL all-criteria-pass verdict at 1.0; it does not loosen the rubric (five scores of 0.8 are five rejections and A1 rejects them; a mean-score-at-0.5 rule is not used anywhere). **A2** logistic model on schema S, which keeps raw scores (mean, all-pass fraction, standard deviation) as features. Regression tests prove A1 and a mean-threshold vote differ (`test_judge_v19_analysis.py`).
- Populations: FIT = 1,050 RAGTruth sentences / 160 groups / 338 unsupported; SCREEN (evaluation-only) = original RouteBench TRAIN 129 + dev 128. Nothing from TofuEval/AggreFact-derived rows is used to fit, select, calibrate or threshold.
- Nested CV, bands, gate and boundary constants are in code (`judge_v19_analysis.py`) and hashed; labels: `fit` never opens RouteBench labels; `screen` opens them only after the model hash is verified (tested).

## Complete call counts and caps
| Stage | Jobs = expected calls | Worst-case attempts (3 per job) | Estimate | **Hard cap** |
|---|---|---|---|---|
| 1a: replicates 2-5 of 257 original RouteBench inputs (replicate 1 = cached judgment) | 1,028 | 3,084 | $8.22 | **$10.00** |
| 1b: replicates 1-5 of 1,050 RAGTruth sentences, after explicit `release-1b`; projection stop after 500 jobs | 5,250 | 15,750 | $42.00 | **$46.00** |
| Phase 1 total | **6,278** | 18,834 | **$50.22** | **$56.00** |
Estimates use $0.0080/call (observed in srcint-eval-v3). Caps bind committed spend (settled charges plus each unsettled reservation at its bound, about $0.047) in an fsynced journal; they are not estimates. Settled campaign spend since 2026-10-01 is about $3.66, so $3.66 + $56.00 = $59.66 is inside the $60 ceiling (no increase requested). Local GPU (separate approval): MiniCheck inference on 1,050 rows plus a 24-row equivalence check, <= 1.5 GPU-h.

## Exact commands (run from `routebench/`; none has been run; every one needs your explicit approval first)
```
# already done (offline):  ..\.venv\Scripts\python.exe -m evalops.judge_v19_phase1 freeze
# Stage 1a (OpenAI, cap $10.00):
..\.venv\Scripts\python.exe -m evalops.judge_v19_phase1 run --stage 1a --protocol-sha256 3d6388088e4b87c0244ed7a29777dcb7a3bdbfa99fffef8ebf3af680f20ceca6 --confirm RUN-JV19-STAGE-1A
# (an ABORTED/INCOMPLETE stage needs an explicit --resume; unsettled reservations need --settle-unsettled)
# Checkpoint: stop and report. Stage 1b exists only after:
..\.venv\Scripts\python.exe -m evalops.judge_v19_phase1 release-1b --protocol-sha256 3d6388088e4b87c0244ed7a29777dcb7a3bdbfa99fffef8ebf3af680f20ceca6 --confirm RELEASE-JV19-STAGE-1B
# Stage 1b (OpenAI, cap $46.00):
..\.venv\Scripts\python.exe -m evalops.judge_v19_phase1 run --stage 1b --protocol-sha256 3d6388088e4b87c0244ed7a29777dcb7a3bdbfa99fffef8ebf3af680f20ceca6 --confirm RUN-JV19-STAGE-1B
# MiniCheck features (GPU inference only, offline weights, MiniCheck virtualenv, own approval):
..\.local\mc-venv\Scripts\python.exe -m evalops.judge_v19_minicheck --work ..\.local\judge-v19-phase1 --protocol-sha256 3d6388088e4b87c0244ed7a29777dcb7a3bdbfa99fffef8ebf3af680f20ceca6 --confirm RUN-JV19-MINICHECK-GPU
# Offline fit, then the one-shot screen (model hash printed by fit):
..\.venv\Scripts\python.exe -m evalops.judge_v19_phase1 fit --protocol-sha256 3d6388088e4b87c0244ed7a29777dcb7a3bdbfa99fffef8ebf3af680f20ceca6
..\.venv\Scripts\python.exe -m evalops.judge_v19_phase1 screen --protocol-sha256 3d6388088e4b87c0244ed7a29777dcb7a3bdbfa99fffef8ebf3af680f20ceca6 --model-sha256 <from fit>
```
The OpenAI stages read the API key from the existing environment/settings; it is never printed.

## Open items found while building (stated, not hidden)
1. **MiniCheck equivalence is unproven.** The program that produced the cached RouteBench MiniCheck features is not in the repository and the upstream `minicheck` package is not installed in the main virtualenv (it is in `.local/mc-venv`, whose `minicheck`, `evalops.checkers` and CUDA availability import cleanly; the model itself was not loaded). The job therefore recomputes 24 cached RouteBench rows first and stops before scoring any RAGTruth row if any sentence split differs or any value differs by more than 0.002. If it stops, the schema cannot be matched and Stage 1b spend would be wasted, so I recommend running the MiniCheck job **before** Stage 1b; that ordering is allowed by the commands above and needs your approval for the GPU step.
2. `load_checker` requires `weights_licence_checked=True`; the job sets it as an operator step. The weights' terms are recorded in `ROUTEBENCH_TRAINING_PERMISSION_STATUS.md` (MIT card; downstream ANLI/synthetic-data terms unresolved; private research only). It is not a clearance.
3. The real MiniCheck path and offline cache location (`.local/ckpts`) are untested; the tests use a fake checker. The offline switches force a stop instead of a download.
4. The runner is serial (as srcint-v3); wall-clock is unmeasured until Stage 1a.
5. Intervals in the screen report are document-grouped paired bootstrap, reported and not gating. The simulation-based operating characteristics are in proposal v2 Section 6.

## Tests
New: `test_judge_v19_analysis.py` (21), `test_judge_v19_phase1.py` (21), earlier `test_judge_improvement_gate_sim.py` (9), `test_judge_improvement_sample.py` (4), `test_judge_improvement_power.py` (7). Full-suite result: see the line appended below by the final run.

**Full-suite result (run after the freeze, code unchanged since): 1343 passed, 6 warnings, 489 s, exit 0; Ruff clean.** The earlier base was 1301, plus 42 new tests for this revision.

## Stage 1a execution attempt (2026-10-05): ABORTED, not resumed
Run from `routebench/` with the exact Stage 1a command. The API key is read from `.env` relative to the working directory, so the key was empty (`Illegal header value b'Bearer '`, a local error raised before any request was sent). The frozen runner recorded a failed row for job `gnd-016e510e0d7fe22d#r2` after its three attempts and stopped (first-error rule). Terminal state ABORTED, 0 of 1,028 jobs ok, committed $0.1401 of the $10.00 cap (three reservations charged at their bound; no provider usage). **Correction to the commands above: run them from the repository root, not from `routebench/`.** A failed row counts as an existing result, so `--resume` would skip that job, spend about $8 and end INCOMPLETE with a missing replicate; it was not used. Artifacts are preserved in `.local/judge-v19-phase1/stage_1a/`. A clean Stage 1a needs a corrective protocol decision from you.

## Stage 1a completion (2026-10-05)
Recovered the empty-token job through `judge_v19_recovery.py` (note, original failed row and journal prefix preserved byte-for-byte), resumed the frozen runner twice after memory-pressure kills (explicit `--settle-unsettled`, stale lock removed only after confirming the PID was dead; the last resume ran as an independent Windows process), and produced `.local/judge-v19-phase1/stage_1a_report.json` with the recovery-aware helper. Terminal state INCOMPLETE (the one preserved failed row), 1,028/1,028 judgments valid, committed $8.1994 of the $10.00 cap, nothing unsettled. Results: see BLOCKERS.md. Stage 1b not released.
