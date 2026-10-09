# Pilot readiness checklist, fourth pass: terminal-result correction (2026-09-30)

One bounded correction pass. The frozen experiment is unchanged: configuration `fe8c51a0...`, freeze v3 `b3af4674...`, manifest v5 `1645dd3f...`
(file hashes of v3/v4/v5 manifests and both freeze files checked before and after). No training, downloads, installations, paid calls, label
changes or new experimental variants. AI-assisted engineering; not legal clearance.

## Defects confirmed and fixed
| Defect (reproduced by review) | Fix |
| --- | --- |
| The worker wrote a COMPLETE result before saving the adapter; an adapter-save exception left COMPLETE behind | `finalize.finalize`: write per-example predictions, save the adapter, **verify** the required adapter files (`adapter_config.json`, `adapter_model.safetensors`, non-empty) and hash everything, THEN publish the result atomically. Any failure withdraws the comparison (`baseline`, `tuned`, `paired_gain_bootstrap`, `continue_decision` are removed and listed under `withheld_keys`) and publishes status **FAILED** |
| The parent trusted the mere existence of the result file; a worker that published and then exited with code 7 counted as success | `finalize.validate_worker_output`: the parent checks the exit status, that the process was not killed, that the result parses and has a known status, that `config_hash`, `manifest_sha256` and `freeze_sha256` equal the frozen values, and, for COMPLETE, that every listed artifact exists and matches its recorded SHA-256 and that the comparison keys are present. It then writes its own `pilot_terminal_v1.json` (never overwritten) |
| The command returned 0 whenever the worker was not killed | Distinct exit codes: **0** VALID_COMPLETE, **3** INCOMPLETE (killed or overrun), **4** NOT_FEASIBLE, **5** FAILED (crash, non-zero exit, finalization failure, malformed or mismatched result) |
| Only aggregate metrics were saved | `predictions_v1.json`: per-example baseline and tuned scores for the selection and validation samples with labels, response and group ids and split; the thresholds; the training-sample ids; the group-to-split assignments; and the config / manifest / freeze hashes |
Finalization is also covered by the external watchdog (announced as the `slack` phase), so a hang while saving is killed and yields a non-COMPLETE terminal status.

## Tests added (12 in `test_pilot_finalize.py`, real subprocesses through the real `execute`)
Successful completion validated; adapter-save failure; missing adapter files; non-zero exit after a published result; termination during finalization (watchdog kill, nothing published by the worker,
parent-written INCOMPLETE, exit 3); malformed result; wrong identity; artifact tampering after publication; terminal and result files never overwritten; unit-level withdrawal when the saver raises;
and two report-recomputation tests: the headline numbers (thresholds, sentence metrics, response-level counts, paired bootstrap bound, continue decision) are reproduced exactly from the saved predictions alone, and a doctored score is exposed.
Sanity check of the tests: disabling the exit-code check fails the non-zero-exit test; letting adapter errors escape the handler fails three tests.
Suite: **700 passed** twice in a row (previously 688); the three process-based test files passed 4 of 4 consecutive stress runs; Ruff clean; `evalops.run_calibration verify` unchanged; no worker or test process left running.
On the real files `execute` still refuses today (`PermissionNotRecorded`), creates no result directory, and the GPU is idle; the simulated `dry-run` on the frozen manifest still completes.

## Not done here, deliberately
- The real backend has never run on hardware (bf16 LoRA dtypes, gradient checkpointing with T5, memory at 2,048 tokens, throughput). The feasibility probe measures this and may stop the run.
- **Reloading the saved adapter and checking it reproduces the tuned scores** is a step for after a real run (not implemented; it needs the trained weights).
- The pilot measures balanced accuracy on a length-filtered RAGTruth validation sample (about 103 unsupported sentences from 45 documents). It does not measure the RouteBench target; a RouteBench development comparison, and a later held-out evaluation, would each need their own frozen protocol and separate approval. There is no honest promise of kappa 0.74.

## Exact approvals needed before any GPU run
1. **Permission decision** written by you in `docs/ROUTEBENCH_TRAINING_PERMISSION_STATUS.md` (line `CNN/DM rows for a private research pilot: accept` or `decline`). A recorded accept is your own risk decision, not a rights clearance. If decline or undecided, the answer is that no permitted training recipe is established.
2. **Installation** of exactly one new package into an isolated copy of the existing environment (plan in `ROUTEBENCH_PILOT_INSTALL_PLAN.md`): `peft==0.21.1`, the only package the resolver reports as new.
3. **One run**, from the repository root with `PYTHONPATH=routebench`:
   `python -m evalops.pilot.cli execute .local/ragtruth-ingest-20260930 .local/token_lens.json <model snapshot 96eafd01...> --python .local/pilot-venv/Scripts/python.exe --confirm RUN-GPU-PILOT`
   within 7,200 s at $0 API. Any outcome (VALID_COMPLETE positive or negative, INCOMPLETE, NOT_FEASIBLE, FAILED) is reported as it is; the run is not repeated until a favourable number appears.
