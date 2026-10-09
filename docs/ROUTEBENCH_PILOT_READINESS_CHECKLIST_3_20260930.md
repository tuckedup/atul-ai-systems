> **SUPERSEDED IN PART** by `ROUTEBENCH_PILOT_READINESS_CHECKLIST_4_20260930.md`: result publication, parent validation, exit codes and per-example artifacts were corrected in the fourth pass. Kept as the third-pass record.

# Pilot readiness checklist, third pass (2026-09-30)

Closes the three execution-path gaps found in review. Supersedes the configuration and freeze identifiers of the second-pass checklist
(`ROUTEBENCH_PILOT_READINESS_CHECKLIST_20260930.md`), which stays as a record. Existing manifests (v3, v4, v5), the original ingestion
report and the earlier freeze file are untouched (file hashes checked). No installation, no GPU work, no paid calls, no downloads, no
original-test evaluation. AI-assisted engineering; not legal clearance and not human adjudication.

## Gap 1: the real command now runs (when permitted)
`python -m evalops.pilot.cli execute DIR TOKEN_LENS MODEL_DIR --python <isolated interpreter> --confirm RUN-GPU-PILOT`
- Refuses unless ALL hold: confirmation token given verbatim; the permission document records an explicit "accept"; the manifest, configuration
  and frozen samples verify; wiring self-check passes; an existing isolated interpreter is named; the result directory does not exist.
  Checked on the real files today: without the token it stops ("refusing: pass --confirm ..."); with the token it stops at
  `PermissionNotRecorded`; no result directory is created; the GPU stays idle.
- Then it starts the worker (`cli child ...`) as a supervised CHILD PROCESS with offline model loading forced, and the worker: re-derives every needed text from the local
  RAGTruth files and refuses any difference from the manifest (file hashes, partition, offsets, label, document fingerprint), builds the real backend
  (`TorchBackend`, pinned-revision directory only), runs the full `run_pilot` path, writes `pilot_result_v1.json`, `environment.json` data inside the result,
  `progress.json`, `child.log`, and the LoRA adapter (never the base weights).
- A worker that is killed or crashes leaves a persisted result with status INCOMPLETE, the phase and reason, and the watchdog record; no comparison and no continue decision.
- Tested (10 tests): every refusal above, completed run with persisted artifacts, killed worker, crashed worker, refusal to run twice, `child_main` with an injected backend, and the text re-derivation checks.
- Still true: nothing has run the real backend, and the permission line is blank, so `execute` cannot start today.

## Gap 2: the feasibility probe now performs real optimizer updates
- The runner passes `micro_batch x grad_accum x probe_optimizer_updates` = 2 x 8 x 2 = 32 training examples, so the probe makes **two full optimizer updates** (the earlier 8 examples gave four micro-batches and none). It also scores 16
  selection sentences. The backend must report `optimizer_updates_done`; below 2, the run ends NOT_FEASIBLE ("probe did not exercise the optimizer"). Peak GPU memory is measured and recorded.
- Final partial accumulation group: it is **stepped, not discarded**, with gradients rescaled by `grad_accum / group size` (a group of 2 micro-batches out of 8 is scaled by 4). The schedule is torch-free
  (`trainloop.py`) and tested: every example seen exactly once per epoch, deterministic order, partial group stepped and rescaled, learning-rate warm-up and linear decay.
- The seed is set **before** the LoRA adapter is initialised (call order checked with stub torch/peft objects), and the adapter is re-seeded for the real training run so the probe cannot change it.
- Every timed section (probe scoring, probe training, full training) is bracketed by `torch.cuda.synchronize()` (checked with the stub). Numerical behaviour, memory and throughput on the GPU remain untested.

## Gap 3: a real external watchdog
- `watchdog.supervise` runs the worker as a subprocess and enforces, on the PARENT's clock, the startup cap (400 s), each announced phase's cap and the total 7,200 s cap, each plus a 15 s grace so the in-process cooperative
  stop fires first and the kill is the backstop. The worker announces phase start/end through an atomic progress file, and the feasibility phase is covered like the others. On Windows the whole process tree is killed.
- Tested against **real subprocesses** with small caps, including a backend that never calls `tick` (it sleeps forever): killed in the feasibility phase, in training, in baseline scoring and in tuned scoring; a worker that never starts a phase is killed at the startup cap; the total cap kills even when the phase cap is larger;
  a cooperating worker finishes and is not killed. After the suite no test process was left running.
- Limits: a hang between phases is bounded only by the total cap; a driver-level GPU hang that survives process kill is not covered.

## Frozen identifiers (this pass)
- Manifest v5 content hash `1645dd3fd9455002db8bec4b2fa0f2dbe40d69a11cac4133ecfecc821e2c4da2` (unchanged); file `training_manifest_v5.json` hash begins `03071041ddfe`.
- Pilot configuration hash `fe8c51a01763e75c2e2270e82cd566c0a08caa35d437b9c831c9ee82686336dc` (adds startup cap, watchdog grace, probe size).
- Freeze v3 `v5/pilot_freeze_v3.json`: `b3af4674b0457c398a68fcb27dc1a1cbc51b480b649ec200a0bfef89299f8812` (samples unchanged: 1,500 training / 823 selection / 1,446 validation sentences; the second-pass freeze file is kept alongside).
- Time budget: startup 400 s + feasibility 300 + baseline scoring 1,100 + training 3,300 + tuned scoring 1,100 + slack 1,000 = 7,200 s. API spend $0.
- Suite: 688 tests pass in the repository (106 in the nine pilot test files), twice in a row; Ruff clean; `evalops.run_calibration verify` unchanged.
- Defect found and fixed during this pass: the full suite failed intermittently (a different process-based test each time). Cause: on Windows the
  watchdog can hold the progress file open at the instant the worker atomically replaces it, raising PermissionError inside the worker and crashing it
  (recorded as INCOMPLETE). Progress reporting is now best-effort: it retries briefly and, if it still cannot write, counts the failure and continues (a
  regression test simulates both cases). After the fix the two process-based test files passed 8 of 8 consecutive runs. If reports are lost the watchdog
  still enforces the startup and total caps but cannot see phase boundaries.

## Unresolved (blocks the GPU run)
1. **Permission**: the decision line in `ROUTEBENCH_TRAINING_PERMISSION_STATUS.md` is blank; a recorded "accept" is the user's own risk decision, not a rights clearance. Recent News stays excluded.
2. **Dependency**: `peft==0.21.1` is not installed (plan: `ROUTEBENCH_PILOT_INSTALL_PLAN.md`; needs approval).
3. **Real backend on hardware**: never run. bf16 LoRA parameter dtypes, gradient checkpointing with T5, memory at micro-batch 2 with up to 2,048 tokens on the 8 GB card and throughput are unknown; the probe exists to find out and may end the run as NOT_FEASIBLE.
4. **Scope and power**: length-filtered sample of 45 validation documents (about 103 unsupported sentences); only a large gain is detectable; domain transfer to RouteBench is untested; a pass would not establish kappa 0.74.

## Approval sought, once item 1 is decided
Install `peft==0.21.1` into an isolated copy per the plan, then one `execute` run within 7,200 s at $0 API. A NOT_FEASIBLE, INCOMPLETE or negative result will be reported as such and is a valid outcome.
