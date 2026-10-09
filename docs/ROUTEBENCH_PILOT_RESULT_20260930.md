# Pilot result: INCOMPLETE at baseline scoring (2026-09-30)

**Outcome: INCOMPLETE (command exit code 3). No trained model, no comparison, no kappa, no balanced-accuracy result exists.** The frozen
stop rule applies: an incomplete run yields no continue decision, so per the authorisation nothing further was started. No API spend; the
held-out RouteBench split, the original labels and cases, and the readiness gate were not touched.

## What ran
- Permission decision recorded at the user's explicit instruction (private-research risk acceptance, not rights clearance).
- `execute` under the external watchdog with the isolated environment (`.local/pilot-venv`, peft 0.21.1), frozen configuration `fe8c51a0...`,
  freeze v3 `b3af4674...`, manifest v5 `1645dd3f...`. The parent validated the output and wrote the terminal verdict: **INCOMPLETE**,
  worker exit code 0, not killed, no integrity problems.
- Feasibility probe (passed): 2 real optimizer updates done, peak GPU memory 2.96 GB, measured 0.195 s per scored example and 1.16 s per
  training example, projected training 1,741 s, each scoring pass 443 s, all inside their caps.
- Baseline scoring then exceeded its 1,100 s cap (cooperative stop at "phase 'baseline_eval' exceeded 1,100 s") and the run ended.
  Wall-clock 1,177 s in total. Training never started.
- Cumulative compute ledger: **0.335 h** of the 40 h ceiling (including the 30 s smoke test); $0 API.
- Artifacts (kept, never overwritten): `.local/ragtruth-ingest-20260930/pilot_run_v1/` (`pilot_result_v1.json`, `pilot_terminal_v1.json`,
  `progress.json`, `child.log`) and `compute_ledger.jsonl`.

## Diagnosis (offline, CPU only)
The failure is a flaw in MY probe design, not evidence about the model or the data.
1. **The probe scored an unrepresentative sample.** It scored the first 16 selection ids in sorted order: mean 573 tokens (max 592). The 2,269
   examples the baseline had to score average 1,007 tokens (max 2,003). Under a simple linear-plus-quadratic attention cost model the probe
   examples cost about 45% of the average example, so 0.195 s per example understates the real rate by roughly 2x (0.195 / 0.45 = 0.43 s).
   The observed overrun means at least 0.485 s per example. Training was timed on the same kind of unrepresentative examples, so its 1,741 s
   projection is also likely too low (plausibly above the 3,300 s cap).
2. **The margin was too thin even with a correct projection.** The evaluation samples were sized from an earlier measured rate of about 2.3
   pairs/s (0.43 s each): 2,269 x 0.43 = about 990 s against a 1,100 s cap, a 10% margin. Other programs also hold about 1.7 GiB of GPU memory and
   GPU time; memory reached about 7.7 of 8.2 GiB during scoring, which can slow allocation.
3. **Observability gap.** The worker logs no scoring heartbeat, so the number of examples scored before the cap is unknown, and the partial
   scores were discarded.

## What would be needed to run a corrected pilot (proposal; needs your approval because it changes the frozen configuration)
- Score a length-stratified probe sample (for example examples at evenly spaced length quantiles) and require projected time <= 80% of the
  remaining cap, for scoring AND training.
- Re-size the pilot so it fits the 7,200 s total with margin: fewer evaluation sentences and/or a smaller training sample, or re-allocated phase
  caps; any such change produces a new configuration hash and freeze and is reported as a new pilot, not as the same run.
- Add a scoring heartbeat (count and time every N examples) and keep partial scores on an overrun for diagnosis.
- Budget: the corrected pilot would again be at most 2 h; ledger would then be about 2.3 h of 40 h. The extension stage stays locked until a
  pilot completes, passes its predeclared gate, and the real trained adapter passes the reload check.
Nothing here has been implemented or run. This is not evidence that the training recipe cannot help; it is a failed feasibility budget.
