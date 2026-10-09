> **SUPERSEDED IN PART** by `ROUTEBENCH_PILOT_READINESS_CHECKLIST_3_20260930.md`: the configuration and freeze identifiers, the probe, the execute command and the watchdog changed in the third pass. Kept as the second-pass record.

# Pilot readiness checklist, second pass (2026-09-30)

Supersedes the manifest, configuration and freeze hashes in `ROUTEBENCH_PILOT_READINESS_20260930.md` (that document stays as the first
pass). No paid calls, no dataset downloads, no GPU training or scoring, no original-test evaluation, no dependency installed.
AI-assisted engineering; not human adjudication and not legal clearance.

## What passes
| Item | Status and evidence |
| --- | --- |
| Tied-score average precision | **Fixed.** Tied scores form one threshold step, so the value no longer depends on row order (the reviewer's `1.0` vs `0.5` case now gives one value). Reference tests compare against scikit-learn's average precision on 25 tie-heavy random sets and check permutation invariance |
| Score direction | **Reworked.** Below-chance performance is reported (`direction_report`, `below_chance` flag) and never rejected as a wiring error. Wiring itself is checked deterministically: `assert_wiring` and `score_from_logits` tests confirm label-token ids differ, P(supported) rises with the supported logit, equal logits give 0.5. A run test with a correctly wired but worse-than-chance model completes and stops per the rule |
| Entry point | **Written and unit-tested against the frozen configuration**: `evalops/pilot/run.py` (runner) with an injected backend, `cli.py` (`freeze`, `dry-run`), and `torch_backend.py` (real backend, imports torch/peft lazily; **not run on hardware**) |
| Guards on the execution path | Manifest and configuration hashes, freeze equality, group-disjoint plan, train-only inputs, score completeness, deterministic wiring check, per-phase and total wall-clock caps (a `tick` reaches the backend so a running phase stops mid-way), feasibility probe inside the total budget, predeclared stop rule, result never overwritten. Ten runner tests with a fake backend and fake clock: complete run, NOT_FEASIBLE before training, INCOMPLETE on a training overrun, INCOMPLETE on missing scores, below-chance model, tampered manifest / configuration / freeze / texts, permission refusal. Two guards were deliberately broken to confirm the tests fail |
| Whole-response metrics | Reported **only for responses whose every in-scope sentence was evaluated** (`n_complete`, `n_partial`, `n_not_evaluated` are stated); partial responses are never scored as complete. The label used is the **authoritative response label** |
| Authoritative response labels | Manifest **v5** carries `response_labels` (3,724 responses; annotated wrapper segments included), so a response with an annotated wrapper stays unsupported. Rows are identical to v4 (verified); only the labels table and hash are new |
| Dry run on the real frozen data | `cli dry-run` executes the full path on the real v5 manifest and frozen samples with a **simulated** backend: COMPLETE, caps respected. It proves the plumbing only; its scores say nothing about a model |
| Suite | 657 tests pass in the repository (75 in the five pilot test files); Ruff clean; `evalops.run_calibration verify` unchanged; v3 and v4 files untouched |

## Frozen identifiers
- Manifest `ragtruth-training-manifest-v5` content hash `1645dd3fd9455002db8bec4b2fa0f2dbe40d69a11cac4133ecfecc821e2c4da2` (policy `ragtruth-sentence-v4`, splitter `regex-v5`).
- Pilot configuration hash `c3ada7234e69574f4cd2a6581dfac7a984f188b97a10bf8bccca378ec79d8c6c` (`evalops/pilot/config.py`).
- Freeze (samples) hash `877269e1660de844bdf995362ba425a3b940a8cfc5f441ddf74460baeff2df0f`: 1,500 training examples (500 unsupported), 823 selection sentences (61
  unsupported, 24 groups), 1,446 validation sentences (103 unsupported, 45 groups); 702 / 193 / 94 over-length sentences excluded, not truncated.

## Scope, stated plainly
The pilot scores a **length-filtered, label-blind sample of 45 validation document groups** of RAGTruth news summaries. Completing every sampled item is
complete coverage of that sample, not of the RAGTruth validation partition, and not the RouteBench benchmark. Response-level numbers cover only fully evaluated responses.
A pass would not establish kappa 0.74 or transfer to meeting and interview material.

## What remains unresolved (blocks the GPU run)
1. **Training-use permission**: undecided. The decision line in `ROUTEBENCH_TRAINING_PERMISSION_STATUS.md` is blank, and the runner refuses to start without a recorded `accept`
   (a test confirms the shipped file is undecided). A recorded accept is the user's own risk decision, not a rights clearance; Recent News stays excluded.
2. **Dependency**: `peft==0.21.1` is not installed. The pinned, isolated plan is `ROUTEBENCH_PILOT_INSTALL_PLAN.md` and needs approval.
3. **Real backend untested on hardware**: bf16 LoRA dtypes, gradient checkpointing with T5, memory at micro-batch 2 and inputs up to 2,048 tokens, training throughput (unmeasured; the 1,500-example
   sample fits the 3,300 s cap only above about 0.46 examples/s). The first phase of the run measures this and stops if infeasible.
4. **Statistical limit**: about 103 unsupported validation sentences from 45 documents detect only a large improvement; an inconclusive stop is the likely outcome for a small effect.
5. **Domain transfer** to RouteBench meeting / interview material is untested.

## Approval sought (after item 1 is settled)
One bounded pilot: install `peft==0.21.1` into an isolated copy per the plan, run the single pilot within 7,200 s total at $0 API, stop or continue per the frozen rule, report a negative or incomplete result as such.
No 40-hour commitment and no sweep.
