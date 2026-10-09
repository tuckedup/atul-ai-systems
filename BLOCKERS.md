# Blocked verification

## MiniCheck equivalence audit correction (2026-10-07)

The saved 24-row check is consistent with the frozen sample and whole-claim pairs, but only 8 rows are multi-sentence (16 are single-sentence), not all 24. Legacy per-sentence vectors and producer digest are absent; the reported zero per-sentence difference is not independently recalculated from retained vectors. Standalone checker/report audit hardened offline, 37 tests and fresh review passed; frozen v19 protocol and old report unchanged. Stage 1b remains unreleased and κ unchanged. Existing review responses are AI-assisted, with zero independent human reviews found in that session. No new GPU/provider run. Detailed corrected evidence and Claude work order: `docs/ROUTEBENCH_MINICHECK_EQ_AUDIT_AND_CLAUDE_HANDOFF_20261007.md`.

## Current RouteBench engineering evidence boundary

- M2/M3 Docker-dependent inference acceptance: BLOCKED — requires docker
- Phoenix UI / localhost:6006 acceptance: BLOCKED — requires docker
- Original mixed-groundedness κ >= 0.74 remains UNMET; Reveal's 0.828125 historical probe is separate. No original held-out test was run in this engineering phase.
- Local safety/control-plane implementation is not a live quality/load/rollback measurement. Production limitations and the remaining CI/calibration/backend work are listed in `docs/ROUTEBENCH_ENGINEERING_COMPLETION_STATUS.md`. No new provider spend, GPU run, downloads or forbidden service checks were performed.

## GEDD and attribution audit (2026-09-29)

- **Quality target still NOT achieved.** Original dev best single 0.6532, inspected
  combination 0.6719; separate attribution test 0.5431 on only 195/206 items.
- Attribution confirmation is compromised: 96 cropped sources, 14 test pairs seen
  in prior probes, and empty annotation/split bindings. Existing artifacts are
  preserved and commands refuse silent rebuild/refreeze. Perfectly resolving the
  11 missing calls could give at most 0.5721, not 0.74.
- Source-complete expert TRAIN examples and unmeasured v17 are prepared offline.
  A new paid dev run needs specific authorization; no paid judge calls were made
  during this audit. Original test remains unmeasured. See
  `docs/ROUTEBENCH_GEDD_AUDIT.md` for diagnosis, changes and the bounded next step.
- A new attribution corpus is not ready: excluding prior probe evidence leaves
  only 47 Wice positive rows against the declared quota of 50. Do not reuse exposed
  documents or alter the sampling protocol silently.

## Current local RouteBench execution status (2026-09-28)

- **API credit blocker CLEARED; authorized dev evaluation COMPLETE.** After the
  owner restored credits and authorized transmission of 128 dev cases for four
  variants within $12, all 512 current-config judgments completed. Recorded
  token-priced cost, including the tiny health probe and retries, is $2.502474.
  No billing settings were changed and no credits were purchased by this agent.
- **Quality target NOT achieved.** Best single-judge dev kappa is 0.6532; best
  inspected unweighted combination is 0.6719, both below 0.74 on all 128 cases.
  No held-out inference or bundle freeze was performed. Dataset, annotations,
  prompts, and scoring code were unchanged during evaluation. See
  `docs/ROUTEBENCH_DEV_RESULTS.md` for the measured results and next decision.
- Quota exhaustion now fails fast through the shared client, judge repair/sample loop,
  concurrent runner and calibration commands. A quota-blocked test leaves its frozen
  bundle/report unchanged and can resume the same candidate from cached successes.
- No ready MiniCheck/AlignScore fallback: the active environment lacks their libraries
  and torch/transformers, and the inspected local model cache lacks their checkpoints.
  Their integration and real-weight validation remain unfinished, as documented in
  `docs/ROUTEBENCH_DEBUG_HANDOFF.md`.

The following sections retain historical observations and other projects' blockers.

- Core M1 infrastructure boot / Phoenix UI proof — BLOCKED: requires Docker
- Core definition-of-done Phoenix trace visibility proof — BLOCKED: requires Docker
- RouteBench M2 self-hosted vLLM backend proof — BLOCKED: requires Docker
- RouteBench M3 live vLLM inference experiment matrix — BLOCKED: requires Docker
- ForgeCode M5 official 20-task SWE-bench Verified run — BLOCKED: the official harness requires a container runtime; Docker is not installed
- ForgeCode M6 ten-organic-trajectory evidence — BLOCKED on source data: `.local/traces.jsonl` contains exactly 1 genuine failed correlated trajectory after excluding 400 expected `GraphInterrupt` spans; no synthetic failures were substituted, so 9 additional genuine failures are still needed
- ForgeCode M6 hosted failing-run URL — BLOCKED: no hosted CI run exists in this local checkout and no remote workflow dispatch or push was authorized
- Incident Commander M5 live GitHub issue proof — BLOCKED: requires an interactive OAuth grant and authorization for the external write
- Incident Commander M6 live accuracy, time-to-root-cause, and cost measurements — NOT MEASURED; the earlier provider-quota blocker has cleared, but that benchmark was outside this ForgeCode task

## Judge calibration (2026-09-27)

- RouteBench M5 live per-model offline quality matrix — NOT MEASURED. `evalops/offline.py` is
  implemented, tested and gated on a valid calibration artifact, but it has not been run against
  three live backends, so no `(model, task_class)` quality matrix is published. `make bench`
  prints `not measured` for those cells. This needs the self-hosted vLLM backend, which is
  Docker-blocked, plus provider budget for the frontier backend.
- RouteBench M5 online sampled-traffic quality — NOT MEASURED. `evalops/online.py` implements
  bounded deterministic sampling, version binding and a minimum observation count, and is unit
  tested, but there is no production traffic on this host to sample.
- Judge calibration against LOCAL hand labels — PENDING HUMAN INPUT, not blocked on code. The
  annotation queue and CLI exist (`make annotate-export`, `make annotate ANNOTATOR=<id>`). Until
  a person labels those items there are no `human_local_annotation` rows, so the human-judgment
  track rests on published expert annotations only. No labels were fabricated to fill the gap.
- `gpt-5` / `gpt-5-mini` as judge candidates — UNAVAILABLE on this key: `/v1/chat/completions`
  returns HTTP 404 for the gpt-5 family. Recorded in the variant grid as skipped rather than
  silently omitted.
- Inter-annotator agreement on the groundedness labels — NOT MEASURED HERE. The published
  subsets ship adjudicated labels without per-rater votes, so human-human agreement cannot be
  recomputed from what was fetched. It is therefore not quoted as a ceiling on judge agreement.

## Judge calibration, second round (2026-09-28)

- **RouteBench κ ≥ 0.74 — NOT ACHIEVED and not measurable in this environment.** No provider
  credential is present (`OPENAI_API_KEY` / `AISYS_OPENAI_API_KEY` unset), so no judge variant can
  be run. The held-out test split was NOT touched, and no calibration bundle was frozen. The best
  agreement measured anywhere in this repository remains a dev-internal estimate well below the
  target: κ ≈ 0.64 on the groundedness track and κ ≈ 0.50 pooled.
- **The recorded dev experiment was written under an older `config_hash` formula, and its prompt
  provenance is unverifiable.** CORRECTED 2026-09-28: an earlier entry here claimed the prompts
  were "no longer in the repository". That was wrong. An exhaustive search over 131,072 candidate
  formulas found exactly one reproducing all eight recorded hashes, and it omits `prompt_template`
  entirely — so a prompt edit was invisible to it and the mismatch is fully explained by the
  formula change. Every `rubric_version` in the cached judgments matches this tree; the `_ROLE` and
  output-format scaffolding is covered by nothing. The 2,968 judgments are therefore of unknown
  prompt provenance: usable for a labelled development probe, NOT admissible behind a frozen
  bundle. Making them admissible needs a re-judge under pinned prompts, which needs a provider key.
  `make calibrate-verify` distinguishes "legacy formula" from "unidentifiable formula"; nothing in
  the code will relabel a legacy judgment as current.
- **No TRAIN judgments exist at all.** The judgment cache holds dev judgments only (0 train, 0
  test at the recorded config hashes). The acceptance protocol fits any learned combiner on train
  groups, so a combiner cannot be fitted under the protocol until the train split is judged. The
  probe in `evalops/data/COMBINER_PROBE.md` works around this with dev-internal grouped
  cross-validation and is labelled, in the artifact itself, as not a held-out result.
- **v8–v16 never run.** The decomposition, source-addressed, contradiction and matched-holistic
  variants are declared in `run_calibration.GRID` with unit-tested prompt construction and parsing,
  but none has been executed. Same blocker: no provider key.
- **Evidence-validity signals are untested as features.** 0 of 5,470 cached judgments carry
  `evidence_location`, because `judge_one` never copied it onto `Judgment` before this round's fix.
  The combiner probe therefore reports the evidence feature family as all-zero and says so; whether
  those signals help is an open question, not a negative result.
- Human-human agreement on the groundedness labels and local hand-labelling remain as previously
  recorded: not measured, pending human input, not fabricated.
- The no-Docker amendment above is unchanged: nothing in this round needed or assumed a container
  runtime, and no Docker-blocked item was reopened or worked around.

## Review response, third round (2026-09-28)

- **κ ≥ 0.74 still NOT ACHIEVED.** Unchanged and unchangeable here: no provider credential, so no
  judge variant can run, the test split is untouched and no bundle is frozen. The best agreement
  measured anywhere in this repository remains a dev-internal cross-validated estimate — κ ≈ 0.64 on
  groundedness over 87 jointly-covered cases, κ ≈ 0.50 pooled.
- **MiniCheck / AlignScore adapters are implemented and unrun.** `evalops/checkers.py` calls upstream
  at the pinned commits, with 53 tests passing against a scripted fake checker and zero heavy
  imports at module load. Running them for real needs (a) a weights-licence decision —
  `CheckerSpec.weights_licence_checked` defaults to False and `load_checker` refuses without it —
  and (b) approved model downloads into an isolated environment. Neither is a code problem.
- **No model weights were downloaded and no upstream package was installed into this project's
  environment.** The upstream repositories were cloned to a scratch directory to read the
  algorithms; `docs/upstream/REPRODUCTION.md` records what was read and the pins it was read at.
- **Prometheus was not adapted.** Its absolute-grading prompt produces a 1–5 ordinal score, and
  RouteBench's target is unweighted binary κ. Converting one to the other needs a declared
  binarisation chosen on dev, which is more experiment than the current round can honestly run.
- **The train split still has zero judgments**, so a learned combiner cannot be fitted under the
  acceptance protocol. Unchanged from the previous round and still the reason the combiner probe is
  labelled dev-internal.
- Docker-dependent milestones remain BLOCKED exactly as recorded above. Nothing in this round needed
  or assumed a container runtime.

## Round 3 (2026-09-29)
- kappa >= 0.74 still NOT ACHIEVED on dev (best single 0.6771, o4-mini high). See docs/ROUTEBENCH_ROUND3_REASONING_JUDGES.md.
- 14/128 dev cases are misjudged by every judge; human adjudication of docs/LABEL_ADJUDICATION_QUEUE.json is needed.

## Phase close-out (2026-09-29)
- kappa >= 0.74 NOT ACHIEVED. Selected comparison baseline v18 (o4-mini, hash 12fc8b550d954359): dev 0.6745 (selected on dev), TRAIN 0.448, fresh upstream-dev documents 0.404 [0.282, 0.525] (weighted, evaluation-only). Highest single-judge dev score: v21, 0.6771. See docs/ROUTEBENCH_FINAL_PHASE_REPORT.md.
- UNRESOLVED dataset terms: TofuEval and LLM-AggreFact forbid training use. The v17/v23 few-shot packets and the MiniCheck+v18 combiner used TofuEval-derived TRAIN data and are not release-ready. Transcripts are CC-BY-NC-SA or research-only and git-ignored.
- No further paid calls, training, downloads or test evaluation without a new approved scope (proposal in the final report).

- HHEM-2.1-Open screening (frozen, 2026-09-29): dev kappa 0.586, fresh-set weighted kappa 0.437 vs v18 0.404 (paired CI [-0.101, +0.171]); no convincing improvement; training provenance undocumented. Pretrained-verifier trials stopped.

- RAGTruth ingestion v3 (2026-09-30): manifest beff83bc frozen (train 418 docs / 2,507 responses / 14,739 sentences, 6.84% unsupported; validation 104 / 629 / 3,466). Training use NOT cleared: CNN/DM article rights unresolved, Recent News excluded. GPU pilot blocked pending that decision. See docs/ROUTEBENCH_INGESTION_AUDIT_V3_20260930.md.

- Pilot readiness (2026-09-30): v4 manifest 2f3325ad frozen; pilot config b93ab024 frozen; guards tested offline. GPU pilot NOT approved: training-use permission unresolved (docs/ROUTEBENCH_TRAINING_PERMISSION_STATUS.md), peft not installed, training script unwritten, throughput unmeasured. See docs/ROUTEBENCH_PILOT_READINESS_20260930.md.

- Pilot readiness pass 2 (2026-09-30): manifest v5 1645dd3f, config c3ada723, freeze 877269e1; AP tie bug fixed; runner + guards tested with fake backend; real backend written, NOT run on hardware; peft not installed (plan in docs/ROUTEBENCH_PILOT_INSTALL_PLAN.md); permission decision still blank. GPU pilot not run.

- Pilot readiness pass 3 (2026-09-30): execute wired (refuses today: permission blank), probe does real optimizer updates + partial-group step, external watchdog tested with real hung subprocesses. Config fe8c51a0, freeze v3 b3af4674. GPU run still blocked: permission undecided, peft not installed, backend never run on hardware. See docs/ROUTEBENCH_PILOT_READINESS_CHECKLIST_3_20260930.md.

- Pilot readiness pass 4 (2026-09-30): terminal-result correction. COMPLETE is published only after artifacts are saved and hashed; the parent validates exit status, identities and artifact hashes and writes its own verdict (exit codes 0/3/4/5); per-example predictions persisted and the report recomputed from them. 700 tests pass. Frozen experiment unchanged. GPU run still needs: permission decision, peft==0.21.1 install approval, one run approval. See docs/ROUTEBENCH_PILOT_READINESS_CHECKLIST_4_20260930.md.

- Pilot run status (2026-09-30): user answered accept (private research only) + authorised peft install and staged scope up to 40 GPU-h. Installed peft==0.21.1 into isolated .local/pilot-venv; real-backend smoke test on GPU with a tiny random T5 passed (incl. adapter reload = identical scores); ledger + ceiling built into execute. Run NOT started: the acceptance line in docs/ROUTEBENCH_TRAINING_PERMISSION_STATUS.md is still blank and the user asked that I not fill it in. See docs/ROUTEBENCH_PILOT_RUN_STATUS_20260930.md.

- Pilot run 1 (2026-09-30): INCOMPLETE at baseline_eval (scoring exceeded its 1,100 s cap; probe used unrepresentative short examples, mean 573 vs 1,007 tokens, so the 443 s projection was ~2x too low). No comparison, no kappa. Ledger 0.335 h of 40 h, $0 API. Stopped per the authorisation; a corrected pilot needs a new frozen configuration and approval. See docs/ROUTEBENCH_PILOT_RESULT_20260930.md.

- Pilot run 2 / v2 (2026-09-30): VALID_COMPLETE, below gate. Tuned vs baseline bal-acc +0.0045 (CI [-0.011,+0.025]); gate needs >=+0.03 and CI>0. Stopped; extended stage not authorized. See docs/ROUTEBENCH_PILOT_V2_RESULT_20260930.md.

- FactCG-DeBERTa-v3-Large screen (2026-09-30): dev kappa 0.549 vs v18 0.675 (paired CI [-0.32,-0.001]); worse than v18, candidate stopped. See docs/ROUTEBENCH_NEXT_ATTEMPT_20260930.md.

- Proposal (2026-09-30, awaiting approval, $2 API cap): error-type-targeted audit of v18 passes. docs/ROUTEBENCH_PROPOSAL_ERROR_TARGETED_AUDIT_20260930.md

- Audit2 v2 (2026-09-30): execution defects fixed offline, frozen; awaiting approval of $3.00 total cap. docs/ROUTEBENCH_PROPOSAL_ERROR_TARGETED_AUDIT_20260930.md

- Audit2 v3 (2026-09-30): single-HTTP-attempt reservations, byte-based token bound, raw-usage validation; frozen; awaiting approval of $3.00 cap.

- Audit2 v3 run (2026-09-30): COMPLETE, spend $0.596; dev 0 flips, kappa 0.6745 unchanged (gate failed, 0.74 not reached). Closed.

- Audit2 diagnostic (2026-09-30): raw verdicts dev kappa 0.729172 (3/3 correct) vs official 0.674518; TRAIN raw 0.448 vs filtered 0.464. Proposed rule drops quote/overlap gates for an independent evaluation (post-hoc; unrun).

- Fresh-document evaluation (2026-10-01): v18 0.5216 -> pipeline 0.5121 weighted kappa (1 corrected, 3 introduced; CI [-0.046,+0.025]); not improved, 0.74 not reached; spend $2.425. Closed.

- Error-cause audit (2026-10-01): 37 false rejections: R rubric-stricter 14, J judge-miss 10, S surface-form 5, L doubtful label 4, M meta 2, U 2; claims_entailed fails in 31/37; overlap-of-criteria hypothesis rejected. Published SOTA on MediaS/MeetB BAcc <=85; kappa 0.74 needs ~0.91. docs/ROUTEBENCH_ERROR_CAUSE_AUDIT_20260930.md

- Campaign 2026-10-01: Opus judge dev 0.65/TRAIN 0.55; v18+Opus AND pooled 0.615; stacker 0.49-0.55; MiniCheck LoRA on 1.7k in-distribution sentences: AUC 0.76->0.76, no gain; GPT-5.4 blocked by OpenAI credit exhaustion. docs/ROUTEBENCH_CAMPAIGN_20261001.md

- Offline implementation pass (2026-10-01): dqdata.py (permission/provenance-aware workflow) + claim_relation.py (claim-relation-v1, mocked end to end); 41 new tests, full suite 904 passed, ruff clean; v18 identity unchanged. One proposal awaiting approval: $6-capped o4-mini TRAIN+dev ablation after API credit is added (docs/ROUTEBENCH_CLAIM_RELATION_PROPOSAL_20261001.md). Vendor request drafted, not sent.

- Claim-relation experiment repaired and frozen (2026-10-02): dev primary, frozen manifest sha 8c32146a..., fail-fast; 931 tests pass; awaiting one clean evaluation approval ($6 cap + API credit).

- Overnight 2026-10-04: frozen claim-relation-v1 run ABORTED at request 1 (OpenAI credit_balance_exhausted); v2 post-processing candidate built offline (950 tests pass); no GPU run; ledger corrected to 4.47 h. See docs/ROUTEBENCH_OVERNIGHT_RESULT_20261004.md.

- Claude-subscription claim-relation experiment frozen (2026-10-04), not run: protocol sha 60b8f8a1...; v2 coverage now flags unreviewed negation/modals/digits. docs/ROUTEBENCH_CLAIM_RELATION_CLAUDE_PROTOCOL_20261004.md

- Claude-subscription claim-relation run (2026-10-04): INCOMPLETE at 82/128 (malformed JSON from the model, fail-fast); no kappa; comparison withheld.

- Claude continuation protocol frozen (2026-10-04), not run: sha 2ae67ff5...; 81 valid imported, 47 pending, <=2 attempts/case; earlier "labels not parsed" claim corrected.

- Claude continuation re-frozen restart-safe (2026-10-04), not run: sha d4a853fe...; reserve-before-launch journal, atomic torn-tail restore; supersedes 2ae67ff5.

- Claude continuation-v3 run (2026-10-04): COMPLETE 128/128; dev kappa 0.3609 vs v18 0.6745 (fixed 7, new errors 31, CI [-0.462,-0.105]); not improved; 0.74 not reached.

- v3 aggregation repair + post-hoc reprocessing (dev kappa 0.5799, not accepted); v16 Reveal peak reproduced (0.8388; threshold 0.30 equals strict rule); matched v16-vs-v18 comparison frozen, not run (sha 281b169d..., ~$2.87 est, needs OpenAI credit).

- Cache-only dev v16 vs v18 (2026-10-04): 0.6403 vs 0.6745, diff -0.034 [-0.146,+0.057]; CRLF explanation corrected; Reveal non-transfer explained; typed-veto intervention proposed (in-sample 0.7134, unvalidated). docs/ROUTEBENCH_CACHE_ONLY_DEV_AND_INTERVENTION_PROPOSAL_20261004.md

- typed-veto-v1 implemented (2026-10-04): whitelist frozen, 32 tests, post-hoc dev reproduction 0.7133825 (3 fixed, 1 new FR; 3 documents), validation proposal = flag-then-adjudicate on new permissioned docs. docs/ROUTEBENCH_TYPED_VETO_V1_20261004.md

- typed-veto-v1.1 (2026-10-04): audit-to-case binding added (whitelist/original results unchanged; dev reproduction identical 0.7133825); power analysis shows 1,500 sentences/150 docs/~265 labels gives ~92-97% power under dev-like flag rates (kappa gain != accuracy gain); amended proposal + vendor quote draft v2 prepared, NOT sent. docs/ROUTEBENCH_TYPED_VETO_V1_20261004.md, docs/ROUTEBENCH_VENDOR_QUOTE_REQUEST_DRAFT_V2.md

- Source-integrity-v2 copy verified independently (257 kept, 63 repaired, 0 removed); fresh-judgment comparison srcint-eval-v1 frozen, not run (sha a52d3716..., ~$1.97 est, cap $3, needs OpenAI credit).

- srcint-eval-v2 frozen (2026-10-05), not run: sha ae824756...; row validation, fail-closed journal, bounded TRAIN/DEV loader, no Fisher/causal claim; v1 preserved. Needs OpenAI credit + approval.

- srcint-eval-v3 frozen (2026-10-05), not run: sha 5cbfa028...; raw-JSON model identity via own adapter, numeric-domain journal validation; v1/v2 preserved. Needs OpenAI credit + approval.

- srcint-eval-v3 run (2026-10-05): ABORTED at request 1 (OpenAI 429 quota); 0/257 judged; spend journal $0.0463 bound; comparison withheld. Needs credit + explicit --resume.

- srcint-eval-v3 COMPLETE (2026-10-05): repaired dev kappa 0.6028 vs cached 0.6745 (CI [-0.201,0.000]); spend $2.0557/$3; repair effect not measurable, run-to-run noise suspected; 0.74 not reached.

- srcint-eval-v3 CLOSED as completed negative result; judge-improvement proposal prepared OFFLINE (2026-10-05), NOT approved: calibrated three-outcome judge (v18 x5 + local verifier + structural features), disjoint T/C/E/S document roles, futility gate before any purchase, predefined criteria and stopping rules; power simulation (7 tests); costs itemised (rights/annotation unpriced; API ~$8 + ~$60, cap $90 requested; GPU <=3 h). No vendor contact, purchase, download, training or paid call. docs/ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_20261005.md

- Judge-improvement proposal v2 (2026-10-05, offline, NOT approved): independent review's four findings confirmed and fixed; full-gate success ~0.5 at true +0.03 (detection power is a different, higher number); single data rule (FIT = 1,050 RAGTruth sentences, sha 7edc80e3...; all TofuEval/AggreFact-derived rows evaluation-only); Phase 1 = 6,278 calls, est $50.22, hard cap $56.00 inside the $60 ceiling; approval requested only for offline build+freeze. docs/ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_V2_20261005.md

- judge-v19 Phase 1 FROZEN offline (2026-10-05), NOT run: protocol sha 3d638808...ceca6; 6,278 calls, est $50.22, caps 1a $10 / 1b $46 / phase $56; A1 = majority vote of original pass/fail verdicts; fit sample sha 7edc80e3ead8e5f6...d66012 (typo fixed); MiniCheck equivalence unproven (GPU job needs approval, run before 1b); 1343 tests pass. docs/ROUTEBENCH_JUDGE_V19_PHASE1_FREEZE_20261005.md

- judge-v19 Stage 1a ABORTED at job 1 (2026-10-05): empty API key because the .env is read relative to the working directory and I ran from routebench/ (my command error; the key loads from the repo root). No request reached the provider; the journal charged 3 attempts at their reservation bounds ($0.1401 of the $10 cap). The failed row is a permanent record, so --resume would skip that job and end INCOMPLETE: not resumed. Stage 1b not released; no GPU, vendor or sealed-test activity.

- judge-v19 Stage 1a DONE (2026-10-05): 1,028/1,028 judgments valid (1,027 via frozen runner + 1 via audited recovery of the empty-token job; terminal INCOMPLETE by design, original failed row byte-identical); committed $8.1994 of $10; two memory-pressure kills, resumed with explicit settle-at-bound; report .local/judge-v19-phase1/stage_1a_report.json. Dev k: A0 0.6745, single runs 0.6216-0.6771, majority vote 0.6403; TRAIN k: A0 0.4476, singles 0.3683-0.4630, majority 0.3999. Voting corrected 0 and introduced 2 (dev) / 3 (TRAIN) errors. Stage 1b NOT released; stopped.

- judge-v19 Stage 1a CLOSED as a completed negative result (2026-10-05): 44 of 54 A0 errors are unanimous across five calls (voting cannot reach them); v18 soft scores are constant (1.0) on the 38 unanimous false passes; only measurable complementary signal is MiniCheck (within-pass AUC 0.69), already used by the failed combiner. Recommendation: no Stage 1b / paid run; first a no-API human adjudication of the 44 unanimous-error cases (+44 controls) with a pre-frozen decision rule. docs/ROUTEBENCH_STAGE1A_CLOSURE_AND_RECOMMENDATION_20261005.md

- blind-review packet prepared OFFLINE (2026-10-05), nothing sent: 88 items (44 persistent + 44 controls, shuffled), reviewer packet separate from reconciliation file; 50%/75% rules withdrawn as decision/spend triggers; Stage 1b stays closed. .local/judge-v19-phase1/blind-review-v1/

- AI-assisted blind review of the 44 persistent errors + 44 controls received (2026-10-05; 0 human reviews): consolidated 68 supported / 15 unsupported / 5 uncertain; reviewers sided with v18 on 35 of 38 persistent false passes but matched the published label on 43 of 43 controls; shared-model bias is the main confound; five unresolved items need speaker/byline metadata or policy decisions (plurals, non-declaratives, inference strength). No relabelling, no spend, Stage 1b closed. docs/ROUTEBENCH_BLIND_REVIEW_AI_DIAGNOSTIC_20261005.md

- Expert-explanation audit (2026-10-06, existing files only): 27 of 35 disputed items have published explanations -> 14 policy mismatch (8 attribution, all one document; 3 document-genre; 3 connotation/simile), 8 genuine reasoning error (hedge/tense/strength/merge/counter-evidence), 3 input-context gap, 1 questionable, 1 unresolved; 'lenient labels' wording corrected; one proposed claim_fidelity criterion + synthetic fixture, NOT run; no spend, no label change, Stage 1b closed. docs/ROUTEBENCH_EXPERT_EXPLANATION_AUDIT_20261006.md

- claim_fidelity variant PREPARED OFFLINE (2026-10-06), not run: opt-in variant v24-claim-fidelity-o4mini = v18 + one rubric criterion (new rubric dir; v18 identity 12fc8b55/d72496fb/ef895d79 and all frozen protocols unchanged); audit corrected (existing rubric already prohibits modal strengthening); fidelity defined by evidential status, not speaker authority; 24 mocked end-to-end tests, ruff clean, full suite 1395 passed. Evaluation proposal (257 calls, cap $3, descriptive only) awaits approval; no spend, GPU, Stage 1b or held-out access. docs/ROUTEBENCH_CLAIM_FIDELITY_VARIANT_20261006.md

- claim-fidelity-eval-v1 FROZEN offline (2026-10-06), NOT run: protocol sha 287d8195f2d2a28c5cccf9aad1d189907f654c1e53497810fe4611ef0d4b6a69; 257 cases original inputs, v24 vs all five stored v18 runs, cap $3.00 cumulative, est $2.11; key preflight + request-not-sent handling + explicit resume/settle; 20 tests, ruff clean, full suite NOT rerun (1.4 GB free). Awaiting explicit run approval. docs/ROUTEBENCH_CLAIM_FIDELITY_EVAL_PROTOCOL_20261006.md

- claim-fidelity-eval-v2 FROZEN offline (2026-10-06), NOT run: protocol sha 619d5547ce4c47f808663a10e36c8ec8609046df2744096b96655c6b27d0a00f supersedes unrun v1 287d8195... (preserved, still verifies); fix = cost reporting from the journal incl. billed retries, bound charges reported separately, attempt counts incl. interrupted; wording caveat on fresh-call verdict changes; 27 tests pass, ruff clean, full suite NOT rerun (935 MB free); cap $3.00 unchanged; awaiting explicit run approval. docs/ROUTEBENCH_CLAIM_FIDELITY_EVAL_V2_20261006.md

- claim-fidelity-eval-v2 COMPLETE (2026-10-06): 257/257 valid, $2.2336 of $3.00, one attempt per case. v24 kappa dev 0.5938 (v18 runs 0.6216-0.6771), TRAIN 0.3889 (0.3683-0.4630), pooled 0.4889 (0.4973-0.5545); fewer false passes but more false rejections (pooled FR 21 vs 9-13); only 11 fidelity-only failures (7 wrong, 4 right); all kappa-difference intervals include 0. Completed negative result; Stage 1b, further runs, GPU and held-out untouched. docs/ROUTEBENCH_CLAIM_FIDELITY_EVAL_V2_RESULT_20261006.md

- v24 (claim_fidelity) CLOSED as a completed negative result (2026-10-06, owner decision); every artifact preserved (claim-fidelity-eval-v1 unrun freeze, claim-fidelity-eval-v2 run + comparison, module/rubric/fixtures/tests); v18 stays the baseline; the result doc no longer claims verdict changes exceed run-to-run noise (paired intervals include zero). ORIGINAL ACCEPTANCE TARGET UNMET: honest binary kappa >= 0.74 not reached by any judge tested (v18 best single-run dev 0.6745, five-run dev mean 0.6576; v24 dev 0.5938); sealed held-out test untouched. No more paid runs, GPU jobs, exclusions, relabelling or minor variants; further quality work only via a separately approved plan grounded in an independently verified annotation policy and evidence. docs/ROUTEBENCH_CLAIM_FIDELITY_EVAL_V2_RESULT_20261006.md

- Reveal historical-probe completion VERIFIED independently by me (2026-10-06): 128/128 valid (124 cached + 4 new), kappa 0.828125, accuracy 117/128, document-grouped 95% interval [0.729, 0.921], new usage-priced cost $0.016062, source hashes match the frozen originals. SEPARATE historical, class-balanced, previously inspected probe (short single-fact claims: median claim 46 chars, source 448 chars); NOT independent validation and NOT the original mixed RouteBench target, which remains UNMET (v18 best single-run dev 0.6745; sealed held-out untouched). docs/ROUTEBENCH_REVEAL_COMPLETION_RESULT_20261006.md

- PROJECT CLOSEOUT STATUS written (2026-10-06): original target UNMET; verified Reveal probe result kept separate with limits; attempts table, diagnosis, supported/unsupported claim wording, and the owner-only decisions that remain (accept/close, human review packet, new benchmark, sealed-test run, new plan). No spend, no runs. docs/ROUTEBENCH_PROJECT_CLOSEOUT_20261006.md

- Final hygiene (2026-10-06): full suite 1503 passed in 578.71 s, ruff clean; added docs/ROUTEBENCH_HUMAN_REVIEW_POLICY_PROPOSAL_20261006.md (proposed policy decisions, nothing decided) and docs/ROUTEBENCH_SUMMARY_FOR_SHARING_20261006.md. No spend, no runs, no config changes; original target still UNMET; remaining routes need owner decisions (see closeout).

- Engineering phase 2 (2026-10-07, offline, local/uncommitted): durable restart-safe tenant budgets, bearer-key tenant authentication, opt-in native token streaming with bound-based settlement, rollout restart-resume + observation worker; 58 new tests, full suite 1587 passed, ruff clean, frozen research protocols still verify. NOT done/unverified: real OpenAI SDK smoke (package not installed; no download), live backends/TTFT/throughput, Docker/Phoenix, multi-worker budgets, canary on real traffic, original kappa >= 0.74 (still UNMET). docs/ROUTEBENCH_ENGINEERING_PHASE2_OFFLINE_20261007.md

- GPU run requested (2026-10-07): owner chose the MiniCheck equivalence-only check (inference only, 24 cached RouteBench rows, no RAGTruth scoring, no training, no API). Prepared as routebench/evalops/judge_v19_equivalence.py (5 tests; refuses below 3,000 MB free RAM, fails closed without psutil, writes a separate report, never writes the frozen minicheck_ragtruth.json). NOT RUN: only about 1.1 GB RAM free (Zoom/Chrome/ChatGPT app resident). Human-review responses were not found in the repo (only the AI-assisted outputs and the reviewer-handoff zips); nothing analysed.

- MiniCheck equivalence-only GPU check RAN (2026-10-07, RTX 4060 shared, ~25 s scoring, peak 3.4 GB GPU, $0): 24/24 cached RouteBench rows reproduced EXACTLY (worst abs difference 0.0, sentence splits identical). Closes the 'equivalence unproven' open item for the v19 feature pipeline; it does NOT revive the closed classifier line (Stage 1b stays closed, no RAGTruth scoring was done, no kappa change). Ledger appended. Report: .local/judge-v19-phase1/minicheck_equivalence_report.json

- CORRECTION (2026-10-07) to the MiniCheck equivalence entry above: only 8 of the 24 rows are multi-sentence (16 single), segmentation identity and the zero per-sentence difference are reported by the runner, not independently shown (legacy report keeps no sentence vectors or producer digest); 24.7 s was scoring time (~120 s process); 3,363 MiB is peak allocated, GPU model unrecorded. Report/protocol/ledger unchanged; auditor + 37 CPU tests pass. docs/ROUTEBENCH_MINICHECK_EQ_NARRATIVE_CORRECTION_20261007.md

- Track B DONE (2026-10-07, offline): durable tenant budgets in the configured database via ModelState (BudgetStore/DbTenantBudget; default for validated startup, ephemeral and file-ledger modes explicit and disclosed): reserve-before-call, exact-once settlement, unknown usage at the bound, overshoots kept, unsettled reservations HELD until an explicit operator resolve, startup verify fails closed, 503 on budget-DB failure with no extra provider call. 35 new tests, two fresh read-only reviews (changes needed -> fixed -> ship), full suite 1659 passed, ruff clean, frozen v19/report hashes unchanged. PostgreSQL NOT executed. Not a judge/kappa result. Track A: policy addendum DRAFT with blank owner fields; no human reviews exist. docs/ROUTEBENCH_DURABLE_BUDGETS_DB_20261007.md, docs/ROUTEBENCH_ANNOTATION_POLICY_ADDENDUM_PROPOSAL_20261007.md

- Budget per-reservation validation DONE (2026-10-08, offline, zero spend): closed the independently reproduced gap (`BudgetStore.verify()` accepted a negative-bound reservation when aggregates balanced; also a NaN-text tenant figure made `reserve` fail open). Every reservation is now validated by state at startup and again before any settle/resolve/reserve/set_limit/ensure_tenant changes balances; malformed rows are refused and never repaired, refunded or discarded. 116 budget tests (43 fail with the check disabled), targeted set 316 passed, full suite 1740 passed in 680.88 s, ruff clean; four read-only reviews, the last on the exact final revision returned ship (non-material limits listed in docs/ROUTEBENCH_DURABLE_BUDGETS_DB_20261007.md). PostgreSQL still never executed. Frozen hashes unchanged, minicheck_ragtruth.json absent. Does not change kappa; the 0.74 target remains unmet and policy/human-review decisions remain open.
