# Blocked verification

## Current local RouteBench execution status (2026-09-28)

- **Live API check: BLOCKED by `credit_balance_exhausted`.** The configured project key
  is now present. One tiny request to the configured `api.openai.com` GPT-4.1 endpoint
  returned HTTP 429, type `insufficient_quota`, code `credit_balance_exhausted`, with
  no usage result. This supersedes the historical "no credential" descriptions below.
  No billing settings were changed and no credits were purchased.
- The 512-case/variant development run was not launched after this confirmation.
  Existing labels/cache remain intact; no held-out judgments or new agreement score
  were produced. Additional data collection is not the current blocker.
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
