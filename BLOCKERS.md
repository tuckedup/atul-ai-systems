# Blocked verification

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
- **The recorded dev experiment cannot be reproduced from the committed tree.** All 8 variants in
  `evalops/data/dev_experiments.json` fail `make calibrate-verify`. The stored `JudgeConfig` fields
  round-trip exactly, so the divergence is in the prompt scaffolding in `judge.py`, which
  `config_hash` covers but the record does not store. The 2,968 cached judgments were produced by
  prompts no longer in the repository. This is not repairable by editing anything: the dev split
  has to be re-judged under the current prompts, which needs a provider key. Diagnosable with
  `make calibrate-verify`; the failure now names the drift instead of reporting "0 paired items".
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
