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
