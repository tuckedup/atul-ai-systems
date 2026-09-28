# RouteBench status

- [x] M1 gateway parity — request, streaming SSE, model list, health, metrics, trace/audit integration, and OpenAI wire-shape tests pass
- [x] M2 backends + vLLM — provider adapters and live self-hosted fallback-model proof completed; vLLM 0.29 was incompatible with Docker Desktop/WSL UVA, so measurements used v0.10.2's V0 engine
- [x] M3 inference experiments — four 50-request GPU configurations measured; p95 TTFT ranged from 69.065400 to 242.420700 ms, throughput from 10.349351 to 24.807449 tokens/s, TPOT from 18.141688 to 20.138633 ms, and peak VRAM from 6,488 to 6,490 MiB
- [x] M4 reliability — cumulative `/v1/usage` cost/token accounting, per-model audit events, three-failures-in-30-seconds circuit breaking with 60-second recovery, fallback routing, 50-request admission threshold, queue timeout, and HTTP 429 `Retry-After`; full suite passes 10/10
- [~] M5 eval control plane — judge calibration rebuilt and measured; live per-model offline/online
      runs against three backends still remain. Detail:
  - The previously reported "κ = 1.0 on 50 committed labels" is WITHDRAWN. It was a fixture in
    which `generate_data.py` wrote one variable into both the `human_score` and `judge_score`
    columns, so κ = 1.0 was arithmetically guaranteed and no model was ever called. Three further
    defects are documented in `docs/KAPPA_DESIGN.md`: `calibrate()` thresholded the human labels
    it was being measured against; an undefined κ (`NaN`) passed the `kappa < 0.6` gate because
    `NaN < 0.6` is `False`; and `EvalSuite.load` silently returned zero cases for a file path.
  - Corpus rebuilt: 1,240 cases from public benchmark sources with recorded provenance and
    licences (`evalops/data/raw/MANIFEST.json`), across summarization/groundedness (real expert
    annotations), coding (HumanEval + MBPP hidden tests), SQL (Spider result-set equivalence on
    vendored SQLite fixtures), reasoning (GSM8K gold final answer) and tool use (BFCL gold call).
    856 independent groups; splits verified free of group and content leakage.
  - Judge rebuilt as a structured rubric judge over six versioned rubrics; the score is computed
    in Python from per-criterion verdicts rather than emitted by the model.
  - Calibration protocol: threshold and variant selected on dev only, bundle frozen with model,
    rubric, dataset and split hashes, test split scored once. Measured result, confusion matrix,
    confidence interval, per-task breakdown, completion rate and the full variant grid (including
    the variants that lost) are in `evalops/data/CALIBRATION_REPORT.md`.
  - Routing wiring fixed: suite tags (`coding`/`summarization`) never matched the router's keys
    (`code`/`summarize`), so every quality lookup missed and the router substituted its 0.5
    default. `evalops/taxonomy.py` now owns one canonical taxonomy and raises on unknown labels.
  - Still outstanding for M5: live per-model offline quality measurement across three backends and
    the online sampled-traffic loop against real traffic. `evalops/offline.py` and
    `evalops/online.py` are implemented and tested but have not been run against three live
    backends, so no per-backend quality matrix is published. `make bench` prints `not measured`
    for those cells rather than a placeholder.
- [x] M6 eval-driven promotion — regression injection automatically rolls back and writes a verified audit event; CI offline gate committed

Phoenix verification: yes. Phoenix GraphQL reported five live spans in the `default` project, including successful `router.route` and `llm.chat` gateway spans at 2026-09-13 13:58:55 UTC. The Windows UI automation helper could not initialize (`failed to write kernel assets`), so trace presence was confirmed against Phoenix's live API rather than by an automated visual inspection.
