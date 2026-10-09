# MiniCheck equivalence run: narrative correction (2026-10-07)

This note corrects what I said about the equivalence-only GPU run. It rewrites **no** artifact: the report (`.local/judge-v19-phase1/minicheck_equivalence_report.json`, sha256 `7ca838a1bdf88ac77d9640292672efc977b8d7af0082fd0245f5faabf24d4a84`), the compute-ledger line, the frozen judge-v19 protocol (`3d6388088e4b87c0244ed7a29777dcb7a3bdbfa99fffef8ebf3af680f20ceca6`) and `BLOCKERS.md` history are unchanged. I recomputed the facts below from the saved report and the frozen cache, read-only.

| What I said | Correction |
|---|---|
| "all 24 rows were multi-sentence, the hardest case" | **Wrong.** 8 of the 24 selected rows have more than one cached sentence score (distribution 1 x16, 2 x4, 3 x2, 4 x2); **16 are single-sentence**. My check counted rows whose split matched, not rows with several sentences |
| "sentence splits identical" | Stronger than the evidence. The checker compared sentence **counts** and score positions. The saved report does not retain the sentence texts or the original splitter's output, so identical segmentation is not independently shown |
| "per-sentence scores reproduced exactly" | The runner **reported** a zero per-sentence difference; the legacy report stores only the whole-claim pair and one maximum difference per row, not the sentence-score vectors, so that zero is not independently recalculable. The 24 whole-claim pairs do equal their frozen references (verified), and the sampled ids match the expected hash order with no duplicates |
| "about 25 seconds" | 24.7 s was **scoring** time. The compute ledger records roughly 120 s for the whole process (model load included). Neither is a benchmark |
| "peak 3.4 GB of GPU memory" | 3,363.3 MiB is **peak allocated** by the process, not total board usage. The report and ledger do not record the GPU model, so the "RTX 4060" statement is not backed by a retained artifact (it came from a live `nvidia-smi` read, unsaved) |
| "closes the equivalence-unproven item" | Narrower: a deterministic, previously cached 24-row sample is **consistent** with the frozen cache. It is not proof of equivalence on every input, and the legacy report lacks a producer digest, so execution provenance is incomplete |

Unchanged and still true: no kappa changed; Stage 1b stays closed (no `release_1b.json`); no RAGTruth row was scored and `minicheck_ragtruth.json` does not exist; the classifier line stays closed; the original mixed-benchmark target (kappa >= 0.74) is unmet; Reveal's 0.828125 is a separate inspected historical probe. The run is a reproducibility prerequisite for a closed line, not evidence of better judgment.

## Checks run for this correction (CPU only, no model, no GPU)
- Read-only auditor `evalops.judge_v19_equivalence_audit`: `CONSISTENT_WITH_FROZEN_SAMPLE`; 24 full pairs checked; **0 sentence vectors checked** (none retained); 8 multi-sentence rows in the cache; `stage_1b_authorization_granted_by_audit: false`; `kappa_improvement_measured: false`; it states the same limitations.
- Targeted tests (main `.venv`, doubles only, `--basetemp` under `.local/pytest-tmp`): `test_judge_v19_equivalence.py` + `test_judge_v19_equivalence_audit.py`, **37 passed**; scoped Ruff clean.
- Hashes after the checks: report `7ca838a1...4a84` and protocol `3d638808...a6` identical to the values above; `minicheck_ragtruth.json` absent.
