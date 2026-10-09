# Result: v18 vs v18 + auditor on the 30 reserved TofuEval upstream-dev documents (run once; protocol in ROUTEBENCH_FRESH_EVAL_PREREG_20260930.md)

Status COMPLETE. Coverage 240/240 v18 judgments and 166/166 audits (all v18 passes), usage verified on every call. **Actual API spend $2.425** (one journal, both judges, all attempts) against the $4.60 hard cap. Output: `.local/fresh-eval-v1/` (rows, journal, terminal, `final_report.json`).

| | Weighted κ (primary, w = 1/π) | Unweighted κ | Sens / Spec (weighted) |
|---|---|---|---|
| Baseline v18 | 0.5216 | 0.5339 | 0.820 / 0.865 |
| v18 + auditor | 0.5121 | 0.5254 | 0.805 / 0.887 |

- Auditor flipped 4 of 166 passes: **1 corrected, 3 introduced errors** (three flipped cases were gold-supported; the one correct flip was an unsupported addition).
- Paired document-bootstrap gain in weighted κ: −0.0095, 95% interval [−0.046, +0.025] (2,000 draws); unweighted −0.0085, [−0.046, +0.027].
- **IMPROVED: no** (corrected < introduced; interval includes 0). **REACHED_0.74: no** (pipeline weighted κ 0.512).
- Descriptive ceiling if every v18 false pass were fixed and nothing else changed: weighted κ 0.600, so this auditor design could not have reached 0.74 on these documents; v18's errors here are mostly false rejections (37 of 42), which the auditor cannot fix.
- Note: v18 on these 30 documents scored 0.52 (weighted), above its 0.40 on the first 40-document sample; the difference is not explained by this evaluation.

Closed: no rule changes, threshold searches, extra candidates or held-out testing follow from this run. The weighted/unweighted κ both use the ordinary binary Cohen's κ; weights only change how cells are counted.
