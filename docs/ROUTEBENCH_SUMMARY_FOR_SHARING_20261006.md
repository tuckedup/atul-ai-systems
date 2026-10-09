# RouteBench groundedness judge: summary for sharing (2026-10-06)

**One line.** Built and froze a reproducible, fail-closed evaluation harness for an LLM source-groundedness judge; the baseline judge reaches Cohen's kappa of about 0.66 on a development split and 0.4 on a training split of a mixed benchmark, and every improvement attempted was tested honestly and found negative. The original 0.74 target was **not** met.

## What was built
- A benchmark of sentence-level source-groundedness cases with preserved original labels and splits, and a sealed held-out test that was never run.
- A judge harness (rubric-based o4-mini baseline, "v18") with hash-bound configurations, cached judgments and paired, document-grouped bootstrap comparisons.
- A discipline of frozen protocols: manifests binding input, code and config hashes; fsynced reserve-before-request spend journals with hard caps; one request per reservation; fail-fast on the first error; labels opened only after complete, validated coverage; explicit resume and settle rules.
- About 1,400 automated tests (as of 2026-10-06), lint-clean, plus written reports for every experiment, including the negative ones.

## Results (honest)
| Measure | Value |
|---|---|
| v18 dev kappa, best single run / five-run mean | 0.6745 / 0.6576 |
| v18 TRAIN kappa, five-run mean | 0.4160 |
| Best alternatives tried (LoRA fine-tunes, second-pass auditor, claim-relation audit, source repair, majority voting, a fidelity criterion) | none improved on v18 |
| Sealed held-out test | not run |
| **Original target (honest binary kappa >= 0.74)** | **unmet** |

## A separate, narrower result
On a historical 128-case Reveal claim-verification probe (short single-fact claims), a frozen GPT-4.1 judge scored **kappa 0.8281** (accuracy 117/128; document-grouped 95% interval about 0.73-0.92; added cost $0.016). It is class-balanced, was inspected before, and was chosen after its high score was known: **it is not independent validation and not the original benchmark result.**

## Findings worth sharing
- 44 of 54 baseline errors repeat identically across five fresh calls, so repeating or voting over the judge cannot fix them.
- A blind AI review and an audit against published expert explanations suggest the remaining gap is partly benchmark-specific policy (for example opinion restated as fact), partly dropped hedges and verb strength, and partly missing context; separating these needs independent human annotation, which has been prepared but not run.

## Do not claim
That the judge meets kappa 0.74 on the original benchmark, that the Reveal number is that result, or anything based on relabelled or AI-adjudicated labels.

## Where to look
`docs/ROUTEBENCH_PROJECT_CLOSEOUT_20261006.md` (full status and decisions that remain), `docs/ROUTEBENCH_REVEAL_COMPLETION_RESULT_20261006.md`, `docs/ROUTEBENCH_CLAIM_FIDELITY_EVAL_V2_RESULT_20261006.md`, `docs/ROUTEBENCH_HUMAN_REVIEW_POLICY_PROPOSAL_20261006.md`, and `BLOCKERS.md` (running log).
