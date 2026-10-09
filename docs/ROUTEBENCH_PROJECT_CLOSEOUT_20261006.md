# RouteBench groundedness judge: project closeout status (2026-10-06)

**Bottom line.** The original acceptance target, an honest binary (unweighted) Cohen's kappa of at least 0.74 against the human labels on the mixed RouteBench groundedness benchmark, is **UNMET**. Every judge and method tested on that benchmark stayed below it, the sealed held-out test has not been run, and the remaining routes need decisions that only the owner can make (listed in Section 6). Nothing below weakens that statement.

## 1. Best honest numbers on the original benchmark
| Measure | Value | Note |
|---|---|---|
| v18 (`v18-reasoning-o4mini`) dev, best single run | 0.6745 | an optimistic draw; five stored runs span 0.6216-0.6771 (mean 0.6576) |
| v18 TRAIN, five runs | 0.3683-0.4630 (mean 0.4160) | cached run 0.4476 |
| v18 on fresh documents (weighted kappa) | about 0.52 | `ROUTEBENCH_FRESH_EVAL_RESULT_20260930.md` |
| Sealed original held-out test | not run | one frozen run needs your separate authorization and the readiness gate |
The dev, TRAIN and every other inspected document are exposed, so none of these numbers can be presented as independent confirmation.

## 2. A separate positive result, with its limits
The historical Reveal claim-verification probe (frozen GPT-4.1 judge, 128 cases, 124 cached + 4 new judgments): **kappa 0.828125**, accuracy 117/128, document-grouped 95% interval [0.729, 0.921] (my recompute; new cost $0.016062). It is a different dataset of short single-fact claims (median claim 46 characters against a 448-character source), class-balanced, previously inspected, and chosen after its high score was known. It is **not independent validation and does not pass the original target**. Report: `ROUTEBENCH_REVEAL_COMPLETION_RESULT_20261006.md`.

## 3. What was tried against the original target (all negative or inconclusive)
| Attempt | Outcome |
|---|---|
| LoRA fine-tunes (RAGTruth pilot, TofuEval-derived), FactCG screen, HHEM/MiniCheck | below v18; MiniCheck + v18 combiner dev 0.593 vs 0.6745 |
| Second-pass auditor (audit2 v1-v3) | raw rejections matched gold on 3 dev cases but filtered result 0.6745; no gain |
| Opus judge, other model panels, stackers | no gain |
| Claim-relation audit (OpenAI and Claude variants) | Claude dev kappa 0.3609; best repaired post-hoc 0.5799; not accepted |
| Typed veto (v1.1) | in-sample 0.7134 on 3 documents; never independently validated |
| Source-integrity repair (srcint-eval-v3) | repaired-input dev kappa 0.6028 vs 0.6745; effect not measurable against run-to-run noise |
| judge-v19 screen, Stage 1a (majority vote over five v18 calls) | corrected 0, introduced 2 (dev) / 3 (TRAIN) errors; 81% of errors unanimous across five calls |
| v24 `claim_fidelity` criterion | dev 0.5938, below all five v18 runs; every paired interval includes 0; more false rejections |

## 4. What the evidence says about why
- 44 of v18's 54 pooled TRAIN+dev errors are identical across five calls; 38 of 45 false passes carry score 1.0 on every call. Repeating or aggregating v18 cannot reach them.
- In an AI-assisted blind review (not human gold; same model family as v18), reviewers sided with v18 against the published label on 35 of 38 persistent false passes. Of 27 with published expert explanations, 14 were policy mismatches (mostly opinion stated as fact from one document), 8 genuine reasoning errors (dropped hedges, tense, verb strength), 3 missing context, 1 questionable annotation, 1 unresolved. So the gap is part judge error, part benchmark-specific policy, and these cannot be separated without independent human annotation.
- v18's own run-to-run spread (0.05-0.06 kappa on dev) is comparable to the effects most variants produced.

## 5. Spend, compute and integrity (what I can document)
Journals under `.local/` hold the exact figures; examples: srcint-eval-v3 $2.0557, judge-v19 Stage 1a $8.1994, v24 evaluation $2.2336 (all within their caps), Reveal completion $0.016. Local GPU ledger about 4.47 of 40 hours; no GPU work since. Every frozen protocol (srcint v1-v3, matched v16/v18, typed-veto, judge-v19, claim-fidelity v1 unrun and v2 run) is preserved with its hash; original cases, labels, splits, caches and the sealed test are unchanged. Last full test-suite result (refreshed 2026-10-06, run serially after a memory check): **1503 passed, 5 warnings, 578.71 s**; `ruff check routebench` is clean.

## 6. What remains, and why I cannot "complete" it alone
Each item needs your explicit decision; none is scheduled.
| Option | What it needs | Honest expectation |
|---|---|---|
| A. Accept the target as unmet and close | nothing further | the state described here |
| B. Independent human review of the 88-item blind packet (44 persistent errors + 44 controls), after written policy decisions on generic plurals, non-declarative sentences and inference strength, and supplying speaker/byline metadata where it exists | human reviewers (in-house avoids a vendor rights question), your approval of the policy | tells whether the residual errors are label/policy-limited or judge errors; cannot itself raise kappa |
| C. A new, independently annotated benchmark (vendor quote draft prepared, unsent, unpriced) | rights decisions, a budget, your approval | a new benchmark, not a pass of the original |
| D. One frozen run on the sealed test | separate authorization and the readiness gate | would very likely report a kappa below 0.74 given dev; it spends the only unexposed data |
| E. Another judge-improvement plan | a separately approved plan grounded in independently verified annotation policy and evidence | no tested route has shown improvement |

## 7. Wording that is supported, and wording that is not
- Supported: "Built and froze a reproducible groundedness-judge evaluation harness with preserved artifacts; the baseline judge (o4-mini rubric) reaches kappa about 0.66 on a development split and 0.4 on a training split of a mixed benchmark; every improvement attempt tested was negative." and the Reveal sentence in Section 2 **with its limitations**.
- Not supported: any statement that the judge meets kappa 0.74 on the original benchmark, any use of the Reveal number as that result, or any claim based on relabelled or AI-adjudicated labels.

## 8. Known defects and open items
Config hashes of path-dependent variants (the rubric directory path); the MiniCheck feature-equivalence check in judge-v19 is unproven and was never run; the judge-v19 runner keeps failed rows permanently (recovered by an audited side path); Stage 1b and the classifier line are closed; the AI-panel and AI-review outputs are diagnostics only.
