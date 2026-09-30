# Next-step options, revised against the actual data (2026-09-29)

Status: PROPOSAL ONLY. Nothing below has been run beyond the MiniCheck inference noted in "Facts". No paid calls,
downloads, training or dev evaluation until an option and budget are approved. Test split untouched; all labels
(including those disputed in the AI review) kept as they are.

## Facts (checked in the repo)
- Headline TRAIN: 129 cases, 40 documents (max 22 cases/document); 35 CNN, 63 MediaS, 31 MeetB; positive rate 0.52.
  Dev: 128 cases / 46 documents, positive rate 0.62. Test: 263 / 50, positive rate 0.43. Kappa depends on
  prevalence, so a threshold or weight tuned on dev may not transfer; dev is also no longer a fresh validation set
  (it has been inspected and tuned on repeatedly).
- TRAIN has ZERO cached v18/o4-mini judgments (only 27-56 older gpt-4o-mini / gpt-4.1 judgments on a subset).
  Any stacking with v18 therefore needs NEW paid judgments; it is not API-free. (This corrects my earlier wording.)
- MiniCheck-Flan-T5-Large scores on TRAIN were computed locally with the already-downloaded weights (no download,
  no spend, 202 sentence passes). MiniCheck alone on TRAIN: kappa 0.496 at its own 0.5 cut; leave-one-document-out
  threshold selection gives 0.45. Consistent with its dev result (0.55), i.e. weaker than the LLM judge.
- With 40 training documents, any grouped-CV estimate is noisy; expect wide intervals and do not read small
  differences as effects.

## Option 1: regularized MiniCheck + v18 combiner (needs approval for paid TRAIN judgments)
- Missing input: v18 judgments on the 129 TRAIN cases. Estimated cost ~ $1.0 (v18 cost $0.988 for 128 dev cases;
  TRAIN prompts are shorter on average); requested cap $1.50 including retries; serial-safe; v18 config hash
  12fc8b550d954359 unchanged. Coverage must be 129/129 or the fit is not run.
- Model (frozen before any dev evaluation): L2-regularized logistic regression on 3 features: logit of MiniCheck
  full-claim probability, logit of MiniCheck weakest-sentence probability, v18 raw score. No subset indicator, no
  interactions. C in {0.01, 0.1, 1, 10} and the decision threshold are chosen by 5-fold document-grouped CV within
  TRAIN (fixed seed) maximizing pooled out-of-fold kappa; final model refit on all TRAIN.
- Then ONE dev evaluation, reported as a non-independent check, with the paired document-bootstrap interval versus
  v18 alone and versus MiniCheck alone, coverage, confusion and corrected/new errors. If it does not clearly beat
  v18, stop. No dev-based re-tuning of weights or threshold.
- Expected limitation: with 129 rows the only reliable finding is whether the two signals are complementary; gains
  over v18 (0.6745 dev) are unproven, and the earlier 0.698 dev blend was tuned on dev.

## Option 2: entirely local alternative (no API spend)
- 2a (recommended local first step, no training): calibrate MiniCheck alone on TRAIN (Platt scaling + threshold by
  grouped CV). Already measured: ~0.45-0.50 on TRAIN. Cheap, but it cannot beat the LLM judge by itself; useful only as
  a sanity check.
- 2b (fine-tuning, higher risk): LoRA fine-tune MiniCheck-Flan-T5-Large (770M, MIT, already downloaded) on the 129
  TRAIN examples (summary claim -> supported/unsupported), bf16 on the 8 GB RTX 4060. To use it in any stack, produce
  5-fold document-grouped OUT-OF-FOLD predictions (5 short runs) plus one final fit. Fixed epochs (<=3) and learning
  rate declared up front, no open-ended loop. Compute: minutes per fold; no downloads needed.
  Limitations: 129 examples from 40 documents is tiny against MiniCheck's original training set; high overfitting
  risk; TRAIN skews to different prevalence than test; disputed labels are included as-is; without v18 the best
  case is a stronger single verifier, which is unlikely to reach 0.74 alone.
- Not included without separate approval: any new model/dataset download (AlignScore weights, RAGTruth, C2D/D2C),
  which would also need document-overlap and licence audits.

## Recommendation
Option 1 first: it directly tests whether MiniCheck and the o4-mini judge are complementary, using the simplest
regularized model and no dev tuning. Option 2b only if Option 1 shows complementarity and the extra verifier signal
is worth the overfitting risk. Neither is a promise of 0.74.
