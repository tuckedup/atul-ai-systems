# Model and dataset options: research decision note

Research date: 2026-09-29. No paid judge inference, downloads of model weights,
training, or evaluation runs were performed. Research connector credits may apply.

## Decision

Kappa 0.74 remains possible but unproven on the fixed RouteBench benchmark.
The evidence supports testing a different verification mechanism, not assuming
that a larger generic model or more prompting will solve the remaining errors.
Improving a judge and replacing the evaluation dataset are different experiments.

## Verified research leads

- MiniCheck trains specialized document-grounding verifiers with synthetic factual
  errors; its project releases 14K C2D/D2C training examples. Consider a pretrained
  verifier baseline before paying to fine-tune a new judge. Multi-sentence inputs
  require an explicit sentence aggregation policy. [Project](https://github.com/Liyan06/MiniCheck),
  [paper](https://arxiv.org/abs/2404.10774).
- AlignScore is a complementary entailment/alignment model, not another prompt for
  the same judge. A complementary error pattern is a hypothesis to measure, not an
  assumed ensemble gain. [Paper](https://arxiv.org/abs/2305.16739).
- FaithJudge uses annotated alternative responses to the same source. Its strong
  published results do not establish unseen-document performance under our split.
  Its FaithBench experiments also use different treatments of ambiguous labels.
  Do not copy its labels/context arrangement into RouteBench held-out evaluation.
  [Paper, sections 5 and 6](https://arxiv.org/html/2505.04847v2).
- ThinknCheck (2026) constructs training data from LLM-AggreFact's development set.
  Since RouteBench imports this dataset family and makes its own splits, audit
  overlap before considering the checkpoint an uncontaminated evaluator.
  [Paper, section 4.1](https://arxiv.org/html/2604.01652).

These papers mostly report balanced accuracy or F1, not Cohen's kappa. Do not
interpret a leaderboard number of 0.77 as evidence that our 0.74 kappa gate passes.

## Dataset shortlist and intended use

| Resource | Suggested role | Required caution |
| --- | --- | --- |
| [RAGTruth](https://arxiv.org/abs/2401.00396) | Human-annotated response/span examples for grounded QA, summarization and data-to-text; candidate external training resource. | Already represented in LLM-AggreFact. Audit source-document overlap and original splits before importing anything. |
| [FaithBench](https://github.com/vectara/FaithBench) | Separate challenging summarization stress test with expert annotations. | Not an easier route to a target. Declare handling of Consistent, Benign, Questionable and Unwanted labels before evaluation. |
| [WiCE](https://arxiv.org/abs/2303.01432) | Evidence selection and subclaim-entailment experiments. | Wikipedia claim/evidence task differs from dialogue summarization; also represented in LLM-AggreFact. |
| [C2D/D2C](https://huggingface.co/datasets/lytang/C2D-and-D2C-MiniCheck) | Synthetic training augmentation. | Not independent human gold or a replacement headline test set. |

The [LLM-AggreFact suite](https://llm-aggrefact.github.io/) includes 11 dataset
components; it is not a wholly new independent source just because downloaded
under a different name. Original FRANK and TofuEval imports also need deduplication.

## Suggested bounded implementation plan

1. Keep the existing corpus, labels, group split and kappa target frozen. Track
   any newly introduced benchmark as a separately named evaluation.
2. Audit candidate model training provenance and external-data overlap using
   document IDs plus normalized text fingerprints, not just case IDs. Verify
   license/usage terms and local runtime requirements before downloading weights.
3. Evaluate one specialized verifier on the existing dev set with fixed chunking,
   sentence aggregation and full coverage. Preserve source evidence; no silent
   truncation. Report kappa, balanced accuracy, confusion counts and per-subset
   errors, plus cost/latency. API-free local inference still uses compute.
4. Compare its error pattern to the existing judge. Consider a simple combination
   only if the models fix different errors; fit on training/out-of-fold data and
   assess on dev, not by hardcoding the reviewed disagreement cases.
5. If performance justifies further work, use diverse train-only examples and/or
   carefully audited external training data. Validate annotation-policy compatibility
   before mixing data; more data with a different target can make the judge worse.
6. Freeze the selected configuration, threshold and integrity-bound artifacts before
   a single held-out test. Repeated dev searches make the best dev score optimistic;
   test results must not become the next tuning loop.

This is a research recommendation, not authorization for paid evaluation, hardware
rental, model installation, training, or a claim that the target has been reached.
