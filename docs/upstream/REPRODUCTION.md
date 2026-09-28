# Upstream reproduction pack

Prepared 2026-09-28, re-verified in the cloud session that added `evalops/checkers.py`.

The upstream repositories were cloned at the pinned commits below and their algorithms read
directly. Nothing has been installed into this project's environment, no weights were downloaded,
no provider calls were made, and no held-out judgment was produced.

## Pinned sources

| Source | Commit | Code licence | Role |
|---|---|---|---|
| https://github.com/Liyan06/MiniCheck | `b58b9fa69acbd1015ec970fa65dd752413a053d2` | Apache-2.0 | specialised document–claim support baseline |
| https://github.com/yuh-zha/AlignScore | `a0936d5afee642a46b22f6c02a163478447aa493` | MIT | chunk/sentence alignment baseline |
| https://github.com/prometheus-eval/prometheus-eval | `dcfb44272d5d0414832f5dbb8c2a05ebc2614234` | Apache-2.0 | reference/rubric evaluator baseline |

Both MiniCheck and AlignScore commits were independently confirmed reachable and checked out in
this session; `git rev-parse HEAD` matched the pins exactly. `tests/test_checkers.py` asserts the
same two hashes, so a silent re-pin breaks the build rather than the comparison.

**Code licences do not license model weights.** The MiniCheck README asks for separate commercial
licensing for Bespoke-MiniCheck-7B; AlignScore distributes checkpoints separately from its
MIT-licensed code. `CheckerSpec.weights_licence_checked` defaults to `False` and `load_checker`
refuses without it, so the weights decision is made deliberately rather than by a default.

## What the code actually does

Read from source, not from the papers.

### MiniCheck (`minicheck/minicheck.py`, `minicheck/inference.py`)

1. `sent_tokenize_with_newlines(doc)` splits the document, preserving paragraph breaks.
2. Sentences are grouped greedily into chunks of approximately `chunk_size` tokens — default 400,
   or 500 for `flan-t5-large`.
3. Each `(chunk, claim)` pair is scored; `softmax(logits)[:, 1]` is P(supported).
4. **Aggregation over chunks is `max`** → `max_support_prob`.
5. The binary label is `max_support_prob > 0.5`.

The claim is **not** split by the library. `Inferencer.inference`'s own docstring says so and
recommends the caller split a multi-sentence claim, adding: *"We leave the user to decide how to
aggregate the results from multiple sentences"*, and *"In general, sentence-level prediction
performance is better than that on the full-response-level."* The README repeats it. So the
response-level aggregation is ours, and it is declared as such in `AGGREGATIONS`.

### AlignScore (`src/alignscore/alignscore.py`, `src/alignscore/inference.py`)

For `nli_sp`, `inference_per_example` does:

1. `sent_tokenize(premise)`; `n_chunk = len(premise.split()) // 350 + 1`, then
   `n_chunk = max(len(premise_sents) // n_chunk, 1)`; premise sentences are grouped into chunks of
   that many sentences.
2. `sent_tokenize(hypo)` gives the response sentences.
3. The full cross product of (premise chunk × response sentence) is scored with the NLI head.
4. The score matrix is reshaped to `(n_premise_chunks, n_hypo_sents)`, then
   **`.max(dim=0).values.mean()`** — max over premise chunks per response sentence, then mean over
   sentences.

So for AlignScore, `mean` over sentences IS upstream's aggregation, and the adapter's
`alignscore-base-nli-sp` spec is faithful rather than adapted. That asymmetry with MiniCheck is
recorded in the spec notes, because "we used mean for both" would be faithful for one and an
invention for the other.

### Prometheus (`libs/prometheus-eval/prometheus_eval/prompts.py`)

Absolute grading supplies an instruction, the response, a reference answer and a score rubric, and
asks for rubric-specific feedback followed by a 1–5 score. Using those prompts with another
backbone is a **prompt baseline**, not a reproduction of the trained Prometheus model. Not adapted
here: RouteBench's target is unweighted binary kappa, and a 1–5 ordinal judge would need its own
declared binarisation before it could be compared.

## What is implemented here

`evalops/checkers.py`. Thin adapters that call upstream rather than reimplementing it:

- No heavy import at module load. `torch`, `transformers`, `nltk`, `vllm`, `minicheck` and
  `alignscore` are imported inside loader functions only, so the 53 adapter tests run with none of
  them present. `test_no_heavy_imports_at_module_load` keeps that true.
- `FakeSupportChecker` scripts support probabilities, so segmentation, aggregation, error handling,
  revision binding and cache separation are all covered before a single weight is downloaded.
- `CheckerResult` is the common record: case id, checker id, config hash, backend, upstream commit,
  model name, source and candidate hashes, segmentation settings, per-sentence scores and
  sentences, the aggregated score, status, error, latency, and cost with an explicit basis.
- A missing or failed checker output is an **error with a status**, never a zero score and never a
  failed human label. Empty input, an over-long source, a checker exception, a score-count
  mismatch and an out-of-range probability are five distinct recorded statuses.
- The premise is `case.context` **only**. Not the reference, not the task input — widening it would
  let a claim be "supported" by a gold answer the annotators never saw.
- `config_hash` includes the upstream commit, so a score from a different revision cannot be
  reused as if it came from this one.

## Still outstanding, and why

Nothing in `checkers.py` has been run against a real model. That needs approved weight downloads
and, for the comparison to mean anything, the groundedness experiment it feeds. Both are gated on
the provider budget and weights-licence decisions that are the user's to make.

## Stop condition

Copying source is complete for the three repositories. **Acceptance is not.** A score reported as
balanced accuracy, correlation, F1, weighted kappa or selective agreement is not evidence of
unweighted binary kappa ≥ 0.74. On exactly balanced binary labels that target requires 87%
accuracy, and published LLM-AggreFact factuality results cluster around 75–78%. Copying an
implementation copies neither its score nor its benchmark. Only RouteBench's own held-out
measurement can close the requirement, and the data and labels must not be changed to force a
passing value.
