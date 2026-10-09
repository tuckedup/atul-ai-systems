# Reveal claim-verification historical probe: completed result

Date: 2026-10-06 (America/New_York client date).

The separately scoped historical Reveal completion reached **ordinary binary
Cohen's kappa = 0.828125**, above its 0.74 point-score target, with **128/128 valid
judgments**. This is not independent validation and does not pass the original
mixed RouteBench summarization acceptance target.

## Measured result

| Measure | Value |
| --- | ---: |
| Cases retained | 128/128 |
| Existing verified cached judgments | 124 |
| New valid judgments | 4 |
| Correct predictions | 117/128 |
| Accuracy | 91.40625% |
| True passes / true rejections | 56 / 61 |
| False passes / false rejections | 3 / 8 |
| Binary Cohen's kappa | **0.828125** |
| New provider requests / retries | 4 / 0 |
| New standard-rate usage-priced cost | **$0.016062** |
| Approved cumulative cap | $0.50 |
| Unknown-cost attempts / unsettled reservations | 0 / 0 |

The four new judgments completed three previously missing negative-label cases
correctly and rejected the remaining positive-label case. No prediction, case or
label in the existing 124 was altered. The old 124-case kappa was 0.8388354562;
the full-coverage result is 0.828125, not an improvement claim about the model.

## Recipe and execution

Variant: `v16-holistic-4.1`; requested alias `gpt-4.1`; exact served snapshot on
all 128 cached/new records: `gpt-4.1-2025-04-14`. Whole-response rubric mode,
temperature 0, max output 2,000 tokens, one sample, boundary examples enabled,
no few-shot packet. All probe references are empty and published support labels
are not rendered in the judge prompt.

- Config hash: `51695bd659270b7f`.
- Rubric: `summarize@1.0.0+d72496fbb2991071`.
- Prompt-template hash: `ef895d791c0bd860`.
- Primary decision rule precommitted before new calls: pass iff score >= 1.0.
- Completion protocol SHA-256:
  `34c6927c22339cdfe77de51141a23162de564411ecd4d55698f97c2ece20c0ec`.

The parent executed the four calls through the existing single-request raw HTTP
adapter against the official OpenAI endpoint, with a journal reservation before
each request. All four settlements have usable provider-reported token counts.
Input/output totals are 4,927 / 776 tokens. At the verified standard GPT-4.1 rates
of $2/$8 per million tokens, the conservative usage-priced amount is $0.016062.
Cached-input discounts are not credited in this amount; it is not a provider
invoice. [Official model pricing](https://developers.openai.com/api/docs/models/gpt-4.1).

## Verification and team

The scope auditor checked the unique 128-case population, exact four missing IDs,
cache/config/rubric/model identity and absence of label exposure in prompts.
The implementer built the isolated completion runner and mocked tests. The parent
inspected the complete implementation, corrected conservative local-error
accounting and scope wording, and ran **14 passing mock tests** plus clean lint.
A fresh read-only reviewer returned **ship** before paid execution. An earlier
review attempt stopped at a native Codex usage limit; it made no provider calls.

Requested delegates used gpt-5.6-sol / high. Realized native model/effort and native
token telemetry were not exposed; requested settings are not runtime or billing
confirmation. The parent independently recomputed the metric and cost from the
stored records, rather than merely copying the comparison report.

The labels contain 64 supported and 64 unsupported cases. For the 128 predictions,
chance agreement is 0.5 and observed agreement is 117/128, giving
`(117/128 - 0.5) / (1 - 0.5) = 0.828125`.

## Boundaries and closure

- Historical, previously inspected, class-balanced probe; **not out-of-sample or
  independent confirmation**. The dataset was chosen after its high probe score
  was already known. Do not present this as an unbiased selection result.
- 124 old cached judgments plus four later completions; **not a fresh rerun of all
  128**, not new human annotation, and not 100 personally hand-labeled items.
- No threshold search, additional candidate, model substitution, case exclusion,
  label change, GPU work, training, download, top-up, purchase or original held-out
  access occurred during completion.
- Original probe cases, labels and cache hashes match the frozen originals.
- **Original mixed RouteBench target remains UNMET.** Its earlier scores and sealed
  held-out split are unchanged; this result must not be substituted into its gate.
- The separately scoped completion is closed. No follow-up calls are authorized
  or scheduled; unused budget is not permission for another experiment.

Accurate short description: "Completed a frozen GPT-4.1 source-support judge on a
128-case historical Reveal claim-verification probe at Cohen's kappa 0.8281, with
full coverage and reproducible artifacts." Include its historical-probe limitation
when using this statement; do not replace the original resume acceptance claim.

## Artifacts and cost receipt

Run files are under `.local/reveal-completion-20261006/work-v2/`: protocol.json,
cases.jsonl, cached_124.json, results.jsonl, spend_journal.jsonl, terminal.json,
comparison.json and independent-verification.json. The isolated runner/tests,
user approval and pricing verification are in the parent directory.

API-equivalent **agent-work** cost receipt: unavailable because native tools did
not expose token usage for the parent, implementer, scope auditor or reviewers.
Routed agent USD, same-token Astra repricing and delegation savings are unavailable;
this is not a subscription-charge estimate. The separate evaluation's observed
token counts and conservative $0.016062 usage-priced cost are reported above.
