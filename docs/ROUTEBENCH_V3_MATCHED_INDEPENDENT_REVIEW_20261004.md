# Independent review: aggregation v3 and proposed matched experiment

2026-10-04. Read-only diagnostics of stored dev/probe results. No provider requests, purchases, new thresholds, label changes or held-out evaluation.

## Findings

The frozen matched protocol `281b169d64519cce2c4b03bad21ddf39ebcbf241dc527f363879ab79f2c4b77f` verifies locally. It contains 635 cached pairs and 391 missing pairs. The missing calls are 129 v16 TRAIN judgments, 6 v16 probe completions, and 256 v18 probe judgments.

Crucially, BOTH judges already have complete cached results on the original 128 dev cases:

| Population | Judge | Coverage | kappa at frozen all-pass rule |
|---|---|---:|---:|
| Original dev | v16 holistic GPT-4.1 | 128/128 | 0.6402569593147751 |
| Original dev | v18 o4-mini | 128/128 | 0.6745182012847966 |
| Reveal probe | v16 | 124/128 | 0.838835456199636 |
| WiCE probe | v16 | 126/128 | 0.44778362133734034 |

The two probe rows are incomplete diagnostics, not release results. No new calls are required to compare the complete dev results. The proposed calls would fill in training and cross-source diagnosis; they do not introduce a new classifier or alter existing dev predictions, so they cannot improve those two dev kappas.

On the 124 completed Reveal items, every cached score is either 1.0 or <=0.25. Consequently threshold 0.30 and threshold 1.0 produce identical predictions. The earlier threshold search occurred, but the exact 0.30 value did not cause the peak. This corrects any suggestion that transferring that threshold would reproduce the gain on other populations.

## V3 discrepancy resolved

V3 differs from the earlier offline diagnostic on exactly one case, `gnd-71ce0e9a4754c173` (published unsupported label). The candidate starts with `(Source: CNN)\r\n` followed by a question. The model declared an uncovered `\n` at offsets 13:14, which actually slice the `\r` of CRLF. The earlier diagnostic conservatively retained a declared span whose offsets/text did not match. V3 finds the uniquely matching newline and recognizes the candidate slices and declared text as non-substantive; it changes uncertain to supported.

Thus v3 releases one MORE demotion, not one fewer. This introduces one additional false pass versus the earlier diagnosis, explaining kappa 0.5799 rather than 0.5976. Do not change the rule specifically to regain this case's published label: the differing treatment is a documented offset-resolution choice, and both metrics are post-hoc dev diagnoses.

## Recommendation

Do not purchase credits or execute the 391-call protocol merely to establish whether v16 beats v18 on dev. That answer is already available: it does not, under the proposed fixed rule.

Publish a separately labelled cache-only diagnostic with coverage, identities, per-subset error counts and paired document-bootstrap uncertainty on the complete dev population. Preserve the frozen matched protocol, original candidate results, and the sealed original held-out set. Resolve the explanatory discrepancy in the write-up. A paid cross-source/training comparison is a separate optional research question, not an implementation that reaches 0.74.

The next model-quality intervention needs a new, specific hypothesis grounded in the remaining semantic errors. Copying the already-measured holistic recipe onto the same dev inputs cannot create a new improvement.
