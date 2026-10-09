# Source-integrity evaluation v2: bounded independent follow-up

Decision: **two fixes before the paid run**, not a new experiment design.

## Verified progress

- Real protocol SHA-256:
  `ae8247561e65b22d7a57184671dec6b6c37bdbf9ff8bfabcb0d8ee00da5f54af`.
- Same 257-case repaired development population; no removals.
- Independently reran all **38 v2 tests**: pass. Targeted Ruff: pass.
- Calibration verification: original recorded variants remain reproducible.
- Original annotation and judgment-cache hashes match the earlier review.
- No real results, spend journal, terminal or comparison files exist in the real
  `.local/srcint-eval-v2/` directory. Read-only comparison returns
  `WITHHELD_INCOMPLETE`, with **zero valid judgments**. No new kappa exists.
- New loader skips held-out IDs before typed parsing. Result re-parsing, duplicate
  refusal, explicit resume, runner lock and dependency binding are real changes.
- The misleading Fisher/causal claims have been removed.
- I did not rerun the reported 1,205-test full suite or query account credits.

## Two remaining issues, reproduced offline

### 1. Model identity is still inferred on the real client path

`srcint_eval_v2.judge_case` rejects an empty `served` value, but its actual default
request function is `matched_v16_v18.http_request`. That wrapper returns `r.model`.
The underlying client (`packages/core/aisys/llm.py:226`) uses:

```python
model=data.get("model", model)
```

Consequently, a provider response with NO `model` field is presented as having
served the requested model. The Fake(served="") test bypasses this substitution.
My additional test replaces only HTTP transport, executes the actual client and
wrapper, and confirms that a valid-looking response without model metadata still
produces `status: ok`. No network or actual provider was used.

Fix the evaluation adapter to extract model identity from the raw provider response,
not the client's fallback-normalized model property. Missing, blank, nonstring or
wrong-family metadata must yield an unverifiable/failed judgment, with its attempt
charge retained. This can be done in the new adapter without editing older frozen
clients or experiments. A present valid model value must still pass.

### 2. Strict journal validation omits numeric-domain checks

`srcint_eval_v2.py:136` accepts any int/float reservation `usd`, without rejecting
boolean, nonfinite or negative amounts. Four tests show settled reservations with
NaN, infinity, -0.01 or True are all accepted by `journal_state`. The settled charge
was a normal finite 0.01 in each test; no journal was modified in a real work dir.
Settlement checks also need explicit boolean exclusion (`bool` is a Python int).

Validate reservation and settlement values as finite, nonnegative, nonboolean
numbers. Reject malformed entries before computing totals or attempting recovery.
Do not replace invalid values with zero, erase entries, reset charges, or repair
this by relaxing the $3 cap. This is a malformed-journal acceptance issue, not
evidence that normal generated reservations overspend.

## Evidence and limits

Additional test file:
`.local/srcint-eval-review/test_v2_followup.py`.
It has **five deliberately failing regression cases** on current v2: one real-client
model-identity check and four invalid reservation amounts. These are independent
checks of untested paths; they do not contradict the 38 existing tests passing.

Original files and both real protocol directories were not modified by this review.
Only scratch tests and this report were added. No model inference, downloads,
vendor contact or provider spend occurred. The model/effort and native token-usage
telemetry are not exposed; API-equivalent review cost is unavailable, not zero.

## Paste into Claude

```text
Most v2 fixes are independently verified. Do one final bounded OFFLINE patch pass
for the TWO remaining findings in docs/ROUTEBENCH_SRCINT_V2_REVIEW_20261005.md.

1. Add an evaluation HTTP adapter that reads served-model identity directly from
   the raw provider JSON, never the client's requested-model fallback. Preserve
   usage accounting and the charge for missing identity; mark that judgment failed.
   Test through the real client path with mocked HTTP, including missing/blank/
   malformed/wrong model metadata and one valid response.
2. Reject boolean, NaN, infinite and negative reservation/settlement money values
   in strict journal validation before totals or recovery. Preserve every charge.

Make .local/srcint-eval-review/test_v2_followup.py pass against the new version
(adapt version/module references, preserve safety assertions and test the actual
default production adapter); rerun the 38 existing safeguard tests,
lint and calibration verification. Preserve v1/v2 code, freezes, original benchmark
files and results. Use a new versioned module/work directory/protocol and return
its new hash. Do not change population, judge, threshold, $3 cap or statistical
scope. No paid calls, downloads, vendor contact or original acceptance evaluation.
Stop after verification and wait for separate run approval.
```

## Decision after the fixes

If approved later, one bounded comparison can measure source-fidelity and
reproducibility on the repaired copy. It is not a promised model-quality gain.
Only four historical DEV errors are among changed inputs, and those DEV repairs
are punctuation-only. A repaired-copy DEV score would not certify the original
benchmark's kappa target. The target remains unmet; no score was fabricated or
improved merely by adding these safeguards.
