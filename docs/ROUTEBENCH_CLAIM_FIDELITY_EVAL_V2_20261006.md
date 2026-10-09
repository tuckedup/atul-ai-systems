# claim-fidelity-eval-v2: corrected frozen protocol (2026-10-06)

**Status: frozen offline, NOT run. $0 committed, no provider call.** Supersedes `claim-fidelity-eval-v1` (hash `287d8195f2d2a28c5cccf9aad1d189907f654c1e53497810fe4611ef0d4b6a69`), which is **preserved unchanged and unrun**: its module `claim_fidelity_eval.py` and `.local/claim-fidelity-eval-v1/` were not touched, and v1 still verifies. Only cost reporting changed; the judge, prompt, rubric, population, baseline, journal, safeguards and the $3.00 cumulative cap are identical.

## Corrected protocol hash
**`619d5547ce4c47f808663a10e36c8ec8609046df2744096b96655c6b27d0a00f`** (`.local/claim-fidelity-eval-v2/protocol.json`; `verify` and the offline key preflight pass: key loaded, 164 characters, format ok, no network call, 257 cases, cap $3.00, $0.00 committed, nothing unsettled).

## The bug and the fix
Your review's reproduction was correct: v1's `compare` summed `usage` only from each case's final successful row, so billed parse retries were omitted (simulated: journal $0.5960, report $0.5937). It was a reporting bug, not a cap breach; the journal always counted every attempt.
v2 derives cost and counts from the **spend journal** (strictly validated first):
- `known_usage_cost_usd`: usage-based cost of **every** journaled attempt that recorded usage, retries included.
- `unknown_cost_attempts` and `unknown_cost_bound_charges_usd`: attempts charged at their reservation bound because no usable usage exists (unsent requests, interrupted attempts settled with `--settle-unsettled`, missing usage), reported **separately and never as zero**.
- `unsettled_attempts` and `unsettled_commitment_usd`: any reservation still open (none at COMPLETE).
- `committed_usd` with a check that known + unknown + unsettled reconcile to it and to the journal.
- Counts: `journal_attempts_reserved` / `_settled`, `successful_final_judgments`, `retry_attempts_within_stored_rows`, `attempts_outside_stored_rows` (unsent or interrupted), `cases_with_more_than_one_attempt`, `max_attempts_for_one_case`.
- The v1 field `actual_usage_usd` no longer exists. Usage-based cost is provider-reported tokens at the frozen prices; bound charges are conservative reservations; **neither is a provider invoice**.

## Wording correction (behaviour)
Adding a fifth criterion cannot turn an identical, fixed set of four old yes/no verdicts from fail to pass. **But a new prompt and fresh stochastic calls can change the four old verdicts as well** (v18's own five runs differ from one another), so v24 can pass cases that a v18 run rejected, and nothing promises monotonic rejection behaviour across old and new runs. This is now in the protocol (`behaviour_caveat`) and the module documentation. My earlier sentence that a fifth criterion "can only add rejections" was too strong and is corrected in the v1 document.

## Checks
- `test_claim_fidelity_eval_v2.py` (20, the v1 suite adapted) + `test_claim_fidelity_eval_v2_cost.py` (7 new regressions): **27 passed**. The regressions: your reproduction (a billed parse retry is now in the reported usage, matching the journal); a resumed interruption (the interrupted attempt is counted, its bound charge reported separately, the case redone); missing usage (charged at the bound, reported separately, known usage not inflated); an unsent request (an attempt outside the stored rows, bound charge); retry + interruption + missing usage together reconcile to the journal total; the v1 freeze is preserved, unrun and still verifies; v2 refuses to freeze if the v1 freeze was edited or has run artifacts.
- `ruff check routebench`: clean.
- **Not run: the full suite.** Free memory was 935 MB, below my threshold, so I did not start it. The last full run (before the claim_fidelity evaluation modules) was 1395 passed; the earlier v1 module and its 20 tests are unchanged.

## Commands (PowerShell, repository root; none has been run; the run needs your explicit approval)
```
cd <repo root>
$env:PYTHONPATH = "routebench"
.\.venv\Scripts\python.exe -m evalops.claim_fidelity_eval_v2 preflight --protocol-sha256 619d5547ce4c47f808663a10e36c8ec8609046df2744096b96655c6b27d0a00f
.\.venv\Scripts\python.exe -m evalops.claim_fidelity_eval_v2 run --protocol-sha256 619d5547ce4c47f808663a10e36c8ec8609046df2744096b96655c6b27d0a00f --confirm RUN-CLAIM-FIDELITY-EVAL-V2
#   after any stop add --resume; after a kill that left a reservation add --settle-unsettled (confirm the recorded PID is dead and remove a stale run.lock first)
.\.venv\Scripts\python.exe -m evalops.claim_fidelity_eval_v2 compare --protocol-sha256 619d5547ce4c47f808663a10e36c8ec8609046df2744096b96655c6b27d0a00f
```
The v1 confirm string and hash must not be used. About 45 minutes at the observed rate; run it in your own window with heavy programs closed (memory was tight again).

## Unchanged limits
Exposed, evaluation-only data; descriptive only; no threshold search, prompt edit after the freeze, automatic follow-up, GPU work or held-out access; v24's net effect on false rejections and kappa is unknown; the variant's config hash depends on this checkout path.
