# Claude-subscription claim-relation experiment: frozen, NOT yet run (2026-10-04)

Separate from the frozen OpenAI experiment (`.local/claim-relation-ablation-v1`, protocol sha `8c32146a...323ce`, verified unchanged; its run ended ABORTED on OpenAI credit). No API key is involved.

## Regression added first (claim_relation_v2, never run before)
v1 treated "not", "no", modals (will/would/can/could) and single digits as ignorable, so an unreviewed "not" in "The board did not approve the plan." went undetected. `claim_relation_v2` now requires every negation (incl. n't contractions), modal, quantifier (all/any/only/...) and any digit-bearing token to be covered by a reviewed assertion; ordinary function words stay ignored. 10 new parametrised tests (omitted not/NOT/didn't/no, will/could/must, single digits 5 and 2) fail on v1 where v1 ignores the word and pass on v2; v2 suite 29 tests. v1 and its frozen manifest are untouched.

## Experiment
- Backend: `claude -p` through the existing claude.ai subscription (pro). `auth` verified: loggedIn, authMethod claude.ai, provider firstParty, subscriptionType pro; ANTHROPIC_API_KEY / AUTH_TOKEN / BASE_URL / cloud-provider variables absent, and the runner refuses if any is set and strips them from the child. Usage is recorded as notional list-price cost; nothing is billed to an API key.
- Model: `claude-opus-5-5`, effort medium (verified by a one-line smoke call: the CLI's own `modelUsage` reported exactly that model). Every case's reported model must equal it or the run stops.
- Judge input: the fixed claim-relation prompt (policy + source + candidate) only. Flags: no tools, no session, no slash commands/skills, strict (empty) MCP, project-only settings from an EMPTY working directory, fixed one-line system prompt. Harness overhead measured at about 480 input tokens (default settings added ~20k, so they were removed). Labels, groups, v18 verdicts and splits are only byte-hashed during the run and parsed first by `compare`, after `judgments.jsonl` is saved; a test enforces this.
- Population: all 128 dev cases (dev-only copy of the frozen 257-case inputs, nothing dropped). Baseline: frozen v18 verdicts. Rule: supported->1, unsupported->0, uncertain->0 (abstentions stay in the denominator). Bootstrap: document-grouped, seed 20261101, 2,000 draws.
- Gate (dev): errors fixed > new errors AND 95% lower bound of the kappa gain > 0; measured dev kappa and "reaches 0.74" reported separately. A development result on inspected cases is not held-out confirmation.
- Failure policy: one attempt per case; the first CLI error, rate limit, timeout, model mismatch, unreadable output or malformed structure stops the run (ABORTED / INCOMPLETE), no further call, comparison withheld. A subscription usage limit would end the run this way.

## Frozen identifiers
protocol file sha256 `60b8f8a1182c65c1ff5fa61bd9bd95a22135e28c8090c81a410576d13f5e1c7c` (`.local/claim-relation-claude-v1/protocol.json`); inputs: cases `9df1ecb1...8396`, labels `6a2c93ad...da96`, groups `4c8f4416...f95f`, baseline `6fdb3739...5c62`, splits `5720fb2e...9063`; candidate claim-relation-v2 content hash `adf22912...384e`; v18 `12fc8b550d954359`; Claude Code CLI 2.1.287.

## Commands (not executed)
```
export PYTHONPATH="packages/core;routebench"; H=60b8f8a1182c65c1ff5fa61bd9bd95a22135e28c8090c81a410576d13f5e1c7c
.venv/Scripts/python.exe -m evalops.claim_relation_claude run --protocol-sha256 $H --confirm RUN-CLAUDE-SUBSCRIPTION
.venv/Scripts/python.exe -m evalops.claim_relation_claude compare --protocol-sha256 $H
```
(defaults point at `.local/claim-relation-claude-v1/{inputs,protocol.json,run}`). Expected load: 128 sequential calls, a few thousand output tokens each at medium effort; it consumes subscription allowance and could hit the plan limit. The CLI version is bound into the protocol, so a Claude Code update before the run requires re-freezing.

Tests: full suite 987 passed, ruff clean; 27 new tests with a fake CLI (auth refusal, metered-variable block, request shape, label non-parsing, fail-fast on 5 anomaly types, timeout, malformed output, altered-file refusal, comparison guards). Mutation checks (dropping the model check, dropping the auth check) fail the tests.

## Result (run once, 2026-10-04): INCOMPLETE - comparison withheld
- The CLI auto-updated 2.1.287 -> 2.1.289 before the run; the frozen check refused (as designed). Only `cli_version` differed; the superseded protocol is kept (`protocol.superseded-cli-2.1.287.json`) and the run used the re-frozen protocol sha `830f6a037d293f6c2c61589785a80adbb908d57d3e824e72d2c6b70e51a2a2bb`. No judgments existed at that point.
- One run. 82 of 128 cases attempted, 81 ok, stopped at case `gnd-9d4d4d09e1f448fe`: the model's JSON was malformed (`Expecting ':' delimiter` at char 3024 of a 7,394-token answer). Frozen fail-fast rule: INCOMPLETE, 46 cases not attempted, no retry, **no kappa, no corrected/new-error counts, no gate**. CORRECTION (2026-10-04, later): the run itself never read labels, but the original `compare` command parsed the labels/groups/baseline/splits files for structural checks BEFORE it noticed the run was incomplete and withheld the result. No label was joined to any prediction and no metric was produced, but the earlier statement that labels were not parsed by anything was wrong. The completeness check now runs first (continuation module). No auth, rate-limit or model problems: every call reported `claude-opus-5-5`, subscription login claude.ai/pro, 0 API spend. Tokens: 293,016 output, notional list cost $7.71 (not an invoice), 2,413 s of model latency.
- Verdict counts on the 81 completed responses (label-blind): 32 supported, 35 unsupported, 14 uncertain (17% abstentions would be counted as not-pass).
- No further runs or experiments were done, per instruction. Held-out test untouched; labels, cases, splits unchanged.


## Continuation protocol (frozen, NOT run; supersedes nothing - the original incomplete run is preserved read-only)
- Module `routebench/evalops/claim_relation_claude_cont.py`, protocol `.local/claim-relation-claude-v1/continuation/protocol.json`, **sha256 `2ae67ff5a99690e6171d754e5e7c1daf50ca5249604efcee13b59fc4701a6491`** (binds the same five inputs, candidate v2 hash `adf22912...384e`, model claude-opus-5-5/medium, CLI 2.1.289 and flags, retry rule, call cap, import provenance).
- Uniform rule: every case gets at most TWO attempts for malformed structured output (one retry), same prompt/model/settings; every attempt is logged (`attempts.jsonl`, fsynced, resumable); the retry decision never sees labels. The original malformed attempt on `gnd-9d4d4d09e1f448fe` counts as that case's first attempt. A case malformed twice is retained as failed (no verdict invented) and the result is INCOMPLETE; the run continues to the other cases. Request errors (rate limit, CLI error, timeout, model mismatch) stop the run ABORTED, do not consume the retry, and are resumable. Hard cap: 100 new model calls.
- Reuse: the 81 valid judgments are imported only after re-verifying input, prompt, judge-code hash, model, session record and that the stored structure re-derives from the stored raw response; the original run files and protocol are hash-bound and only read (verified unchanged after freezing). Disclosed in the protocol. 47 cases need valid judgments.
- Publication: kappa only at 128/128 valid judgments; `compare` establishes completeness from cases.json and the attempts log BEFORE any label/group/baseline/split file is parsed (tests enforce this); otherwise it returns WITHHELD_INCOMPLETE and writes no comparison.
- Usage estimate (from the 82 original calls: 3,573 output tokens and 29.4 s per call, notional $0.094): expected about 48 calls, ~170k output tokens, ~23 minutes, ~$4.47 notional; worst case 94 calls, ~336k tokens, ~46 minutes, ~$8.84 notional. Subscription allowance, not an invoice; the plan limit may stop the run.
- Tests: 42 new tests (malformed output, retry limits incl. restart, tampered imports at record level, original-file preservation, crash/torn-log recovery, request-error resume, call cap, label non-parsing, publication once); full suite 1019 passed, ruff clean; two mutation checks (removing the attempt limit, removing import re-derivation) fail the tests.
- Command (not executed; needs your go-ahead):
```
export PYTHONPATH="packages/core;routebench"; H=2ae67ff5a99690e6171d754e5e7c1daf50ca5249604efcee13b59fc4701a6491
.venv/Scripts/python.exe -m evalops.claim_relation_claude_cont run --protocol-sha256 $H --confirm RUN-CLAUDE-CONTINUATION
.venv/Scripts/python.exe -m evalops.claim_relation_claude_cont compare --protocol-sha256 $H
```
Re-running `run` after an ABORTED stop resumes without re-judging valid cases. A Claude Code CLI update before the run would again require re-freezing (the version is bound).

## Restart-safety repairs and the NEW frozen continuation (supersedes the 2ae67ff5 continuation; nothing run)
Two independently reproduced defects in the continuation runner (reproductions: `.local/review_claude_cont/test_restart_review.py`, both now pass):
1. A model call whose result row was lost (crash/disk failure) was invisible after restart, so the case could get an extra malformed-output attempt. Repair: every attempt writes an fsynced `reserve` row BEFORE the model is launched, and a `result` row afterwards. A reservation without a result is UNSETTLED: it counts toward the 100-call cap and consumes one of the case's two attempts, is never replayed, and is listed in the terminal record (`unsettled_reservations`). A case whose lost attempt plus earlier malformed attempt reach two is retained as failed (INCOMPLETE); a case with one lost attempt still gets its one remaining attempt.
2. Torn-tail recovery was in-memory only: appending after a recovered torn line produced an unreadable journal on the next restart. Repair: the damaged bytes are preserved in `attempts.jsonl.torn.N` (fsynced) and the journal is restored atomically (temp file + fsync + `os.replace`); read-only callers (`compare`) never rewrite, and the first append through a read-only handle restores first. A complete final line missing only its newline is kept and normalised. Import seeding is one atomic step, and an interrupted/partial seed is completed idempotently (each verified original record exactly once, unchanged; a modified imported row is refused).
- Usage figures are now labelled as ESTIMATES (average- and max-observed-based); only the call counts are bounds (94 if no request errors; protocol cap 100).
- Tests: 54 -> 55 in `test_claim_relation_claude_cont.py` (crash before settlement, lost result, cap with unsettled reservations, interrupted launch, torn tail then further restarts, second tear, read-only handle, atomic swap crash, atomic/idempotent seeding, modified imported row); the two reviewer reproductions pass; full suite 1041 passed, ruff clean. Mutation checks (ignore unsettled reservations, exclude them from the cap, non-atomic restore) each fail tests.
- Preserved unchanged and re-hashed after freezing: original run, original protocol `830f6a03...`, superseded continuation protocol `2ae67ff5...`.

**New frozen continuation:** `.local/claim-relation-claude-v1/continuation-v3/protocol.json`, sha256 `d4a853fe1ea0917ffb8350dc55987ab9000ce089caf7f28c7b915cd47dfa0361` (81 valid imported, 47 pending, CLI 2.1.289, cap 100 calls, at most two attempts per case).
Usage ESTIMATES (from 82 original calls; not upper bounds): expected about 48 calls, ~170k output tokens, ~23 min, ~$4.47 notional; two attempts for all 47 at mean cost ~336k tokens, ~46 min, ~$8.84; at the maximum observed per-call cost ~873k tokens, ~107 min, ~$19.34 notional (subscription allowance, not an invoice).
```
export PYTHONPATH="packages/core;routebench"; H=d4a853fe1ea0917ffb8350dc55987ab9000ce089caf7f28c7b915cd47dfa0361
.venv/Scripts/python.exe -m evalops.claim_relation_claude_cont run --protocol-sha256 $H --confirm RUN-CLAUDE-CONTINUATION
.venv/Scripts/python.exe -m evalops.claim_relation_claude_cont compare --protocol-sha256 $H
```

## RESULT of the frozen continuation-v3 run (run once, 2026-10-04; protocol d4a853fe...)
- Status COMPLETE: **128/128 valid judgments** (81 imported after identity re-verification + 47 new), exactly 47 new model calls, 127 cases needed one attempt and 1 case (the originally malformed one) needed its retry. No failures, no unsettled reservations, no torn log. Model reported on every new call: `claude-opus-5-5`; auth claude.ai/pro; no API spend.
- Usage (new calls): 185,378 output tokens, 1,520 s model latency, ~25 min wall, notional list cost $4.75 (not an invoice). Within the pre-run estimates.
- **Dev result (primary, 128 cases, 46 documents; uncertain -> not-pass, frozen rule):** v18 κ **0.6745**; claim-relation (Claude Opus 5.5) κ **0.3609**. False passes fixed 7, false rejections fixed 0; **new false rejections 31, new false passes 0** (errors fixed 7 < new errors 31). Document-grouped 95% interval of the κ gain: **[-0.462, -0.105]** (2,000 draws). **IMPROVED_ON_DEV: no. DEV κ reaches 0.74: no.**
- 26 of 128 judgments (20%) were "uncertain" abstentions; counted as not-pass under the frozen rule, they produced essentially all of the new false rejections. Supplementary, non-gating view excluding the 26 abstentions (n=102): v18 κ 0.6795 vs candidate 0.6086 - the candidate is still below v18 there, and this view cannot be used to claim a pass.
- This is a development result on inspected cases, not held-out confirmation. Labels, cases, splits and the held-out test were not changed or opened. No further runs were made, per instruction.
