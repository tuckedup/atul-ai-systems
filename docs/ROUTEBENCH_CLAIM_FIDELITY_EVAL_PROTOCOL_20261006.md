> **SUPERSEDED 2026-10-06 by `ROUTEBENCH_CLAIM_FIDELITY_EVAL_V2_20261006.md` (hash `619d5547ce4c47f808663a10e36c8ec8609046df2744096b96655c6b27d0a00f`).** This v1 freeze was never run and is preserved unchanged. Its cost report omitted billed retries (reporting bug, not a cap breach), and the sentence below saying a fifth criterion "can only add rejections" was too strong: a new prompt and fresh stochastic calls can change the four old verdicts as well, so monotonic behaviour across old and new runs is not promised. Do not run v1.

# claim-fidelity-eval-v1: frozen 257-case v24 evaluation protocol (2026-10-06)

**Status: frozen offline, NOT run. Waiting for your explicit run approval.** No provider call, GPU work, download, training, relabelling, exclusion or held-out access occurred. The only thing read from the network layer is the key's presence and shape (length and format), never its value.

## Protocol hash
`protocol.json` sha256 = **`287d8195f2d2a28c5cccf9aad1d189907f654c1e53497810fe4611ef0d4b6a69`** (`.local/claim-fidelity-eval-v1/protocol.json`; `verify` and `preflight` pass offline).

## What is frozen
| Item | Value |
|---|---|
| Question | Does adding `claim_fidelity` change v18's behaviour in the intended direction without a net loss? **Descriptive** on already-inspected, evaluation-only data; not a pass of 0.74; not held-out |
| Population | The original TRAIN (129) + dev (128) cases, **original unrepaired inputs, original labels and splits**, 0 removed; the sealed test is excluded by the bounded loader |
| Variant | `v24-claim-fidelity-o4mini` = v18's config with only the rubric directory changed; config hash `4c9c23cd3b42e03e`, rubric hash `f47d0ab5d07233a8`, prompt-template hash `ef895d791c0bd860` (same as v18) |
| v18 preserved | config `12fc8b550d954359`, rubric `d72496fbb2991071`, template `ef895d791c0bd860`; re-checked at every verify |
| Decision rule | pass iff every one of the five criteria is yes (score 1.0); no threshold search; no rubric or prompt edit after the freeze (the rubric file and all code are hash-bound) |
| Baseline | All **five** stored v18 runs per case: run 1 = the cached frozen judgment, runs 2-5 = judge-v19 Stage 1a (recovered job included); the six underlying artifacts are bound by sha256 and re-verified, and the judge-v19 protocol hash `3d638808...` is recorded |
| Calls | 257 (one per case), expected cost about **$2.11** (257 x $0.0082, an estimate), worst case per reservation about $0.047 |
| **Cap** | **$3.00 cumulative** over retries and interruptions: the fsynced reserve-before-request journal counts settled charges plus every unsettled reservation at its bound; a reservation that would cross the cap is refused |
| Config-hash caveat | `config_hash` includes the rubric directory path, so the variant's config hash is specific to this checkout location; the rubric hash is path-independent and is stored in every judgment |

## Safeguards (tested with a mocked provider; 20 tests)
- **Key preflight before anything is reserved.** `preflight` and `run` check the loaded key (present, `sk-` format, ascii, no whitespace or quotes) without printing it. A missing or malformed key **refuses the run with no lock, no reservation, no row and no charge**. The earlier failure is reproduced by the check: from `routebench/` the same preflight reports `key_loaded: false`; from the repository root it reports `key_loaded: true, key_length: 164`.
- **A request that provably was not sent** (local protocol or URL error) settles its reservation at the bound, writes **no** result row, aborts and leaves the job pending; resuming judges it, so nothing is skipped and nothing becomes a permanent failed row (the Stage 1a defect).
- **Interruptions:** a killed process leaves an unsettled reservation; the run refuses to continue until `--settle-unsettled` charges it at its bound; the interrupted case is redone; `--resume` is required after any ABORTED state; a COMPLETE run is final; one runner at a time (a stale `run.lock` must be removed by hand after confirming its PID is dead).
- **Row validation:** every stored row is re-derived from its raw response (all five criteria required, `claim_fidelity` included); prompt, config and rubric hashes, settled attempt ids and the served model (from the raw provider JSON) are checked; duplicates, foreign ids or tampering stop everything.
- **A missing `claim_fidelity` verdict is a parse error, never a pass** (retried up to the config's 3 attempts, then a failed row that stops the run and withholds kappa).
- **Complete, valid coverage is required before kappa:** `compare` opens neither labels nor the baseline until the terminal record is COMPLETE, the journal is clean and 257/257 rows validate; otherwise it returns `WITHHELD_INCOMPLETE` and no metric exists. It is written once.
- **No automatic follow-up:** the module has no second stage, no retry beyond the frozen per-case attempts, and no scheduling.

## What `compare` reports (TRAIN and dev separately, pooled supplementary)
For v24 and for each of the five v18 runs: kappa, accuracy, false passes, false rejections; for each run, errors **corrected** and **new** errors introduced by v24 and verdicts changed; the 95% **document-grouped** paired bootstrap interval of kappa(v24) - kappa(run) (2,000 draws, fixed seed) and v24's own document-grouped kappa interval; the five-run kappa range and mean; **fidelity-only failures** (all four original criteria yes, `claim_fidelity` no) with their label split and how many v18's run 1 had passed; failures per criterion; **cost** (committed, actual usage, per-case mean, retried cases). Nothing gates anything: the output only decides whether a further test on new, independently annotated documents is worth proposing.

## Exact commands (PowerShell, from the repository root; none has been run)
```
cd <repo root>
$env:PYTHONPATH = "routebench"
# 1. offline check, no network:
.\.venv\Scripts\python.exe -m evalops.claim_fidelity_eval preflight --protocol-sha256 287d8195f2d2a28c5cccf9aad1d189907f654c1e53497810fe4611ef0d4b6a69
# 2. the run (needs your explicit approval; about 45 minutes at the observed 6 judgments per minute):
.\.venv\Scripts\python.exe -m evalops.claim_fidelity_eval run --protocol-sha256 287d8195f2d2a28c5cccf9aad1d189907f654c1e53497810fe4611ef0d4b6a69 --confirm RUN-CLAIM-FIDELITY-EVAL-V1
#    after any stop: add --resume; after a kill that left a reservation: --resume --settle-unsettled
# 3. only after COMPLETE:
.\.venv\Scripts\python.exe -m evalops.claim_fidelity_eval compare --protocol-sha256 287d8195f2d2a28c5cccf9aad1d189907f654c1e53497810fe4611ef0d4b6a69
```
Because Claude's background job manager was reaped twice for memory pressure, the safest way to run step 2 is this standalone command in your own PowerShell window with other heavy programs closed (memory was 1.4 GB free when I last checked).

## Checks I ran
`test_claim_fidelity_eval.py`: **20 passed** (freeze contents; baseline holds five runs with run 1 equal to the cached judgment; original inputs only; verify refuses changed inputs, code or baseline; preflight leaks nothing; missing or malformed key refuses with no side effects; complete run on original inputs; the cap; request-not-sent; interruption and explicit settle; a missing `claim_fidelity` verdict withholds kappa without opening labels; missing model identity; tampered row; full comparison structure and consistency identities; fidelity-only counts and cost against the scripted run; CLI). Earlier frozen artifacts (srcint-v3 `5cbfa028...`, judge-v19 `3d638808...`) are unchanged and still verify. `ruff check routebench`: clean.
**Not run:** the full suite. Free memory was 1.4 GB, below the threshold I set for a heavy serial job, so I did not start it. The last full run (before this module) was 1395 passed.

## Uncertainties
- The data are exposed; nothing here can validate the variant or approach 0.74.
- A fifth criterion cannot turn an identical fixed set of four old verdicts from fail to pass, but fresh stochastic calls can change those old verdicts too (corrected in v2), so false rejections may rise or fall; the report will show it, but the net effect on kappa is unknown.
- The estimate ($2.11) assumes v24's reasoning cost stays near v18's; the cap leaves $0.89 of margin, which covers at most about 19 worst-case reservations. If the cap is reached the run ends ABORTED and no kappa is reported.
- Wall-clock time and Claude's background manager are the practical risks, not money.
