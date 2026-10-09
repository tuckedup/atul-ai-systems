"""`claim-fidelity-eval-v2` (supersedes the UNRUN v1 freeze `287d8195...`, which is preserved unchanged; its module `claim_fidelity_eval.py` is untouched): the frozen, fail-closed 257-case evaluation of the opt-in `claim_fidelity` variant (v24) against v18's five stored runs. PREPARED OFFLINE; nothing here has run against a provider.

Population: the original TRAIN (129) + dev (128) cases, ORIGINAL unrepaired inputs, original labels and splits, evaluation-only and already inspected (a descriptive result, never a pass of the 0.74 target); the sealed
test is excluded by the bounded loader. One v24 judgment per case. Baselines: all five stored v18 runs (run 1 = the cached frozen judgment; runs 2-5 = judge-v19 Stage 1a, recovered job included), bound by file hash.

Safeguards (the tested srcint-v3 / judge-v19 discipline, adapted to v24 without editing any frozen module):
  * reserve-before-request, fsynced journal, one HTTP request per reservation, committed spend = settled charges + unsettled reservations at their bound, hard cap $3.00 CUMULATIVE over retries and interruptions;
  * key preflight before anything is reserved: an empty or malformed key (including the working-directory `.env` pitfall) refuses the run with no reservation, no row and no charge; the key is never printed;
  * a request that provably was not sent (local protocol / URL error) settles its reservation at the bound, writes NO result row, aborts, and leaves the job pending, so it is never skipped; resuming needs --resume,
    an interrupted reservation needs --settle-unsettled, a COMPLETE run is final, one runner at a time;
  * every row is re-validated from its stored raw response (all FIVE criteria required, score and criteria re-derived, prompt/config/rubric bound, served model verified from the raw provider JSON);
  * labels and the five-run baseline are opened only by `compare`, after completeness and every integrity gate; COMPLETE needs 257/257 valid rows, otherwise kappa is withheld;
  * no threshold search (pass iff every criterion is yes, as in v18), no edits to the prompt or rubric after the freeze (hash-bound), no automatic follow-up run, no held-out access.

v2 correction (the only functional change): COST REPORTING is derived from the spend journal, every attempt included. v1 summed only each case's final successful response and so omitted billed retries. v2 reports usage-based
cost from every journaled attempt that has recorded usage, reports attempts charged at their reservation bound because no usage was recorded (unsent, interrupted, or missing-usage attempts) SEPARATELY and never as zero, reports
unsettled commitments separately, and derives attempt counts from the journal (including interrupted attempts). Usage-based cost is computed from provider-reported tokens at the frozen prices; it is not a provider invoice.

Behaviour caveat (corrected wording): adding a fifth criterion cannot turn an identical fixed set of four old yes/no verdicts from fail to pass, BUT a new prompt and fresh stochastic calls can change the four old verdicts as
well (v18's own five runs differ from one another). v24 can therefore pass cases that a v18 run rejected, and monotonic behaviour across old and new runs is not promised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from . import audit2_v3 as a3
from . import claim_fidelity as cf
from . import judge
from . import judge_v19_analysis as an
from . import judge_v19_phase1 as jv
from . import judge_v19_recovery as rec
from . import matched_v16_v18 as mv
from . import srcint_eval_v3 as sv3
from .audit2_v2 import Journal, SpendError, kappa
from .dataset import CaseRecord

ROOT = sv3.ROOT
WORK = ROOT / ".local" / "claim-fidelity-eval-v2"
V1_WORK = ROOT / ".local" / "claim-fidelity-eval-v1"
V1_PROTOCOL_SHA256 = "287d8195f2d2a28c5cccf9aad1d189907f654c1e53497810fe4611ef0d4b6a69"
PROTOCOL_NAME = "claim-fidelity-eval-v2"
CONFIRM = "RUN-CLAIM-FIDELITY-EVAL-V2"
CAP_USD = 3.0
JV19_PROTOCOL_SHA256 = "3d6388088e4b87c0244ed7a29777dcb7a3bdbfa99fffef8ebf3af680f20ceca6"
EST_USD_PER_CALL = 0.0082  # observed $0.0078-0.0082 per v18 call; v24's prompt adds about 250 tokens, so this is deliberately not lower
BOOT_DRAWS = 2000
ROW_KEYS = {"case_id", "protocol_sha256", "record"}
INPUT_FILES = ("cases.jsonl", "population.json", "labels.json", "baseline_v18_five_runs.json")
BASELINE_ARTIFACTS = ("protocol.json", "cached_v18.json", "stage_1a/results.jsonl", "stage_1a/recovery_results.jsonl", "stage_1a/spend_journal.jsonl", "stage_1a_recovery_note.json")
NOT_SENT = (httpx.LocalProtocolError, httpx.UnsupportedProtocol, httpx.InvalidURL)
CRITERIA = ("no_contradiction", "claims_entailed", "specifics_grounded", "no_invented_relations", "claim_fidelity")


class ClaimFidelityEvalError(ValueError):
    pass


class PreflightError(ClaimFidelityEvalError):
    pass


class RequestNotSent(SpendError):
    """The request provably never left this machine; the job stays pending (never skipped, never a failed row)."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    here = {"claim_fidelity_eval_v2.py": _sha(Path(__file__)), "claim_fidelity.py": _sha(Path(cf.__file__)), "rubrics_claim_fidelity_v1/summarize.yaml": _sha(cf.RUBRIC_DIR / "summarize.yaml"),
            "judge_v19_recovery.py": _sha(Path(rec.__file__))}
    return jv.code_hashes() | here


def _abs_baseline(rel: str) -> Path:
    base = jv.WORK
    return base / rel if rel != "stage_1a_recovery_note.json" else base / rec.NOTE


# ---------------------------------------------------------------------------------------------------------------------------- freeze / verify
def build_population() -> dict[str, Any]:
    items, cases, labels, cached, _mc = jv._routebench_cases()
    scores, info = rec.assemble_scores(jv.WORK, JV19_PROTOCOL_SHA256)  # replicates 2-5, validated, recovery-aware
    five = {c: [float(cached[c]["score"]), *scores[c]] for c in cases}
    if any(len(v) != an.K for v in five.values()) or set(five) != set(cases):
        raise ClaimFidelityEvalError("the five-run v18 baseline is not complete for every case")
    return {"items": items, "cases": cases, "labels": labels, "five": five, "baseline_info": info}


def freeze(work: Path = WORK) -> dict[str, Any]:
    work = Path(work)
    if (work / "protocol.json").exists():
        raise ClaimFidelityEvalError("protocol exists; a frozen protocol is never overwritten")
    pop = build_population()
    items, cases = pop["items"], pop["cases"]
    work.mkdir(parents=True, exist_ok=True)
    (work / "cases.jsonl").write_text("".join(cases[i["case_id"]].model_dump_json() + "\n" for i in items), encoding="utf8")
    (work / "population.json").write_text(json.dumps(items, sort_keys=True), encoding="utf8")
    (work / "labels.json").write_text(json.dumps(pop["labels"], sort_keys=True), encoding="utf8")  # opened only by `compare`
    (work / "baseline_v18_five_runs.json").write_text(json.dumps(pop["five"], sort_keys=True), encoding="utf8")  # opened only by `compare`
    cfg, ident = cf.config(), cf.identity()
    if ident["v18_config_hash"] != cf.V18_IDENTITY["config_hash"] or ident["v18_rubric_hash"] != cf.V18_IDENTITY["rubric_hash"]:
        raise ClaimFidelityEvalError("v18's identity changed; refusing to freeze")
    expected = round(len(items) * EST_USD_PER_CALL, 2)
    v1_protocol = V1_WORK / "protocol.json"
    if v1_protocol.exists() and (_sha(v1_protocol) != V1_PROTOCOL_SHA256 or any((V1_WORK / n).exists() for n in ("results.jsonl", "spend_journal.jsonl", "terminal.json"))):
        raise ClaimFidelityEvalError("the v1 freeze was edited or has run artifacts; it must stay preserved and unrun")
    body = {"protocol": PROTOCOL_NAME, "status": "frozen offline; not run",
            "supersedes": f"claim-fidelity-eval-v1 {V1_PROTOCOL_SHA256} (never run; preserved unchanged); v2 corrects cost reporting only",
            "behaviour_caveat": "adding a fifth criterion cannot turn an identical set of four old yes/no verdicts from fail to pass, but a new prompt and fresh stochastic calls can change the four old verdicts as well; "
                                "v24 can pass cases a v18 run rejected; monotonic behaviour across old and new runs is not promised",
            "question": "does adding claim_fidelity change v18's behaviour in the intended direction without a net loss? DESCRIPTIVE on exposed evaluation-only data; not a pass of the 0.74 target; not held-out",
            "judge": {"variant": cf.VARIANT_ID, "config_hash": cfg.config_hash, "model": cfg.model, "max_tokens": cfg.max_tokens, "reasoning_effort": cfg.reasoning_effort, "max_attempts": cfg.max_attempts,
                      "prompt_template_hash": judge.prompt_template_hash(cfg.mode), "rubric_hash": cf.rubric().hash, "rubric_version": f"{cf.rubric().id}@{cf.rubric().version}",
                      "criteria": [c.id for c in cf.rubric().criteria], "v18_identity": cf.V18_IDENTITY, "identity": ident},
            "decision_rule": {"pass_iff_score_at_least": mv.PASS_THRESHOLD, "thresholds_searched": False, "rubric_edits_after_freeze": False},
            "population": {"cases": len(items), "dev": sum(i["pop"] == "dev" for i in items), "train": sum(i["pop"] == "train" for i in items), "original_inputs": True, "removed_cases": 0,
                           "sealed_test_excluded_by_bounded_loader": True},
            "population_digest": mv._digest(items), "input_sha256": {n: _sha(work / n) for n in INPUT_FILES},
            "baseline": {"description": "five stored v18 runs per case: run1 = cached frozen judgment, runs 2-5 = judge-v19 Stage 1a (recovered job included)", "judge_v19_protocol_sha256": JV19_PROTOCOL_SHA256,
                         "artifact_sha256": {n: _sha(_abs_baseline(n)) for n in BASELINE_ARTIFACTS}, "assembled_with": pop["baseline_info"]},
            "cap_usd": CAP_USD, "price_usd_per_1m": mv.PRICES[cfg.model],
            "estimate": {"calls": len(items), "expected_usd": expected, "basis": f"${EST_USD_PER_CALL}/call (observed v18 average, prompt slightly larger); an estimate, not a bound",
                         "hard_bound": "fsynced spend journal; one HTTP request per reservation; cumulative cap $3.00 across retries and interruptions; an unsettled reservation counts at its bound",
                         "per_reservation_bound_usd": "about 0.047"},
            "safeguards": ["key preflight before any reservation", "no row and no charge when the key is missing", "request-not-sent aborts without a row; the job stays pending", "explicit --resume / --settle-unsettled",
                           "rows re-validated from raw responses with all five criteria required", "labels and baseline opened only after completeness", "COMPLETE needs 257/257 valid rows"],
            "reporting": {"populations": ["TRAIN", "dev", "pooled (supplementary)"],
                          "per_v18_run": ["kappa", "accuracy", "false passes", "false rejections", "errors corrected by v24", "new errors introduced by v24", "verdict changes",
                                          "95% document-grouped paired bootstrap interval of kappa(v24) - kappa(run)"],
                          "v24": ["kappa with a document-grouped interval", "false passes", "false rejections", "per-criterion failures", "fidelity-only failures (all four original criteria yes, claim_fidelity no) with their label split",
                                  ("cost from the journal: usage-based cost of every attempt with recorded usage (retries included), attempts charged at their bound because usage is unknown (reported separately, never zero), "
                                   "unsettled commitments, journaled attempt counts (including interrupted attempts)")], "bootstrap": {"seed": an.SEED, "draws": BOOT_DRAWS, "unit": "document"}},
            "boundary": ["no threshold search", "no prompt or rubric edit after the freeze", "no automatic follow-up run", "no held-out access", "no label, case or split change", "a failed or missing judgment withholds kappa"],
            "code": code_hashes()}
    (work / "protocol.json").write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf8")
    return body


def verify(work: Path, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, CaseRecord]]:
    work = Path(work)
    if _sha(work / "protocol.json") != expected_sha256:
        raise ClaimFidelityEvalError("protocol hash differs from the hash recorded at freeze time")
    frozen = json.loads((work / "protocol.json").read_text(encoding="utf8"))
    for n, d in frozen["input_sha256"].items():
        if _sha(work / n) != d:
            raise ClaimFidelityEvalError(f"input file {n} changed after the freeze")
    for n, d in frozen["baseline"]["artifact_sha256"].items():
        if _sha(_abs_baseline(n)) != d:
            raise ClaimFidelityEvalError(f"baseline artifact {n} changed after the freeze")
    if frozen["code"] != code_hashes():
        raise ClaimFidelityEvalError("code, dependency, rubric or pricing changed after the freeze")
    cfg = cf.config()
    j = frozen["judge"]
    if (j["config_hash"], j["rubric_hash"], j["prompt_template_hash"]) != (cfg.config_hash, cf.rubric().hash, judge.prompt_template_hash(cfg.mode)):
        raise ClaimFidelityEvalError("judge identity changed")
    if {k: cf.identity()[k] for k in ("v18_config_hash", "v18_rubric_hash", "v18_prompt_template_hash")} != {"v18_config_hash": cf.V18_IDENTITY["config_hash"], "v18_rubric_hash": cf.V18_IDENTITY["rubric_hash"],
                                                                                                          "v18_prompt_template_hash": cf.V18_IDENTITY["prompt_template_hash"]}:
        raise ClaimFidelityEvalError("v18's identity changed")
    items = json.loads((work / "population.json").read_text(encoding="utf8"))
    cases = {}
    for line in (work / "cases.jsonl").read_text(encoding="utf8").splitlines():
        c = CaseRecord.model_validate_json(line)
        cases[c.case_id] = c
    if mv._digest(items) != frozen["population_digest"] or {i["case_id"] for i in items} != set(cases) or len(items) != 257:
        raise ClaimFidelityEvalError("population differs from the frozen population")
    return frozen, items, cases


# ---------------------------------------------------------------------------------------------------------------------------- preflight
def key_status() -> dict[str, Any]:
    """Facts about the loaded key without exposing it. Never makes a network call."""
    from aisys import llm

    key = llm.settings.openai_api_key or ""
    ok = bool(key) and key.startswith("sk-") and key.isascii() and not re.search(r"[\s'\"]", key) and 20 <= len(key) <= 400
    return {"key_loaded": bool(key), "key_length": len(key), "key_format_ok": ok, "working_directory": os.getcwd(), "env_file_in_working_directory": Path(".env").exists(),
            "base_url": llm.settings.openai_base_url, "network_call_made": False}


def require_key() -> dict[str, Any]:
    st = key_status()
    if not st["key_loaded"]:
        raise PreflightError("no API key is loaded: the .env file is read relative to the working directory, so run from the repository root; nothing was reserved, written or charged")
    if not st["key_format_ok"]:
        raise PreflightError("the loaded API key has an unexpected format (expected an ascii sk- key without whitespace or quotes); nothing was reserved, written or charged")
    return st


def preflight(work: Path, expected_sha256: str) -> dict[str, Any]:
    frozen, items, _cases = verify(work, expected_sha256)
    st = key_status()
    state = sv3.journal_state(Path(work) / "spend_journal.jsonl", frozen["cap_usd"]) if (Path(work) / "spend_journal.jsonl").exists() else {"committed": 0.0, "unsettled": []}
    return st | {"protocol_verified": True, "cases": len(items), "cap_usd": frozen["cap_usd"], "committed_usd": round(state["committed"], 4), "unsettled": state["unsettled"],
                 "ready_to_run": st["key_loaded"] and st["key_format_ok"] and not state["unsettled"]}


# ---------------------------------------------------------------------------------------------------------------------------- judgments
def judge_case(case: CaseRecord, journal: Journal, request: Callable[..., Any], sha: str) -> dict[str, Any]:
    """The frozen v24 recipe (judge.build_prompt / judge.parse with the claim_fidelity rubric). One journaled HTTP request per attempt; raw response and attempt ids retained."""
    from aisys import llm

    cfg, rub = cf.config(), cf.rubric()
    pin, pout = mv.PRICES[cfg.model]
    prompt = judge.build_prompt(case, rub, cfg)
    base = {"prompt_sha256": hashlib.sha256(prompt.encode("utf8")).hexdigest(), "config_hash": cfg.config_hash, "rubric_hash": rub.hash}
    messages, last, ids = [{"role": "user", "content": prompt}], "", []
    for attempt in range(1, cfg.max_attempts + 1):
        bound_in = sum(len(m["content"].encode("utf8")) for m in messages) + a3.MESSAGE_OVERHEAD_TOKENS
        reserve = (bound_in * pin + cfg.max_tokens * pout) / 1e6
        rid = journal.reserve(f"{cf.VARIANT_ID}:{case.case_id}", attempt, reserve)
        ids.append(rid)
        try:
            text, raw_usage, served = request(cfg.model, cfg.reasoning_effort, cfg.temperature, cfg.max_tokens, messages)
        except llm.QuotaExceededError as exc:
            journal.settle(rid, reserve, None)
            raise SpendError(f"provider quota error: {exc}") from exc
        except NOT_SENT as exc:
            journal.settle(rid, reserve, None)  # conservative: charged at the bound although nothing was sent
            raise RequestNotSent(f"the request could not be sent ({type(exc).__name__}); reservation settled at its bound; the job stays pending") from exc
        except Exception as exc:  # noqa: BLE001 - a failed request may still have billed
            journal.settle(rid, reserve, None)
            last = f"{type(exc).__name__}: {exc}"
            continue
        usage = a3.validated_usage(raw_usage)
        if usage is None:
            journal.settle(rid, reserve, None)
        else:
            cost = (usage[0] * pin + usage[1] * pout) / 1e6
            journal.settle(rid, reserve, max(cost, reserve * (1 + 1e-9)) if usage[0] > bound_in or usage[1] > cfg.max_tokens else cost)
        if not isinstance(served, str) or not served.strip():
            return base | {"status": "failed", "reason": "served model identity missing, blank or not a string in the raw provider response; judgment unverifiable", "attempt_ids": ids, "identity_failure": True}
        if not served.startswith(cfg.model):
            return base | {"status": "failed", "reason": f"served model {served!r} is not in the {cfg.model!r} family; judgment unverifiable", "attempt_ids": ids, "identity_failure": True}
        try:
            parsed = judge.parse(text, rub, cfg, case)  # requires EVERY criterion, claim_fidelity included
        except (judge.JudgeOutputError, judge.RubricError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": text[:2000]},
                        {"role": "user", "content": f"That response was rejected by the parser: {exc}\nReturn ONLY the JSON object in the exact shape requested, with no prose and no code fence."}]
            continue
        return base | {"status": "ok", "score": float(parsed.score), "criteria": parsed.criteria_detail, "raw_response": text, "usage": list(usage) if usage else None, "model_served": served,
                       "attempts": attempt, "attempt_ids": ids}
    return base | {"status": "failed", "reason": last or "no usable response", "attempt_ids": ids}


def validate_rows(work: Path, items: Sequence[dict[str, Any]], cases: dict[str, CaseRecord], state: dict[str, Any], sha: str) -> dict[str, dict[str, Any]]:
    path = Path(work) / "results.jsonl"
    if not path.exists():
        return {}
    cfg, rub = cf.config(), cf.rubric()
    expected = {i["case_id"] for i in items}
    rows: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClaimFidelityEvalError(f"results line {n} is unreadable") from exc
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ClaimFidelityEvalError(f"results line {n}: unexpected row shape")
        cid, r = row["case_id"], row["record"]
        if cid not in expected:
            raise ClaimFidelityEvalError(f"results line {n}: foreign case id {cid!r}")
        if cid in rows:
            raise ClaimFidelityEvalError(f"results line {n}: duplicate result for {cid} (conflicting duplicates are never merged)")
        if row["protocol_sha256"] != sha or not isinstance(r, dict):
            raise ClaimFidelityEvalError(f"results line {n}: row is not bound to this protocol")
        ids = r.get("attempt_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(a, str) for a in ids):
            raise ClaimFidelityEvalError(f"results line {n}: missing attempt ids")
        for a in ids:
            res = state["reserves"].get(a)
            if res is None or a not in state["settled"] or res.get("case") != f"{cf.VARIANT_ID}:{cid}" or a in used:
                raise ClaimFidelityEvalError(f"results line {n}: attempt {a} is not a settled reservation of this case")
            used.add(a)
        prompt = judge.build_prompt(cases[cid], rub, cfg)
        if r.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf8")).hexdigest() or r.get("config_hash") != cfg.config_hash or r.get("rubric_hash") != rub.hash:
            raise ClaimFidelityEvalError(f"results line {n}: prompt/config/rubric binding does not match")
        if r.get("status") == "failed":
            if not isinstance(r.get("reason"), str):
                raise ClaimFidelityEvalError(f"results line {n}: failed row without a reason")
        elif r.get("status") == "ok":
            score, served = r.get("score"), r.get("model_served")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ClaimFidelityEvalError(f"results line {n}: invalid score {score!r}")
            if not isinstance(served, str) or not served or not served.startswith(cfg.model):
                raise ClaimFidelityEvalError(f"results line {n}: served model identity missing or wrong")
            raw = r.get("raw_response")
            if not isinstance(raw, str) or not raw:
                raise ClaimFidelityEvalError(f"results line {n}: raw response not retained")
            try:
                parsed = judge.parse(raw, rub, cfg, cases[cid])
            except (judge.JudgeOutputError, judge.RubricError) as exc:
                raise ClaimFidelityEvalError(f"results line {n}: stored raw response does not parse ({exc})") from exc
            if abs(float(parsed.score) - float(score)) > 1e-9 or parsed.criteria_detail != r.get("criteria") or {c["id"] for c in parsed.criteria_detail} != set(CRITERIA):
                raise ClaimFidelityEvalError(f"results line {n}: stored score/criteria differ from the stored raw response or lack a required criterion")
        else:
            raise ClaimFidelityEvalError(f"results line {n}: unknown status {r.get('status')!r}")
        rows[cid] = r
    return rows


def run(work: Path, expected_sha256: str, request: Callable[..., Any] = sv3.http_request_raw, *, resume: bool = False, settle_unsettled: bool = False) -> dict[str, Any]:
    frozen, items, cases = verify(work, expected_sha256)
    work = Path(work)
    cap = frozen["cap_usd"]
    term_path = work / "terminal.json"
    prior = json.loads(term_path.read_text(encoding="utf8")) if term_path.exists() else None
    if prior is not None:
        if prior.get("status") == "COMPLETE":
            raise ClaimFidelityEvalError("a COMPLETE terminal record exists; the run is final and is never repeated")
        if prior.get("status") == "REFUSED_OVER_CAP":
            raise ClaimFidelityEvalError("an over-cap run was recorded; it is permanently refused")
        if not resume:
            raise ClaimFidelityEvalError(f"a {prior.get('status')} terminal record exists; resuming needs an explicit --resume")
    require_key()  # BEFORE the lock, any reservation or any row: a missing key charges and records nothing
    lock = work / "run.lock"
    try:
        with lock.open("x", encoding="utf8") as fh:
            fh.write(json.dumps({"pid": os.getpid()}))
    except FileExistsError as exc:
        raise ClaimFidelityEvalError("another runner holds run.lock (or a crash left it): confirm the recorded PID is dead, then remove it manually") from exc
    try:
        if settle_unsettled:
            sv3.settle_unsettled_at_bound(work, cap)
        state = sv3.journal_state(work / "spend_journal.jsonl", cap)
        if state["over_cap"]:
            sv3._write_terminal(work, {"status": "REFUSED_OVER_CAP", "committed_usd": round(state["committed"], 4), "cap_usd": cap, "protocol_sha256": expected_sha256})
            raise ClaimFidelityEvalError(f"committed spend ${state['committed']:.4f} exceeds the ${cap:.2f} cap: refused")
        if state["unsettled"]:
            raise ClaimFidelityEvalError(f"{len(state['unsettled'])} unsettled reservation(s) in the journal; settle them at their bound explicitly with --settle-unsettled before resuming")
        rows = validate_rows(work, items, cases, state, expected_sha256)
        journal = Journal(work / "spend_journal.jsonl", cap)
        status, stop = "COMPLETE", None
        with (work / "results.jsonl").open("a", encoding="utf8") as fh:
            for it in items:
                cid = it["case_id"]
                if cid in rows:
                    continue
                try:
                    r = judge_case(cases[cid], journal, request, expected_sha256)
                except RequestNotSent as exc:
                    status, stop = "ABORTED", f"{exc} (case {cid})"
                    break
                except SpendError as exc:
                    status, stop = "ABORTED", str(exc)
                    break
                fh.write(json.dumps({"case_id": cid, "protocol_sha256": expected_sha256, "record": r}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                rows[cid] = r
                if r.get("identity_failure"):
                    status, stop = "ABORTED", f"model identity failure on {cid}: {r['reason']}"
                    break
                if r["status"] != "ok":
                    status, stop = "ABORTED", f"first failed case {cid}: {r.get('reason')}"
                    break
        state = sv3.journal_state(work / "spend_journal.jsonl", cap)
        if state["over_cap"]:
            status, stop = "REFUSED_OVER_CAP", f"committed spend ${state['committed']:.4f} exceeds the ${cap:.2f} cap"
        elif status == "COMPLETE" and state["unsettled"]:
            status, stop = "ABORTED", "journal inconsistent after the run (unsettled reservation)"
        valid = validate_rows(work, items, cases, state, expected_sha256)
        failed = sorted(c for c, r in valid.items() if r["status"] != "ok")
        missing = [i["case_id"] for i in items if i["case_id"] not in valid]
        if status == "COMPLETE" and (failed or missing):
            status = "INCOMPLETE"
        t = {"status": status, "cases": len(items), "ok": sum(1 for r in valid.values() if r["status"] == "ok"), "failed": failed, "not_attempted": len(missing), "stopped_on": stop,
             "committed_usd": round(state["committed"], 4), "cap_usd": cap, "unsettled": state["unsettled"], "protocol_sha256": expected_sha256}
        sv3._write_terminal(work, t)
        with (work / "invocations.jsonl").open("a", encoding="utf8") as fh:
            fh.write(json.dumps({"resume": resume, "status": status, "ok": t["ok"], "committed_usd": t["committed_usd"]}) + "\n")
        return t
    finally:
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------------------------- compare
def _metrics(y: Sequence[int], p: Sequence[int]) -> dict[str, Any]:
    return {"n": len(y), "kappa": kappa(list(y), list(p)) if len(set(y)) > 1 else None, "accuracy": sum(a == b for a, b in zip(y, p, strict=True)) / len(y),
            "false_passes": sum(1 for a, b in zip(y, p, strict=True) if a == 0 and b == 1), "false_rejections": sum(1 for a, b in zip(y, p, strict=True) if a == 1 and b == 0)}


def _self_interval(groups: Sequence[str], y: Sequence[int], p: Sequence[int]) -> list[float] | None:
    import numpy as np

    rng = np.random.default_rng(an.SEED)
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    keys = sorted(members)
    vals = []
    for _ in range(BOOT_DRAWS):
        idx = [i for g in rng.choice(keys, size=len(keys)) for i in members[g]]
        yy = [y[i] for i in idx]
        k = kappa(yy, [p[i] for i in idx]) if len(set(yy)) > 1 else None
        if k is not None:
            vals.append(k)
    if len(vals) < BOOT_DRAWS // 2:
        return None
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[max(0, int(0.975 * len(vals)) - 1)]]


def block(sel: Sequence[dict[str, Any]], labels: dict[str, int], five: dict[str, list[float]], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pure: everything reported for one population (TRAIN, dev, or pooled)."""
    ids = [i["case_id"] for i in sel]
    groups = [i["group"] for i in sel]
    y = [labels[c] for c in ids]
    v24 = [an.verdict(rows[c]["score"]) for c in ids]
    crit = {c: {d["id"]: d["verdict"] for d in rows[c]["criteria"]} for c in ids}
    out: dict[str, Any] = {"cases": len(ids), "documents": len(set(groups)), "v24": _metrics(y, v24) | {"kappa_95_document_bootstrap": _self_interval(groups, y, v24)}, "v18_runs": {}}
    for r in range(an.K):
        p = [an.verdict(five[c][r]) for c in ids]
        out["v18_runs"][f"run{r + 1}"] = _metrics(y, p) | {
            "errors_corrected_by_v24": sum(1 for a, u, v in zip(y, p, v24, strict=True) if u != a and v == a), "new_errors_introduced_by_v24": sum(1 for a, u, v in zip(y, p, v24, strict=True) if u == a and v != a),
            "verdicts_changed": sum(1 for u, v in zip(p, v24, strict=True) if u != v), "v24_minus_run_kappa_95_document_bootstrap": an.paired_bootstrap(groups, y, p, v24, BOOT_DRAWS)}
    ks = [m["kappa"] for m in out["v18_runs"].values() if m["kappa"] is not None]
    out["v18_five_run_summary"] = {"kappa_min": min(ks), "kappa_max": max(ks), "kappa_mean": sum(ks) / len(ks)} if ks else None
    cf_fail = [c for c in ids if crit[c]["claim_fidelity"] is False]
    only = [c for c in cf_fail if all(crit[c][k] for k in CRITERIA[:4])]
    out["v24_criteria"] = {"failures_by_criterion": {k: sum(1 for c in ids if crit[c][k] is False) for k in CRITERIA}, "claim_fidelity_failures": len(cf_fail), "fidelity_only_failures": len(only),
                           "fidelity_only_failures_published_supported": sum(1 for c in only if labels[c] == 1), "fidelity_only_failures_published_unsupported": sum(1 for c in only if labels[c] == 0),
                           "fidelity_only_failures_where_run1_of_v18_passed": sum(1 for c in only if an.verdict(five[c][0]) == 1), "fidelity_only_failure_case_ids": sorted(only)}
    return out


def journal_cost(work: Path, rows: dict[str, dict[str, Any]], cap: float) -> dict[str, Any]:
    """Cost and attempt accounting derived from the SPEND JOURNAL, every attempt included (billed parse retries, unsent requests, interrupted attempts). Usage-based cost is summed only where usage was recorded;
    attempts charged at their reservation bound because no usable usage exists are reported SEPARATELY and are never treated as zero; unsettled commitments are separate. Not a provider invoice."""
    path = Path(work) / "spend_journal.jsonl"
    state = sv3.journal_state(path, cap)  # strict validation first (duplicates, unknown ids, malformed amounts all raise)
    reserves: dict[str, dict[str, Any]] = {}
    settles: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        (reserves if e["op"] == "reserve" else settles)[e["id"]] = e
    known = [x for x in settles.values() if x.get("actual") is not None]
    unknown = [x for x in settles.values() if x.get("actual") is None]
    unsettled = [r for i, r in reserves.items() if i not in settles]
    known_usd = sum(float(x["actual"]) for x in known)
    unknown_usd = sum(float(x["charged"]) for x in unknown)
    unsettled_usd = sum(float(r["usd"]) for r in unsettled)
    committed = sum(float(x["charged"]) for x in settles.values()) + unsettled_usd
    if abs(committed - state["committed"]) > 1e-9:
        raise ClaimFidelityEvalError("journal totals do not reconcile")
    attempts_in_rows = {a for r in rows.values() for a in r["attempt_ids"]} & set(reserves)
    ok = [r for r in rows.values() if r["status"] == "ok"]
    per_case: dict[str, int] = {}
    for r in reserves.values():
        per_case[r["case"]] = per_case.get(r["case"], 0) + 1
    return {"committed_usd": round(committed, 4), "cap_usd": cap,
            "known_usage_cost_usd": round(known_usd, 4), "attempts_with_known_usage": len(known),
            "unknown_cost_attempts": len(unknown), "unknown_cost_bound_charges_usd": round(unknown_usd, 4),
            "unsettled_attempts": len(unsettled), "unsettled_commitment_usd": round(unsettled_usd, 4),
            "components_reconcile_to_committed": abs(known_usd + unknown_usd + unsettled_usd - committed) < 1e-9,
            "journal_attempts_reserved": len(reserves), "journal_attempts_settled": len(settles), "successful_final_judgments": len(ok),
            "attempts_in_stored_rows": len(attempts_in_rows), "retry_attempts_within_stored_rows": len(attempts_in_rows) - len(ok),
            "attempts_outside_stored_rows": len(reserves) - len(attempts_in_rows), "attempts_outside_stored_rows_meaning": "unsent requests or interrupted attempts (an interrupted reservation is settled at its bound)",
            "cases_with_more_than_one_attempt": sum(1 for n in per_case.values() if n > 1), "max_attempts_for_one_case": max(per_case.values(), default=0),
            "mean_known_usage_usd_per_successful_judgment": round(known_usd / max(1, len(ok)), 5),
            "basis": "usage-based cost = provider-reported tokens x the frozen prices, summed over every journaled attempt with recorded usage; bound charges are conservative reservations, not measured cost; neither is a provider invoice"}


def compare(work: Path, expected_sha256: str) -> dict[str, Any]:
    frozen, items, cases = verify(work, expected_sha256)
    work = Path(work)
    cap = frozen["cap_usd"]
    # every integrity gate runs BEFORE labels or the baseline are opened
    state = sv3.journal_state(work / "spend_journal.jsonl", cap)
    term = json.loads((work / "terminal.json").read_text(encoding="utf8")) if (work / "terminal.json").exists() else None
    problems = []
    if state["over_cap"]:
        problems.append("committed spend exceeds the cap")
    if state["unsettled"]:
        problems.append("unsettled reservations in the journal")
    if term is None or term.get("status") != "COMPLETE" or term.get("protocol_sha256") != expected_sha256:
        problems.append("no COMPLETE terminal record for this protocol")
    rows = validate_rows(work, items, cases, state, expected_sha256)
    missing = [i["case_id"] for i in items if i["case_id"] not in rows or rows[i["case_id"]]["status"] != "ok"]
    if missing:
        problems.append(f"{len(missing)} cases lack a valid judgment")
    if problems:
        return {"status": "WITHHELD_INCOMPLETE", "problems": problems, "valid": len(items) - len(missing), "expected": len(items), "note": "labels and the baseline were not read; no kappa exists"}
    target = work / "comparison.json"
    if target.exists():
        raise ClaimFidelityEvalError("comparison.json exists; the comparison is written once")
    labels = json.loads((work / "labels.json").read_text(encoding="utf8"))
    five = json.loads((work / "baseline_v18_five_runs.json").read_text(encoding="utf8"))
    cost = journal_cost(work, rows, cap)
    result = {"status": "COMPLETE", "protocol_sha256": expected_sha256,
              "label": "DESCRIPTIVE result of the opt-in claim_fidelity variant on already-inspected, evaluation-only original inputs; not held-out; not a pass of the 0.74 target; no threshold search; no follow-up run authorized",
              "coverage": {"cases": len(items), "valid_judgments": len(rows), "removed_cases": 0},
              "cost": cost,
              "dev": block([i for i in items if i["pop"] == "dev"], labels, five, rows), "train": block([i for i in items if i["pop"] == "train"], labels, five, rows),
              "pooled_supplementary": block(items, labels, five, rows)}
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "preflight", "run", "compare"])
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--protocol-sha256")
    p.add_argument("--confirm", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--settle-unsettled", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.command == "freeze":
            body = freeze(a.work)
            print(json.dumps({"protocol_sha256": _sha(a.work / "protocol.json"), "cases": body["population"]["cases"], "cap_usd": body["cap_usd"], "estimate": body["estimate"],
                              "identity": body["judge"]["identity"]}, indent=2))
            return 0
        if not a.protocol_sha256:
            print("REFUSED: --protocol-sha256 is required", file=sys.stderr)
            return 2
        if a.command == "preflight":
            print(json.dumps(preflight(a.work, a.protocol_sha256), indent=2))
        elif a.command == "run":
            if a.confirm != CONFIRM:
                print(f"REFUSED: needs --confirm {CONFIRM}", file=sys.stderr)
                return 2
            t = run(a.work, a.protocol_sha256, resume=a.resume, settle_unsettled=a.settle_unsettled)
            print(json.dumps(t, indent=2))
            return 0 if t["status"] == "COMPLETE" else 3
        else:
            print(json.dumps(compare(a.work, a.protocol_sha256), indent=2))
    except (ClaimFidelityEvalError, sv3.SrcIntV3Error, mv.MatchedError, SpendError, rec.RecoveryError, jv.Phase1Error) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
