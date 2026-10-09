"""One MATCHED development comparison of the historical `v16-holistic-4.1` recipe and the current baseline `v18-reasoning-o4mini` on every original groundedness subset.

Populations (fixed before any call, every case kept): dev (128: TofuEval-MediaS 63, AggreFact-CNN 41, TofuEval-MeetB 24), TRAIN (129: 63/35/31) and the two
published-label diagnostic probes the v16 peak came from (Reveal 128, WiCE 128). Each judge must hold a valid judgment for every case. Cached judgments are
re-verified (current-style cache key, config hash, rubric hash, served model, status) and reused; only missing pairs are called (v16: TRAIN + the 6 probe cases that failed
with provider errors; v18: both probes). Both judges use the SAME frozen decision rule - pass iff every rubric criterion passes (score >= 1.0) - on every subset; no threshold
is searched or transferred (on Reveal the historical 0.30 threshold and the strict rule give identical predictions - verified). Judges are never mixed across providers: v16 is
gpt-4.1, v18 is o4-mini. Labels live in a separate file that is parsed only after completeness (128+129+128+128 valid judgments per judge) is established.
Results are DEVELOPMENT evidence on inspected cases, and per-subset/pooled numbers across different label sources are never merged into a claim about the original target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import audit2_v3 as a3
from . import experiments, judge, rubrics
from .audit2_v2 import Journal, SpendError, kappa
from .dataset import CaseRecord
from .experiments import JudgmentCache, _cache_key
from .multi_judge import DATA, load_cases
from .run_calibration import GRID

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).parent / "data" / "probe"
WORK = ROOT / ".local" / "matched-v16-v18-v1"
JUDGES = ("v16-holistic-4.1", "v18-reasoning-o4mini")
PRICES = {"gpt-4.1": (2.0, 8.0), "o4-mini": (1.10, 4.40)}  # USD per 1M tokens; checked against aisys/pricing.yaml at freeze
CAP_USD = 6.0
CONFIRM = "RUN-MATCHED-OPENAI"
PASS_THRESHOLD = 1.0
SENSITIVITY_V16_THRESHOLD = 0.30  # the historical Reveal threshold: reported as a labelled sensitivity analysis only
BOOT_SEED, BOOT_DRAWS = 20261104, 2000
POPS = ("dev", "train", "reveal", "wice")


class MatchedError(ValueError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _digest(v: Any) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cfg_of(variant: str):
    return next(c for c in GRID if c.variant_id == variant)


# ---------------------------------------------------------------------------------------------------------- population / cache
def build_population() -> tuple[list[dict[str, Any]], dict[str, int], dict[str, CaseRecord]]:
    items, labels, cases = [], {}, {}
    for case, label, split in load_cases(["train", "dev"]):
        items.append({"case_id": case.case_id, "pop": split, "subset": case.meta.get("subset", "?"), "group": case.group_id})
        labels[case.case_id], cases[case.case_id] = label, case
    for name in ("reveal", "wice"):
        lab = {json.loads(x)["case_id"]: int(json.loads(x)["label"]) for x in (PROBE / f"annotations_{name}.jsonl").read_text(encoding="utf8").splitlines() if x.strip()}
        for line in (PROBE / f"cases_{name}.jsonl").read_text(encoding="utf8").splitlines():
            case = CaseRecord.model_validate_json(line)
            items.append({"case_id": case.case_id, "pop": name, "subset": case.meta.get("subset", name), "group": case.group_id})
            labels[case.case_id], cases[case.case_id] = lab[case.case_id], case
    ids = [i["case_id"] for i in items]
    if len(set(ids)) != len(ids):
        raise MatchedError("duplicate case ids across populations")
    return items, labels, cases


_CACHES: dict[str, JudgmentCache] = {}


def cache_for(pop: str) -> JudgmentCache:
    path = PROBE / "probe_judgment_cache.jsonl" if pop in ("reveal", "wice") else DATA / "judgment_cache.jsonl"
    key = str(path)
    if key not in _CACHES:  # loading is expensive; the files are read-only inputs, hash-bound through the cached-reuse digest
        _CACHES[key] = JudgmentCache(path)
    return _CACHES[key]


def verified_cached(case: CaseRecord, variant: str, pop: str) -> dict[str, Any] | None:
    """A cached judgment usable as-is: found under the CURRENT cache key (which covers case content, config hash incl. prompt template, rubric hash), ok status, matching
    variant/config/rubric/served model. Anything else is treated as missing (it will be re-judged), never silently accepted."""
    cfg = cfg_of(variant)
    j = cache_for(pop).get(_cache_key(case, cfg))
    if j is None or not j.usable:
        return None
    rub = rubrics.for_task(case.task_class, cfg.rubric_dir)
    ok = (j.variant_id == variant and j.config_hash == cfg.config_hash and j.rubric_version == f"{rub.id}@{rub.version}+{rub.hash}" and j.model_served.startswith(cfg.model)
          and j.case_id == case.case_id and j.raw_score is not None)
    return {"score": float(j.raw_score), "criteria": j.criteria, "usage": [j.prompt_tokens, j.completion_tokens], "model_served": j.model_served, "origin": "cached",
            "judgment_sha256": _digest(j.model_dump(mode="json"))} if ok else None


# ---------------------------------------------------------------------------------------------------------- freeze
def code_hashes() -> dict[str, str]:
    from aisys import llm

    here = Path(__file__)
    return {"matched_v16_v18.py": _sha(here), "judge.py": _sha(Path(judge.__file__)), "rubrics.py": _sha(Path(rubrics.__file__)), "experiments.py": _sha(Path(experiments.__file__)),
            "audit2_v2.py": _sha(Path(a3.v2.__file__)), "aisys/llm.py": _sha(Path(llm.__file__)), "aisys/pricing.yaml": _sha(Path(llm.__file__).parent / "pricing.yaml"),
            "rubric_bundle": rubrics.bundle_hash()}


def _plan(items: Sequence[Mapping[str, Any]], cases: Mapping[str, CaseRecord]) -> tuple[dict[str, dict[str, Any]], list[list[str]]]:
    cached: dict[str, dict[str, Any]] = {}
    todo: list[list[str]] = []
    for it in items:
        for v in JUDGES:
            rec = verified_cached(cases[it["case_id"]], v, it["pop"])
            if rec is None:
                todo.append([v, it["case_id"]])
            else:
                cached[f"{v}|{it['case_id']}"] = rec
    return cached, todo


def estimate(todo: Sequence[Sequence[str]], cases: Mapping[str, CaseRecord], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """ESTIMATES (not bounds) from observed cached calls of the same judge; plus the hard cap."""
    out: dict[str, Any] = {}
    obs: dict[str, list[tuple[int, int]]] = {v: [] for v in JUDGES}
    for it in items:
        for v in JUDGES:
            r = verified_cached(cases[it["case_id"]], v, it["pop"])
            if r:
                obs[v].append((r["usage"][0], r["usage"][1]))
    total = 0.0
    for v in JUDGES:
        pin, pout = PRICES[cfg_of(v).model]
        n = sum(1 for t in todo if t[0] == v)
        if not n:
            out[v] = {"calls": 0}
            continue
        # per-pop mean prompt tokens from the same judge where available, else from the other judge's prompt length
        mean_out = sum(o for _p, o in obs[v]) / max(1, len(obs[v]))
        cost = 0.0
        for t in (t for t in todo if t[0] == v):
            prompt_tokens = len(judge.build_prompt(cases[t[1]], rubrics.for_task(cases[t[1]].task_class, cfg_of(v).rubric_dir), cfg_of(v)).encode("utf8")) / 4
            cost += (prompt_tokens * pin + mean_out * pout) / 1e6
        out[v] = {"calls": n, "estimated_usd": round(cost, 2), "assumed_completion_tokens_per_call": round(mean_out), "basis": f"{len(obs[v])} cached {v} calls; prompt tokens ~ bytes/4"}
        total += cost
    out["expected_usd_estimate"] = round(total, 2)
    out["two_attempt_estimate_usd"] = round(2 * total, 2)
    out["hard_cap_usd"] = CAP_USD
    out["note"] = "estimates, not bounds; the hard bound is the fsynced spend journal (reservation per HTTP request, unsettled counts at its bound)"
    return out


def freeze(work: Path = WORK) -> dict[str, Any]:
    work = Path(work)
    if (work / "protocol.json").exists():
        raise MatchedError("protocol exists; a frozen protocol is never overwritten")
    from aisys import llm

    for model, (pin, pout) in PRICES.items():
        p = llm._PRICING.get(model, {})
        if (p.get("input_per_1m"), p.get("output_per_1m")) != (pin, pout):
            raise MatchedError(f"pricing for {model} differs from the protocol constants")
    work.mkdir(parents=True, exist_ok=True)
    items, labels, cases = build_population()
    cached, todo = _plan(items, cases)
    (work / "cases.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in (cases[i["case_id"]] for i in items)), encoding="utf8")
    (work / "population.json").write_text(json.dumps(items, sort_keys=True), encoding="utf8")
    (work / "labels.json").write_text(json.dumps(labels, sort_keys=True), encoding="utf8")  # parsed only by `compare`, after completeness
    cfgs = {v: {"config_hash": cfg_of(v).config_hash, "model": cfg_of(v).model, "mode": cfg_of(v).mode, "max_tokens": cfg_of(v).max_tokens, "temperature": cfg_of(v).temperature,
                "reasoning_effort": cfg_of(v).reasoning_effort, "samples": cfg_of(v).samples, "max_attempts": cfg_of(v).max_attempts, "prompt_template_hash": judge.prompt_template_hash(cfg_of(v).mode)}
            for v in JUDGES}
    counts = {p: sum(1 for i in items if i["pop"] == p) for p in POPS}
    body = {"protocol": "matched-v16-v18-dev-v1", "judges": cfgs, "rubric": {"id": "summarize", "hash": rubrics.for_task("summarize").hash}, "populations": counts,
            "subsets": {s: sum(1 for i in items if i["subset"] == s) for s in sorted({i["subset"] for i in items})}, "population_digest": _digest(items),
            "input_sha256": {n: _sha(work / n) for n in ("cases.jsonl", "population.json", "labels.json")},
            "cached_reuse": {"pairs": len(cached), "digest": _digest({k: v["judgment_sha256"] for k, v in sorted(cached.items())})},
            "new_calls": {"pairs": len(todo), "digest": _digest(todo), "v16": sum(1 for t in todo if t[0] == JUDGES[0]), "v18": sum(1 for t in todo if t[0] == JUDGES[1])},
            "decision_rule": {"pass_iff_score_at_least": PASS_THRESHOLD, "applied_to": "both judges, every subset", "thresholds_searched": False, "uncertain_handling": "not applicable (binary rubric)"},
            "sensitivity_not_primary": {"v16_threshold": SENSITIVITY_V16_THRESHOLD, "reason": "historical Reveal threshold; reported separately, never selected or transferred"},
            "bootstrap": {"seed": BOOT_SEED, "draws": BOOT_DRAWS, "unit": "group (document), paired across judges, within population"},
            "cap_usd": CAP_USD, "prices_usd_per_1m": PRICES, "code": code_hashes(), "estimate": estimate(todo, cases, items),
            "completeness": "every (judge, case) pair needs a valid judgment; labels are not parsed before that is established; failed pairs are retained and reported, never dropped",
            "scope": "development evidence on inspected cases; not held-out; not a claim about the original target"}
    (work / "protocol.json").write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf8")
    return body


def verify(work: Path, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, CaseRecord], dict[str, dict[str, Any]], list[list[str]]]:
    work = Path(work)
    if _sha(work / "protocol.json") != expected_sha256:
        raise MatchedError("protocol hash differs from the hash recorded at freeze time")
    frozen = json.loads((work / "protocol.json").read_text(encoding="utf8"))
    for n, d in frozen["input_sha256"].items():
        if _sha(work / n) != d:
            raise MatchedError(f"input file {n} changed after the freeze")
    if frozen["code"] != code_hashes():
        raise MatchedError("code, rubric or pricing changed after the freeze")
    items = json.loads((work / "population.json").read_text(encoding="utf8"))
    cases = {}
    for line in (work / "cases.jsonl").read_text(encoding="utf8").splitlines():
        c = CaseRecord.model_validate_json(line)
        cases[c.case_id] = c
    if _digest(items) != frozen["population_digest"] or {i["case_id"] for i in items} != set(cases):
        raise MatchedError("population differs from the frozen population")
    cached, todo = _plan(items, cases)
    if _digest({k: v["judgment_sha256"] for k, v in sorted(cached.items())}) != frozen["cached_reuse"]["digest"] or _digest(todo) != frozen["new_calls"]["digest"]:
        raise MatchedError("cached judgments or the call plan differ from the frozen ones")
    for v in JUDGES:
        c = cfg_of(v)
        if frozen["judges"][v]["config_hash"] != c.config_hash or frozen["judges"][v]["prompt_template_hash"] != judge.prompt_template_hash(c.mode):
            raise MatchedError(f"judge identity of {v} changed")
    return frozen, items, cases, cached, todo


# ---------------------------------------------------------------------------------------------------------- running
Attempt = tuple[str, Any, str]  # text, raw usage, served model


def http_request(model: str, effort: str, temperature: float, max_tokens: int, messages: list[dict[str, str]]) -> Attempt:
    from aisys import llm

    extra = {"reasoning_effort": effort} if effort else {}
    r = llm._call(model, messages, None, temperature, max_tokens, extra)  # exactly one HTTP request, no client retries or fallback
    return r.text, (r.raw.get("usage") if isinstance(r.raw, dict) else None), r.model or ""


def judge_pair(variant: str, case: CaseRecord, journal: Journal, request: Callable[..., Attempt] = http_request) -> dict[str, Any]:
    """The historical recipe unchanged: judge.build_prompt / judge.parse with the registered config (model, max_tokens, temperature, effort); one journaled HTTP request per attempt."""
    from aisys import llm

    cfg = cfg_of(variant)
    pin, pout = PRICES[cfg.model]
    rub = rubrics.for_task(case.task_class, cfg.rubric_dir)
    prompt = judge.build_prompt(case, rub, cfg)
    messages, last = [{"role": "user", "content": prompt}], ""
    for attempt in range(1, cfg.max_attempts + 1):
        bound_in = sum(len(m["content"].encode("utf8")) for m in messages) + a3.MESSAGE_OVERHEAD_TOKENS
        reserve = (bound_in * pin + cfg.max_tokens * pout) / 1e6
        rid = journal.reserve(f"{variant}:{case.case_id}", attempt, reserve)
        try:
            text, raw_usage, served = request(cfg.model, cfg.reasoning_effort, cfg.temperature, cfg.max_tokens, messages)
        except llm.QuotaExceededError as exc:
            journal.settle(rid, reserve, None)
            raise SpendError(f"provider quota error: {exc}") from exc
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
        if served and not served.startswith(cfg.model):
            raise SpendError(f"served model {served!r} is not {cfg.model!r}")
        try:
            parsed = judge.parse(text, rub, cfg, case)
        except (judge.JudgeOutputError, rubrics.RubricError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": text[:2000]},
                        {"role": "user", "content": f"That response was rejected by the parser: {exc}\nReturn ONLY the JSON object in the exact shape requested, with no prose and no code fence."}]
            continue
        return {"status": "ok", "score": float(parsed.score), "criteria": parsed.criteria_detail, "usage": list(usage) if usage else None, "model_served": served, "origin": "new", "attempts": attempt}
    return {"status": "failed", "reason": last, "origin": "new"}


def run(work: Path, expected_sha256: str, request: Callable[..., Attempt] = http_request) -> dict[str, Any]:
    frozen, _items, cases, _cached, todo = verify(work, expected_sha256)
    work = Path(work)
    results = work / "results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if results.exists():
        for line in results.read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[f"{row['judge']}|{row['case_id']}"] = row["record"]
    journal = Journal(work / "spend_journal.jsonl", frozen["cap_usd"])
    status, stop = "COMPLETE", None
    with results.open("a", encoding="utf8") as fh:
        for variant, cid in todo:
            key = f"{variant}|{cid}"
            if key in done and done[key]["status"] == "ok":
                continue
            if key in done:  # failed after the judge's own attempts: retained, not retried
                continue
            try:
                rec = judge_pair(variant, cases[cid], journal, request)
            except SpendError as exc:
                status, stop = "ABORTED", str(exc)
                break
            fh.write(json.dumps({"judge": variant, "case_id": cid, "record": rec}) + "\n")
            fh.flush()
            done[key] = rec
    failed = [k for k, r in done.items() if r["status"] != "ok"]
    missing = [f"{v}|{c}" for v, c in todo if f"{v}|{c}" not in done]
    if status == "COMPLETE" and (failed or missing):
        status = "INCOMPLETE"
    terminal = {"status": status, "new_pairs_planned": len(todo), "new_ok": sum(1 for r in done.values() if r["status"] == "ok"), "failed": failed, "not_attempted": missing, "stopped_on": stop,
                "spent_usd_journal": round(journal.committed(), 4), "cap_usd": frozen["cap_usd"], "protocol_sha256": expected_sha256}
    (work / "terminal.json").write_text(json.dumps(terminal, indent=2), encoding="utf8")
    return terminal


# ---------------------------------------------------------------------------------------------------------- compare
def _scores(work: Path, cached: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    s = {k: v["score"] for k, v in cached.items()}
    results = Path(work) / "results.jsonl"
    if results.exists():
        for line in results.read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["record"]["status"] == "ok":
                    s[f"{row['judge']}|{row['case_id']}"] = row["record"]["score"]
    return s


def _kap(y: Sequence[int], p: Sequence[int]) -> float | None:
    return kappa(list(y), list(p)) if len(y) else None


def compare(work: Path, expected_sha256: str) -> dict[str, Any]:
    frozen, items, _cases, cached, _todo = verify(work, expected_sha256)
    work = Path(work)
    scores = _scores(work, cached)
    missing = [f"{v}|{i['case_id']}" for i in items for v in JUDGES if f"{v}|{i['case_id']}" not in scores]
    if missing:  # completeness FIRST: no label file has been parsed at this point
        return {"status": "WITHHELD_INCOMPLETE", "valid_pairs": len(items) * 2 - len(missing), "expected_pairs": len(items) * 2, "missing_or_failed": missing,
                "note": "labels were not read; no metric exists"}
    target = work / "comparison.json"
    if target.exists():
        raise MatchedError("comparison.json exists; the comparison is written once")
    labels = json.loads((work / "labels.json").read_text(encoding="utf8"))
    rng = random.Random(BOOT_SEED)

    def block(sel: Sequence[Mapping[str, Any]], with_ci: bool) -> dict[str, Any]:
        ids = [i["case_id"] for i in sel]
        y = [labels[c] for c in ids]
        pv = {v: [int(scores[f"{v}|{c}"] >= PASS_THRESHOLD) for c in ids] for v in JUDGES}
        out: dict[str, Any] = {"n": len(ids), "documents": len({i["group"] for i in sel}), "prevalence_supported": round(sum(y) / len(y), 4),
                               **{v: {"kappa": _kap(y, pv[v]), "false_passes": sum(1 for a, b in zip(y, pv[v], strict=True) if a == 0 and b == 1),
                                      "false_rejections": sum(1 for a, b in zip(y, pv[v], strict=True) if a == 1 and b == 0)} for v in JUDGES}}
        p16s = [int(scores[f"{JUDGES[0]}|{c}"] >= SENSITIVITY_V16_THRESHOLD) for c in ids]
        out["sensitivity_v16_at_0.30_not_primary"] = {"kappa": _kap(y, p16s), "predictions_identical_to_primary": p16s == pv[JUDGES[0]]}
        d = (out[JUDGES[0]]["kappa"] or 0) - (out[JUDGES[1]]["kappa"] or 0)
        out["kappa_difference_v16_minus_v18"] = d
        if with_ci:
            groups = sorted({i["group"] for i in sel})
            members = {g: [k for k, i in enumerate(sel) if i["group"] == g] for g in groups}
            diffs = []
            for _ in range(BOOT_DRAWS):
                idx = [k for g in (rng.choice(groups) for _ in groups) for k in members[g]]
                a, b = _kap([y[k] for k in idx], [pv[JUDGES[0]][k] for k in idx]), _kap([y[k] for k in idx], [pv[JUDGES[1]][k] for k in idx])
                if a is not None and b is not None:
                    diffs.append(a - b)
            diffs.sort()
            out["difference_95_group_bootstrap"] = [diffs[int(0.025 * len(diffs))], diffs[max(0, int(0.975 * len(diffs)) - 1)]] if diffs else None
        return out

    result: dict[str, Any] = {"status": "COMPLETE", "scope": frozen["scope"], "decision_rule": frozen["decision_rule"], "populations": {}, "subsets": {}}
    for p in POPS:
        result["populations"][p] = block([i for i in items if i["pop"] == p], True)
    for s in sorted({i["subset"] for i in items}):
        result["subsets"][s] = block([i for i in items if i["subset"] == s], False)
    result["original_groundedness_dev_plus_train_supplementary"] = block([i for i in items if i["pop"] in ("dev", "train")], True)
    result["note"] = ("per-population numbers are primary; dev+train pooling is supplementary; probe populations use different label sources and are never pooled into a claim about the "
                      "original target; nothing here is held-out confirmation")
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "estimate", "run", "compare"])
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--protocol-sha256")
    p.add_argument("--confirm", default="")
    a = p.parse_args(argv)
    try:
        if a.command == "freeze":
            body = freeze(a.work)
            print(json.dumps({"protocol_sha256": _sha(a.work / "protocol.json"), **body}, indent=2))
        elif a.command == "estimate":
            print(json.dumps(json.loads((a.work / "protocol.json").read_text(encoding="utf8"))["estimate"], indent=2))
        else:
            if not a.protocol_sha256:
                print("REFUSED: --protocol-sha256 is required", file=sys.stderr)
                return 2
            if a.command == "run":
                if a.confirm != CONFIRM:
                    print(f"REFUSED: needs --confirm {CONFIRM}", file=sys.stderr)
                    return 2
                t = run(a.work, a.protocol_sha256)
                print(json.dumps(t, indent=2))
                return 0 if t["status"] == "COMPLETE" else 3
            print(json.dumps(compare(a.work, a.protocol_sha256), indent=2))
    except (MatchedError, SpendError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
