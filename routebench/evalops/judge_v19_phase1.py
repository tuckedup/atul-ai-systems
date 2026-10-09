"""judge-v19 Phase 1 screening: frozen protocol, staged resumable runner, local MiniCheck feature job, nested-CV fit and one-shot screen.
(docs/ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_V2_20261005.md). NOTHING here has been run against a provider or a GPU; `freeze` reads local files only.

Stages (separate work directories, journals, caps and terminal records; NO automatic transition):
  1a  1,028 v18 calls: replicates 2-5 of the 257 original RouteBench TRAIN+dev inputs (replicate 1 is the cached frozen v18 judgment).        cap $10.00
  1b  5,250 v18 calls: replicates 1-5 of the 1,050 RAGTruth fitting sentences; starts only after an explicit `release-1b` by the operator.    cap $46.00
Reuses the srcint-eval-v3 safeguards without editing that frozen module: raw-JSON model identity, strict numeric-domain journal, one HTTP request per reservation, reserve-before-launch,
fail-fast on the first error, row re-validation from the stored raw response, terminal-state resume rules, single-runner lock. Estimates are not caps: the journal enforces the cap on
committed spend (settled charges plus every unsettled reservation at its bound).
Evaluation boundary: original TRAIN/dev labels are read only by `screen`, after the model is frozen and fully validated; nothing from them reaches `fit`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import judge_improvement_sample as sample_mod
from . import judge_v19_analysis as an
from . import judge_v19_minicheck as mcjob
from . import matched_v16_v18 as mv
from . import srcint_eval_v3 as sv3
from .audit2_v2 import Journal, SpendError, kappa
from .dataset import CaseRecord

ROOT = sv3.ROOT
WORK = ROOT / ".local" / "judge-v19-phase1"
PROTOCOL_NAME = "judge-v19-phase1"
VARIANT = sv3.VARIANT
STAGES = {"1a": {"cap_usd": 10.0, "confirm": "RUN-JV19-STAGE-1A"}, "1b": {"cap_usd": 46.0, "confirm": "RUN-JV19-STAGE-1B"}}
PHASE_CAP_USD, RELEASE_CONFIRM = 56.0, "RELEASE-JV19-STAGE-1B"
EST_USD_PER_CALL = 0.0080  # observed: srcint-eval-v3 settled $2.0094 over 257 calls
PROJECTION_AFTER_JOBS = 500
RAGTRUTH_DIR = ROOT / ".local" / "ragtruth-ingest-20260930"
MANIFEST_PATH = RAGTRUTH_DIR / "v5" / "training_manifest_v5.json"
SAMPLE_PATH = ROOT / ".local" / "judge-improvement" / "ragtruth_fit_sample_phase1-v1.json"
MANIFEST_SHA256 = "1645dd3fd9455002db8bec4b2fa0f2dbe40d69a11cac4133ecfecc821e2c4da2"
SAMPLE_SHA256 = "7edc80e3ead8e5f6bc308807078e6a3a7a40d9f0c6afcff9d3f7445452d66012"
TASK_INPUT = "Summarise the source document faithfully. Every statement in your summary must be supported by the document."
BOOT_SEED, BOOT_DRAWS = 20261105, 2000
INPUT_FILES = ("population.json", "cases_routebench.jsonl", "cases_ragtruth.jsonl", "labels_routebench.json", "labels_ragtruth.json", "cached_v18.json", "minicheck_cached_routebench.json")


class Phase1Error(ValueError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def code_hashes() -> dict[str, str]:
    from . import checkers
    from .pilot import ingest_audit, inputs, ragtruth

    here = {n: _sha(Path(m.__file__)) for n, m in (("judge_v19_phase1.py", sys.modules[__name__]), ("judge_v19_analysis.py", an), ("judge_improvement_sample.py", sample_mod), ("judge_v19_minicheck.py", mcjob),
                                                  ("checkers.py", checkers), ("pilot/inputs.py", inputs), ("pilot/ragtruth.py", ragtruth), ("pilot/ingest_audit.py", ingest_audit))}
    return sv3.code_hashes() | here


# ------------------------------------------------------------------------------------------------------------------------------ population
def _ragtruth_cases(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, CaseRecord], dict[str, int]]:
    from .pilot.inputs import load_texts

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf8"))
    ids = [r[0] for r in sample["rows"]]
    texts = load_texts(RAGTRUTH_DIR, manifest, ids)  # re-derives every example from the local files and refuses any difference from the manifest
    items, cases, labels = [], {}, {}
    for eid, _resp, _src, group, label, start, end in sample["rows"]:
        doc, sent = texts[eid]
        cid = f"ragtruth:{eid}"
        cases[cid] = CaseRecord(case_id=cid, group_id=group, task_class="summarize", task_input=TASK_INPUT, context=doc, candidate_output=sent, generator_id="ragtruth-llm-response",
                                source="RAGTruth/CNN-DM", source_license="unresolved; private-research risk acceptance only (not a rights clearance)",
                                reference_provenance="none; the source document is the only admissible support", meta={"example_id": eid, "start": start, "end": end})
        items.append({"case_id": cid, "example_id": eid, "group": group, "run_order": _h(f"{sample_mod.RULE}|run|{eid}"), "context_sha256": _h(doc), "candidate_sha256": _h(sent)})
        labels[cid] = int(label)
    items.sort(key=lambda i: i["run_order"])
    return items, cases, labels


def _routebench_cases() -> tuple[list[dict[str, Any]], dict[str, CaseRecord], dict[str, int], dict[str, Any], dict[str, Any]]:
    from . import multi_judge

    items, cases, labels, cached, mc = [], {}, {}, {}, {}
    mcfiles = {pop: {r["id"]: r for r in json.loads((Path(multi_judge.DATA) / f"minicheck_flan_t5_large_{pop}_scores.json").read_text(encoding="utf8"))} for pop in ("train", "dev")}
    for case, y, split in multi_judge.load_cases(["train", "dev"]):  # bounded loader: held-out cases are excluded before any annotation is parsed
        rec = mv.verified_cached(case, VARIANT, split)
        if rec is None:
            raise Phase1Error(f"{case.case_id}: no verified cached v18 judgment")
        row = mcfiles[split].get(case.case_id)
        if row is None:
            raise Phase1Error(f"{case.case_id}: no cached MiniCheck scores")
        items.append({"case_id": case.case_id, "pop": split, "subset": case.meta.get("subset", "?"), "group": case.group_id, "context_sha256": _h(case.context),
                      "candidate_sha256": _h(case.candidate_output)})
        cases[case.case_id], labels[case.case_id] = case, int(y)
        cached[case.case_id] = {"score": rec["score"], "judgment_sha256": rec["judgment_sha256"]}
        mc[case.case_id] = {"full": row["full"], "sent": row["sent"]}
    if len(items) != 257 or sum(i["pop"] == "train" for i in items) != 129 or sum(i["pop"] == "dev" for i in items) != 128:
        raise Phase1Error("expected 129 TRAIN + 128 dev original cases")
    return items, cases, labels, cached, mc


def build_population() -> dict[str, Any]:
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf8"))
    if sample["sample_sha256"] != SAMPLE_SHA256 or hashlib.sha256(json.dumps(sample["rows"], separators=(",", ":")).encode()).hexdigest() != SAMPLE_SHA256 or sample["manifest_sha256"] != MANIFEST_SHA256:
        raise Phase1Error("the RAGTruth fitting sample differs from the verified sample hash or manifest binding")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf8"))
    if manifest["manifest_sha256"] != MANIFEST_SHA256:
        raise Phase1Error("the RAGTruth training manifest differs from the verified manifest")
    if sample_mod.draw(manifest)["sample_sha256"] != SAMPLE_SHA256:
        raise Phase1Error("re-drawing the fitting sample does not reproduce the verified sample hash")
    rt_items, rt_cases, rt_labels = _ragtruth_cases(sample)
    rb_items, rb_cases, rb_labels, cached, mc = _routebench_cases()
    if len(rt_items) != 1050 or sum(1 for v in rt_labels.values() if v == 0) != 338 or len({i["group"] for i in rt_items}) != 160:
        raise Phase1Error("RAGTruth sample is not 1,050 sentences / 338 unsupported / 160 groups")
    return {"rb_items": rb_items, "rb_cases": rb_cases, "rb_labels": rb_labels, "cached": cached, "mc": mc, "rt_items": rt_items, "rt_cases": rt_cases, "rt_labels": rt_labels}


def jobs_for(stage: str, rb_items: Sequence[dict[str, Any]], rt_items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if stage == "1a":
        return [{"job_id": f"{i['case_id']}#r{r}", "base": i["case_id"], "replicate": r} for i in sorted(rb_items, key=lambda x: x["case_id"]) for r in range(2, an.K + 1)]
    if stage == "1b":  # sentence-major in hash order: the first 500 jobs are a representative random 100 sentences
        return [{"job_id": f"{i['case_id']}#r{r}", "base": i["case_id"], "replicate": r} for i in rt_items for r in range(1, an.K + 1)]
    raise Phase1Error(f"unknown stage {stage!r}")


def job_cases(jobs: Sequence[dict[str, Any]], base_cases: dict[str, CaseRecord]) -> dict[str, CaseRecord]:
    """A job's record is its base case with only the id changed: the prompt does not contain the id, so replicates send identical requests."""
    return {j["job_id"]: base_cases[j["base"]].model_copy(update={"case_id": j["job_id"]}) for j in jobs}


# ------------------------------------------------------------------------------------------------------------------------------ freeze / verify
def _estimate(n_jobs: int) -> dict[str, Any]:
    return {"jobs": n_jobs, "expected_calls": n_jobs, "max_attempts_per_job": mv.cfg_of(VARIANT).max_attempts, "estimated_usd": round(n_jobs * EST_USD_PER_CALL, 2),
            "basis": f"${EST_USD_PER_CALL}/call observed in srcint-eval-v3; an estimate, not a bound"}


def dry_run() -> dict[str, Any]:
    pop = build_population()
    stages = {s: _estimate(len(jobs_for(s, pop["rb_items"], pop["rt_items"]))) | {"cap_usd": STAGES[s]["cap_usd"]} for s in STAGES}
    return {"stages": stages, "total_calls": sum(v["expected_calls"] for v in stages.values()), "total_estimated_usd": round(sum(v["estimated_usd"] for v in stages.values()), 2),
            "phase_cap_usd": PHASE_CAP_USD, "routebench_cases": len(pop["rb_items"]), "ragtruth_sentences": len(pop["rt_items"])}


def freeze(work: Path = WORK) -> dict[str, Any]:
    work = Path(work)
    if (work / "protocol.json").exists():
        raise Phase1Error("protocol exists; a frozen protocol is never overwritten")
    if abs(sum(s["cap_usd"] for s in STAGES.values()) - PHASE_CAP_USD) > 1e-9:
        raise Phase1Error("stage caps do not sum to the phase cap")
    if an.PASS_THRESHOLD != mv.PASS_THRESHOLD:
        raise Phase1Error("analysis pass threshold differs from the original v18 threshold")
    pop = build_population()
    work.mkdir(parents=True, exist_ok=True)
    (work / "population.json").write_text(json.dumps({"routebench": pop["rb_items"], "ragtruth": pop["rt_items"]}, sort_keys=True), encoding="utf8")
    (work / "cases_routebench.jsonl").write_text("".join(pop["rb_cases"][i["case_id"]].model_dump_json() + "\n" for i in pop["rb_items"]), encoding="utf8")
    (work / "cases_ragtruth.jsonl").write_text("".join(pop["rt_cases"][i["case_id"]].model_dump_json() + "\n" for i in pop["rt_items"]), encoding="utf8")
    (work / "labels_routebench.json").write_text(json.dumps(pop["rb_labels"], sort_keys=True), encoding="utf8")  # read only by `screen`
    (work / "labels_ragtruth.json").write_text(json.dumps(pop["rt_labels"], sort_keys=True), encoding="utf8")
    (work / "cached_v18.json").write_text(json.dumps(pop["cached"], sort_keys=True), encoding="utf8")
    (work / "minicheck_cached_routebench.json").write_text(json.dumps(pop["mc"], sort_keys=True), encoding="utf8")
    cfg = mv.cfg_of(VARIANT)
    stages = {}
    for s, spec in STAGES.items():
        jobs = jobs_for(s, pop["rb_items"], pop["rt_items"])
        stages[s] = _estimate(len(jobs)) | {"cap_usd": spec["cap_usd"], "confirm": spec["confirm"], "job_digest": mv._digest(jobs),
                                            "projection_stop": {"after_jobs": PROJECTION_AFTER_JOBS, "rule": "stop if committed/jobs_done x total_jobs exceeds the stage cap"} if s == "1b" else None}
    body = {"protocol": PROTOCOL_NAME, "proposal": "docs/ROUTEBENCH_JUDGE_IMPROVEMENT_PROPOSAL_V2_20261005.md",
            "judge": {"variant": VARIANT, "config_hash": cfg.config_hash, "model": cfg.model, "max_tokens": cfg.max_tokens, "reasoning_effort": cfg.reasoning_effort,
                      "prompt_template_hash": mv.judge.prompt_template_hash(cfg.mode)}, "rubric_hash": mv.rubrics.for_task("summarize").hash,
            "decision_rule": {"pass_iff_score_at_least": mv.PASS_THRESHOLD, "thresholds_searched": False},
            "data_policy": {"fit": "1,050 RAGTruth CNN/DM sentences (160 groups, 338 unsupported) only; private-research risk acceptance, not a rights clearance",
                            "evaluation_only": "original RouteBench TRAIN 129 + dev 128, original unrepaired inputs; never used to fit, select, calibrate or threshold",
                            "untouched": ["70 TofuEval upstream documents", "RAGTruth select/validation/test partitions", "sealed test", "all frozen v1-v3 artifacts"]},
            "ragtruth": {"manifest_sha256": MANIFEST_SHA256, "manifest_file_sha256": _sha(MANIFEST_PATH), "sample_sha256": SAMPLE_SHA256, "sample_file_sha256": _sha(SAMPLE_PATH), "sample_rule": sample_mod.RULE,
                        "documents": len({i["group"] for i in pop["rt_items"]}), "sentences": len(pop["rt_items"]), "unsupported": sum(1 for v in pop["rt_labels"].values() if v == 0)},
            "routebench": {"cases": len(pop["rb_items"]), "train": sum(i["pop"] == "train" for i in pop["rb_items"]), "dev": sum(i["pop"] == "dev" for i in pop["rb_items"]), "digest": mv._digest(pop["rb_items"])},
            "arms": {"A0": "cached frozen v18 verdict (score >= 1.0)", "A1": "majority vote (>= 3 of 5) over each call's original all-criteria-pass verdict; NOT a mean-score threshold",
                     "A2": "class-balanced L2 logistic regression on schema S (keeps raw scores as features), bands from pooled out-of-fold class-conditional rates"},
            "features": list(an.FEATURES), "nested_cv": {"seed": an.SEED, "outer": an.OUTER_FOLDS, "inner": an.INNER_FOLDS, "final_inner": an.FINAL_INNER_FOLDS, "c_grid": list(an.C_GRID)},
            "bands": {"alpha_pass": an.ALPHA_PASS, "alpha_reject": an.ALPHA_REJECT, "min_confident_coverage": an.MIN_CONFIDENT_COVERAGE},
            "screening_gate": an.GATE, "minicheck": {"equivalence_rows": mcjob.EQUIVALENCE_ROWS, "tolerance": mcjob.EQUIVALENCE_TOL, "rule": "recompute cached RouteBench rows first; any difference above the tolerance stops before RAGTruth is scored"},
            "stages": stages, "phase_cap_usd": PHASE_CAP_USD, "total_expected_calls": sum(s["expected_calls"] for s in stages.values()),
            "total_estimated_usd": round(sum(s["estimated_usd"] for s in stages.values()), 2), "price_usd_per_1m": mv.PRICES["o4-mini"],
            "input_sha256": {n: _sha(work / n) for n in INPUT_FILES}, "code": code_hashes(),
            "boundary": ["no provider call, GPU inference or download occurred at freeze", "stage 1b needs an explicit release-1b", "RouteBench labels are read only by `screen`"]}
    (work / "protocol.json").write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf8")
    return body


def verify(work: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, CaseRecord]]]:
    work = Path(work)
    if _sha(work / "protocol.json") != expected_sha256:
        raise Phase1Error("protocol hash differs from the hash recorded at freeze time")
    frozen = json.loads((work / "protocol.json").read_text(encoding="utf8"))
    for n, d in frozen["input_sha256"].items():
        if _sha(work / n) != d:
            raise Phase1Error(f"input file {n} changed after the freeze")
    if frozen["code"] != code_hashes():
        raise Phase1Error("code, dependency, rubric or pricing changed after the freeze")
    cfg = mv.cfg_of(VARIANT)
    if frozen["judge"]["config_hash"] != cfg.config_hash or frozen["judge"]["prompt_template_hash"] != mv.judge.prompt_template_hash(cfg.mode):
        raise Phase1Error("judge identity changed")
    population = json.loads((work / "population.json").read_text(encoding="utf8"))
    base: dict[str, CaseRecord] = {}
    for n in ("cases_routebench.jsonl", "cases_ragtruth.jsonl"):
        for line in (work / n).read_text(encoding="utf8").splitlines():
            c = CaseRecord.model_validate_json(line)
            base[c.case_id] = c
    jobs = {s: jobs_for(s, population["routebench"], population["ragtruth"]) for s in STAGES}
    for s in STAGES:
        if mv._digest(jobs[s]) != frozen["stages"][s]["job_digest"] or {j["base"] for j in jobs[s]} - set(base):
            raise Phase1Error(f"stage {s} job list differs from the frozen job list")
    return frozen, population, jobs, {s: job_cases(jobs[s], base) for s in STAGES}


# ------------------------------------------------------------------------------------------------------------------------------ run
def _terminal(stage_dir: Path) -> dict[str, Any] | None:
    p = stage_dir / "terminal.json"
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else None


def release_1b(work: Path, expected_sha256: str, confirm: str) -> dict[str, Any]:
    if confirm != RELEASE_CONFIRM:
        raise Phase1Error(f"release needs --confirm {RELEASE_CONFIRM}")
    frozen, _pop, jobs, jcases = verify(work, expected_sha256)
    sd = Path(work) / "stage_1a"
    t = _terminal(sd)
    if t is None or t.get("status") != "COMPLETE" or t.get("protocol_sha256") != expected_sha256:
        raise Phase1Error("stage 1a has no COMPLETE terminal record for this protocol")
    state = sv3.journal_state(sd / "spend_journal.jsonl", frozen["stages"]["1a"]["cap_usd"])
    rows = sv3.validate_rows(sd, frozen, [{"case_id": j["job_id"]} for j in jobs["1a"]], jcases["1a"], state, expected_sha256)
    if len(rows) != len(jobs["1a"]) or any(r["status"] != "ok" for r in rows.values()) or state["unsettled"]:
        raise Phase1Error("stage 1a is not fully valid")
    target = Path(work) / "release_1b.json"
    if target.exists():
        raise Phase1Error("release_1b.json exists")
    body = {"protocol_sha256": expected_sha256, "stage_1a_committed_usd": round(state["committed"], 4), "released_by": "explicit operator command"}
    target.write_text(json.dumps(body, indent=2), encoding="utf8")
    return body


def _need_release(work: Path, expected_sha256: str) -> None:
    p = Path(work) / "release_1b.json"
    if not p.exists() or json.loads(p.read_text(encoding="utf8")).get("protocol_sha256") != expected_sha256:
        raise Phase1Error("stage 1b requires an explicit release-1b for this protocol after stage 1a is reported")


def run_stage(work: Path, stage: str, expected_sha256: str, request: Callable[..., Any] = sv3.http_request_raw, *, resume: bool = False, settle_unsettled: bool = False) -> dict[str, Any]:
    work = Path(work)
    frozen, _pop, jobs, jcases = verify(work, expected_sha256)
    if stage not in STAGES:
        raise Phase1Error(f"unknown stage {stage!r}")
    if stage == "1b":
        _need_release(work, expected_sha256)
    sd = work / f"stage_{stage}"
    sd.mkdir(exist_ok=True)
    cap, mine, cases = frozen["stages"][stage]["cap_usd"], jobs[stage], jcases[stage]
    items = [{"case_id": j["job_id"]} for j in mine]
    prior = _terminal(sd)
    if prior is not None:
        if prior.get("status") == "COMPLETE":
            raise Phase1Error("a COMPLETE terminal record exists; the stage is final and is never repeated")
        if prior.get("status") == "REFUSED_OVER_CAP":
            raise Phase1Error("an over-cap run was recorded; it is permanently refused")
        if not resume:
            raise Phase1Error(f"a {prior.get('status')} terminal record exists; resuming needs an explicit --resume")
    lock = sd / "run.lock"
    try:
        with lock.open("x", encoding="utf8") as fh:
            fh.write(json.dumps({"pid": os.getpid()}))
    except FileExistsError as exc:
        raise Phase1Error("another runner holds run.lock (or a crash left it): inspect and remove it manually") from exc
    try:
        if settle_unsettled:
            sv3.settle_unsettled_at_bound(sd, cap)
        state = sv3.journal_state(sd / "spend_journal.jsonl", cap)
        if state["over_cap"]:
            sv3._write_terminal(sd, {"status": "REFUSED_OVER_CAP", "committed_usd": round(state["committed"], 4), "cap_usd": cap, "protocol_sha256": expected_sha256})
            raise Phase1Error(f"committed spend ${state['committed']:.4f} exceeds the ${cap:.2f} stage cap: refused")
        if state["unsettled"]:
            raise Phase1Error(f"{len(state['unsettled'])} unsettled reservation(s); settle them at their bound explicitly with --settle-unsettled before resuming")
        rows = sv3.validate_rows(sd, frozen, items, cases, state, expected_sha256)
        journal = Journal(sd / "spend_journal.jsonl", cap)
        status, stop = "COMPLETE", None
        with (sd / "results.jsonl").open("a", encoding="utf8") as fh:
            for j in mine:
                jid = j["job_id"]
                if jid in rows:
                    continue
                try:
                    rec = sv3.judge_case(cases[jid], journal, request, expected_sha256)
                except SpendError as exc:
                    status, stop = "ABORTED", str(exc)
                    break
                fh.write(json.dumps({"case_id": jid, "protocol_sha256": expected_sha256, "record": rec}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                rows[jid] = rec
                if rec.get("identity_failure"):
                    status, stop = "ABORTED", f"model identity failure on {jid}: {rec['reason']}"
                    break
                if rec["status"] != "ok":
                    status, stop = "ABORTED", f"first failed job {jid}: {rec.get('reason')}"
                    break
                if stage == "1b" and len(rows) == PROJECTION_AFTER_JOBS:
                    spent = sv3.journal_state(sd / "spend_journal.jsonl", cap)["committed"]
                    projected = spent / len(rows) * len(mine)
                    if projected > cap:
                        status, stop = "ABORTED", f"projection stop: ${spent:.4f} after {len(rows)} jobs projects to ${projected:.2f} > cap ${cap:.2f}"
                        break
        state = sv3.journal_state(sd / "spend_journal.jsonl", cap)
        if state["over_cap"]:
            status, stop = "REFUSED_OVER_CAP", f"committed spend ${state['committed']:.4f} exceeds the ${cap:.2f} cap"
        elif status == "COMPLETE" and state["unsettled"]:
            status, stop = "ABORTED", "journal inconsistent after the run (unsettled reservation)"
        valid = sv3.validate_rows(sd, frozen, items, cases, state, expected_sha256)
        failed = sorted(c for c, r in valid.items() if r["status"] != "ok")
        missing = [i["case_id"] for i in items if i["case_id"] not in valid]
        if status == "COMPLETE" and (failed or missing):
            status = "INCOMPLETE"
        t = {"status": status, "stage": stage, "jobs": len(items), "ok": sum(1 for r in valid.values() if r["status"] == "ok"), "failed": failed, "not_attempted": len(missing),
             "stopped_on": stop, "committed_usd": round(state["committed"], 4), "cap_usd": cap, "unsettled": state["unsettled"], "protocol_sha256": expected_sha256}
        sv3._write_terminal(sd, t)
        with (sd / "invocations.jsonl").open("a", encoding="utf8") as fh:
            fh.write(json.dumps({"resume": resume, "status": status, "ok": t["ok"], "committed_usd": t["committed_usd"]}) + "\n")
        return t
    finally:
        lock.unlink(missing_ok=True)


def stage_scores(work: Path, stage: str, expected_sha256: str) -> dict[str, list[float]]:
    """{base case id: [score of replicate 1..] } from a COMPLETE, fully re-validated stage."""
    frozen, _pop, jobs, jcases = verify(work, expected_sha256)
    sd = Path(work) / f"stage_{stage}"
    t = _terminal(sd)
    if t is None or t.get("status") != "COMPLETE" or t.get("protocol_sha256") != expected_sha256:
        raise Phase1Error(f"stage {stage} has no COMPLETE terminal record")
    state = sv3.journal_state(sd / "spend_journal.jsonl", frozen["stages"][stage]["cap_usd"])
    rows = sv3.validate_rows(sd, frozen, [{"case_id": j["job_id"]} for j in jobs[stage]], jcases[stage], state, expected_sha256)
    if len(rows) != len(jobs[stage]) or state["unsettled"] or any(r["status"] != "ok" for r in rows.values()):
        raise Phase1Error(f"stage {stage} is not fully valid")
    out: dict[str, dict[int, float]] = {}
    for j in jobs[stage]:
        out.setdefault(j["base"], {})[j["replicate"]] = float(rows[j["job_id"]]["score"])
    return {b: [v[r] for r in sorted(v)] for b, v in out.items()}


# ------------------------------------------------------------------------------------------------------------------------------ fit / screen
def _matrix(base_ids: Sequence[str], scores: dict[str, list[float]], mc: dict[str, Any], cases: dict[str, CaseRecord]) -> np.ndarray:
    return np.array([an.feature_row(scores[b], mc[b]["full"], mc[b]["sent"], cases[b].candidate_output, cases[b].context) for b in base_ids], float)


def fit(work: Path, expected_sha256: str) -> dict[str, Any]:
    """Uses only RAGTruth results, features and labels. RouteBench labels are never opened here."""
    work = Path(work)
    verify(work, expected_sha256)
    if (work / "model.json").exists():
        raise Phase1Error("model.json exists; the model is fitted once")
    rt = json.loads((work / "population.json").read_text(encoding="utf8"))["ragtruth"]
    cases = {}
    for line in (work / "cases_ragtruth.jsonl").read_text(encoding="utf8").splitlines():
        c = CaseRecord.model_validate_json(line)
        cases[c.case_id] = c
    mcj = json.loads((work / "minicheck_ragtruth.json").read_text(encoding="utf8"))
    frozen_code = json.loads((work / "protocol.json").read_text(encoding="utf8"))["code"]
    if mcj["protocol_sha256"] != expected_sha256 or mcj.get("producer_sha256") != frozen_code["judge_v19_minicheck.py"]:
        raise Phase1Error("MiniCheck features are bound to a different protocol or were not produced by the frozen job")
    if mcj["equivalence_worst_abs_difference"] > mcjob.EQUIVALENCE_TOL or mcj["equivalence_rows"] != mcjob.EQUIVALENCE_ROWS:
        raise Phase1Error("the MiniCheck equivalence check was not passed")
    scores = stage_scores(work, "1b", expected_sha256)
    ids = [i["case_id"] for i in rt]
    if set(ids) - set(scores) or set(ids) - set(mcj["features"]):
        raise Phase1Error("some RAGTruth rows lack v18 or MiniCheck features")
    y_all = json.loads((work / "labels_ragtruth.json").read_text(encoding="utf8"))
    X, y, groups = _matrix(ids, scores, mcj["features"], cases), np.array([y_all[b] for b in ids]), [i["group"] for i in rt]
    nested = an.nested_oof(X, y, groups)
    b = an.bands(nested["oof"], y)
    model = an.final_model(X, y, groups, nested["oof"])
    diag = {"ragtruth_oof_kappa_at_0.5": kappa(y.tolist(), (nested["oof"] >= 0.5).astype(int).tolist()), "outer_fold_sizes": nested["outer_fold_sizes"],
            "chosen_c_per_outer_fold": nested["chosen_c_per_outer_fold"], "bands": {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in b.items()}}
    (work / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf8")
    (work / "fit_diagnostics.json").write_text(json.dumps(diag, indent=2, sort_keys=True), encoding="utf8")
    return {"model_sha256": model["model_sha256"], "diagnostics": diag}


def screen(work: Path, expected_sha256: str, model_sha256: str) -> dict[str, Any]:
    """The one-shot screen. Every integrity gate runs BEFORE RouteBench labels are opened; the model hash must be the one `fit` recorded."""
    work = Path(work)
    verify(work, expected_sha256)
    mpath = work / "model.json"
    model = json.loads(mpath.read_text(encoding="utf8"))
    if model.get("model_sha256") != model_sha256:
        raise Phase1Error("model hash differs from the hash recorded by `fit`")
    check = {k: v for k, v in model.items() if k != "model_sha256"}
    if hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest() != model_sha256:
        raise Phase1Error("model.json content does not match its recorded hash")
    target = work / "screen.json"
    if target.exists():
        raise Phase1Error("screen.json exists; the screen is run once")
    scores = stage_scores(work, "1a", expected_sha256)  # replicates 2-5; replicate 1 is the cached frozen judgment
    cached = json.loads((work / "cached_v18.json").read_text(encoding="utf8"))
    pop = json.loads((work / "population.json").read_text(encoding="utf8"))["routebench"]
    cases = {}
    for line in (work / "cases_routebench.jsonl").read_text(encoding="utf8").splitlines():
        c = CaseRecord.model_validate_json(line)
        cases[c.case_id] = c
    ids = [i["case_id"] for i in pop]
    five = {b: [float(cached[b]["score"]), *scores[b]] for b in ids}
    mc = json.loads((work / "minicheck_cached_routebench.json").read_text(encoding="utf8"))
    X = _matrix(ids, five, mc, cases)
    out = an.predict(model, X)
    a0 = [an.a0_verdict(cached[b]["score"]) for b in ids]
    a1 = [an.a1_majority(five[b]) for b in ids]
    a2 = out["binary"].tolist()
    labels = json.loads((work / "labels_routebench.json").read_text(encoding="utf8"))  # opened only now
    y, pops = [labels[b] for b in ids], [i["pop"] for i in pop]
    singles = {f"run{r + 1}": [int(five[b][r] >= an.PASS_THRESHOLD) for b in ids] for r in range(an.K)}
    rep = {"status": "COMPLETE", "label": "SCREEN on already-inspected, evaluation-only data: a futility screen, never a validation, not a pass of the original benchmark target",
           "gate_A2_vs_A0": an.screen_gate(pops, y, a0, a2), "gate_A1_vs_A0_descriptive": an.screen_gate(pops, y, a0, a1),
           "kappa_difference_95_document_bootstrap_reported_not_gating": {"A2_minus_A0": an.paired_bootstrap([i["group"] for i in pop], y, a0, a2), "A1_minus_A0": an.paired_bootstrap([i["group"] for i in pop], y, a0, a1)},
           "kappa": {"A0_cached": kappa(y, a0), "A1_majority": kappa(y, a1), "A2": kappa(y, a2), "v18_single_runs": {k: kappa(y, v) for k, v in singles.items()}},
           "uncertain_band": {"share": float(np.mean(out["uncertain"])), "error_rate_uncertain": float(np.mean([a != b for a, b, u in zip(y, a2, out["uncertain"], strict=True) if u] or [float("nan")])),
                              "error_rate_confident": float(np.mean([a != b for a, b, u in zip(y, a2, out["uncertain"], strict=True) if not u] or [float("nan")]))},
           "model_sha256": model_sha256, "protocol_sha256": expected_sha256}
    target.write_text(json.dumps(rep, indent=2), encoding="utf8")
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["dry-run", "freeze", "run", "release-1b", "fit", "screen"])
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--stage", choices=sorted(STAGES))
    p.add_argument("--protocol-sha256")
    p.add_argument("--model-sha256")
    p.add_argument("--confirm", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--settle-unsettled", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.command == "dry-run":
            print(json.dumps(dry_run(), indent=2))
        elif a.command == "freeze":
            body = freeze(a.work)
            print(json.dumps({"protocol_sha256": _sha(a.work / "protocol.json"), "stages": body["stages"], "total_expected_calls": body["total_expected_calls"],
                              "total_estimated_usd": body["total_estimated_usd"], "phase_cap_usd": body["phase_cap_usd"]}, indent=2))
        else:
            if not a.protocol_sha256:
                print("REFUSED: --protocol-sha256 is required", file=sys.stderr)
                return 2
            if a.command == "run":
                if not a.stage or a.confirm != STAGES[a.stage]["confirm"]:
                    print(f"REFUSED: needs --stage and --confirm {STAGES.get(a.stage or '', {}).get('confirm', '<stage confirm>')}", file=sys.stderr)
                    return 2
                t = run_stage(a.work, a.stage, a.protocol_sha256, resume=a.resume, settle_unsettled=a.settle_unsettled)
                print(json.dumps(t, indent=2))
                return 0 if t["status"] == "COMPLETE" else 3
            if a.command == "release-1b":
                print(json.dumps(release_1b(a.work, a.protocol_sha256, a.confirm), indent=2))
            elif a.command == "fit":
                print(json.dumps(fit(a.work, a.protocol_sha256), indent=2))
            else:
                if not a.model_sha256:
                    print("REFUSED: --model-sha256 is required", file=sys.stderr)
                    return 2
                print(json.dumps(screen(a.work, a.protocol_sha256, a.model_sha256), indent=2))
    except (Phase1Error, sv3.SrcIntV3Error, an.AnalysisError, mv.MatchedError, SpendError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
