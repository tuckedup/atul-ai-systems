"""One frozen evaluation of v18 and v18 + second-pass auditor on the 30 RESERVED TofuEval upstream-dev documents (240 cases).

Frozen before any provider call (see `prepare`): document ids, sampling seed, inclusion probabilities, v18 config hash, auditor prompt hash, decision rule,
code/data hashes, one hard USD cap shared by BOTH judges (failed attempts and retries included), bootstrap seed, outcome definitions.
Labels are written to a separate file at `prepare` and are NOT read until `report`: no judge input contains a label, and neither stage reads one.

Sampling (label-independent): the reserve is the 15 documents per source that the first validation sample did not take. Within each document a simple
random sample of 8 summary sentences (seed SEED_SENT) uses only the document's row indices. Inclusion probability pi = 8 / N_doc (document stage is a census
of the reserve). Weight w = 1 / pi.

Metrics. Cohen's kappa is the ORDINARY binary definition, kappa = (po - pe) / (1 - pe), on a 2x2 table. Two tables are reported, declared in advance:
  weighted   (primary population estimate): cell counts are sums of w = 1/pi (pi from the manifest), then the ordinary formula.
  unweighted (secondary): each of the 240 cases counts once, then the ordinary formula.
Paired document bootstrap: within each source resample its 15 documents with replacement (same draw for baseline and pipeline), BOOT_N draws.
Outcomes, defined now and reported separately: IMPROVED = corrected > introduced AND the 95% interval of the weighted kappa gain has lower bound > 0.
REACHED_0.74 = pipeline weighted kappa point estimate >= 0.74 (the interval is reported too).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import audit2 as a1
from . import audit2_v2 as a2
from . import audit2_v3 as a3
from . import judge, rubrics
from . import upstream_validation as uv
from .audit2 import ERROR_TYPES, build_prompt, norm
from .audit2_v2 import AuditParseError, FreezeError, Journal, SpendError, parse
from .run_calibration import GRID

ROOT = a2.ROOT
DATA = a2.DATA
WORK = ROOT / ".local" / "fresh-eval-v1"
CONFIRM = "RUN-FRESH-EVAL-PAID"
CAP_USD = 4.60
SEED_SENT, BOOT_SEED, BOOT_N, N_SENT = 20261101, 20261102, 2000, 8
V18, V18_HASH, PASS_THRESHOLD = uv.V18, uv.V18_HASH, uv.PASS_THRESHOLD
AUDIT_MODEL, AUDIT_MAX_TOKENS, AUDIT_ATTEMPTS = a1.MODEL, a1.MAX_TOKENS, a1.MAX_ATTEMPTS
MIN_EXPLANATION_WORDS = 8
TARGET_KAPPA = 0.74
OVERHEAD_TOKENS = a3.MESSAGE_OVERHEAD_TOKENS
FIRST_SAMPLE_MANIFEST = DATA / "tofueval_upstream_dev_sample_manifest.json"
CSV_FILES = uv.CSV_FILES


class ModelMismatch(SpendError):
    pass


# ----------------------------------------------------------------------------------------------------------- sampling
def reserve_documents() -> dict[str, list[str]]:
    """Per source, the dev documents NOT taken by the first sample. Reads only the doc_id column."""
    first = json.loads(FIRST_SAMPLE_MANIFEST.read_text(encoding="utf8"))
    out = {}
    for source, name in CSV_FILES.items():
        with (DATA / "raw" / "authoritative" / name).open(encoding="utf8", newline="") as fh:
            docs = sorted({row["doc_id"] for row in csv.DictReader(fh)})
        taken = set(first["sources"][source]["sampled_docs"])
        out[source] = [d for d in docs if d not in taken]
    return out


def sample_manifest() -> dict[str, Any]:
    """Label-independent: only (doc_id, row position) are used to draw. The label column is never accessed here."""
    rng = np.random.default_rng(SEED_SENT)
    reserve, rows = reserve_documents(), []
    for source, name in CSV_FILES.items():
        with (DATA / "raw" / "authoritative" / name).open(encoding="utf8", newline="") as fh:
            positions: dict[str, list[tuple[int, str, str]]] = {}
            for i, row in enumerate(csv.DictReader(fh)):
                positions.setdefault(row["doc_id"], []).append((i, row["sent_idx"], row["annotation_id"]))
        for doc in reserve[source]:
            drows = positions[doc]
            for k in sorted(rng.choice(len(drows), min(N_SENT, len(drows)), replace=False).tolist()):
                i, sent_idx, annotation_id = drows[k]
                rows.append({"source": source, "doc_id": doc, "row_index": i, "sent_idx": sent_idx, "annotation_id": annotation_id,
                             "inclusion_prob": min(N_SENT, len(drows)) / len(drows)})
    return {"seed": SEED_SENT, "n_per_doc": N_SENT, "reserve_docs": reserve, "rows": rows}


def _csv_rows() -> dict[str, list[dict[str, str]]]:
    out = {}
    for source, name in CSV_FILES.items():
        with (DATA / "raw" / "authoritative" / name).open(encoding="utf8", newline="") as fh:
            out[source] = list(csv.DictReader(fh))
    return out


def build_labels(manifest: Mapping[str, Any]) -> dict[str, int]:
    """The gold labels for the sampled rows. Called only by `prepare` (to write the hashed labels file) and `report`."""
    rows = _csv_rows()
    return {f"fe1-{r['source'][:5].lower()}-{r['row_index']}": {"yes": 1, "no": 0}[rows[r["source"]][r["row_index"]]["sent_label"].strip().lower()]
            for r in manifest["rows"]}


def build_items(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Judge inputs. Never touches the label column."""
    docs = {(d["source"], d["doc_id"]): d for d in
            (json.loads(line) for line in uv.DOCS.read_text(encoding="utf8").splitlines() if line.strip())}
    csv_rows = _csv_rows()
    items = []
    for r in manifest["rows"]:
        raw = csv_rows[r["source"]][r["row_index"]]
        assert raw["doc_id"] == r["doc_id"] and raw["sent_idx"] == r["sent_idx"] and raw["annotation_id"] == r["annotation_id"]
        doc = docs[(r["source"], r["doc_id"])]
        assert doc["upstream_part"] == "dev"
        case_id = f"fe1-{r['source'][:5].lower()}-{r['row_index']}"
        items.append({"id": case_id, "source": r["source"], "doc_id": r["doc_id"], "row_index": r["row_index"], "text": doc["text"],
                      "claim": raw["summ_sent"].strip(), "weight": 1.0 / r["inclusion_prob"]})
    return items


def to_case(item: Mapping[str, Any]):
    case = uv.to_case({"source": item["source"], "doc_id": item["doc_id"], "text": item["text"], "claim": item["claim"],
                       "row_index": item["row_index"]}, 0)
    return case.model_copy(update={"case_id": item["id"], "group_id": f"fe1-{item['doc_id']}"})


# ------------------------------------------------------------------------------------------------------------ calls
Attempt = tuple[str, Any, str]


def http_call(model: str, extra: Mapping[str, Any]) -> Callable[[list[dict[str, str]], int], Attempt]:
    """Exactly one HTTP request per call through the project client (`llm._call`): no client retries, no fallback."""
    def call(messages: list[dict[str, str]], max_tokens: int) -> Attempt:
        from aisys import llm

        result = llm._call(model, messages, None, 0.0, max_tokens, dict(extra))
        usage = result.raw.get("usage") if isinstance(result.raw, dict) else None
        return result.text, usage, result.model or ""
    return call


def bound_usd(messages: Sequence[Mapping[str, str]], max_tokens: int) -> tuple[int, float]:
    tokens = sum(len(m["content"].encode("utf8")) for m in messages) + OVERHEAD_TOKENS
    return tokens, a3.usd(tokens, max_tokens)


def _settle(journal: Journal, rid: str, reserve: float, raw_usage: Any, bound_in: int, max_tokens: int) -> bool:
    """Charge the real validated cost; missing/malformed usage -> the reservation; above the token bounds -> record then abort."""
    usage = a3.validated_usage(raw_usage)
    if usage is None:
        journal.settle(rid, reserve, None)
        return False
    cost = a3.usd(*usage)
    if usage[0] > bound_in or usage[1] > max_tokens:
        journal.settle(rid, reserve, max(cost, reserve * (1 + 1e-9)))
    journal.settle(rid, reserve, cost)
    return True


def judge_v18(item: Mapping[str, Any], call: Callable[..., Attempt], journal: Journal, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Frozen v18 on one case: judge.build_prompt / judge.parse unchanged; one journaled HTTP request per attempt (with the judge's repair retry)."""
    from aisys import llm

    cfg = next(c for c in GRID if c.variant_id == V18)
    case = to_case(item)
    rubric = rubrics.for_task(case.task_class, cfg.rubric_dir)
    prompt = judge.build_prompt(case, rubric, cfg)
    messages, last = [{"role": "user", "content": prompt}], ""
    for attempt in range(1, cfg.max_attempts + 1):
        if attempt > 1:
            sleep(a3.BACKOFF_SECONDS * (attempt - 1))
        bound_in, reserve = bound_usd(messages, cfg.max_tokens)
        rid = journal.reserve(f"v18:{item['id']}", attempt, reserve)
        try:
            text, raw_usage, served = call(messages, cfg.max_tokens)
        except llm.QuotaExceededError as exc:
            journal.settle(rid, reserve, None)
            raise SpendError(f"provider quota error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - a failed request may still have billed
            journal.settle(rid, reserve, None)
            last = f"{type(exc).__name__}: {exc}"
            continue
        verified = _settle(journal, rid, reserve, raw_usage, bound_in, cfg.max_tokens)
        if served and not served.startswith(cfg.model):
            raise ModelMismatch(f"v18 served {served!r}, not {cfg.model!r}")
        try:
            parsed = judge.parse(text, rubric, cfg, case)
        except (judge.JudgeOutputError, rubrics.RubricError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": text[:2000]},
                        {"role": "user", "content": f"That response was rejected by the parser: {exc}\nReturn ONLY the JSON object in the exact shape "
                                                    "requested, with no prose and no code fence."}]
            continue
        return {"id": item["id"], "status": "ok", "score": float(parsed.score), "pass": int(parsed.score >= PASS_THRESHOLD),
                "attempts": attempt, "usage_verified": verified}
    return {"id": item["id"], "status": "failed", "attempts": cfg.max_attempts, "reason": last}


def decide_final(result: a1.AuditResult, source: str, candidate: str) -> tuple[bool, str]:
    """FROZEN decision rule: flip a v18 pass to unsupported iff the auditor's own verdict is 'unsupported' with a known error type, the disputed
    assertion is verbatim in the candidate, claim_says / source_says are present and differ, and the explanation has >= 8 words.
    Quote authenticity and lexical overlap are NOT gates (recorded as diagnostics by `diagnostics`)."""
    if result.verdict != "unsupported":
        return False, "auditor_supported"
    if result.error_type not in ERROR_TYPES:
        return False, "unknown_error_type"
    if not result.disputed_assertion or norm(result.disputed_assertion) not in norm(candidate):
        return False, "assertion_not_in_candidate"
    if len(result.explanation.split()) < MIN_EXPLANATION_WORDS:
        return False, "explanation_too_short"
    if not result.claim_says or not result.source_says or norm(result.claim_says) == norm(result.source_says):
        return False, "claim_source_statements_missing_or_equal"
    return True, "flip_final_rule"


def diagnostics(result: a1.AuditResult, source: str) -> dict[str, Any]:
    quote = result.evidence_quote
    return {"quote_present": bool(quote), "quote_in_source": bool(quote) and norm(quote) in norm(source),
            "quote_overlaps_assertion": bool(quote and result.disputed_assertion and a1.content_tokens(quote) & a1.content_tokens(result.disputed_assertion))}


def audit_final(item: Mapping[str, Any], call: Callable[..., Attempt], journal: Journal, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    from aisys import llm

    prompt, last = build_prompt(item["text"], item["claim"]), ""
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(1, AUDIT_ATTEMPTS + 1):
        if attempt > 1:
            sleep(a3.BACKOFF_SECONDS * (attempt - 1))
        bound_in, reserve = bound_usd(messages, AUDIT_MAX_TOKENS)
        rid = journal.reserve(f"audit:{item['id']}", attempt, reserve)
        try:
            text, raw_usage, served = call(messages, AUDIT_MAX_TOKENS)
        except llm.QuotaExceededError as exc:
            journal.settle(rid, reserve, None)
            raise SpendError(f"provider quota error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            journal.settle(rid, reserve, None)
            last = f"{type(exc).__name__}: {exc}"
            continue
        verified = _settle(journal, rid, reserve, raw_usage, bound_in, AUDIT_MAX_TOKENS)
        if served and not served.startswith(AUDIT_MODEL):
            raise ModelMismatch(f"auditor served {served!r}, not {AUDIT_MODEL!r}")
        try:
            result = parse(text)
            flip, reason = decide_final(result, item["text"], item["claim"])
        except (AuditParseError, TypeError, AttributeError) as exc:
            last = f"parse: {exc}"
            continue
        return {"id": item["id"], "status": "ok", "attempts": attempt, "flip": flip, "reason": reason, "usage_verified": verified,
                "diagnostics": diagnostics(result, item["text"]),
                "result": result.__dict__ | {"unsupported_details": list(result.unsupported_details)}}
    return {"id": item["id"], "status": "failed", "attempts": AUDIT_ATTEMPTS, "flip": False, "reason": last}


# ------------------------------------------------------------------------------------------------------------ freeze
def _sha(value: Any) -> str:
    return a2._sha(value)


def items_digest(items: Sequence[Mapping[str, Any]]) -> str:
    return _sha(sorted(([i["id"], i["source"], i["doc_id"], i["row_index"], i["text"], i["claim"], i["weight"]] for i in items), key=lambda r: r[0]))


def code_hashes() -> dict[str, str]:
    cfg = next(c for c in GRID if c.variant_id == V18)
    files = {name: a2.file_sha(Path(mod.__file__)) for name, mod in
             {"fresh_eval.py": sys.modules[__name__], "audit2.py": a1, "audit2_v2.py": a2, "audit2_v3.py": a3, "judge.py": judge,
              "rubrics.py": rubrics, "upstream_validation.py": uv}.items()}
    files["aisys/llm.py"] = a2.file_sha(Path(__import__("aisys.llm", fromlist=["x"]).__file__))
    files["aisys/pricing.yaml"] = a2.file_sha(Path(__import__("aisys.llm", fromlist=["x"]).__file__).parent / "pricing.yaml")
    files["rubric_bundle"] = rubrics.bundle_hash()
    files["v18_config_hash"] = cfg.config_hash
    files["v18_prompt_template"] = judge.prompt_template_hash("rubric")
    files["auditor_prompt"] = hashlib.sha256(a1.PROMPT.encode()).hexdigest()
    files["manifest_first_sample"] = a2.file_sha(FIRST_SAMPLE_MANIFEST)
    files["docs.jsonl"] = a2.file_sha(uv.DOCS)
    files.update({f"csv/{n}": a2.file_sha(DATA / "raw" / "authoritative" / n) for n in CSV_FILES.values()})
    files["settings"] = _sha([CAP_USD, SEED_SENT, BOOT_SEED, BOOT_N, N_SENT, AUDIT_MODEL, AUDIT_MAX_TOKENS, AUDIT_ATTEMPTS, MIN_EXPLANATION_WORDS,
                              TARGET_KAPPA, PASS_THRESHOLD, OVERHEAD_TOKENS, "single_http_attempt=llm._call"])
    return files


def prepare(work: Path = WORK) -> dict[str, Any]:
    if not a3._pricing_matches():
        raise FreezeError("pricing differs from the protocol constants")
    manifest = sample_manifest()
    items, labels = build_items(manifest), build_labels(manifest)
    assert len(items) == 240 and len({i["id"] for i in items}) == 240
    first_docs = {d for v in json.loads(FIRST_SAMPLE_MANIFEST.read_text(encoding="utf8"))["sources"].values() for d in v["sampled_docs"]}
    assert not first_docs & {i["doc_id"] for i in items}, "reserve overlaps the first validation sample"
    work.mkdir(parents=True, exist_ok=False)
    (work / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    (work / "items.json").write_text(json.dumps(items), encoding="utf8")
    (work / "labels.json").write_text(json.dumps(labels, sort_keys=True), encoding="utf8")  # not read until `report`
    body = {"protocol": "fresh-eval-v1", "n_cases": len(items), "n_docs": len({i["doc_id"] for i in items}), "cap_usd": CAP_USD,
            "hashes": code_hashes(), "items_digest": items_digest(items), "labels_sha256": a2.file_sha(work / "labels.json"),
            "manifest_sha256": a2.file_sha(work / "manifest.json"), "reserve_documents": manifest["reserve_docs"],
            "expected_usd": {"v18": round(240 * 0.0077, 2), "auditor": round(180 * 0.0055, 2)}}
    (work / "protocol.json").write_text(json.dumps(body, indent=2), encoding="utf8")
    return body


def verify_freeze(work: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads((work / "protocol.json").read_text(encoding="utf8"))
    if protocol["hashes"] != code_hashes():
        raise FreezeError("code, rubric, data or settings changed since the freeze")
    items = json.loads((work / "items.json").read_text(encoding="utf8"))
    ids = [i["id"] for i in items]
    if len(items) != protocol["n_cases"] or len(set(ids)) != len(ids) or items_digest(items) != protocol["items_digest"]:
        raise FreezeError("items differ from the frozen set")
    if a2.file_sha(work / "manifest.json") != protocol["manifest_sha256"] or a2.file_sha(work / "labels.json") != protocol["labels_sha256"]:
        raise FreezeError("manifest or labels file changed since the freeze")
    # re-derive from the raw files and compare, so the frozen files cannot drift from the sampling procedure (labels are NOT read here)
    if items_digest(build_items(sample_manifest())) != protocol["items_digest"]:
        raise FreezeError("re-derived sample differs from the frozen sample")
    return protocol, items


# ----------------------------------------------------------------------------------------------------------- running
def run_all(work: Path, call_v18: Callable[..., Attempt], call_audit: Callable[..., Attempt], *, cap_usd: float = CAP_USD,
            resume: bool = False, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Both stages in one process under ONE journal. Labels are never read here."""
    protocol, items = verify_freeze(work)
    cap = min(cap_usd, protocol["cap_usd"])
    terminal_path = work / "terminal.json"
    if terminal_path.exists():
        raise FreezeError("a terminal record exists; the evaluation is run once")
    paths = {"v18": work / "rows_v18.jsonl", "audit": work / "rows_audit.jsonl"}
    if any(p.exists() for p in paths.values()) and not resume:
        raise FreezeError("partial rows exist; resume explicitly (spend is already journaled)")
    journal = Journal(work / "spend_journal.jsonl", cap)

    def load(path: Path) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        if path.exists():
            for line in path.read_text(encoding="utf8").splitlines():
                if line.strip() and json.loads(line)["status"] == "ok":
                    rows[json.loads(line)["id"]] = json.loads(line)
        return rows

    status, stage_rows = "COMPLETE", {"v18": load(paths["v18"]), "audit": load(paths["audit"])}

    def stage(name: str, todo: Sequence[Mapping[str, Any]], fn: Callable[..., dict[str, Any]], call: Callable[..., Attempt]) -> bool:
        nonlocal status
        with paths[name].open("a", encoding="utf8") as stream:
            for item in todo:
                if item["id"] in stage_rows[name]:
                    continue
                try:
                    row = fn(item, call, journal, sleep)
                except SpendError as exc:
                    status = f"ABORTED: {exc}"
                    return False
                stream.write(json.dumps(row) + chr(10))
                stream.flush()
                os.fsync(stream.fileno())
                stage_rows[name][item["id"]] = row
        return True

    if stage("v18", items, judge_v18, call_v18):
        passes = [i for i in items if stage_rows["v18"].get(i["id"], {}).get("pass") == 1]
        stage("audit", passes, audit_final, call_audit)
    v18_ok = sum(1 for r in stage_rows["v18"].values() if r["status"] == "ok")
    passes = [i["id"] for i in items if stage_rows["v18"].get(i["id"], {}).get("pass") == 1]
    audit_ok = sum(1 for i in passes if stage_rows["audit"].get(i, {}).get("status") == "ok")
    if status == "COMPLETE" and (v18_ok != len(items) or audit_ok != len(passes)):
        status = "INCOMPLETE: failed judgments"
    terminal = {"status": status, "spent_usd_total": round(journal.committed(), 5), "cap_usd": cap, "cases": len(items), "v18_ok": v18_ok,
                "v18_passes": len(passes), "audits_ok": audit_ok, "flips": sum(bool(r.get("flip")) for r in stage_rows["audit"].values()),
                "usage_unverified": sum(1 for s in stage_rows.values() for r in s.values() if not r.get("usage_verified", True))}
    terminal_path.write_text(json.dumps(terminal, indent=2), encoding="utf8")
    return terminal


# ------------------------------------------------------------------------------------------------------------ metrics
def kappa_counts(tp: float, fp: float, fn: float, tn: float) -> float | None:
    """Ordinary binary Cohen's kappa from a 2x2 table of (possibly weighted) counts; None when chance agreement is total."""
    n = tp + fp + fn + tn
    if n <= 0:
        return None
    po, pe = (tp + tn) / n, ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return None if pe >= 1 - 1e-12 else (po - pe) / (1 - pe)


def table(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float]:
    return (float(w[(p == 1) & (y == 1)].sum()), float(w[(p == 1) & (y == 0)].sum()),
            float(w[(p == 0) & (y == 1)].sum()), float(w[(p == 0) & (y == 0)].sum()))


def rates(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float | None]:
    tp, fp, fn, tn = table(y, p, w)
    sens, spec = tp / (tp + fn) if tp + fn else None, tn / (tn + fp) if tn + fp else None
    return {"sensitivity": sens, "specificity": spec, "balanced_accuracy": None if sens is None or spec is None else (sens + spec) / 2}


def paired_bootstrap(y: np.ndarray, base: np.ndarray, new: np.ndarray, w: np.ndarray, src: np.ndarray, doc: np.ndarray) -> dict[str, Any]:
    """Resample documents within each source with replacement; the same draw scores baseline and pipeline (paired)."""
    rng = np.random.default_rng(BOOT_SEED)
    by_src = {s: sorted(set(doc[src == s].tolist())) for s in sorted(set(src.tolist()))}
    idx = {d: np.where(doc == d)[0] for d in set(doc.tolist())}
    gains_w, gains_u, used = [], [], 0
    ones = np.ones(len(y))
    for _ in range(BOOT_N):
        ks = np.concatenate([idx[d] for ds in by_src.values() for d in rng.choice(ds, len(ds))])
        kb_w, kn_w = kappa_counts(*table(y[ks], base[ks], w[ks])), kappa_counts(*table(y[ks], new[ks], w[ks]))
        kb_u, kn_u = kappa_counts(*table(y[ks], base[ks], ones[ks])), kappa_counts(*table(y[ks], new[ks], ones[ks]))
        if None in (kb_w, kn_w, kb_u, kn_u):
            continue
        used += 1
        gains_w.append(kn_w - kb_w)
        gains_u.append(kn_u - kb_u)
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"seed": BOOT_SEED, "draws_requested": BOOT_N, "draws_used": used, "gain_weighted_ci95": pct(gains_w), "gain_unweighted_ci95": pct(gains_u)}


def report(work: Path = WORK) -> dict[str, Any]:
    """Join labels for the first time. Withheld unless coverage is complete. Written once."""
    target = work / "final_report.json"
    if target.exists():
        raise FreezeError("final_report.json exists; refusing to overwrite")
    terminal = json.loads((work / "terminal.json").read_text(encoding="utf8"))
    _, items = verify_freeze(work)
    if terminal["status"] != "COMPLETE" or terminal["v18_ok"] != len(items) or terminal["audits_ok"] != terminal["v18_passes"]:
        result = {"status": "WITHHELD_INCOMPLETE", "terminal": terminal, "note": "no comparison is reported for an incomplete evaluation"}
        target.write_text(json.dumps(result, indent=2), encoding="utf8")
        return result
    labels = json.loads((work / "labels.json").read_text(encoding="utf8"))
    if labels != build_labels(sample_manifest()):
        raise FreezeError("labels differ from the frozen labels file")
    v18 = {json.loads(x)["id"]: json.loads(x) for x in (work / "rows_v18.jsonl").read_text(encoding="utf8").splitlines() if x.strip()}
    aud = {json.loads(x)["id"]: json.loads(x) for x in (work / "rows_audit.jsonl").read_text(encoding="utf8").splitlines() if x.strip()}
    y = np.array([labels[i["id"]] for i in items])
    base = np.array([v18[i["id"]]["pass"] for i in items])
    new = np.array([0 if aud.get(i["id"], {}).get("flip") else v18[i["id"]]["pass"] for i in items])
    w, ones = np.array([i["weight"] for i in items]), np.ones(len(items))
    src, doc = np.array([i["source"] for i in items]), np.array([i["doc_id"] for i in items])
    corrected, introduced = int(((base != y) & (new == y)).sum()), int(((base == y) & (new != y)).sum())
    k = lambda p, wt: kappa_counts(*table(y, p, wt))
    boot = paired_bootstrap(y, base, new, w, src, doc)
    gain_w = (k(new, w) or 0) - (k(base, w) or 0)
    improved = corrected > introduced and boot["gain_weighted_ci95"][0] > 0
    oracle = np.where((base == 1) & (y == 0), 0, base)  # descriptive ceiling: an auditor that fixes every false pass and nothing else
    result = {
        "status": "COMPLETE", "coverage": f"{len(items)}/{len(items)} v18; {terminal['audits_ok']}/{terminal['v18_passes']} audits",
        "spent_usd_total": terminal["spent_usd_total"], "cap_usd": terminal["cap_usd"],
        "weighting": "weighted = counts summed with w = 1/inclusion_prob (pi = 8/N_doc, document stage a census of the 30 reserve documents); "
                     "unweighted = each case once. Both use the ordinary binary Cohen's kappa formula.",
        "baseline_v18": {"kappa_weighted": k(base, w), "kappa_unweighted": k(base, ones), "confusion_unweighted_tp_fp_fn_tn": list(table(y, base, ones)),
                         "rates_weighted": rates(y, base, w)},
        "pipeline_v18_plus_auditor": {"kappa_weighted": k(new, w), "kappa_unweighted": k(new, ones),
                                      "confusion_unweighted_tp_fp_fn_tn": list(table(y, new, ones)), "rates_weighted": rates(y, new, w)},
        "corrected": corrected, "introduced": introduced, "flips": int(((base == 1) & (new == 0)).sum()),
        "gain_weighted": gain_w, "gain_unweighted": (k(new, ones) or 0) - (k(base, ones) or 0), "paired_document_bootstrap": boot,
        "outcomes": {"IMPROVED": bool(improved), "REACHED_0.74": bool((k(new, w) or 0) >= TARGET_KAPPA),
                     "definition_improved": "corrected > introduced AND lower bound of the 95% weighted-kappa gain interval > 0"},
        "descriptive_ceiling_if_every_v18_false_pass_were_fixed": {"kappa_weighted": k(oracle, w), "kappa_unweighted": k(oracle, ones)},
        "per_source": {s: {"n": int((src == s).sum()), "kappa_weighted_baseline": kappa_counts(*table(y[src == s], base[src == s], w[src == s])),
                           "kappa_weighted_pipeline": kappa_counts(*table(y[src == s], new[src == s], w[src == s]))} for s in sorted(set(src.tolist()))},
        "flip_details": [{"id": i, "gold": labels[i], "reason": r["reason"], "error_type": r["result"]["error_type"], "diagnostics": r["diagnostics"]}
                         for i, r in sorted(aud.items()) if r.get("flip")],
        "note": "fresh reserved documents; separate from the original RouteBench benchmark; no rule changes after this report",
    }
    target.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


# ---------------------------------------------------------------------------------------------------------------- CLI
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "report"])
    parser.add_argument("--confirm", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2))
        return 0
    if args.command == "report":
        print(json.dumps(report(), indent=2))
        return 0
    if args.confirm != CONFIRM:
        raise SystemExit(f"paid run requires --confirm {CONFIRM}")
    cfg = next(c for c in GRID if c.variant_id == V18)
    terminal = run_all(WORK, http_call(cfg.model, {"reasoning_effort": cfg.reasoning_effort} if cfg.reasoning_effort else {}),
                       http_call(AUDIT_MODEL, {"reasoning_effort": "medium"}), resume=args.resume)
    print(json.dumps(terminal, indent=2))
    return 0 if terminal["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    sys.exit(main())
