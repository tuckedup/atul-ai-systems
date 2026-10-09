"""Permission- and provenance-aware workflow for NEW source-groundedness data (offline; development folders only).

Files in a workspace (all JSONL/JSON, kept separate on purpose): sources, cases, annotations, adjudications, rejected, quality_report.
    python -m evalops.dqdata init     --root DIR
    python -m evalops.dqdata import   --root DIR --incoming DIR
    python -m evalops.dqdata report   --root DIR
    python -m evalops.dqdata packets  --root DIR --salt TEXT
    python -m evalops.dqdata overlap  --root DIR --index FILE
    python -m evalops.dqdata export-training --root DIR --out NAME --uncertain-policy exclude_from_binary_export
    python -m evalops.dqdata build-protected-index --out FILE        (reads protected corpora READ-ONLY, writes fingerprints only)

Principles: nothing is silently dropped (malformed rows go to rejected.jsonl with reasons and count in the report denominators); uncertain stays
uncertain; AI annotations are never exported as human; training export refuses unresolved/prohibited rights; blind packets are built from a strict
allow-list; quote authenticity is reported apart from semantic support; overlap checks are lexical fingerprints, not proof of independence.
Writes into the original corpus, acceptance annotations, caches or frozen artifacts are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .pilot.audit import ReferenceIndex

ROOT = Path(__file__).resolve().parents[2]
PROTECTED_ROOTS = (
    Path(__file__).parent / "data",  # original corpus, annotations, splits, caches
    ROOT / ".local" / "fresh-eval-v1", ROOT / ".local" / "audit2-v1", ROOT / ".local" / "audit2-v2", ROOT / ".local" / "audit2-v3",
    ROOT / ".local" / "ai-panel-20261001-v1", ROOT / ".local" / "ragtruth-ingest-20260930", ROOT / ".local" / "factcg-screen-v1",
)
POLICY_VERSION = "source-groundedness-policy-v1"
OUTCOMES = ("supported", "unsupported", "uncertain")
PROVENANCE = ("natural", "constructed", "synthetic")
PERMISSION_STATUS = ("permitted", "prohibited", "unresolved")
PERMISSION_KINDS = ("annotate", "train", "retain", "redistribute")
ANNOTATION_KINDS = ("human", "ai")
PACKET_ALLOW = ("packet_id", "source_text", "candidate", "context", "policy")
PACKET_POLICY = (
    "Question: does the source support every factual assertion in the marked candidate sentence? Judge source support, not style or world truth. "
    "Outcomes: supported, unsupported, uncertain. Accept faithful paraphrase; check values with their entity, unit, date and relation; do not equate "
    "planned/possible/alleged with actual events; do not supply unstated causes or motives. Neighbouring summary sentences may resolve references but "
    "are not evidence.")


class DataQualityError(ValueError):
    pass


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


# ------------------------------------------------------------------------------------------------------ workspace
def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class Workspace:
    """A separately versioned development folder. Every write goes through here so protected locations cannot be modified."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        for protected in PROTECTED_ROOTS:
            if _is_inside(self.root, protected) or _is_inside(protected, self.root):
                raise DataQualityError(f"workspace {self.root} overlaps protected location {protected}; choose a new development folder")

    def path(self, name: str) -> Path:
        target = (self.root / name).resolve()
        if not _is_inside(target, self.root):
            raise DataQualityError(f"{name!r} escapes the workspace")
        return target

    def write_jsonl(self, name: str, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> Path:
        target = self.path(name)
        if target.exists() and not overwrite:
            raise DataQualityError(f"{name} exists; refusing to overwrite")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf8")
        return target

    def write_json(self, name: str, value: Any, *, overwrite: bool = False) -> Path:
        target = self.path(name)
        if target.exists() and not overwrite:
            raise DataQualityError(f"{name} exists; refusing to overwrite")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf8")
        return target

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        p = self.path(name)
        return [json.loads(line) for line in p.read_text(encoding="utf8").splitlines() if line.strip()] if p.exists() else []


# ------------------------------------------------------------------------------------------------------ validation
def _str(row: Mapping[str, Any], key: str, problems: list[str], *, required: bool = True) -> str | None:
    value = row.get(key)
    if value is None or value == "":
        if required:
            problems.append(f"missing {key}")
        return None
    if not isinstance(value, str):
        problems.append(f"{key} must be a string")
        return None
    return value


def _permissions(row: Mapping[str, Any], key: str, problems: list[str]) -> None:
    perms = row.get(key)
    if not isinstance(perms, dict):
        problems.append(f"missing {key}")
        return
    for kind in PERMISSION_KINDS:
        entry = perms.get(kind)
        if not isinstance(entry, dict) or entry.get("status") not in PERMISSION_STATUS:
            problems.append(f"{key}.{kind}.status must be one of {PERMISSION_STATUS}")
        elif entry["status"] != "unresolved" and not str(entry.get("evidence", "")).strip():
            problems.append(f"{key}.{kind} needs evidence for status {entry['status']}")


def validate_source(row: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    _str(row, "source_id", problems)
    _str(row, "document_family", problems)
    text = _str(row, "text", problems)
    given = _str(row, "text_sha256", problems)
    if text is not None and given is not None and sha256(text) != given:
        problems.append("text_sha256 does not match text (altered source)")
    if row.get("provenance") not in PROVENANCE:
        problems.append(f"provenance must be one of {PROVENANCE}")
    _str(row, "policy_version", problems)
    _str(row, "collection_record", problems)
    _permissions(row, "source_permissions", problems)
    return problems


def validate_case(row: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    _str(row, "case_id", problems)
    sid = _str(row, "source_id", problems)
    cand = _str(row, "candidate", problems)
    if cand is not None and row.get("candidate_sha256") not in (None, sha256(cand)):
        problems.append("candidate_sha256 does not match candidate")
    if row.get("provenance") not in PROVENANCE:
        problems.append(f"provenance must be one of {PROVENANCE}")
    if sid is not None:
        src = sources.get(sid)
        if src is None:
            problems.append("source_id not in accepted sources")
        elif row.get("source_sha256") != src["text_sha256"]:
            problems.append("source_sha256 does not match the accepted source (altered or wrong version)")
    return problems


def span_authenticity(text: str, spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Exact offsets only: zero-based, end-exclusive code points, and the quote must equal text[start:end]. Authenticity, not support."""
    out = []
    for span in spans:
        start, end, quote = span.get("start"), span.get("end"), span.get("text", span.get("quote"))
        ok = isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text) and isinstance(quote, str) and text[start:end] == quote
        out.append({"start": start, "end": end, "authentic": bool(ok)})
    return out


def derive_outcome(assertion_outcomes: Sequence[str]) -> str:
    if not assertion_outcomes:
        raise DataQualityError("no assertions")
    if "unsupported" in assertion_outcomes:
        return "unsupported"
    return "uncertain" if "uncertain" in assertion_outcomes else "supported"


def validate_annotation(row: Mapping[str, Any], cases: Mapping[str, Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    _str(row, "annotation_id", problems)
    cid = _str(row, "case_id", problems)
    _str(row, "annotator_id", problems)
    kind = row.get("kind")
    if kind not in ANNOTATION_KINDS:
        problems.append(f"kind must be one of {ANNOTATION_KINDS}")
    elif kind == "human":
        if not str(row.get("collection_process_id", "")).strip():
            problems.append("human annotation needs a collection_process_id (a real collection record, not a self-asserted label)")
    else:
        model = row.get("model_identity")
        if not isinstance(model, dict) or not str(model.get("name", "")).strip() or "version" not in model:
            problems.append("ai annotation needs model_identity {name, version} (version may be 'unknown', never omitted)")
    outcome = row.get("outcome")
    if outcome not in OUTCOMES:
        problems.append(f"outcome must be one of {OUTCOMES} (uncertain is not mapped to pass/fail)")
    _str(row, "policy_version", problems)
    case = cases.get(cid) if cid else None
    if cid is not None and case is None:
        problems.append("case_id not in accepted cases")
    assertions = row.get("assertions")
    if assertions is not None:
        if not isinstance(assertions, list) or not assertions:
            problems.append("assertions must be a non-empty list when present")
        else:
            outs = []
            for i, a in enumerate(assertions):
                if not isinstance(a, dict) or a.get("outcome") not in OUTCOMES:
                    problems.append(f"assertion {i}: outcome must be one of {OUTCOMES}")
                    continue
                outs.append(a["outcome"])
                if case is not None:
                    cand = case["candidate"]
                    if not span_authenticity(cand, [a.get("candidate_span", {})])[0]["authentic"]:
                        problems.append(f"assertion {i}: candidate_span offsets do not match the stored candidate text")
                    src_text = sources[case["source_id"]]["text"]
                    for j, ev in enumerate(a.get("evidence", []) or []):
                        if not span_authenticity(src_text, [ev])[0]["authentic"]:
                            problems.append(f"assertion {i} evidence {j}: offsets/quote do not match the stored source text")
            if len(outs) == len(assertions) and outcome in OUTCOMES and derive_outcome(outs) != outcome:
                problems.append(f"case outcome {outcome!r} inconsistent with assertion outcomes {sorted(set(outs))}")
    return problems


def validate_adjudication(row: Mapping[str, Any], annotations: Mapping[str, Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    _str(row, "adjudication_id", problems)
    cid = _str(row, "case_id", problems)
    _str(row, "reviewer_id", problems)
    _str(row, "reason", problems)
    if row.get("final_outcome") not in OUTCOMES:
        problems.append(f"final_outcome must be one of {OUTCOMES}")
    if row.get("reviewer_kind") not in ANNOTATION_KINDS:
        problems.append(f"reviewer_kind must be one of {ANNOTATION_KINDS}")
    ids = row.get("annotation_ids")
    if not isinstance(ids, list) or len(ids) < 2:
        problems.append("annotation_ids must list the original annotations being adjudicated")
    else:
        for a in ids:
            if a not in annotations:
                problems.append(f"annotation {a} not accepted")
            elif cid is not None and annotations[a]["case_id"] != cid:
                problems.append(f"annotation {a} belongs to another case")
            elif annotations[a]["annotator_id"] == row.get("reviewer_id"):
                problems.append("adjudicator must differ from the annotators")
    if cid is not None and cid not in cases:
        problems.append("case_id not in accepted cases")
    return problems


# ------------------------------------------------------------------------------------------------------ import
def import_records(ws: Workspace, incoming: Path) -> dict[str, Any]:
    """Validate every incoming row. Accepted rows go to the workspace; rejected rows are KEPT with reasons. Nothing is dropped silently."""
    incoming = Path(incoming)

    def rows_of(name: str) -> list[dict[str, Any]]:
        p = incoming / name
        out = []
        if p.exists():
            for n, line in enumerate(p.read_text(encoding="utf8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    out.append({"_malformed_line": n, "_raw": line[:200], "_error": str(exc)})
                    continue
                out.append(value if isinstance(value, dict) else {"_malformed_line": n, "_raw": line[:200], "_error": "row is not an object"})
        return out

    rejected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}

    def accept(kind: str, key: str, rows: list[dict[str, Any]], validator) -> dict[str, dict[str, Any]]:
        accepted: dict[str, dict[str, Any]] = {}
        seen: Counter[str] = Counter(r.get(key) for r in rows if isinstance(r.get(key), str))
        for r in rows:
            problems = [f"malformed line {r['_malformed_line']}: {r['_error']}"] if "_malformed_line" in r else validator(r)
            if isinstance(r.get(key), str) and seen[r[key]] > 1:
                problems.append(f"duplicate {key}")
            if problems:
                rejected.append({"kind": kind, "id": r.get(key), "reasons": problems, "row": r})
            else:
                accepted[r[key]] = r
        counts[kind] = {"attempted": len(rows), "accepted": len(accepted), "rejected": len(rows) - len(accepted)}
        return accepted

    sources = accept("source", "source_id", rows_of("sources.jsonl"), validate_source)
    cases = accept("case", "case_id", rows_of("cases.jsonl"), lambda r: validate_case(r, sources))
    annotations = accept("annotation", "annotation_id", rows_of("annotations.jsonl"), lambda r: validate_annotation(r, cases, sources))
    adjudications = accept("adjudication", "adjudication_id", rows_of("adjudications.jsonl"), lambda r: validate_adjudication(r, annotations, cases))
    ws.write_jsonl("sources.jsonl", sources.values())
    ws.write_jsonl("cases.jsonl", cases.values())
    ws.write_jsonl("annotations.jsonl", annotations.values())
    ws.write_jsonl("adjudications.jsonl", adjudications.values())
    ws.write_jsonl("rejected.jsonl", rejected)
    return {"counts": counts, "rejected_rows": len(rejected)}


# ------------------------------------------------------------------------------------------------------ report
def quality_report(ws: Workspace, *, min_human: int = 2) -> dict[str, Any]:
    sources, cases = ws.read_jsonl("sources.jsonl"), ws.read_jsonl("cases.jsonl")
    annotations, adjudications, rejected = ws.read_jsonl("annotations.jsonl"), ws.read_jsonl("adjudications.jsonl"), ws.read_jsonl("rejected.jsonl")
    by_case: dict[str, list[dict[str, Any]]] = {c["case_id"]: [] for c in cases}
    for a in annotations:
        by_case[a["case_id"]].append(a)
    adj_by_case = {a["case_id"]: a for a in adjudications}
    inventory = []
    for c in cases:
        humans = [a for a in by_case[c["case_id"]] if a["kind"] == "human"]
        outcomes = [a["outcome"] for a in humans]
        disagree = len(set(outcomes)) > 1
        adj = adj_by_case.get(c["case_id"])
        status = "incomplete" if len(humans) < min_human else ("disagreement_unadjudicated" if disagree and adj is None else "complete")
        inventory.append({"case_id": c["case_id"], "human_annotations": len(humans), "ai_annotations": len(by_case[c["case_id"]]) - len(humans),
                          "human_outcomes": outcomes, "disagreement": disagree, "adjudicated": adj is not None,
                          "final_outcome": adj["final_outcome"] if adj else (outcomes[0] if len(set(outcomes)) == 1 and len(humans) >= min_human else None),
                          "status": status})
    perm_counts = {kind: dict(Counter(s["source_permissions"][kind]["status"] for s in sources)) for kind in PERMISSION_KINDS}
    reasons = Counter(r for rej in rejected for r in rej["reasons"])
    report = {
        "policy_version": POLICY_VERSION,
        "inventory": {"sources": len(sources), "cases": len(cases), "annotations": len(annotations), "adjudications": len(adjudications),
                      "rejected_rows": len(rejected), "rejected_by_kind": dict(Counter(r["kind"] for r in rejected))},
        "provenance": dict(Counter(c["provenance"] for c in cases)),
        "annotation_kinds": dict(Counter(a["kind"] for a in annotations)),
        "outcomes_all_annotations": dict(Counter(a["outcome"] for a in annotations)),
        "case_status": dict(Counter(i["status"] for i in inventory)),
        "disagreements": sum(i["disagreement"] for i in inventory), "adjudicated": sum(i["adjudicated"] for i in inventory),
        "uncertain_retained": sum(1 for a in annotations if a["outcome"] == "uncertain"),
        "final_outcomes": dict(Counter(str(i["final_outcome"]) for i in inventory)),
        "source_permissions": perm_counts, "rejection_reasons": dict(reasons), "case_inventory": inventory,
        "note": "AI annotations are listed but never counted as human; uncertain outcomes stay in every denominator; rejected rows are retained in rejected.jsonl.",
    }
    return report


# ------------------------------------------------------------------------------------------------------ blind packets
def build_packets(ws: Workspace, salt: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Reviewer packets from an ALLOW-LIST only. Returns (packets, parent-only packet_id->case_id mapping)."""
    sources = {s["source_id"]: s for s in ws.read_jsonl("sources.jsonl")}
    packets, mapping = [], {}
    for c in ws.read_jsonl("cases.jsonl"):
        pid = "P" + sha256(f"{salt}:{c['case_id']}")[:12]
        packet = {"packet_id": pid, "source_text": sources[c["source_id"]]["text"], "candidate": c["candidate"],
                  "context": c.get("summary_context") or "", "policy": PACKET_POLICY}
        assert set(packet) == set(PACKET_ALLOW)
        packets.append(packet)
        mapping[pid] = c["case_id"]
    return packets, mapping


def assert_packet_clean(packet: Mapping[str, Any]) -> None:
    extra = set(packet) - set(PACKET_ALLOW)
    if extra:
        raise DataQualityError(f"packet carries non-allow-listed fields: {sorted(extra)}")


# ------------------------------------------------------------------------------------------------------ overlap and groups
def overlap_report(ws: Workspace, index: ReferenceIndex) -> dict[str, Any]:
    flagged = []
    for s in ws.read_jsonl("sources.jsonl"):
        for m in index.matches(s["source_id"], s["text"]):
            flagged.append({"source_id": s["source_id"], "reference": m.reference_name, "kind": m.kind, "jaccard": round(m.jaccard, 4),
                            "containment_of_candidate": round(m.containment_of_candidate, 4), "containment_of_reference": round(m.containment_of_reference, 4)})
    return {"flagged": flagged, "sources_checked": len(ws.read_jsonl("sources.jsonl")),
            "limits": "lexical exact/near-duplicate fingerprints only; no claim about semantic or pretraining independence"}


def check_split_groups(sources: Sequence[Mapping[str, Any]], split_of_family: Mapping[str, str]) -> dict[str, Any]:
    """Every source of one document family must be in the same split; missing assignments are reported, never defaulted."""
    families: dict[str, set[str]] = {}
    missing = []
    for s in sources:
        fam = s["document_family"]
        if fam not in split_of_family:
            missing.append(s["source_id"])
        else:
            families.setdefault(fam, set()).add(split_of_family[fam])
    leaked = sorted(f for f, splits in families.items() if len(splits) > 1)
    return {"leaked_families": leaked, "sources_without_assignment": missing}


def check_case_split_consistency(cases: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]], split_of_case: Mapping[str, str]) -> dict[str, Any]:
    fam_splits: dict[str, set[str]] = {}
    for c in cases:
        fam_splits.setdefault(sources[c["source_id"]]["document_family"], set()).add(split_of_case[c["case_id"]])
    return {"leaked_families": sorted(f for f, s in fam_splits.items() if len(s) > 1)}


# ------------------------------------------------------------------------------------------------------ training export
def export_training(ws: Workspace, out_name: str, *, uncertain_policy: str) -> dict[str, Any]:
    """Human-labelled binary export. Refuses unresolved/prohibited rights; never includes AI labels; declares the uncertain-handling policy."""
    if uncertain_policy not in ("exclude_from_binary_export",):
        raise DataQualityError("uncertain_policy must be declared as 'exclude_from_binary_export'")
    report = quality_report(ws)
    sources = {s["source_id"]: s for s in ws.read_jsonl("sources.jsonl")}
    cases = {c["case_id"]: c for c in ws.read_jsonl("cases.jsonl")}
    blocked = []
    for cid, c in cases.items():
        src = sources[c["source_id"]]
        status = src["source_permissions"]["train"]["status"]
        if status != "permitted":
            blocked.append({"case_id": cid, "source_id": c["source_id"], "source_train_permission": status})
        ann_perm = c.get("annotation_permissions", {}).get("train", {}).get("status", "unresolved")
        if ann_perm != "permitted":
            blocked.append({"case_id": cid, "annotation_train_permission": ann_perm})
    if blocked:
        raise DataQualityError(f"training export refused: {len(blocked)} rights blocks (unresolved or prohibited): {blocked[:5]}")
    incomplete = [i["case_id"] for i in report["case_inventory"] if i["status"] != "complete"]
    if incomplete:
        raise DataQualityError(f"training export refused: {len(incomplete)} cases incomplete or unadjudicated: {incomplete[:5]}")
    rows, excluded = [], []
    for i in report["case_inventory"]:
        if i["final_outcome"] == "uncertain":
            excluded.append({"case_id": i["case_id"], "reason": "uncertain outcome excluded from the BINARY export only (kept in inventory)"})
            continue
        c = cases[i["case_id"]]
        rows.append({"case_id": i["case_id"], "source_text": sources[c["source_id"]]["text"], "candidate": c["candidate"],
                     "label": 1 if i["final_outcome"] == "supported" else 0, "label_origin": "human"})
    manifest = {"cases_in_inventory": len(cases), "exported": len(rows), "excluded_uncertain": len(excluded), "uncertain_policy": uncertain_policy,
                "excluded": excluded, "ai_labels_exported": 0}
    ws.write_jsonl(out_name + ".jsonl", rows)
    ws.write_json(out_name + ".manifest.json", manifest)
    return manifest


# ------------------------------------------------------------------------------------------------------ protected index
def build_protected_index(out: Path) -> dict[str, Any]:
    """Fingerprints (hashes only) of every protected source document, read-only. Output must not be inside a protected location."""
    from .dataset import CaseRecord

    out = Path(out)
    for protected in PROTECTED_ROOTS:
        if _is_inside(out, protected):
            raise DataQualityError("index output must not be written inside a protected location")
    index, seen = ReferenceIndex(), set()
    cases_path = Path(__file__).parent / "data" / "cases.jsonl"
    for line in cases_path.read_text(encoding="utf8").splitlines():
        if line.strip():
            case = CaseRecord.model_validate_json(line)
            if case.context.strip() and case.group_id not in seen:
                seen.add(case.group_id)
                index.add(f"routebench:{case.group_id}", case.context)
    for rel in (ROOT / ".local" / "tofu70" / "rows.jsonl", Path(__file__).parent / "data" / "raw" / "tofueval_upstream" / "docs.jsonl"):
        if rel.exists():
            for line in rel.read_text(encoding="utf8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    name = f"tofu:{row.get('doc_id')}"
                    if name not in seen and row.get("text"):
                        seen.add(name)
                        index.add(name, row["text"])
    out.parent.mkdir(parents=True, exist_ok=True)
    index.save(out)
    return {"documents": len(index), "path": str(out)}


# ------------------------------------------------------------------------------------------------------ CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "import", "report", "packets", "overlap", "export-training", "build-protected-index"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--incoming", type=Path)
    parser.add_argument("--salt", default="")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--out", default="training_export")
    parser.add_argument("--uncertain-policy", default="")
    args = parser.parse_args(argv)
    if args.command == "build-protected-index":
        print(json.dumps(build_protected_index(Path(args.out)), indent=2))
        return 0
    if args.root is None:
        parser.error("--root required")
    ws = Workspace(args.root)
    ws.root.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "init":
            print(f"workspace ready: {ws.root}")
        elif args.command == "import":
            print(json.dumps(import_records(ws, args.incoming), indent=2))
        elif args.command == "report":
            report = quality_report(ws)
            ws.write_json("quality_report.json", report, overwrite=True)
            print(json.dumps({k: v for k, v in report.items() if k != "case_inventory"}, indent=2))
        elif args.command == "packets":
            if not args.salt:
                parser.error("--salt required")
            packets, mapping = build_packets(ws, args.salt)
            ws.write_jsonl("packets/packets.jsonl", packets, overwrite=True)
            ws.write_json("parent_only_mapping.json", mapping, overwrite=True)
            print(f"{len(packets)} blind packets written")
        elif args.command == "overlap":
            print(json.dumps(overlap_report(ws, ReferenceIndex.load(args.index)), indent=2))
        else:
            print(json.dumps(export_training(ws, args.out, uncertain_policy=args.uncertain_policy), indent=2))
    except DataQualityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
