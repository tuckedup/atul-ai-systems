"""Local packet preparation/validation for an AI-only diagnostic panel, never gold-label export.

No model calls. Reads only an already evaluated upstream sample, not RouteBench test.
The original artifacts are read-only; every output goes into a new, separate directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / ".local" / "fresh-eval-v1"
WORK = ROOT / ".local" / "ai-panel-20261001-v1"
SEED = 2026100117
VERDICTS = {"supported", "unsupported", "uncertain"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def prepare() -> dict:
    if WORK.exists():
        raise ValueError("panel directory already exists; refusing to overwrite")
    items = json.loads((INPUT / "items.json").read_text(encoding="utf8"))
    labels = json.loads((INPUT / "labels.json").read_text(encoding="utf8"))
    baseline = {r["id"]: r for r in map(json.loads, (INPUT / "rows_v18.jsonl").read_text().splitlines())}
    errors = [r for r in items if labels[r["id"]] != baseline[r["id"]]["pass"]]
    controls = [r for r in items if labels[r["id"]] == baseline[r["id"]]["pass"]]
    if len(errors) != 42:
        raise ValueError("expected the known 42-error diagnostic population")
    rng = random.Random(SEED)
    selected = errors + rng.sample(controls, 18)
    rng.shuffle(selected)
    packet = WORK / "blind"
    packet.mkdir(parents=True)
    mapping = {}
    for index, item in enumerate(selected, 1):
        cid = f"C{index:03d}"
        save(packet / f"{cid}.json", {"case_id": cid, "source_text": item["text"], "candidate": item["claim"]})
        mapping[cid] = {"original_case_id": item["id"], "reference_label": labels[item["id"]],
                        "baseline_prediction": baseline[item["id"]]["pass"],
                        "selection_stratum": "error" if item in errors else "control"}
    save(WORK / "parent_only_mapping.json", mapping)
    manifest = {"provenance": "ai_assisted_diagnostic_not_human", "seed": SEED, "n": len(selected),
                "selection": "all 42 known baseline errors plus 18 sampled correct cases; not representative",
                "source_hashes": {name: digest(INPUT / name) for name in ("items.json", "labels.json", "rows_v18.jsonl")},
                "packet_hashes": {p.name: digest(p) for p in sorted(packet.glob("*.json"))},
                "mapping_sha256": digest(WORK / "parent_only_mapping.json"),
                "limitations": ["instruction-level blinding, shared filesystem", "AI errors may be correlated",
                                "parent has prior benchmark exposure", "not an independent human or held-out benchmark"]}
    save(WORK / "manifest.json", manifest)
    return {"directory": str(WORK), "n": len(selected)}


def _validated_output(name: str) -> tuple[dict, list[dict]]:
    """Read only blind packets and one panel output; do not join reference labels."""
    if name not in {"sol", "luna", "astra"}:
        raise ValueError("unknown panel member")
    manifest = json.loads((WORK / "manifest.json").read_text())
    filenames = set(manifest["packet_hashes"])
    if filenames != {p.name for p in (WORK / "blind").glob("*.json")} or len(filenames) != manifest["n"]:
        raise ValueError("packet inventory differs from frozen manifest")
    for filename, expected in manifest["packet_hashes"].items():
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("invalid packet filename")
        if digest(WORK / "blind" / filename) != expected:
            raise ValueError(f"packet changed: {filename}")
    packets = {Path(name).stem: json.loads((WORK / "blind" / name).read_text(encoding="utf8")) for name in filenames}
    if any(case.get("case_id") != cid for cid, case in packets.items()):
        raise ValueError("packet case ID differs from filename")
    payload = (WORK / f"{name}.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf8").splitlines() if line.strip()]
    ids = [r["case_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(packets):
        raise ValueError("duplicate, missing or unknown case IDs")
    for row in rows:
        case = packets[row["case_id"]]
        if row["verdict"] not in VERDICTS or row.get("confidence") not in {"low", "medium", "high"}:
            raise ValueError(f"invalid verdict/confidence: {row['case_id']}")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError(f"missing rationale: {row['case_id']}")
        assertions = row.get("assertions")
        if not isinstance(assertions, list) or not assertions:
            raise ValueError(f"missing assertions: {row['case_id']}")
        for assertion in assertions:
            if not isinstance(assertion.get("text"), str) or not assertion["text"] or assertion["text"] not in case["candidate"]:
                raise ValueError(f"assertion is not candidate text: {row['case_id']}")
            if assertion.get("verdict") not in VERDICTS or not isinstance(assertion.get("explanation"), str) or not assertion["explanation"].strip():
                raise ValueError(f"invalid assertion verdict/explanation: {row['case_id']}")
            quotes = assertion.get("source_quotes")
            if not isinstance(quotes, list) or any(not isinstance(q, str) or not q or q not in case["source_text"] for q in quotes):
                raise ValueError(f"inauthentic source quote: {row['case_id']}")
            if assertion["verdict"] == "supported" and not quotes:
                raise ValueError(f"supported assertion lacks source evidence: {row['case_id']}")
        expected = "unsupported" if any(a["verdict"] == "unsupported" for a in assertions) else (
            "uncertain" if any(a["verdict"] == "uncertain" for a in assertions) else "supported")
        if row["verdict"] != expected:
            raise ValueError(f"case/assertion verdict inconsistency: {row['case_id']}")
    return ({"member": name, "rows": len(rows), "verdicts": dict(Counter(r["verdict"] for r in rows)),
             "sha256": hashlib.sha256(payload).hexdigest(),
             "validation": "structure and quote authenticity only; not entailment or independence proof"}, rows)


def validate(name: str) -> dict:
    return _validated_output(name)[0]


def compare() -> dict:
    """Only after three complete outputs: descriptive comparison, never a new gold export."""
    target = WORK / "comparison.json"
    if target.exists():
        raise ValueError("comparison already exists; refusing to overwrite")
    snapshots = {member: _validated_output(member) for member in ("sol", "luna", "astra")}
    checks = {member: snapshot[0] for member, snapshot in snapshots.items()}
    manifest = json.loads((WORK / "manifest.json").read_text())
    if digest(WORK / "parent_only_mapping.json") != manifest["mapping_sha256"]:
        raise ValueError("reference mapping changed")
    for name, expected in manifest["source_hashes"].items():
        if digest(INPUT / name) != expected:
            raise ValueError(f"original evaluated artifact changed: {name}")
    mapping = json.loads((WORK / "parent_only_mapping.json").read_text())
    if set(mapping) != {Path(name).stem for name in manifest["packet_hashes"]}:
        raise ValueError("reference mapping differs from frozen population")
    outputs = {member: {row["case_id"]: row for row in snapshot[1]} for member, snapshot in snapshots.items()}
    cases = []
    for cid, original in sorted(mapping.items()):
        reference = "supported" if original["reference_label"] else "unsupported"
        final = outputs["astra"][cid]["verdict"]
        cases.append({"case_id": cid, **original, "reference_verdict": reference,
                      "sol": outputs["sol"][cid]["verdict"], "luna": outputs["luna"][cid]["verdict"], "astra": final,
                      "astra_reference_relation": "uncertain" if final == "uncertain" else "agree" if final == reference else "disagree"})
    report = {"provenance": "AI_ONLY_DIAGNOSTIC_NOT_HUMAN_GOLD", "n": len(cases), "validation": checks,
              "sol_luna_exact_agreement": sum(c["sol"] == c["luna"] for c in cases),
              "sol_luna_disagreement": sum(c["sol"] != c["luna"] for c in cases),
              "astra_vs_reference_by_selection_stratum": {
                  group: dict(Counter(c["astra_reference_relation"] for c in cases if c["selection_stratum"] == group))
                  for group in ("error", "control")},
              "cases": cases,
              "limitations": manifest["limitations"] + [
                  "42-error + 18-control selection is not population representative",
                  "reference disagreement is not proof the reference label is wrong",
                  "no kappa claim, benchmark relabeling, training authorization or human-review substitution"]}
    save(target, report)
    return {key: value for key, value in report.items() if key != "cases"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "validate", "compare"])
    parser.add_argument("--member", choices=["sol", "luna", "astra"])
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "compare":
        result = compare()
    else:
        result = validate(args.member)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
