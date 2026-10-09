"""Offline blind-review packet for the residual-error adjudication (docs/ROUTEBENCH_STAGE1A_CLOSURE_AND_RECOMMENDATION_20261005.md, as corrected by the owner's review).
No model call, no GPU, no purchase, no label change, no held-out access. It reads the validated Stage 1a artifacts and writes two SEPARATE directories:

  packet/           what a reviewer receives: opaque review ids, the sentence and its source document, instructions, a response format. Nothing else.
  reconciliation/   kept away from reviewers: id -> case mapping, selection group (persistent error or control), published label, the five v18 verdicts and scores, selection manifest.

Selection (deterministic; no random-number state): PERSISTENT = the original TRAIN/dev cases on which all five v18 verdicts agree and are wrong against the published label (44 expected);
CONTROLS = the same number of unanimous-correct cases, first by sha256("blind-review-v1|control|<case id>"); ORDER of the combined list = sha256("blind-review-v1|order|<case id>"). Seed string: blind-review-v1.
Outcomes stay three-valued (supported / unsupported / uncertain). The reconciliation file states that results are DESCRIPTIVE: no threshold converts them into a label ceiling or authorises spending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SEED = "blind-review-v1"
OUTCOMES = ("supported", "unsupported", "uncertain")
REVIEWER_KINDS = ("human", "ai_assisted")
PACKET_KEYS = ("review_id", "sentence", "source")
INSTRUCTIONS = """# Source-support review: instructions

You are asked for an independent judgement on {n} short sentences. Each sentence is one sentence taken from a summary of the document shown with it. Some items will be easy and some hard;
you will not be told which. Work alone. Please do not discuss items with anyone, and do not look anything up.

## What to decide
For each item decide whether the DOCUMENT supports the marked SENTENCE, using only the document:
- **supported**: every factual assertion in the sentence is stated in the document, or follows from it by an obvious, direct inference (no speculation, no outside knowledge).
- **unsupported**: at least one factual assertion in the sentence is contradicted by the document, or is not stated and does not follow directly from it.
- **uncertain**: you cannot decide from the document alone, or reasonable reviewers could differ (for example ambiguous wording, a reference that cannot be resolved without the rest of the summary, conflicting passages, or a judgement that depends on how strict the standard is). Uncertain is a legitimate outcome, not a failure; say exactly why.

## How to decide
1. Split the sentence into its factual assertions (who, what, when, how many, cause, attitude, quotation) and check each one against the document.
2. Paraphrase is fine. Different spellings or transcription variants of the same name or title are the same entity.
3. Numbers, dates, quantities, tense (did / will), strength (may / will / must) and attribution (who said or did it) must match.
4. A sentence that talks about the document itself ("The document says...", "The speaker argues...") is judged on whether the document does what is described.
5. Do not use general knowledge, the web, or any AI tool to decide. If you used any assistance, you must mark your response `reviewer_kind` as `ai_assisted`; such responses are kept separate and are never counted as independent human annotation.
6. Do not try to guess what a "correct answer" is expected to be. There is none that you can see.

## What to record (one JSON object per line in `responses.jsonl`)
| field | required | meaning |
|---|---|---|
| `review_id` | yes | the id shown with the item (R001 ...), each exactly once |
| `reviewer_id` | yes | a pseudonymous id, the same for all your responses |
| `reviewer_kind` | yes | `human` or `ai_assisted` |
| `outcome` | yes | `supported`, `unsupported` or `uncertain` |
| `evidence` | yes | a list of EXACT quotations copied from the document that you relied on (the passage that supports it, the passage that contradicts it, or the closest passage). If the document contains nothing relevant, use an empty list and say so in the explanation |
| `explanation` | yes | at least one full sentence: which assertion(s) you checked and why the evidence leads to your outcome; for `uncertain`, what is missing or ambiguous |
| `minutes` | no | active time spent on the item |
| `notes` | no | anything else |

Quotations must be verbatim (a validator checks them against the document text, ignoring only differences in whitespace). Keep each quotation short but sufficient to check your reading.

## Handling
The documents are research material under their own licences; use them privately for this review, do not redistribute them, and do not paste them into any online service, including AI tools.
Every item should be reviewed independently by at least two reviewers who do not see one another's responses; disagreements and every `uncertain` response go to a third, later adjudication.
"""


class PacketError(ValueError):
    pass


def _h(tag: str, case_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{tag}|{case_id}".encode()).hexdigest()


def select(ids: Sequence[str], labels: dict[str, int], verdicts: dict[str, list[int]]) -> dict[str, Any]:
    """Pure selection. `labels`: 1 = supported. `verdicts`: five original pass/fail verdicts per case."""
    unanimous = [c for c in ids if len(set(verdicts[c])) == 1]
    persistent = sorted((c for c in unanimous if verdicts[c][0] != labels[c]), key=lambda c: _h("persistent", c))
    pool = [c for c in unanimous if verdicts[c][0] == labels[c]]
    controls = sorted(pool, key=lambda c: _h("control", c))[: len(persistent)]
    if len(controls) != len(persistent):
        raise PacketError("not enough unanimous-correct cases to draw an equal-size control set")
    order = sorted([*persistent, *controls], key=lambda c: _h("order", c))
    return {"persistent": sorted(persistent), "controls": sorted(controls), "order": order, "pool_size": len(pool), "unanimous_cases": len(unanimous)}


def build_items(order: Sequence[str], cases: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    items, mapping = [], {}
    for n, c in enumerate(order, start=1):
        rid = f"R{n:03d}"
        items.append({"review_id": rid, "sentence": cases[c].candidate_output, "source": cases[c].context})
        mapping[rid] = c
    return items, mapping


def render_markdown(items: Sequence[dict[str, str]]) -> str:
    parts = ["# Items for review\n\nRead `INSTRUCTIONS.md` first. Respond in `responses.jsonl`.\n"]
    for it in items:
        parts.append(f"\n---\n\n## {it['review_id']}\n\n**Sentence to judge:** {it['sentence']}\n\n**Document:**\n\n{it['source']}\n")
    return "".join(parts)


def write_packet(out: Path, items: Sequence[dict[str, str]]) -> dict[str, str]:
    out.mkdir(parents=True)
    body = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    (out / "review_items.jsonl").write_text(body, encoding="utf8")
    (out / "review_items.md").write_text(render_markdown(items), encoding="utf8")
    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS.format(n=len(items)), encoding="utf8")
    example = {"review_id": "R001", "reviewer_id": "reviewer-A", "reviewer_kind": "human", "outcome": "uncertain", "evidence": ["<exact quotation from the document>"],
               "explanation": "<which assertion you checked, and why the evidence leads to this outcome>", "minutes": 0}
    (out / "response_format_example.jsonl").write_text(json.dumps(example) + "\n", encoding="utf8")
    hashes = {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ("review_items.jsonl", "review_items.md", "INSTRUCTIONS.md", "response_format_example.jsonl")}
    (out / "packet_manifest.json").write_text(json.dumps({"items": len(items), "item_fields": list(PACKET_KEYS), "outcomes": list(OUTCOMES), "reviewer_kinds": list(REVIEWER_KINDS), "sha256": hashes}, indent=2), encoding="utf8")
    return hashes


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def validate_responses(packet_dir: Path, responses: Path) -> dict[str, Any]:
    """Format and evidence check only (no label, no reconciliation). Returns {problems, counts}."""
    items = {json.loads(x)["review_id"]: json.loads(x) for x in (Path(packet_dir) / "review_items.jsonl").read_text(encoding="utf8").splitlines() if x.strip()}
    problems, seen, counts, kinds = [], set(), Counter(), Counter()
    for n, line in enumerate(Path(responses).read_text(encoding="utf8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {n}: not valid JSON")
            continue
        if not isinstance(r, dict):
            problems.append(f"line {n}: not an object")
            continue
        rid = r.get("review_id")
        if rid not in items:
            problems.append(f"line {n}: unknown review_id {rid!r}")
            continue
        key = (rid, r.get("reviewer_id"))
        if key in seen:
            problems.append(f"line {n}: duplicate response for {rid} by {r.get('reviewer_id')!r}")
        seen.add(key)
        if not isinstance(r.get("reviewer_id"), str) or not r["reviewer_id"].strip():
            problems.append(f"line {n} ({rid}): reviewer_id missing")
        if r.get("reviewer_kind") not in REVIEWER_KINDS:
            problems.append(f"line {n} ({rid}): reviewer_kind must be one of {REVIEWER_KINDS}")
        if r.get("outcome") not in OUTCOMES:
            problems.append(f"line {n} ({rid}): outcome must be one of {OUTCOMES}")
        ev = r.get("evidence")
        if not isinstance(ev, list) or not all(isinstance(q, str) and q.strip() for q in ev):
            problems.append(f"line {n} ({rid}): evidence must be a list of non-empty quotations (empty list allowed)")
        else:
            doc = _squash(items[rid]["source"])
            for q in ev:
                if _squash(q) not in doc:
                    problems.append(f"line {n} ({rid}): a quotation is not verbatim in the document: {q[:60]!r}")
        if not isinstance(r.get("explanation"), str) or len(r["explanation"].split()) < 5:
            problems.append(f"line {n} ({rid}): explanation is required (at least a full sentence)")
        counts[r.get("outcome")] += 1
        kinds[r.get("reviewer_kind")] += 1
    return {"problems": problems, "responses": sum(counts.values()), "outcome_counts": dict(counts), "reviewer_kind_counts": dict(kinds), "items_with_a_response": len({k[0] for k in seen}),
            "note": "format and quotation check only; ai_assisted responses are never counted as independent human annotation"}


def build(work: Path, out_root: Path, protocol_sha: str) -> dict[str, Any]:
    from . import judge_v19_analysis as an
    from . import judge_v19_recovery as rec
    from .dataset import CaseRecord

    out_root = Path(out_root)
    if out_root.exists():
        raise PacketError("output directory exists; the packet is written once")
    scores, info = rec.assemble_scores(work, protocol_sha)  # validated rows + recovery row; refuses on any inconsistency
    cached = json.loads((Path(work) / "cached_v18.json").read_text(encoding="utf8"))
    pop = json.loads((Path(work) / "population.json").read_text(encoding="utf8"))["routebench"]
    labels = json.loads((Path(work) / "labels_routebench.json").read_text(encoding="utf8"))
    cases = {}
    for line in (Path(work) / "cases_routebench.jsonl").read_text(encoding="utf8").splitlines():
        c = CaseRecord.model_validate_json(line)
        cases[c.case_id] = c
    ids = [i["case_id"] for i in pop]
    five = {c: [float(cached[c]["score"]), *scores[c]] for c in ids}
    verdicts = {c: [an.verdict(s) for s in five[c]] for c in ids}
    sel = select(ids, labels, verdicts)
    items, mapping = build_items(sel["order"], cases)
    hashes = write_packet(out_root / "packet", items)
    meta = {i["case_id"]: i for i in pop}
    recon = {"status": "DO NOT SHARE WITH REVIEWERS",
             "purpose": "descriptive characterisation of residual errors; original labels, cases and splits are unchanged; AI-assisted responses are not independent human annotation",
             "analysis_rule": "Outcomes are reported descriptively (counts by group, reviewer agreement, uncertain share). No threshold (for example 50% or 75%) converts them into a label ceiling or authorises any spending; "
                              "'uncertain' does not mean the published label is wrong, and reviewers siding with v18 would not show that 0.74 is unreachable. The reviewers' written policy is plain-language and is not the original annotators' guideline.",
             "seed": SEED, "protocol_sha256": protocol_sha, "stage_1a_committed_usd": info["committed_usd"], "packet_sha256": hashes,
             "selection": {"persistent_errors": len(sel["persistent"]), "controls": len(sel["controls"]), "unanimous_cases": sel["unanimous_cases"], "unanimous_correct_pool": sel["pool_size"],
                           "persistent_false_passes": sum(1 for c in sel["persistent"] if verdicts[c][0] == 1), "persistent_false_rejections": sum(1 for c in sel["persistent"] if verdicts[c][0] == 0),
                           "controls_published_supported": sum(1 for c in sel["controls"] if labels[c] == 1), "controls_published_unsupported": sum(1 for c in sel["controls"] if labels[c] == 0)},
             "items": {rid: {"case_id": c, "group": "persistent_error" if c in set(sel["persistent"]) else "control", "published_label": labels[c], "published_label_meaning": "1 = supported",
                             "v18_verdicts": verdicts[c], "v18_scores": five[c], "population": meta[c]["pop"], "subset": meta[c]["subset"], "document_group": meta[c]["group"]} for rid, c in mapping.items()}}
    (out_root / "reconciliation").mkdir()
    (out_root / "reconciliation" / "reconciliation.json").write_text(json.dumps(recon, indent=2, ensure_ascii=False), encoding="utf8")
    return {"items": len(items), "selection": recon["selection"], "packet_dir": str(out_root / "packet"), "reconciliation": str(out_root / "reconciliation" / "reconciliation.json")}


def main(argv: Sequence[str] | None = None) -> int:
    from . import judge_v19_phase1 as jv

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["build", "validate"])
    p.add_argument("--work", type=Path, default=jv.WORK)
    p.add_argument("--out", type=Path, default=jv.WORK / "blind-review-v1")
    p.add_argument("--protocol-sha256")
    p.add_argument("--packet", type=Path)
    p.add_argument("--responses", type=Path)
    a = p.parse_args(argv)
    try:
        if a.command == "build":
            if not a.protocol_sha256:
                print("REFUSED: --protocol-sha256 is required", file=sys.stderr)
                return 2
            print(json.dumps(build(a.work, a.out, a.protocol_sha256), indent=2))
        else:
            if not a.packet or not a.responses:
                print("REFUSED: --packet and --responses are required", file=sys.stderr)
                return 2
            res = validate_responses(a.packet, a.responses)
            print(json.dumps(res, indent=2))
            return 0 if not res["problems"] else 3
    except (PacketError, jv.Phase1Error, jv.sv3.SrcIntV3Error) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
