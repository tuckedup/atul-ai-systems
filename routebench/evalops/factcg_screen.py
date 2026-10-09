"""One fixed-threshold FactCG dev screen. Prepare/report use the project environment;
download/run use the isolated GPU environment. No training or test-split command.

The prompt, 550-word source chunking and max-over-chunks aggregation follow
https://github.com/derenlei/FactCG/tree/41f185413312ed9b3904ec03963fe899693498e0
(MIT; see docs/FACTCG_NOTICE.txt). Unlike upstream, overlength prompts fail rather
than truncate. This is a screening adaptation, not a leaderboard reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
REVISION = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
THRESHOLD = 0.5
MAX_SECONDS = 3600
PROMPT = ('{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that '
          '"{text_b}"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is ')
FILES = ["README.md", "config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "spm.model"]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def source_chunks(text: str, sentences: list[str], word_counts: list[int]) -> list[str]:
    """Keep every source sentence, including an individually long one; never add an empty chunk."""
    if not text.strip() or not sentences or len(sentences) != len(word_counts):
        raise ValueError("empty or malformed source")
    # Verify the sentence tokenizer did not omit non-whitespace content.
    compact = lambda s: "".join(s.split())
    if compact(text) != compact("".join(sentences)):
        raise ValueError("sentence tokenizer lost source content")
    out, current, words = [], [], 0
    for sentence, count in zip(sentences, word_counts, strict=True):
        if current and words + count > 550:
            out.append("\n".join(current))
            current, words = [], 0
        current.append(sentence)
        words += count
    out.append("\n".join(current))
    return out


def prepare(work: Path) -> None:
    from .dataset import HEADLINE_PROVENANCE, Annotation, CaseRecord, Corpus
    from .experiments import JudgmentCache, _cache_key
    from .run_calibration import GRID

    assignment = read(DATA / "splits.json")["assignment"]
    cases = []
    for line in (DATA / "cases.jsonl").read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        if assignment[row["group_id"]] == "dev":
            cases.append(CaseRecord.model_validate(row))
    ids = {c.case_id for c in cases}
    annotations = []
    for line in (DATA / "annotations.jsonl").read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        if row["case_id"] in ids:
            annotations.append(Annotation.model_validate(row))
    labels = Corpus(cases, annotations).resolved_labels(allow=HEADLINE_PROVENANCE)
    expected = {a.case_id for a in annotations if a.provenance in HEADLINE_PROVENANCE}
    if set(labels) != expected:
        raise ValueError("unresolved dev labels; refusing to reduce denominator")
    cases = sorted([c for c in cases if c.case_id in labels], key=lambda c: c.case_id)
    if len(cases) != 128:
        raise ValueError(f"expected frozen 128-case headline dev population, got {len(cases)}")
    config = next(c for c in GRID if c.variant_id == "v18-reasoning-o4mini")
    if config.config_hash != "12fc8b550d954359":
        raise ValueError("v18 identity changed")
    cache = JudgmentCache(DATA / "judgment_cache.jsonl")
    pairs, metadata = [], []
    for c in cases:
        j = cache.get(_cache_key(c, config))
        if j is None or not j.usable:
            raise ValueError(f"missing frozen v18 prediction: {c.case_id}")
        pairs.append({"id": c.case_id, "premise": c.context, "hypothesis": c.candidate_output.strip()})
        metadata.append({"id": c.case_id, "group": c.group_id, "label": int(labels[c.case_id].label),
                         "v18": int(float(j.raw_score) >= 1.0)})
    body = {"model": MODEL, "revision": REVISION, "threshold": THRESHOLD, "max_seconds": MAX_SECONDS,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pairs_sha256": digest(pairs), "metadata_sha256": digest(metadata),
            "n": len(pairs), "scope": "repeatedly inspected headline dev; screening only"}
    work.mkdir(parents=True, exist_ok=False)
    write_new(work / "pairs.json", pairs)
    write_new(work / "metadata.json", metadata)
    write_new(work / "protocol.json", body)
    print(json.dumps(body, indent=2))


def check_protocol(work: Path) -> dict[str, Any]:
    protocol = read(work / "protocol.json")
    if (protocol["script_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or protocol["pairs_sha256"] != digest(read(work / "pairs.json"))
            or protocol["metadata_sha256"] != digest(read(work / "metadata.json"))
            or protocol["revision"] != REVISION or protocol["threshold"] != THRESHOLD):
        raise ValueError("prepared protocol or inputs changed")
    return protocol


def validate_scores(expected: list[str], rows: list[dict[str, Any]]) -> dict[str, float]:
    if len(expected) != len(set(expected)) or len(rows) != len(expected):
        raise ValueError("incomplete or duplicate score population")
    scores = {}
    for row in rows:
        value = float(row["score"])
        if row["id"] in scores or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("duplicate or invalid score")
        scores[row["id"]] = value
    if set(scores) != set(expected):
        raise ValueError("score IDs differ from frozen population")
    return scores


def worker(work: Path, model_dir: Path) -> None:
    check_protocol(work)
    import nltk
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model_dir.name != REVISION:
        raise ValueError("model path must be the pinned Hugging Face snapshot directory")
    # Assets must already be installed; there are no implicit network downloads here.
    nltk.sent_tokenize("Tokenizer asset check.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True).to("cuda").eval()
    started = time.perf_counter()
    pairs = read(work / "pairs.json")
    with (work / "partial_scores.jsonl").open("x", encoding="utf8") as stream:
        for n, pair in enumerate(pairs, 1):
            sentences = nltk.sent_tokenize(pair["premise"])
            chunks = source_chunks(pair["premise"], sentences, [len(nltk.word_tokenize(s)) for s in sentences])
            scores, lengths = [], []
            for chunk in chunks:
                enc = tok(PROMPT.format(text_a=chunk, text_b=pair["hypothesis"]),
                          return_tensors="pt", truncation=False).to("cuda")
                length = enc["input_ids"].shape[1]
                if length > 2048:
                    raise ValueError(f"{pair['id']}: {length} tokens exceeds 2048; no truncation allowed")
                with torch.inference_mode():
                    value = torch.softmax(model(**enc).logits.float(), dim=-1)[0, 1].item()
                if not math.isfinite(value):
                    raise ValueError("non-finite model output")
                scores.append(value)
                lengths.append(length)
            row = {"id": pair["id"], "score": max(scores), "chunks": len(chunks), "max_tokens": max(lengths)}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps({"done": n, "total": len(pairs), "seconds": time.perf_counter() - started}), flush=True)


def run(work: Path, model_dir: Path) -> int:
    from .pilot import ledger

    check_protocol(work)
    if (work / "terminal.json").exists() or (work / "partial_scores.jsonl").exists():
        raise ValueError("run already exists; refusing a second attempt or overwrite")
    ledger_path = ROOT / ".local" / "ragtruth-ingest-20260930" / ledger.LEDGER_NAME
    ledger.check_budget(ledger_path, MAX_SECONDS)
    started = time.perf_counter()
    status, problem, code = "FAILED", None, None
    with (work / "run.log").open("x", encoding="utf8") as log:
        try:
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--work", str(work),
                                    "--model-dir", str(model_dir)], cwd=ROOT, stdout=log, stderr=log,
                                   timeout=MAX_SECONDS, check=False)
            code = child.returncode
            if code == 0:
                rows = [json.loads(s) for s in (work / "partial_scores.jsonl").read_text().splitlines()]
                validate_scores([p["id"] for p in read(work / "pairs.json")], rows)
                status = "COMPLETE"
        except subprocess.TimeoutExpired:
            status, problem = "INCOMPLETE", "one-hour wall-clock cap reached"
        except (ValueError, KeyError, OSError) as exc:
            problem = str(exc)
    terminal = {"status": status, "exit_code": code, "problem": problem,
                "wall_seconds": time.perf_counter() - started, "api_usd": 0,
                "protocol_sha256": digest(read(work / "protocol.json"))}
    if status == "COMPLETE":
        terminal["scores_sha256"] = file_digest(work / "partial_scores.jsonl")
    write_new(work / "terminal.json", terminal)
    ledger.append(ledger_path, kind="factcg_frozen_dev_screen", wall_seconds=terminal["wall_seconds"],
                  verdict=status, note="128 headline dev cases; fixed 0.5 threshold; no training")
    print(json.dumps(terminal))
    return 0 if status == "COMPLETE" else 3


def report(work: Path) -> None:
    check_protocol(work)
    import numpy as np
    from sklearn.metrics import (
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
    )

    terminal = read(work / "terminal.json")
    if terminal["status"] != "COMPLETE" or terminal["protocol_sha256"] != digest(read(work / "protocol.json")):
        raise ValueError("no complete, matching terminal result")
    if terminal["scores_sha256"] != file_digest(work / "partial_scores.jsonl"):
        raise ValueError("scores changed after completed run")
    meta = read(work / "metadata.json")
    rows = [json.loads(s) for s in (work / "partial_scores.jsonl").read_text().splitlines()]
    scores = validate_scores([p["id"] for p in meta], rows)
    y, baseline = np.array([p["label"] for p in meta]), np.array([p["v18"] for p in meta])
    pred = np.array([int(scores[p["id"]] > THRESHOLD) for p in meta])
    def metric(p: Any) -> dict[str, Any]:
        return {"kappa": float(cohen_kappa_score(y, p)), "balanced_accuracy": float(balanced_accuracy_score(y, p)),
                "errors": int(sum(y != p)), "confusion_labels_0_1": confusion_matrix(y, p, labels=[0, 1]).tolist()}
    groups = sorted({p["group"] for p in meta})
    indices = {g: np.array([i for i, p in enumerate(meta) if p["group"] == g]) for g in groups}
    rng, deltas = np.random.default_rng(20260930), []
    for _ in range(2000):
        ix = np.concatenate([indices[g] for g in rng.choice(groups, len(groups))])
        if len(set(y[ix])) == 2:
            deltas.append(cohen_kappa_score(y[ix], pred[ix]) - cohen_kappa_score(y[ix], baseline[ix]))
    result = {"scope": "128 previously inspected dev cases; no independent acceptance claim", "n": len(meta),
              "threshold": THRESHOLD, "factcg": metric(pred), "v18": metric(baseline),
              "corrected": int(sum((baseline != y) & (pred == y))),
              "introduced": int(sum((baseline == y) & (pred != y))),
              "paired_kappa_delta_ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
              "provenance": "published training recipe reviewed; exact training-document overlap not certified"}
    write_new(work / "report.json", result)
    print(json.dumps(result, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "worker", "report"])
    parser.add_argument("--work", type=Path, default=ROOT / ".local" / "factcg-screen-v1")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    if args.command == "prepare":
        prepare(work)
    elif args.command == "report":
        report(work)
    else:
        if args.model_dir is None:
            parser.error("--model-dir required")
        if args.command == "worker":
            worker(work, args.model_dir.resolve())
        else:
            return run(work, args.model_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
