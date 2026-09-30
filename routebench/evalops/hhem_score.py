"""HHEM-2.1-Open scorer (runs in the ISOLATED .local/hhem-venv; no project imports, no API, no training).

Frozen input handling (docs/ROUTEBENCH_HHEM_SCREEN_PROTOCOL.md):
  premise    = the case's full source document, unmodified
  hypothesis = the case's candidate text, stripped; multi-sentence summaries are NOT split
  prompt     = the model's own template; the remote code applies no truncation, and none is added here
  score      = P(consistent) from the model's two output neurons (softmax), fp32, batch size 4 sorted by length
Reads  .local/hhem/pairs.json  [{"id","premise","hypothesis"}]
Writes .local/hhem/scores.json {"scores": {id: p}, "seconds": s, "max_tokens": n, "n_over_512": k, ...}
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".local/hhem")
BATCH = 4


def main() -> None:
    pairs = json.loads((ROOT / "pairs.json").read_text(encoding="utf8"))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ROOT / "model"), trust_remote_code=True, local_files_only=True)
    model.t5.to("cuda")
    tok = model.tokenzier  # (sic) attribute name used by the upstream remote code
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]["premise"]) + len(pairs[i]["hypothesis"]))
    scores: dict[str, float] = {}
    lengths: list[int] = []
    started = time.perf_counter()
    for start in range(0, len(order), BATCH):
        batch = [pairs[i] for i in order[start:start + BATCH]]
        texts = [model.prompt.format(text1=p["premise"], text2=p["hypothesis"]) for p in batch]
        lengths += [len(tok(t)["input_ids"]) for t in texts]
        out = model.predict([(p["premise"], p["hypothesis"]) for p in batch])
        for p, s in zip(batch, out.float().cpu().tolist()):
            scores[p["id"]] = s
    seconds = time.perf_counter() - started
    result = {"scores": scores, "seconds": seconds, "n": len(scores), "max_tokens": max(lengths),
              "n_over_512": sum(n > 512 for n in lengths), "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__}
    (ROOT / "scores.json").write_text(json.dumps(result), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}))


if __name__ == "__main__":
    main()
