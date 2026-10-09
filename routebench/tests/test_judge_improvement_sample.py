"""Tests for the deterministic RAGTruth fitting-sample rule (synthetic manifest; no real data)."""
from __future__ import annotations

import json

from evalops import judge_improvement_sample as js

COLS = ["example_id", "response_id", "source_id", "partition", "document_fp16", "group_id", "label", "start", "end", "implicit_true", "crosses_boundary"]


def manifest(n_groups=30, per=12):
    rows = []
    for g in range(n_groups):
        for i in range(per):
            label = 0 if i % 6 == 0 else 1
            rows.append([f"{g}#{i}", str(g), str(g), "train" if g < n_groups - 3 else "validation", "fp", f"g{g}", label, 0, 10, False, i == 7 and g % 5 == 0])
    return {"columns": COLS, "rows": rows, "manifest_sha256": "x" * 64}


def test_sample_is_deterministic_and_document_grouped():
    a, b = js.draw(manifest(), documents=10, supported=40), js.draw(manifest(), documents=10, supported=40)
    assert a == b and a["counts"]["documents"] == 10
    groups = {r[3] for r in a["rows"]}
    assert len(groups) <= 10


def test_every_unsupported_sentence_of_chosen_documents_is_kept_and_supported_is_capped():
    s = js.draw(manifest(), documents=10, supported=15)
    assert s["counts"]["supported"] == 15 and sum(1 for r in s["rows"] if r[4] == 0) == s["counts"]["unsupported"]
    chosen = {r[3] for r in s["rows"]}
    full = manifest()
    all_unsup = [r for r in full["rows"] if r[3] != "validation" and r[6] == 0 and f"g{r[0].split('#')[0]}" in chosen]
    assert len(all_unsup) >= s["counts"]["unsupported"]


def test_validation_partition_and_boundary_crossing_sentences_never_enter():
    s = js.draw(manifest(), documents=27, supported=10_000)
    ids = {r[0] for r in s["rows"]}
    assert not any(int(i.split("#")[0]) >= 27 for i in ids)
    assert s["counts"]["dropped_crosses_boundary"] > 0
    for g in range(0, 27, 5):
        assert f"{g}#7" not in ids


def test_cli_refuses_to_overwrite(tmp_path):
    m, out = tmp_path / "m.json", tmp_path / "o.json"
    m.write_text(json.dumps(manifest()), encoding="utf8")
    assert js.main(["--manifest", str(m), "--out", str(out)]) == 0
    assert js.main(["--manifest", str(m), "--out", str(out)]) == 2
