import json
from pathlib import Path

import pytest
from evalops.dataset import Corpus
from evalops.gedd_exemplars import COMPLETENESS_MARKERS, build_diverse
from evalops.run_calibration import GRID, _validate_exemplar_provenance
from evalops.splits import SplitPlan, select

DATA = Path(__file__).parent.parent / "evalops" / "data"
pytestmark = pytest.mark.skipif(
    not (DATA / "gedd_diverse_exemplars.json").exists()
    or not any((DATA / "raw/authoritative").glob("tofueval_*.csv")),
    reason="optional local-only expert packet and annotation CSVs are not redistributed",
)


def _packet():
    return json.loads((DATA / "gedd_diverse_exemplars.json").read_text(encoding="utf8"))


def test_one_example_per_document_train_only_and_both_classes():
    corpus, sp = Corpus.load(DATA), SplitPlan.load(DATA / "splits.json")
    ex = _packet()["exemplars"]["summarize"]
    assert len({e["group_id"] for e in ex}) == len(ex) == 6
    train_ids = {c.case_id for c in select(corpus.cases, sp, "train")}
    assert all(e["case_id"] in train_ids for e in ex)
    assert {e["verdict"] for e in ex} == {"PASS", "FAIL"}
    held = {c.group_id for s in ("dev", "test") for c in select(corpus.cases, sp, s)}
    assert not {e["group_id"] for e in ex} & held


def test_completeness_exclusion_is_a_general_rule_and_recorded():
    p = _packet()
    assert "gnd-33aed12bc35c7bcd" in p["excluded_completeness_case_ids"]
    chosen = {e["case_id"] for e in p["exemplars"]["summarize"]}
    assert not chosen & set(p["excluded_completeness_case_ids"])
    for e in p["exemplars"]["summarize"]:
        assert e["verdict"] == "PASS" or not COMPLETENESS_MARKERS.search(e["why"])


def test_packet_is_reproducible_and_provenance_validates():
    corpus, sp = Corpus.load(DATA), SplitPlan.load(DATA / "splits.json")
    from evalops.gedd_exemplars import _load_inputs
    _, _, notes, raw = _load_inputs()
    assert build_diverse(corpus, sp, notes, raw) == _packet()
    cfg = next(c for c in GRID if c.variant_id == "v23-diverse-fewshot-o4mini")
    _validate_exemplar_provenance(cfg, corpus, sp)
