"""Synthetic checks for diagnostic AI packets. No inference or benchmark access."""
import json

import pytest
from evalops import ai_panel as panel


@pytest.fixture
def packet(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "WORK", tmp_path)
    blind = tmp_path / "blind"
    blind.mkdir()
    case = {"case_id": "C001", "candidate": "The council voted on Monday.",
            "source_text": "The council voted on Monday. The proposal passed."}
    panel.save(blind / "C001.json", case)
    panel.save(tmp_path / "manifest.json", {"n": 1, "packet_hashes": {"C001.json": panel.digest(blind / "C001.json")}})
    row = {"case_id": "C001", "verdict": "supported", "confidence": "high", "rationale": "Directly stated.",
           "assertions": [{"text": case["candidate"], "verdict": "supported",
                           "source_quotes": [case["candidate"]], "explanation": "The source states this event and date."}]}
    return tmp_path, row


def write_rows(packet, rows):
    (packet[0] / "sol.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")


def test_validate_requires_no_reference_labels(packet):
    write_rows(packet, [packet[1]])
    result = panel.validate("sol")
    assert result["rows"] == 1
    assert not (packet[0] / "parent_only_mapping.json").exists()


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown"])
def test_exact_population_required(packet, problem):
    row = packet[1]
    if problem == "unknown":
        row["case_id"] = "C999"
    write_rows(packet, [] if problem == "missing" else [row, row] if problem == "duplicate" else [row])
    with pytest.raises(ValueError, match="case IDs"):
        panel.validate("sol")


def test_fabricated_quote_rejected(packet):
    packet[1]["assertions"][0]["source_quotes"] = ["The vote was unanimous."]
    write_rows(packet, [packet[1]])
    with pytest.raises(ValueError, match="inauthentic"):
        panel.validate("sol")


def test_unsupported_addition_need_not_have_contradicting_quote(packet):
    row = packet[1]
    row["verdict"] = row["assertions"][0]["verdict"] = "unsupported"
    row["assertions"][0]["source_quotes"] = []
    write_rows(packet, [row])
    assert panel.validate("sol")["rows"] == 1


def test_supported_assertion_needs_evidence(packet):
    packet[1]["assertions"][0]["source_quotes"] = []
    write_rows(packet, [packet[1]])
    with pytest.raises(ValueError, match="lacks source evidence"):
        panel.validate("sol")


def test_uncertainty_is_preserved(packet):
    row = packet[1]
    row["verdict"] = row["assertions"][0]["verdict"] = "uncertain"
    write_rows(packet, [row])
    assert panel.validate("sol")["verdicts"] == {"uncertain": 1}


def test_case_verdict_matches_assertions(packet):
    packet[1]["verdict"] = "unsupported"
    write_rows(packet, [packet[1]])
    with pytest.raises(ValueError, match="inconsistency"):
        panel.validate("sol")


def test_changed_packet_refused(packet):
    path = packet[0] / "blind" / "C001.json"
    path.write_text(path.read_text() + " ")
    write_rows(packet, [packet[1]])
    with pytest.raises(ValueError, match="packet changed"):
        panel.validate("sol")


@pytest.mark.parametrize("problem", ["extra", "missing", "count", "internal_id"])
def test_frozen_packet_inventory_required(packet, problem):
    work, row = packet
    manifest = json.loads((work / "manifest.json").read_text())
    case = json.loads((work / "blind" / "C001.json").read_text())
    rows = [row]
    if problem == "extra":
        case["case_id"] = "C002"
        panel.save(work / "blind" / "C002.json", case)
        rows.append({**row, "case_id": "C002"})
    elif problem == "missing":
        (work / "blind" / "C001.json").unlink()
    elif problem == "count":
        manifest["n"] = 2
    else:
        case["case_id"] = "C999"
        panel.save(work / "blind" / "C001.json", case)
        manifest["packet_hashes"]["C001.json"] = panel.digest(work / "blind" / "C001.json")
    panel.save(work / "manifest.json", manifest)
    write_rows(packet, rows)
    with pytest.raises(ValueError, match="inventory|case ID differs"):
        panel.validate("sol")


def test_prepare_keeps_labels_out_of_packet_and_preserves_sources(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    work = tmp_path / "panel"
    monkeypatch.setattr(panel, "INPUT", source)
    monkeypatch.setattr(panel, "WORK", work)
    items = [{"id": f"i{i}", "text": f"Source {i}", "claim": f"Claim {i}"} for i in range(70)]
    panel.save(source / "items.json", items)
    panel.save(source / "labels.json", {i["id"]: 1 for i in items})
    (source / "rows_v18.jsonl").write_text("\n".join(json.dumps({"id": f"i{i}", "pass": int(i >= 42)}) for i in range(70)))
    before = {p.name: panel.digest(p) for p in source.iterdir()}
    assert panel.prepare()["n"] == 60
    for p in (work / "blind").glob("*.json"):
        assert set(json.loads(p.read_text())) == {"case_id", "source_text", "candidate"}
    assert before == {p.name: panel.digest(p) for p in source.iterdir()}
    with pytest.raises(ValueError, match="refusing to overwrite"):
        panel.prepare()


@pytest.fixture
def completed_panel(packet, monkeypatch):
    work, row = packet
    for member in ("sol", "luna", "astra"):
        (work / f"{member}.jsonl").write_text(json.dumps(row) + "\n\n", encoding="utf8")
    panel.save(work / "parent_only_mapping.json", {"C001": {"original_case_id": "original",
               "reference_label": 1, "baseline_prediction": 0, "selection_stratum": "error"}})
    source = work / "original"
    source.mkdir()
    monkeypatch.setattr(panel, "INPUT", source)
    panel.save(source / "labels.json", {"original": 1})
    manifest = json.loads((work / "manifest.json").read_text())
    manifest.update(mapping_sha256=panel.digest(work / "parent_only_mapping.json"),
                    source_hashes={"labels.json": panel.digest(source / "labels.json")}, limitations=["synthetic test"])
    panel.save(work / "manifest.json", manifest)
    return work


def test_compare_preserves_uncertainty_and_refuses_overwrite(completed_panel):
    path = completed_panel / "astra.jsonl"
    row = json.loads(path.read_text())
    row["verdict"] = row["assertions"][0]["verdict"] = "uncertain"
    path.write_text(json.dumps(row))
    result = panel.compare()
    assert result["astra_vs_reference_by_selection_stratum"]["error"] == {"uncertain": 1}
    assert result["provenance"] == "AI_ONLY_DIAGNOSTIC_NOT_HUMAN_GOLD"
    assert "kappa" not in result
    with pytest.raises(ValueError, match="refusing to overwrite"):
        panel.compare()


def test_compare_requires_all_reviewers(completed_panel):
    (completed_panel / "astra.jsonl").write_text("")
    with pytest.raises(ValueError, match="case IDs"):
        panel.compare()
    assert not (completed_panel / "comparison.json").exists()


def test_compare_refuses_changed_reference(completed_panel):
    (completed_panel / "original" / "labels.json").write_text('{"original": 0}')
    with pytest.raises(ValueError, match="original evaluated artifact changed"):
        panel.compare()


def test_compare_requires_mapping_population_even_with_matching_digest(completed_panel):
    panel.save(completed_panel / "parent_only_mapping.json", {})
    manifest = json.loads((completed_panel / "manifest.json").read_text())
    manifest["mapping_sha256"] = panel.digest(completed_panel / "parent_only_mapping.json")
    panel.save(completed_panel / "manifest.json", manifest)
    with pytest.raises(ValueError, match="mapping differs"):
        panel.compare()
    assert not (completed_panel / "comparison.json").exists()


def test_compare_uses_exact_validated_reviewer_snapshot(completed_panel, monkeypatch):
    original = panel._validated_output
    frozen_hash = panel.digest(completed_panel / "astra.jsonl")

    def change_after_validation(name):
        snapshot = original(name)
        if name == "astra":
            altered = {**snapshot[1][0], "verdict": "unsupported"}
            (completed_panel / "astra.jsonl").write_text(json.dumps(altered), encoding="utf8")
        return snapshot

    monkeypatch.setattr(panel, "_validated_output", change_after_validation)
    result = panel.compare()
    assert result["validation"]["astra"]["sha256"] == frozen_hash
    assert result["astra_vs_reference_by_selection_stratum"]["error"] == {"agree": 1}
    assert panel.digest(completed_panel / "astra.jsonl") != frozen_hash
