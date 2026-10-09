"""Regression tests for the repaired prospective experiment: dev-primary reporting, frozen protocol manifest, fail-fast spend. Mocked transport only."""
import json
from pathlib import Path

import pytest
from evalops import claim_relation as cr

FIXTURES = cr.synthetic_fixtures()
DEV_FAMILIES = {f"fam-{i}" for i in range(1, 7)}
BY_PROMPT = {cr.sha256(cr.build_prompt(f["source"], f["candidate"])): f for f in FIXTURES}


def make_inputs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    cases = [{"case_id": f["case_id"], "source": f["source"], "candidate": f["candidate"]} for f in FIXTURES]
    files = {"cases.json": cases, "labels.json": {f["case_id"]: f["label"] for f in FIXTURES},
             "groups.json": {f["case_id"]: f["document_family"] for f in FIXTURES}, "baseline_v18.json": {f["case_id"]: 1 for f in FIXTURES},
             "splits.json": {f["case_id"]: "dev" if f["document_family"] in DEV_FAMILIES else "train" for f in FIXTURES}}
    for name, value in files.items():
        (root / name).write_text(json.dumps(value, sort_keys=True), encoding="utf8")
    return root


def frozen(tmp_path, cap=6.0):
    inputs = make_inputs(tmp_path / "in")
    proto = tmp_path / "protocol.json"
    cr.freeze_protocol(inputs, proto, cap_usd=cap)
    return inputs, proto


def scripted(prompt, usage=None, model="o4-mini-2025-04-16"):
    f = BY_PROMPT[cr.sha256(prompt)]
    return cr.Response(cr.scripted_response(f), usage=usage or {"prompt_tokens": 500, "completion_tokens": 300}, model=model)


# ---------------------------------------------------------------------------------------------- dev is primary
def _rec(cid, verdict):
    return {"case_id": cid, "status": "ok", "structured": {"verdict": verdict}}


def test_pooled_improvement_cannot_hide_a_dev_regression():
    dev = [f"d{i}" for i in range(12)]
    train = [f"t{i}" for i in range(12)]
    ids = dev + train
    labels = {i: (1 if k % 2 == 0 else 0) for k, i in enumerate(ids)}
    baseline = {i: labels[i] for i in dev} | {i: 1 - labels[i] for i in train}  # baseline right on dev, wrong on train
    verdict = {}
    for k, i in enumerate(dev):
        verdict[i] = "supported" if (labels[i] == 1) != (k < 4) else "unsupported"  # candidate flips 4 correct dev decisions
    for i in train:
        verdict[i] = "supported" if labels[i] == 1 else "unsupported"  # candidate fixes every train decision
    groups = {i: f"g{k % 4}" + ("d" if i in dev else "t") for k, i in enumerate(ids)}
    splits = {i: "dev" if i in dev else "train" for i in ids}
    out = cr.compare_to_baseline({"status": "COMPLETE"}, [_rec(i, verdict[i]) for i in ids], labels, baseline, groups, splits, uncertain_as=0, draws=200)
    pooled, d = out["pooled_supplementary_not_for_gating"], out["dev_primary"]
    assert pooled["kappa_candidate"] > pooled["kappa_baseline"]  # pooled looks better ...
    assert d["kappa_candidate"] < d["kappa_baseline"] and d["new_errors"] > d["errors_fixed"]  # ... while dev got worse
    assert out["gate"]["IMPROVED_ON_DEV"] is False and out["gate"]["DEV_KAPPA_REACHES_0.74"] is False
    assert "never drive the gate" in out["gate"]["note"] and out["train_diagnostic_only"]["errors_fixed"] == 12


def test_the_gate_requires_dev_errors_fixed_and_a_positive_lower_bound():
    dev = [f"d{i}" for i in range(24)]
    labels = {i: 0 if k % 2 else 1 for k, i in enumerate(dev)}
    baseline = {i: 1 for i in dev}  # passes everything: 12 false passes
    groups = {i: f"g{k // 2}" for k, i in enumerate(dev)}
    splits = {i: "dev" for i in dev}
    good = [_rec(i, "supported" if labels[i] == 1 else "unsupported") for i in dev]
    out = cr.compare_to_baseline({"status": "COMPLETE"}, good, labels, baseline, groups, splits, uncertain_as=0, draws=500)
    assert out["gate"]["IMPROVED_ON_DEV"] is True and out["dev_primary"]["errors_fixed"] == 12 and out["dev_primary"]["new_errors"] == 0
    assert out["train_diagnostic_only"] is None
    nothing = [_rec(i, "supported") for i in dev]
    assert cr.compare_to_baseline({"status": "COMPLETE"}, nothing, labels, baseline, groups, splits, uncertain_as=0, draws=200)["gate"]["IMPROVED_ON_DEV"] is False


def test_incomplete_or_mismatched_populations_withhold_every_number():
    ids = ["a", "b"]
    labels, base = {"a": 1, "b": 0}, {"a": 1, "b": 1}
    groups, splits = {"a": "g1", "b": "g2"}, {"a": "dev", "b": "dev"}
    recs = [_rec(i, "supported") for i in ids]
    assert cr.compare_to_baseline({"status": "ABORTED"}, recs, labels, base, groups, splits, uncertain_as=0)["status"] == "WITHHELD_INCOMPLETE"
    assert cr.compare_to_baseline({"status": "COMPLETE"}, recs[:1], labels, base, groups, splits, uncertain_as=0)["status"] == "WITHHELD_INCOMPLETE"
    assert cr.compare_to_baseline({"status": "COMPLETE"}, [*recs, recs[0]], labels, base, groups, splits, uncertain_as=0)["status"] == "WITHHELD_INCOMPLETE"
    assert cr.compare_to_baseline({"status": "COMPLETE"}, recs, labels, base, groups, {"a": "dev", "b": "test"}, uncertain_as=0)["status"] == "WITHHELD_INCOMPLETE"


# ---------------------------------------------------------------------------------------------- frozen manifest
def test_the_manifest_binds_inputs_code_identity_model_rule_bootstrap_prices_and_cap(tmp_path):
    inputs, proto = frozen(tmp_path)
    m = json.loads(proto.read_text())
    assert set(m["input_sha256"]) == set(cr.INPUT_FILES) and m["population"]["per_split"] == {"train": 7, "dev": 12} and m["population"]["cases"] == 19
    assert m["candidate"]["content_hash"] == cr.content_hash() and m["v18_identity"]["unchanged"] and m["v18_identity"]["v18_config_hash"] == cr.V18_CONFIG_HASH
    assert (m["model"], m["reasoning_effort"], m["cap_usd"]) == ("o4-mini", "medium", 6.0) and m["decision_rule"]["uncertain_as"] == 0
    assert m["bootstrap"] == {"seed": 20261101, "draws": 2000} and m["price_usd_per_1m"] == {"input": 1.1, "output": 4.4}
    assert cr.verify_protocol(inputs, proto)["protocol"] == "claim-relation-v1-ablation"
    with pytest.raises(cr.ClaimRelationError, match="never overwritten"):
        cr.freeze_protocol(inputs, proto)


@pytest.mark.parametrize("name,mutate", [
    ("cases.json", lambda d: d.__setitem__(0, {**d[0], "candidate": d[0]["candidate"] + " (edited)"})),
    ("labels.json", lambda d: d.__setitem__("syn-01", 1 - d["syn-01"])),
    ("groups.json", lambda d: d.__setitem__("syn-01", "other-doc")),
    ("baseline_v18.json", lambda d: d.__setitem__("syn-02", 1 - d["syn-02"])),
    ("splits.json", lambda d: d.__setitem__("syn-01", "train" if d["syn-01"] == "dev" else "dev")),
])
def test_altered_files_after_the_freeze_are_refused_before_any_request(tmp_path, name, mutate):
    inputs, proto = frozen(tmp_path)
    data = json.loads((inputs / name).read_text())
    mutate(data)
    (inputs / name).write_text(json.dumps(data, sort_keys=True), encoding="utf8")
    posts = []
    with pytest.raises(cr.ProtocolError):
        cr.run_frozen(inputs, proto, tmp_path / "run", lambda p: posts.append(p) or scripted(p))
    assert posts == []


@pytest.mark.parametrize("name,mutate,needle", [
    ("labels.json", lambda d: d.__setitem__("syn-extra", 1), "missing or extra"),
    ("baseline_v18.json", lambda d: d.pop("syn-03"), "missing or extra"),
    ("cases.json", lambda d: d.append(d[0]), "duplicate"),
    ("cases.json", lambda d: d.__setitem__(0, {**d[0], "label": 1}), "no labels"),
    ("splits.json", lambda d: d.__setitem__("syn-01", "test"), "only train and dev"),
])
def test_missing_duplicated_extra_or_leaky_records_are_refused(tmp_path, name, mutate, needle):
    inputs, proto = frozen(tmp_path)
    data = json.loads((inputs / name).read_text())
    mutate(data)
    (inputs / name).write_text(json.dumps(data, sort_keys=True), encoding="utf8")
    with pytest.raises(cr.ProtocolError, match=needle):
        cr.verify_protocol(inputs, proto)


def _edit(original, path, value):
    m = json.loads(json.dumps(original))
    node = m
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return m


def test_changed_code_identity_model_or_rule_in_the_manifest_is_detected(tmp_path):
    inputs, proto = frozen(tmp_path)
    anchor = cr._file_sha(proto)
    original = json.loads(proto.read_text())
    p2 = tmp_path / "p2.json"
    for path, value in (("candidate.code_sha256", "0" * 64), ("candidate.content_hash", "0" * 64), ("v18_identity.v18_config_hash", "0" * 16),
                        ("decision_rule.uncertain_as", 1), ("bootstrap.seed", 1), ("price_usd_per_1m.output", 0.1), ("max_tokens", 1), ("model", "o3"),
                        ("reasoning_effort", "high"), ("cap_usd", 60.0), ("cap_usd", 5.0), ("candidate.variant", "claim-relation-v2")):
        p2.write_text(json.dumps(_edit(original, path, value)), encoding="utf8")
        with pytest.raises(cr.ProtocolError):  # anchored by the freeze-time hash: any edit to the manifest itself is refused, even a plausible one
            cr.verify_protocol(inputs, p2, anchor)
    for path, value in (("cap_usd", 60.0), ("decision_rule.uncertain_as", 1), ("price_usd_per_1m.output", 0.1), ("candidate.code_sha256", "0" * 64)):
        p2.write_text(json.dumps(_edit(original, path, value)), encoding="utf8")
        with pytest.raises(cr.ProtocolError):  # also refused without the anchor: checked against the code's own constants
            cr.verify_protocol(inputs, p2)
    with pytest.raises(cr.ClaimRelationError, match="cap must be"):
        cr.freeze_protocol(inputs, tmp_path / "p3.json", cap_usd=60.0)


# ---------------------------------------------------------------------------------------------- fail-fast spending
class Transport:
    """Mocked HTTP transport for aisys.llm: counts real requests; the n-th reply is chosen by `reply(n, prompt)`."""

    def __init__(self, monkeypatch, reply):
        from aisys import llm

        self.posts = []
        outer = self

        class Resp:
            def __init__(self, status, body):
                self.status_code, self._body, self.headers = status, body, {}

            def json(self):
                return self._body

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

        class Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *e):
                return False

            def post(self, url, headers=None, json=None):
                outer.posts.append(json)
                status, body = reply(len(outer.posts), json["messages"][0]["content"])
                return Resp(status, body)

        monkeypatch.setattr(llm.httpx, "Client", Client)
        monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("llm.chat retries internally")))


def good_body(prompt, usage=None, model="o4-mini-2025-04-16"):
    body = {"choices": [{"message": {"content": cr.scripted_response(BY_PROMPT[cr.sha256(prompt)])}}], "model": model}
    body["usage"] = usage if usage is not None else {"prompt_tokens": 500, "completion_tokens": 300}
    return 200, body


def run_provider(tmp_path, monkeypatch, reply, cap=6.0):
    inputs, proto = frozen(tmp_path, cap=cap)
    transport = Transport(monkeypatch, reply)
    frozen_m = cr.verify_protocol(inputs, proto)
    request = cr.provider_request_factory(frozen_m["model"], frozen_m["reasoning_effort"], frozen_m["cap_usd"], tmp_path / "run" / "spend_journal.jsonl", (1.10, 4.40))
    terminal = cr.run_frozen(inputs, proto, tmp_path / "run", request)
    return inputs, proto, transport, terminal


def test_a_complete_mocked_provider_run_makes_exactly_one_request_per_case_and_compares(tmp_path, monkeypatch):
    inputs, proto, transport, terminal = run_provider(tmp_path, monkeypatch, lambda n, p: good_body(p))
    assert terminal["status"] == "COMPLETE" and len(transport.posts) == 19 and terminal["not_attempted"] == []
    out = cr.compare_run(inputs, tmp_path / "run", proto)
    assert out["status"] == "COMPLETE" and set(out) >= {"dev_primary", "train_diagnostic_only", "pooled_supplementary_not_for_gating", "gate"}
    assert out["dev_primary"]["n"] == 12 and out["train_diagnostic_only"]["n"] == 7 and out["pooled_supplementary_not_for_gating"]["n"] == 19
    with pytest.raises(cr.ClaimRelationError, match="written once"):
        cr.compare_run(inputs, tmp_path / "run", proto)


def test_one_malformed_early_response_causes_zero_subsequent_http_calls(tmp_path, monkeypatch):
    def reply(n, prompt):
        return (200, {"choices": [{"message": {"content": "this is not json"}}], "model": "o4-mini-x", "usage": {"prompt_tokens": 500, "completion_tokens": 300}}) if n == 2 \
            else good_body(prompt)

    inputs, proto, transport, terminal = run_provider(tmp_path, monkeypatch, reply)
    assert len(transport.posts) == 2 and terminal["status"] == "INCOMPLETE" and terminal["attempted"] == 2 and len(terminal["not_attempted"]) == 17
    assert cr.compare_run(inputs, tmp_path / "run", proto)["status"] == "WITHHELD_INCOMPLETE"


@pytest.mark.parametrize("label,reply_for_first,expected_posts", [
    ("quota", lambda p: (429, {"error": {"code": "insufficient_quota"}}), 1),
    ("model mismatch", lambda p: good_body(p, model="gpt-4.1-mini"), 1),
    ("missing usage", lambda p: (200, {"choices": [{"message": {"content": "{}"}}], "model": "o4-mini-x"}), 1),
    ("invalid usage", lambda p: good_body(p, usage={"prompt_tokens": -5, "completion_tokens": "x"}), 1),
    ("cost above reservation", lambda p: good_body(p, usage={"prompt_tokens": 10_000_000, "completion_tokens": 500}), 1),
    ("server error", lambda p: (503, {}), 1),
])
def test_spend_and_provider_anomalies_stop_with_zero_subsequent_calls(tmp_path, monkeypatch, label, reply_for_first, expected_posts):
    inputs, proto, transport, terminal = run_provider(tmp_path, monkeypatch, lambda n, p: reply_for_first(p) if n == 1 else good_body(p))
    assert len(transport.posts) == expected_posts, label
    assert terminal["status"] in ("ABORTED", "INCOMPLETE") and terminal["attempted"] == 1 and len(terminal["not_attempted"]) == 18
    assert cr.compare_run(inputs, tmp_path / "run", proto)["status"] == "WITHHELD_INCOMPLETE"


def test_cap_exhaustion_stops_before_the_first_request(tmp_path, monkeypatch):
    _inputs, _proto, transport, terminal = run_provider(tmp_path, monkeypatch, lambda n, p: good_body(p), cap=0.0001)
    assert transport.posts == [] and terminal["status"] == "ABORTED" and "cap" in terminal["stopped_on"]["error"]


def test_cap_exhaustion_midway_stops_immediately(tmp_path, monkeypatch):
    # three requests fit under the cap at their settled cost; the fourth reservation does not
    one = (cr.MAX_TOKENS * 4.4 + 2200 * 1.1) / 1e6
    _inputs, _proto, transport, terminal = run_provider(tmp_path, monkeypatch, lambda n, p: good_body(p), cap=one * 1.4)
    assert terminal["status"] == "ABORTED" and len(transport.posts) == terminal["attempted"] - 1 and len(transport.posts) >= 1
    assert terminal["ok"] == len(transport.posts) and terminal["stopped_on"]["status"] == "aborted"


# ---------------------------------------------------------------------------------------------- comparison guards
def _complete(tmp_path, monkeypatch):
    return run_provider(tmp_path, monkeypatch, lambda n, p: good_body(p))


def test_comparison_refuses_a_changed_decision_rule(tmp_path, monkeypatch):
    inputs, proto, _, _ = _complete(tmp_path, monkeypatch)
    with pytest.raises(cr.ProtocolError, match="decision rule"):
        cr.compare_run(inputs, tmp_path / "run", proto, uncertain_as=1)
    assert cr.compare_run(inputs, tmp_path / "run", proto, uncertain_as=0)["uncertain_mapping"] == "uncertain->0"


def test_comparison_refuses_a_changed_population_or_label_file(tmp_path, monkeypatch):
    inputs, proto, _, _ = _complete(tmp_path, monkeypatch)
    labels = json.loads((inputs / "labels.json").read_text())
    labels["syn-01"] = 1 - labels["syn-01"]
    (inputs / "labels.json").write_text(json.dumps(labels, sort_keys=True), encoding="utf8")
    with pytest.raises(cr.ProtocolError):
        cr.compare_run(inputs, tmp_path / "run", proto)


def test_comparison_refuses_extra_duplicated_or_altered_run_records(tmp_path, monkeypatch):
    inputs, proto, _, _ = _complete(tmp_path, monkeypatch)
    path = tmp_path / "run" / "judgments.jsonl"
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["case_id"] = "syn-not-in-protocol"
    path.write_text("\n".join([*lines, json.dumps(rec)]) + "\n", encoding="utf8")
    with pytest.raises(cr.ProtocolError, match="extra, altered or duplicated"):
        cr.compare_run(inputs, tmp_path / "run", proto)


def test_comparison_refuses_a_run_from_a_different_protocol(tmp_path, monkeypatch):
    inputs, proto, _, _ = _complete(tmp_path, monkeypatch)
    other = tmp_path / "other_protocol.json"
    other.write_text(proto.read_text() + " ", encoding="utf8")  # different bytes -> different protocol identity
    with pytest.raises(cr.ProtocolError, match="not produced under this frozen protocol"):
        cr.compare_run(inputs, tmp_path / "run", other)


def test_run_command_refuses_without_confirmation_and_verifies_the_freeze(tmp_path, capsys):
    inputs, proto = frozen(tmp_path)
    assert cr.main(["run", "--inputs", str(inputs), "--protocol", str(proto), "--out", str(tmp_path / "o"), "--protocol-sha256", cr._file_sha(proto)]) == 2
    assert "REFUSED" in capsys.readouterr().err and not (tmp_path / "o").exists()
    (inputs / "labels.json").write_text(json.dumps({}), encoding="utf8")
    assert cr.main(["run", "--inputs", str(inputs), "--protocol", str(proto), "--out", str(tmp_path / "o2"), "--confirm", "RUN-CLAIM-RELATION-PAID",
                    "--protocol-sha256", cr._file_sha(proto)]) == 2
    assert not (tmp_path / "o2" / "judgments.jsonl").exists()


def test_build_cases_writes_the_split_file_and_freeze_binds_the_real_population(tmp_path):
    info = cr.build_cases(["train", "dev"], tmp_path / "in")
    assert info["files"] == list(cr.INPUT_FILES)
    splits = json.loads((tmp_path / "in" / "splits.json").read_text())
    assert sum(v == "dev" for v in splits.values()) == 128 and sum(v == "train" for v in splits.values()) == 129
    body = cr.freeze_protocol(tmp_path / "in", tmp_path / "p.json")
    assert body["population"]["per_split"] == {"train": 129, "dev": 128} and body["v18_identity"]["unchanged"]


def test_run_command_requires_the_freeze_time_protocol_hash(tmp_path, capsys):
    inputs, proto = frozen(tmp_path)
    assert cr.main(["run", "--inputs", str(inputs), "--protocol", str(proto), "--out", str(tmp_path / "o"), "--confirm", "RUN-CLAIM-RELATION-PAID"]) == 2
    assert "--protocol-sha256" in capsys.readouterr().err
    assert cr.main(["run", "--inputs", str(inputs), "--protocol", str(proto), "--out", str(tmp_path / "o"), "--confirm", "RUN-CLAIM-RELATION-PAID",
                    "--protocol-sha256", "0" * 64]) == 2
    assert not (tmp_path / "o" / "judgments.jsonl").exists()
