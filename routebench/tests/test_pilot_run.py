"""The pilot execution path with a fake backend and a fake clock. No model, no GPU, no network."""
import dataclasses
import json

import numpy as np
import pytest
from evalops.pilot import harness as h
from evalops.pilot import run as r
from evalops.pilot.config import PILOT


def make_manifest(train_groups=60, val_groups=30, per_group=8, every=6):
    rows, resp_labels, k = [], {}, 0
    for part, n in (("train", train_groups), ("validation", val_groups), ("test", 4)):
        for g in range(n):
            for j in range(per_group):
                label = 0 if k % every == 0 else 1
                rid = f"{part}{g}-r{j // 4}"
                resp_labels[rid] = min(resp_labels.get(rid, 1), label)
                rows.append([f"{part}{g}#{j}", rid, "s", part, "fp", f"g-{part}{g}", label, 0, 5, False, False])
                k += 1
    m = {"version": "ragtruth-training-manifest-v5",
         "columns": ["example_id", "response_id", "source_id", "partition", "document_fp16", "group_id", "label", "start", "end",
                     "implicit_true", "crosses_boundary"], "rows": rows, "response_labels": resp_labels, "counts": {}, "meta": {}}
    m["manifest_sha256"] = h.content_hash(m)
    return m


def make_cfg(m):
    return dataclasses.replace(PILOT, manifest_version=m["version"], manifest_sha256=m["manifest_sha256"],
                               train_unsupported_max=15, train_supported_per_unsupported=2, select_eval_sentences=30,
                               validation_eval_sentences=60, bootstrap_draws=200,
                               phase_seconds={"feasibility": 60, "baseline_eval": 200, "training": 400, "tuned_eval": 200, "slack": 140},
                               total_seconds=1000)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class FakeBackend:
    def __init__(self, labels, clock, *, base_sep=0.15, tuned_sep=0.7, score_s=0.05, train_s=0.2, drop=0, train_overrun=False,
                 probe_train_s=None, updates_done=2):
        self.updates_done = updates_done
        self.labels, self.clock, self.tuned = labels, clock, False
        self.base_sep, self.tuned_sep, self.score_s, self.train_s = base_sep, tuned_sep, score_s, train_s
        self.drop, self.train_overrun, self.probe_train_s = drop, train_overrun, probe_train_s
        self.trained_on: list[str] = []
        self.rng = np.random.default_rng(0)

    def probe(self, train_examples, score_examples):
        self.probe_sizes = (len(train_examples), len(score_examples))
        return {"score_s_per_example": self.score_s, "train_s_per_example": self.probe_train_s or self.train_s,
                "optimizer_updates_done": self.updates_done, "peak_memory_gb": 5.0}

    def score(self, examples, tick):
        out = {}
        sep = self.tuned_sep if self.tuned else self.base_sep
        for eid, _doc, _sent in examples:
            self.clock.t += self.score_s
            tick()
            y = self.labels[eid]
            out[eid] = float(np.clip(0.5 + (sep / 2 if y else -sep / 2) + self.rng.normal(0, 0.12), 0, 1))
        for eid in list(out)[: self.drop]:
            del out[eid]
        return out

    def train(self, examples, tick):
        for eid, _doc, _sent, _label in examples:
            self.clock.t += self.train_s * (50 if self.train_overrun else 1)
            self.trained_on.append(eid)
            tick()
        self.tuned = True


def setup(**backend_kw):
    m = make_manifest()
    cfg = make_cfg(m)
    rows = h.rows_from_manifest(m)
    lens = {row.example_id: 800 for row in rows}
    freeze = r.build_freeze(m, lens, cfg)
    texts = {row.example_id: ("doc", "sentence") for row in rows}
    clock = Clock()
    backend = FakeBackend({row.example_id: row.label for row in rows}, clock, **backend_kw)
    return m, cfg, lens, freeze, texts, clock, backend


def go(m, cfg, lens, freeze, texts, clock, backend, **kw):
    kw.setdefault("require_permission", False)
    return r.run_pilot(cfg, m, freeze, lens, texts, backend, clock=clock, **kw)


def test_complete_run_produces_the_comparison_and_stays_inside_the_caps():
    m, cfg, lens, freeze, texts, clock, backend = setup()
    res = go(m, cfg, lens, freeze, texts, clock, backend)
    assert res["status"] == "COMPLETE" and res["feasibility"]["feasible"]
    assert res["tuned"]["sentence"]["balanced_accuracy"] > res["baseline"]["sentence"]["balanced_accuracy"]
    assert res["continue_decision"]["checks"]["complete_coverage"] is True
    assert res["total_seconds"] <= cfg.total_seconds and all(v <= cfg.phase_seconds[k] for k, v in res["seconds_by_phase"].items())
    assert set(backend.trained_on) == set(freeze["train_sample"]["example_ids"])
    assert res["scope"]["validation_groups"] > 0 and "not the RouteBench benchmark" in res["scope"]["note"]
    assert "not a rights clearance" in res["permission_note"]
    assert res["tuned"]["response"]["scope"] == "complete responses only"


def test_the_training_sample_contains_only_train_fit_examples():
    m, cfg, lens, freeze, texts, clock, backend = setup()
    go(m, cfg, lens, freeze, texts, clock, backend)
    rows = {row.example_id: row for row in h.rows_from_manifest(m)}
    plan = h.make_plan(list(rows.values()), cfg)
    assert set(backend.trained_on) <= plan.fit_ids()
    assert not {i for i in backend.trained_on if rows[i].partition != "train"}


def test_infeasible_projection_stops_before_training():
    m, cfg, lens, freeze, texts, clock, backend = setup(probe_train_s=10.0)  # 10 s/example x 45 examples >> 400 s
    res = go(m, cfg, lens, freeze, texts, clock, backend)
    assert res["status"] == "NOT_FEASIBLE" and res["continue_decision"] is None and "baseline" not in res
    assert backend.trained_on == [] and not backend.tuned


def test_training_overrun_is_incomplete_and_yields_no_comparison():
    m, cfg, lens, freeze, texts, clock, backend = setup(train_overrun=True)
    res = go(m, cfg, lens, freeze, texts, clock, backend)
    assert res["status"] == "INCOMPLETE" and res["incomplete_phase"] == "training"
    assert res["continue_decision"] is None and "tuned" not in res


def test_incomplete_scoring_is_reported_incomplete_never_evaluated():
    m, cfg, lens, freeze, texts, clock, backend = setup(drop=3)
    res = go(m, cfg, lens, freeze, texts, clock, backend)
    assert res["status"] == "INCOMPLETE" and res["incomplete_phase"] == "baseline_eval" and "baseline" not in res
    assert backend.trained_on == []


def test_a_below_chance_model_is_reported_as_a_result():
    m, cfg, lens, freeze, texts, clock, backend = setup(tuned_sep=-0.6)  # correctly wired, but worse than chance
    res = go(m, cfg, lens, freeze, texts, clock, backend)
    assert res["status"] == "COMPLETE" and res["tuned"]["direction_on_validation"]["below_chance"] is True
    assert res["continue_decision"]["continue"] is False


def test_tampering_is_refused():
    m, cfg, lens, freeze, texts, clock, backend = setup()
    with pytest.raises(h.ManifestMismatch):
        go({**m, "rows": m["rows"][:-1]}, cfg, lens, freeze, texts, clock, backend)
    with pytest.raises(h.ConfigMismatch):
        go(m, dataclasses.replace(cfg, learning_rate=1e-3), lens, freeze, texts, clock, backend)
    doctored = json.loads(json.dumps(freeze))
    doctored["train_sample"]["example_ids"].append(next(x.example_id for x in h.rows_from_manifest(m) if x.partition == "validation"))
    with pytest.raises(h.ConfigMismatch):
        go(m, cfg, lens, doctored, texts, clock, backend)
    with pytest.raises(h.IncompleteEvaluation):
        go(m, cfg, lens, freeze, {k: v for k, v in list(texts.items())[:-50]}, clock, backend)


def test_the_run_refuses_without_a_recorded_permission_decision(tmp_path):
    m, cfg, lens, freeze, texts, clock, backend = setup()
    empty = tmp_path / "perm.md"
    empty.write_text("CNN/DM rows for a private research pilot: ______ (accept / decline)\n", encoding="utf8")
    with pytest.raises(r.PermissionNotRecorded):
        go(m, cfg, lens, freeze, texts, clock, backend, require_permission=True, permission_record=empty)
    decline = tmp_path / "decline.md"
    decline.write_text("CNN/DM rows for a private research pilot: decline\n", encoding="utf8")
    with pytest.raises(r.PermissionNotRecorded):
        go(m, cfg, lens, freeze, texts, clock, backend, require_permission=True, permission_record=decline)
    with pytest.raises(r.PermissionNotRecorded):
        go(m, cfg, lens, freeze, texts, clock, backend, require_permission=True, permission_record=tmp_path / "missing.md")
    ok = tmp_path / "ok.md"
    ok.write_text("CNN/DM rows for a private research pilot: accept\n", encoding="utf8")
    assert go(m, cfg, lens, freeze, texts, clock, backend, require_permission=True, permission_record=ok)["status"] == "COMPLETE"


def test_results_are_never_overwritten(tmp_path):
    r.write_result({"status": "COMPLETE"}, tmp_path)
    with pytest.raises(h.PilotError):
        r.write_result({"status": "COMPLETE"}, tmp_path)


def test_the_shipped_permission_document_records_the_users_decision_and_its_limits():
    from pathlib import Path

    doc = Path(__file__).parents[2] / "docs" / "ROUTEBENCH_TRAINING_PERMISSION_STATUS.md"
    r.check_permission_record(doc)  # recorded on 2026-09-30 at the user's explicit instruction
    text = doc.read_text(encoding="utf8")
    assert "NOT a rights clearance" in text and "Recent News" in text and "accept, date 2026-09-30" in text


def test_the_probe_receives_enough_examples_for_real_optimizer_updates_and_must_prove_them():
    m, cfg, lens, freeze, texts, clock, backend = setup()
    go(m, cfg, lens, freeze, texts, clock, backend)
    need = cfg.micro_batch * cfg.grad_accum * cfg.probe_optimizer_updates
    assert backend.probe_sizes == (need, cfg.probe_score_examples) and need == 32
    m, cfg, lens, freeze, texts, clock, backend = setup(updates_done=0)
    res = go(m, cfg, lens, freeze, texts, clock, backend)
    assert res["status"] == "NOT_FEASIBLE" and "optimizer updates" in res["note"] and backend.trained_on == []
