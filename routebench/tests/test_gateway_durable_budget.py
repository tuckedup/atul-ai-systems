"""Restart-safe tenant budgets (durable ledger). Local files only; no provider call."""
from __future__ import annotations

import json
import threading

import pytest
from gateway.durable_budget import BudgetLedgerError, DurableBudgetLedger


def open_ledger(tmp_path, **kw):
    return DurableBudgetLedger(tmp_path / "budget.jsonl", **kw)


def lines(tmp_path):
    return [json.loads(x) for x in (tmp_path / "budget.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]


def test_spend_survives_a_restart_and_the_cap_still_binds(tmp_path):
    led = open_ledger(tmp_path)
    b = led.budget("acme", 1.0)
    r = b.reserve(0.4)
    assert r is not None and r.settle(0.3) is True
    led.close()
    led2 = open_ledger(tmp_path)
    b2 = led2.budget("acme", 1.0)
    assert b2.spent_usd == pytest.approx(0.3) and b2.remaining() == pytest.approx(0.7)
    assert b2.reserve(0.8) is None  # a restart does not refund the tenant
    r2 = b2.reserve(0.7)
    assert r2 is not None and r2.settle(0.7) is True and b2.remaining() == pytest.approx(0.0)
    led2.close()


def test_a_crash_with_a_reservation_in_flight_is_charged_at_its_bound_durably(tmp_path):
    led = open_ledger(tmp_path)
    b = led.budget("acme", 1.0)
    assert b.reserve(0.25) is not None  # never settled: the process "dies" here
    led.lock_path.unlink()  # simulate the dead process's leftover lock being cleared by the operator
    led2 = open_ledger(tmp_path)
    assert len(led2.recovered) == 1 and led2.spent("acme") == pytest.approx(0.25)
    assert led2.budget("acme", 1.0).spent_usd == pytest.approx(0.25)
    recovered = [e for e in lines(tmp_path) if e.get("recovered")]
    assert len(recovered) == 1 and recovered[0]["charged"] == pytest.approx(0.25)
    led2.close()
    led3 = open_ledger(tmp_path)  # recovery is idempotent: nothing is charged twice
    assert led3.spent("acme") == pytest.approx(0.25) and led3.recovered == []
    led3.close()


def test_overshoot_is_recorded_and_the_breach_count_survives(tmp_path):
    led = open_ledger(tmp_path)
    b = led.budget("acme", 1.0)
    r = b.reserve(0.1)
    assert r.settle(0.4) is False  # actual cost above the reservation: kept, flagged
    assert b.reservation_breaches == 1 and b.spent_usd == pytest.approx(0.4)
    led.close()
    b2 = open_ledger(tmp_path).budget("acme", 1.0)
    assert b2.reservation_breaches == 1 and b2.spent_usd == pytest.approx(0.4)


def test_tenants_are_independent_and_lowering_a_limit_below_spend_leaves_none_remaining(tmp_path):
    led = open_ledger(tmp_path)
    a, c = led.budget("a", 1.0), led.budget("c", 1.0)
    a.reserve(0.6).settle(0.6)
    assert c.spent_usd == 0
    led.close()
    led2 = open_ledger(tmp_path)
    low = led2.budget("a", 0.5)
    assert low.remaining() == 0.0 and low.cap_exceeded and low.reserve(0.01) is None
    assert led2.budget("a", 5.0).remaining() == pytest.approx(4.4)
    led2.close()


def test_one_shot_charges_are_durable(tmp_path):
    led = open_ledger(tmp_path)
    b = led.budget("a", 1.0)
    assert b.charge(0.2) is True and b.charge(0.9) is False
    led.close()
    assert open_ledger(tmp_path).budget("a", 1.0).spent_usd == pytest.approx(0.2)


def test_a_failed_ledger_write_admits_nothing_and_changes_no_memory(tmp_path, monkeypatch):
    led = open_ledger(tmp_path)
    b = led.budget("a", 1.0)
    monkeypatch.setattr(led, "_append", lambda entry: (_ for _ in ()).throw(BudgetLedgerError("disk full")))
    with pytest.raises(BudgetLedgerError):
        b.reserve(0.1)
    assert b.reserved_usd == 0 and b.spent_usd == 0
    monkeypatch.undo()
    r = b.reserve(0.1)
    monkeypatch.setattr(led, "_append", lambda entry: (_ for _ in ()).throw(BudgetLedgerError("disk full")))
    with pytest.raises(BudgetLedgerError):
        r.settle(0.05)
    assert not r.settled and b.reserved_usd == pytest.approx(0.1) and b.spent_usd == 0  # still open, still counted at its reservation
    monkeypatch.undo()
    assert r.settle(0.05) is True and b.spent_usd == pytest.approx(0.05)
    with pytest.raises(RuntimeError):
        r.settle(0.05)  # a second settlement is refused and writes nothing
    assert sum(1 for e in lines(tmp_path) if e["op"] == "settle") == 1


def test_a_corrupt_ledger_stops_startup_and_is_left_untouched(tmp_path):
    p = tmp_path / "budget.jsonl"
    good = {"v": 1, "op": "reserve", "id": "x", "tenant": "a", "usd": 0.1, "ts": 1}
    for bad in ("not json", json.dumps({**good, "usd": -1}), json.dumps({**good, "usd": float("nan")}), json.dumps({"v": 2, "op": "charge", "tenant": "a", "usd": 1}),
                json.dumps({"v": 1, "op": "settle", "id": "ghost", "tenant": "a", "charged": 0.1}), json.dumps({"v": 1, "op": "wat", "tenant": "a"}),
                json.dumps(good) + "\n" + json.dumps(good)):
        p.write_text(bad + "\n", encoding="utf8")
        before = p.read_bytes()
        with pytest.raises(BudgetLedgerError):
            open_ledger(tmp_path)
        assert p.read_bytes() == before and not (tmp_path / "budget.jsonl.lock").exists()  # nothing repaired, lock released


def test_only_one_owner_may_hold_a_ledger_and_takeover_is_explicit(tmp_path):
    led = open_ledger(tmp_path)
    with pytest.raises(BudgetLedgerError, match="single worker"):
        open_ledger(tmp_path)
    led.close()
    open_ledger(tmp_path).close()  # a clean close releases the lock
    led.lock_path.write_text(json.dumps({"pid": 99999999, "since": 0}), encoding="utf8")  # leftover from a dead process
    with pytest.raises(BudgetLedgerError, match="takeover=True"):
        open_ledger(tmp_path)
    open_ledger(tmp_path, takeover=True).close()


def test_takeover_is_refused_while_the_recorded_process_is_alive(tmp_path):
    import os

    led = open_ledger(tmp_path)
    led.lock_path.write_text(json.dumps({"pid": os.getppid() or 1, "since": 0}), encoding="utf8")
    try:
        import psutil  # noqa: F401
    except ImportError:
        pytest.skip("psutil unavailable")
    with pytest.raises(BudgetLedgerError, match="still running"):
        open_ledger(tmp_path, takeover=True)


def test_concurrent_reservations_never_overspend_and_every_one_is_journaled(tmp_path):
    led = open_ledger(tmp_path)
    b = led.budget("a", 0.10)
    granted = []

    def worker():
        r = b.reserve(0.02)
        if r is not None:
            granted.append(r)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(granted) == 5
    for r in granted:
        assert r.settle(0.02) in (True, False)
    assert b.spent_usd == pytest.approx(0.10)
    led.close()
    ent = lines(tmp_path)
    assert sum(1 for e in ent if e["op"] == "reserve") == 5 and sum(1 for e in ent if e["op"] == "settle") == 5
    assert open_ledger(tmp_path).budget("a", 0.10).spent_usd == pytest.approx(0.10)


def test_the_durable_budget_works_inside_the_gateway_request_path(tmp_path):
    from fastapi.testclient import TestClient
    from gateway.app import app, build_runtime, configure_runtime
    from gateway.router import Backend
    from test_gateway_spend_safety import FakeBackend

    led = open_ledger(tmp_path)
    backend = FakeBackend(cost_usd=0.0002)
    clients = {"model-a": backend}
    value = build_runtime([Backend("model-a", "fake", 1.0, 1.0, 1)], clients, {"model-a": {"chat": 1.0}}, budgets={"acme": led.budget("acme", 1.0)}, default_budget_usd=0.0)
    configure_runtime(value)
    body = {"messages": [{"role": "user", "content": "hello"}]}
    r = TestClient(app).post("/v1/chat/completions", json=body, headers={"x-tenant-id": "acme"})
    assert r.status_code == 200 and backend.calls == 1
    spent = value.budgets["acme"].spent_usd
    assert spent == pytest.approx(0.0002)
    led.close()
    assert open_ledger(tmp_path).budget("acme", 1.0).spent_usd == pytest.approx(spent)  # the same books after a restart
