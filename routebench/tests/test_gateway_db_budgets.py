# ruff: noqa: RUF059
"""Database-backed durable tenant budgets through the real gateway path: fake providers and temporary local SQLite only (no PostgreSQL, no provider call, no frozen module touched)."""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest
from aisys.llm import ChatResult, Usage
from fastapi.testclient import TestClient
from gateway import budget_admin
from gateway.app import app, configure_runtime
from gateway.bootstrap import GatewayStartupError, runtime_from_state
from gateway.budget_store import BudgetStateError, BudgetStore
from gateway.reliability import TenantBudget
from gateway.router import Backend
from gateway.state import ModelState
from test_rollout_controller import _calibration, _state

BODY = {"messages": [{"role": "user", "content": "calculate 2+2"}]}
LIMIT = 0.01
BOUND = None  # filled from the first measured reservation


class Crash(BaseException):
    """Simulates the process dying (not an Exception, so no handler may swallow it)."""


class Provider:
    name = "fake"

    def __init__(self, cost=0.0002, raw_usage=None, fail=None):
        self.cost, self.calls, self.raw_usage, self.fail = cost, 0, raw_usage, fail

    def chat(self, payload):
        self.calls += 1
        if self.fail:
            raise self.fail
        raw = {} if self.raw_usage is None else {"model": payload["model"], "usage": self.raw_usage}
        if self.raw_usage == "missing":
            raw = {"model": payload["model"], "choices": []}
        return ChatResult(text="4", usage=Usage(prompt_tokens=10, completion_tokens=5), model=payload["model"], provider=self.name, cost_usd=self.cost, latency_ms=1, raw=raw)


class World:
    """One database file; `boot()` opens a NEW connection and a NEW runtime each time = a restart or a second runtime."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.artifact, self.payload = _calibration(tmp_path)
        _state(tmp_path, str(self.payload["bundle_id"]))

    def state(self) -> ModelState:
        st = ModelState(f"sqlite:///{self.tmp_path / 'state.db'}")
        st.apply_weight_map({"candidate": 0, "incumbent": 1, "other": 0})
        return st

    def boot(self, provider=None, *, limits=None, **kw):
        st = self.state()
        backends = [Backend(m, "local-test", 1.0, 1.0, 1) for m in st.weight_map()]
        provider = provider or Provider()
        runtime = runtime_from_state(st, backends, {b.model: provider for b in backends}, calibration=self.artifact, expected_dataset_hash="dataset-1",
                                     tenant_limits=limits or {"default": LIMIT}, **kw)
        return runtime, provider, st

    def db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.tmp_path / "state.db")


def post(runtime, tenant="default", body=None):
    configure_runtime(runtime)
    return TestClient(app).post("/v1/chat/completions", json=body or BODY, headers={"x-tenant-id": tenant})


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# ------------------------------------------------------------------------------------------------ migration and startup
def test_first_start_creates_the_budget_tables_without_touching_existing_state(world):
    con = world.db()
    tables = lambda: {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "tenant_budgets" not in tables()
    before = con.execute("SELECT model, quality, weight FROM model_state ORDER BY model").fetchall()
    rt, _p, _st = world.boot()
    assert {"budget_meta", "tenant_budgets", "budget_reservations", "budget_events"} <= tables()
    assert con.execute("SELECT value FROM budget_meta WHERE key='schema_version'").fetchone()[0] == "1"
    assert con.execute("SELECT model, quality, weight FROM model_state ORDER BY model").fetchall() != [] and [r[0] for r in before] == ["candidate", "incumbent", "other"]
    assert rt.budget_mode == "durable_database" and rt.unresolved_reservations == 0
    events = con.execute("SELECT COUNT(*) FROM budget_events").fetchone()[0]
    world.boot()  # idempotent: a second start creates nothing new
    assert con.execute("SELECT COUNT(*) FROM tenant_budgets").fetchone()[0] == 1 and con.execute("SELECT COUNT(*) FROM budget_events").fetchone()[0] == events


def test_spend_survives_a_restart_and_the_cap_still_binds(world):
    rt1, p1, _ = world.boot()
    for i in (2, 3):  # distinct prompts: identical ones would be served from the response cache and cost nothing
        assert post(rt1, body={"messages": [{"role": "user", "content": f"calculate {i}+{i}"}]}).status_code == 200
    spent = rt1.budgets["default"].spent_usd
    assert spent == pytest.approx(0.0004)
    rt2, p2, _ = world.boot()
    assert rt2.budgets["default"].spent_usd == pytest.approx(spent) and rt2.budgets["default"].limit_usd == LIMIT
    held = rt2.budgets["default"].reserve(LIMIT - spent - 0.001)  # leave less headroom than one call bound (about $0.003)
    assert held is not None
    r = post(rt2)
    assert r.status_code == 402 and p2.calls == 0  # the restarted runtime does not get a fresh budget (and refuses before any provider call)


def test_the_configured_limit_never_silently_overwrites_the_persisted_one(world):
    world.boot()
    with pytest.raises(GatewayStartupError, match="differs from the configured limit"):
        world.boot(limits={"default": 999.0})
    BudgetStore(world.state()).set_limit("default", 999.0, operator="alice", reason="test raise")
    rt, _p, _st = world.boot(limits={"default": 999.0})
    assert rt.budgets["default"].limit_usd == 999.0
    with pytest.raises(BudgetStateError, match="named operator"):
        BudgetStore(world.state()).set_limit("default", 1.0, operator=" ", reason="x")


@pytest.mark.parametrize("sql", ["UPDATE tenant_budgets SET reserved_usd = 0.5", "UPDATE tenant_budgets SET spent_usd = 0.5", "UPDATE tenant_budgets SET spent_usd = 'nan'",
                                 "UPDATE tenant_budgets SET spent_usd = -1", "UPDATE budget_meta SET value = '99'", "UPDATE budget_reservations SET state = 'weird'",
                                 "DELETE FROM tenant_budgets"])
def test_startup_fails_closed_when_the_durable_books_do_not_verify(world, sql):
    rt, _p, _st = world.boot()
    assert post(rt).status_code == 200  # a settled reservation exists, so tamper cases touch real rows
    con = world.db()
    con.execute(sql)
    con.commit()
    con.close()
    with pytest.raises(GatewayStartupError, match="durable budget state blocks startup"):
        world.boot()


def test_ephemeral_budgets_are_an_explicit_disclosed_opt_out(world):
    rt, _p, _st = world.boot(ephemeral_budgets=True)
    assert isinstance(rt.budgets["default"], TenantBudget) and rt.budget_mode == "ephemeral_disclosed"
    configure_runtime(rt)
    body = TestClient(app).get("/readyz").json()
    assert body["budget_mode"] == "ephemeral_disclosed"
    durable, _p2, _st2 = world.boot()
    configure_runtime(durable)
    assert TestClient(app).get("/readyz").json()["budget_mode"] == "durable_database"


# ------------------------------------------------------------------------------------------------ concurrency
def test_two_runtimes_compete_for_one_cap_and_never_exceed_it(world):
    rt_a, _pa, _ = world.boot()
    rt_b, _pb, _ = world.boot()
    amount, granted, errors = 0.003, [], []

    def worker(budget):
        try:
            r = budget.reserve(amount)
            if r is not None:
                granted.append(r)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=((rt_a if i % 2 else rt_b).budgets["default"],)) for i in range(24)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(granted) == 3  # 3 x 0.003 fits under 0.01, a fourth does not, across both runtimes
    snap = rt_a.budget_store.snapshot("default")
    assert snap["reserved_usd"] == pytest.approx(0.009) and snap["reserved_usd"] <= LIMIT
    for r in granted:
        assert r.settle(0.003) is True
    assert rt_b.budget_store.verify({"default": LIMIT}) == [] and rt_b.budgets["default"].spent_usd == pytest.approx(0.009)


# ------------------------------------------------------------------------------------------------ crash boundaries (real gateway path)
def test_crash_after_reserve_before_the_response_holds_the_funds_and_is_never_retried(world):
    rt1, p1, _ = world.boot(Provider(fail=Crash()))
    configure_runtime(rt1)
    with pytest.raises(Crash):
        TestClient(app).post("/v1/chat/completions", json=BODY)
    assert p1.calls == 1
    rt2, p2, _ = world.boot()
    held = rt2.budget_store.unresolved()
    assert rt2.unresolved_reservations == len(held) == 1 and held[0]["amount_usd"] > 0 and not held[0]["held_by_this_runtime"]
    snap = rt2.budget_store.snapshot("default")
    assert snap["spent_usd"] == 0 and snap["reserved_usd"] == pytest.approx(held[0]["amount_usd"])  # held, not free and not spent
    assert p2.calls == 0 and p1.calls == 1  # nothing was retried or re-sent
    again, _p3, _ = world.boot()
    assert again.budget_store.snapshot("default")["reserved_usd"] == pytest.approx(held[0]["amount_usd"])  # a further restart does not release it either
    out = budget_admin.main(["resolve", "--reservation", held[0]["reservation_id"], "--operator", "alice", "--reason", "process died"], state_factory=world.state)
    assert out == 0
    final = again.budget_store.snapshot("default")
    assert final["reserved_usd"] == 0 and final["spent_usd"] == pytest.approx(held[0]["amount_usd"])  # default resolution charges the full bound
    assert again.budget_store.verify({"default": LIMIT}) == []


def test_crash_after_the_provider_answered_but_before_settlement_also_stays_held(world, monkeypatch):
    rt1, p1, _ = world.boot()

    def crash(*a, **k):
        raise Crash()

    monkeypatch.setattr(rt1.budget_store, "settle", crash)
    configure_runtime(rt1)
    with pytest.raises(Crash):
        TestClient(app).post("/v1/chat/completions", json=BODY)
    assert p1.calls == 1
    rt2, p2, _ = world.boot()
    assert rt2.unresolved_reservations == 1 and rt2.budget_store.snapshot("default")["spent_usd"] == 0 and p2.calls == 0
    assert post(rt2).status_code == 200 and p2.calls == 1  # the held amount reduced headroom but did not block an affordable request


def test_only_an_explicit_operator_decision_can_release_held_funds_and_only_once(world):
    rt, _p, _ = world.boot()
    store = rt.budget_store
    rid = store.reserve("default", 0.004)
    with pytest.raises(BudgetStateError, match="operator"):
        store.resolve(rid, operator="", reason="x")
    with pytest.raises(BudgetStateError, match="finite"):
        store.resolve(rid, operator="a", reason="r", charge_usd=float("nan"))
    res = store.resolve(rid, operator="alice", reason="confirmed no billing, partial refund", charge_usd=0.001)
    assert res["charged_usd"] == 0.001
    snap = store.snapshot("default")
    assert snap["reserved_usd"] == 0 and snap["spent_usd"] == pytest.approx(0.001)
    with pytest.raises(BudgetStateError, match="already resolved"):
        store.resolve(rid, operator="alice", reason="again")
    with pytest.raises(BudgetStateError, match="already resolved"):
        store.settle(rid, 0.004)
    assert store.snapshot("default") == snap


# ------------------------------------------------------------------------------------------------ settlement semantics
def test_duplicate_settlement_is_refused_and_changes_nothing(world):
    rt, _p, _ = world.boot()
    r = rt.budgets["default"].reserve(0.003)
    assert r.settle(0.001) is True
    before = rt.budget_store.snapshot("default")
    with pytest.raises(BudgetStateError, match="already settled"):
        r.settle(0.001)
    with pytest.raises(BudgetStateError, match="already settled"):
        rt.budget_store.settle(r.reservation_id, 0.002)
    assert rt.budget_store.snapshot("default") == before
    con = world.db()
    assert con.execute("SELECT COUNT(*) FROM budget_events WHERE kind='settled'").fetchone()[0] == 1


@pytest.mark.parametrize("raw_usage", ["missing", {"prompt_tokens": 0, "completion_tokens": 3}, {"prompt_tokens": -1, "completion_tokens": 3}, {"prompt_tokens": True, "completion_tokens": 3}])
def test_missing_or_malformed_usage_is_charged_at_the_bound_not_zero(world, raw_usage):
    rt, p, _ = world.boot(Provider(raw_usage=raw_usage))
    assert post(rt).status_code == 200 and p.calls == 1
    snap = rt.budget_store.snapshot("default")
    assert snap["spent_usd"] > 0.002 and snap["reserved_usd"] == 0  # the bound is about $0.003, never the fake provider's $0.0002
    row = world.db().execute("SELECT actual_usd, charged_usd FROM budget_reservations WHERE state='settled'").fetchone()
    assert row[0] is None and row[1] == pytest.approx(snap["spent_usd"])  # usage unknown is recorded as unknown, charged at the bound


def test_an_overshoot_is_recorded_in_full_counted_and_survives_restart(world):
    rt, p, _ = world.boot(Provider(cost=0.5), limits={"default": LIMIT})
    r = post(rt)
    assert r.status_code == 402 and p.calls == 1  # the response is refused (cap exceeded) but the cost is kept
    snap = rt.budget_store.snapshot("default")
    assert snap["spent_usd"] == pytest.approx(0.5) and snap["breaches"] == 1
    rt2, p2, _ = world.boot()
    assert rt2.budgets["default"].spent_usd == pytest.approx(0.5) and rt2.budgets["default"].reservation_breaches == 1 and rt2.budgets["default"].cap_exceeded
    assert post(rt2).status_code == 402 and p2.calls == 0


def test_a_provider_failure_charges_the_bound_and_persists(world):
    rt, p, _ = world.boot(Provider(fail=RuntimeError("provider down")))
    assert post(rt).status_code in (502, 503)
    snap = rt.budget_store.snapshot("default")
    assert snap["spent_usd"] > 0.002 and snap["reserved_usd"] == 0
    assert world.boot()[0].budgets["default"].spent_usd == pytest.approx(snap["spent_usd"])


# ------------------------------------------------------------------------------------------------ failure of the budget database itself
def test_a_database_failure_at_reserve_refuses_before_any_provider_call(world, monkeypatch):
    rt, p, _ = world.boot()

    def boom(*a, **k):
        raise BudgetStateError("disk I/O error")

    monkeypatch.setattr(rt.budget_store, "reserve", boom)
    r = post(rt)
    assert r.status_code == 503 and r.json()["error"]["type"] == "budget_state_error" and p.calls == 0


def test_a_settlement_failure_never_triggers_another_provider_call(world, monkeypatch):
    rt, p, _ = world.boot()

    def boom(*a, **k):
        raise BudgetStateError("disk I/O error")

    monkeypatch.setattr(rt.budget_store, "settle", boom)
    r = post(rt)
    assert r.status_code == 503 and p.calls == 1  # three backends are configured, yet exactly one provider call was made
    monkeypatch.undo()
    snap = rt.budget_store.snapshot("default")
    assert snap["spent_usd"] == 0 and snap["reserved_usd"] > 0 and len(rt.budget_store.unresolved()) == 1  # still held; an operator must decide


# ------------------------------------------------------------------------------------------------ tenants, cache, routing
def test_an_unknown_tenant_is_refused_without_creating_budget_rows(world):
    rt, p, _ = world.boot()
    r = post(rt, tenant="ghost")
    assert r.status_code == 403 and p.calls == 0
    assert world.db().execute("SELECT COUNT(*) FROM tenant_budgets").fetchone()[0] == 1


def test_cache_hits_do_not_touch_the_budget_and_tenants_stay_isolated(world):
    rt, p, _ = world.boot(limits={"default": LIMIT, "other": LIMIT})
    first = post(rt)
    events = world.db().execute("SELECT COUNT(*) FROM budget_events").fetchone()[0]
    again = post(rt)
    assert first.headers["x-routebench-cache"] == "miss" and again.headers["x-routebench-cache"] == "hit" and p.calls == 1
    assert world.db().execute("SELECT COUNT(*) FROM budget_events").fetchone()[0] == events  # a cache hit reserves and spends nothing
    other = post(rt, tenant="other")
    assert other.headers["x-routebench-cache"] == "miss" and p.calls == 2
    assert rt.budgets["other"].spent_usd == pytest.approx(0.0002) and rt.budgets["default"].spent_usd == pytest.approx(0.0002)


# ------------------------------------------------------------------------------------------------ operator CLI
def test_the_operator_cli_lists_resolves_and_sets_limits_with_a_named_operator(world, capsys):
    rt, _p, _ = world.boot()
    rid = rt.budget_store.reserve("default", 0.004)
    assert budget_admin.main(["unresolved"], state_factory=world.state) == 0
    assert json.loads(capsys.readouterr().out)[0]["reservation_id"] == rid
    assert budget_admin.main(["resolve", "--reservation", rid, "--operator", " ", "--reason", "x"], state_factory=world.state) == 2
    assert budget_admin.main(["resolve", "--reservation", rid, "--operator", "alice", "--reason", "died", "--charge-usd", "0.002"], state_factory=world.state) == 0
    assert budget_admin.main(["resolve", "--reservation", rid, "--operator", "alice", "--reason", "again"], state_factory=world.state) == 2
    assert budget_admin.main(["set-limit", "--tenant", "default", "--limit", "5", "--operator", "alice", "--reason", "raise"], state_factory=world.state) == 0
    capsys.readouterr()
    assert budget_admin.main(["snapshot", "--tenant", "default"], state_factory=world.state) == 0
    snap = json.loads(capsys.readouterr().out)
    assert snap["limit_usd"] == 5.0 and snap["spent_usd"] == pytest.approx(0.002) and snap["reserved_usd"] == 0


# ------------------------------------------------------------------------------------------------ fixes from the fresh read-only review
def test_a_settlement_failure_mid_stream_never_leaks_the_admission_slot_or_the_upstream_connection(world, monkeypatch):
    from gateway import streaming
    from gateway.app import ChatRequest
    from test_gateway_native_streaming import StreamBackend

    backend = StreamBackend()
    rt, _p, _ = world.boot(backend, native_streaming=True)
    monkeypatch.setattr(rt.budget_store, "settle", lambda *a, **k: (_ for _ in ()).throw(BudgetStateError("disk I/O error")))
    configure_runtime(rt)
    opened = streaming.open_stream(ChatRequest(messages=BODY["messages"], stream=True), "default", rt.budgets["default"], "chatcmpl-x")
    released = []
    out = list(opened.events(lambda: released.append(1)))
    assert released == [1] and backend.closed  # the slot was returned and the upstream connection closed despite the failing ledger
    assert json.loads(out[-2][len("data: "):])["error"]["type"] == "upstream_error" and out[-1] == "data: [DONE]\n\n"
    assert rt.breakers[opened.backend].failures == 0  # a budget-database failure is not the model's fault
    monkeypatch.undo()
    assert len(rt.budget_store.unresolved()) == 1  # nothing was freed or guessed: the reservation is still open (held)


def test_repeated_settlement_failures_during_streams_cannot_exhaust_gateway_capacity(world, monkeypatch):
    from test_gateway_native_streaming import StreamBackend

    rt, _p, _ = world.boot(StreamBackend(), native_streaming=True, limits={"default": 100.0})
    configure_runtime(rt)
    real = rt.budget_store.settle
    monkeypatch.setattr(rt.budget_store, "settle", lambda *a, **k: (_ for _ in ()).throw(BudgetStateError("disk I/O error")))
    client = TestClient(app)
    for i in range(60):  # more failures than the 50 admission slots
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": f"calculate {i}+1"}], "stream": True}).status_code == 200
    monkeypatch.setattr(rt.budget_store, "settle", real)
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "calculate 99+1"}], "stream": True})
    assert r.status_code == 200  # not 429: no slot leaked


def test_startup_verification_is_a_consistent_snapshot_while_another_runtime_reserves_and_settles(world):
    rt_a, _pa, _ = world.boot(limits={"default": 1000.0})
    errors, stop = [], threading.Event()

    def churn():
        budget = rt_a.budgets["default"]
        for _ in range(150):
            r = budget.reserve(0.0003)
            if r is not None:
                r.settle(0.0001)
        stop.set()

    t = threading.Thread(target=churn)
    t.start()
    verifier = BudgetStore(world.state())
    n = 0
    while not stop.is_set() or n < 5:
        try:
            verifier.verify({"default": 1000.0})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        n += 1
    t.join()
    assert not errors and n >= 5


def test_float_residue_cannot_accumulate_in_the_held_funds(world):
    rt, _p, _ = world.boot(limits={"default": 1000.0})
    store, budget = rt.budget_store, rt.budgets["default"]
    amounts = [0.1, 0.2, 0.3, 0.7, 1e-7, 0.15, 3.3333333, 0.30000000000000004]
    for _ in range(40):
        held = [budget.reserve(a) for a in amounts]
        assert all(h is not None for h in held)
        for h, a in zip(reversed(held), reversed(amounts), strict=True):
            h.settle(a / 3)
    snap = store.snapshot("default")
    assert snap["reserved_usd"] == 0.0 and store.verify({"default": 1000.0}) == []  # exactly zero, not -1e-16
    open_one = budget.reserve(0.1)
    assert store.snapshot("default")["reserved_usd"] == 0.1
    open_one.settle(0.05)


def test_two_runtimes_starting_together_create_the_tenant_once_and_a_mismatch_is_still_refused(world):
    errors, stores = [], [BudgetStore(world.state()) for _ in range(6)]

    def go(s):
        try:
            s.ensure_tenant("newcomer", 5.0)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(s,)) for s in stores]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and world.db().execute("SELECT COUNT(*) FROM tenant_budgets WHERE tenant='newcomer'").fetchone()[0] == 1
    assert world.db().execute("SELECT COUNT(*) FROM budget_events WHERE kind='tenant_created' AND tenant='newcomer'").fetchone()[0] == 1
    with pytest.raises(BudgetStateError, match="differs from the configured limit"):
        stores[0].ensure_tenant("newcomer", 6.0)


def test_readiness_reports_the_live_unresolved_count_and_fails_when_the_budget_database_is_unreachable(world, monkeypatch):
    rt, _p, _ = world.boot()
    configure_runtime(rt)
    client = TestClient(app)
    assert client.get("/readyz").json()["unresolved_budget_reservations"] == 0
    rt.budget_store.reserve("default", 0.002)  # created after startup by "another runtime"
    assert client.get("/readyz").json()["unresolved_budget_reservations"] == 1
    monkeypatch.setattr(rt.budget_store, "unresolved", lambda: (_ for _ in ()).throw(BudgetStateError("db down")))
    down = client.get("/readyz")
    assert down.status_code == 503 and down.json()["ready"] is False and down.json()["unresolved_budget_reservations"] is None


def test_startup_refuses_conflicting_budget_modes_nonfinite_limits_and_orphan_reservations(world, tmp_path):
    with pytest.raises(GatewayStartupError, match="mutually exclusive"):
        world.boot(budget_ledger=tmp_path / "x.jsonl", ephemeral_budgets=True)
    with pytest.raises(GatewayStartupError, match="durable budget state"):
        world.boot(limits={"default": float("nan")})
    world.boot()
    con = world.db()
    con.execute("INSERT INTO budget_reservations (reservation_id, tenant, amount_usd, state, owner, created_at) VALUES ('orphan', 'ghost-tenant', 0.1, 'open', 'x', 1.0)")
    con.commit()
    with pytest.raises(GatewayStartupError, match="no budget row"):
        world.boot()


def test_the_operator_cli_refuses_cleanly_when_the_database_cannot_be_opened(capsys):
    def broken():
        raise OSError("unable to open database file")

    assert budget_admin.main(["unresolved"], state_factory=broken) == 2
    assert "REFUSED" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ per-reservation validation (independently reproduced gap)
# A tenant total that still equals the sum of the rows must not be enough: every row is validated on its own, by state.
REBALANCE = (
    "UPDATE tenant_budgets SET "
    "reserved_usd=(SELECT COALESCE(SUM(amount_usd),0) FROM budget_reservations r WHERE r.tenant=tenant_budgets.tenant AND r.state='open'), "
    "spent_usd=(SELECT COALESCE(SUM(charged_usd),0) FROM budget_reservations r WHERE r.tenant=tenant_budgets.tenant AND r.state IN ('settled','resolved')), "
    "breaches=(SELECT COALESCE(SUM(breach),0) FROM budget_reservations r WHERE r.tenant=tenant_budgets.tenant AND r.state IN ('settled','resolved'))"
)


def seeded(world):
    """A database holding every kind of row: known and unknown-usage settlements, an overshoot, two operator resolutions and one open reservation."""
    rt, provider, _ = world.boot(limits={"default": 10.0})
    s = rt.budget_store
    ids = {}
    ids["known"] = s.reserve("default", 0.004)
    s.settle(ids["known"], 0.001)
    ids["known2"] = s.reserve("default", 0.004)
    s.settle(ids["known2"], 0.002)
    ids["unknown"] = s.reserve("default", 0.003)
    s.settle(ids["unknown"], None)
    ids["breach"] = s.reserve("default", 0.001)
    s.settle(ids["breach"], 0.5)
    ids["resolved_bound"] = s.reserve("default", 0.005)
    s.resolve(ids["resolved_bound"], operator="alice", reason="died")
    ids["resolved_stated"] = s.reserve("default", 0.005)
    s.resolve(ids["resolved_stated"], operator="alice", reason="partial", charge_usd=0.002)
    ids["open"] = s.reserve("default", 0.006)
    ids["open2"] = s.reserve("default", 0.3)  # a second open row, so a negative bound on one can be offset in the held-funds total
    assert len(s.verify({"default": 10.0})) == 2  # two unresolved reservations, the books are consistent
    return rt, provider, ids


def dump(world):
    con = world.db()
    try:
        return [con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in ("tenant_budgets", "budget_reservations", "budget_events")]
    finally:
        con.close()


# name -> (SQL statements separated by "; ", whether to rebalance the tenant totals so ONLY per-row validation can catch it)
TAMPERS = {
    "negative bound on a settled row": ("UPDATE budget_reservations SET amount_usd=-0.25 WHERE reservation_id='{known}'", False),
    "negative bound on an open row, offset by its sibling so held funds stay non-negative": ("UPDATE budget_reservations SET amount_usd=-0.25 WHERE reservation_id='{open}'", True),
    "resolution with a null operator and reason": ("UPDATE budget_reservations SET resolution='{{\"operator\":null,\"reason\":null,\"charge_stated_by_operator\":false}}' WHERE reservation_id='{resolved_bound}'", False),
    "resolution with non-string operator and reason": ("UPDATE budget_reservations SET resolution='{{\"operator\":[],\"reason\":{{}},\"charge_stated_by_operator\":false}}' WHERE reservation_id='{resolved_bound}'", False),
    "text nan bound": ("UPDATE budget_reservations SET amount_usd='nan' WHERE reservation_id='{known}'", False),
    "missing booked charge on a settled row, spend rebalanced": ("UPDATE budget_reservations SET charged_usd=NULL WHERE reservation_id='{known}'", True),
    "missing booked charge on a resolved row, spend rebalanced": ("UPDATE budget_reservations SET charged_usd=NULL WHERE reservation_id='{resolved_bound}'", True),
    "negative charge offset by an inflated one, totals unchanged": (
        ("UPDATE budget_reservations SET charged_usd=-0.2, actual_usd=-0.2 WHERE reservation_id='{known}'; "
         "UPDATE budget_reservations SET charged_usd=charged_usd+0.201, actual_usd=actual_usd+0.201 WHERE reservation_id='{known2}'"), False),
    "known actual differs from the booked charge, spend rebalanced": ("UPDATE budget_reservations SET charged_usd=actual_usd+0.01 WHERE reservation_id='{known}'", True),
    "unknown usage no longer charged at its bound, spend rebalanced": ("UPDATE budget_reservations SET charged_usd=0.0001 WHERE reservation_id='{unknown}'", True),
    "charge moved between two rows, totals unchanged": (
        ("UPDATE budget_reservations SET charged_usd=charged_usd-0.001, actual_usd=actual_usd-0.001 WHERE reservation_id='{known2}'; "
         "UPDATE budget_reservations SET charged_usd=charged_usd+0.001 WHERE reservation_id='{unknown}'"), False),
    "known actual differs from the booked charge but the breach flag still agrees": ("UPDATE budget_reservations SET actual_usd=actual_usd+0.0005 WHERE reservation_id='{known}'", False),
    "settled row without a settlement time": ("UPDATE budget_reservations SET settled_at=NULL WHERE reservation_id='{known}'", False),
    "resolution with a text charge declaration": ("UPDATE budget_reservations SET resolution='{{\"operator\":\"a\",\"reason\":\"r\",\"charge_stated_by_operator\":\"false\"}}' WHERE reservation_id='{resolved_bound}'", False),
    "resolution without a charge declaration": ("UPDATE budget_reservations SET resolution='{{\"operator\":\"a\",\"reason\":\"r\"}}' WHERE reservation_id='{resolved_bound}'", False),
    "unrecognised state": ("UPDATE budget_reservations SET state='OPEN', actual_usd=NULL, charged_usd=NULL, breach=NULL, settled_at=NULL WHERE reservation_id='{unknown}'", False),
    "wrong breach flag, count rebalanced": ("UPDATE budget_reservations SET breach=1 WHERE reservation_id='{known}'", True),
    "breach flag cleared on a real overshoot, count rebalanced": ("UPDATE budget_reservations SET breach=0 WHERE reservation_id='{breach}'", True),
    "open row carrying a booked charge": ("UPDATE budget_reservations SET charged_usd=0.1 WHERE reservation_id='{open}'", False),
    "settled row carrying a resolution": ("UPDATE budget_reservations SET resolution='{{\"operator\":\"x\"}}' WHERE reservation_id='{known}'", False),
    "resolved row without a resolution": ("UPDATE budget_reservations SET resolution=NULL WHERE reservation_id='{resolved_bound}'", False),
    "resolved row with an empty operator": ("UPDATE budget_reservations SET resolution='{{\"operator\":\"\",\"reason\":\"r\",\"charge_stated_by_operator\":false}}' WHERE reservation_id='{resolved_bound}'", False),
    "resolution contradicting the stored actual cost": ("UPDATE budget_reservations SET resolution='{{\"operator\":\"a\",\"reason\":\"r\",\"charge_stated_by_operator\":false}}' WHERE reservation_id='{resolved_stated}'", False),
    "non-numeric creation time": ("UPDATE budget_reservations SET created_at='yesterday' WHERE reservation_id='{known}'", False),
    "missing owner": ("UPDATE budget_reservations SET owner='' WHERE reservation_id='{known}'", False),
}


def tamper(world, ids, name):
    sql, rebalance = TAMPERS[name]
    con = world.db()
    for stmt in sql.format(**ids).split("; "):
        con.execute(stmt)
    if rebalance:
        con.execute(REBALANCE)
    con.commit()
    con.close()


@pytest.mark.parametrize("name", list(TAMPERS))
def test_startup_rejects_a_malformed_reservation_even_when_the_aggregate_balances_match(world, name):
    _rt, _provider, ids = seeded(world)
    tamper(world, ids, name)
    before = dump(world)
    with pytest.raises(GatewayStartupError, match="durable budget state blocks startup"):
        world.boot(limits={"default": 10.0})
    assert dump(world) == before  # nothing was repaired, refunded or discarded


def test_the_seeded_books_with_every_row_type_pass_verification(world):
    rt, _provider, _ids = seeded(world)
    again = world.boot(limits={"default": 10.0})[0]
    assert len(again.budget_store.unresolved()) == 2
    assert rt.budget_store.snapshot("default")["breaches"] == 1


@pytest.mark.parametrize("name", ["negative bound on an open row, offset by its sibling so held funds stay non-negative", "text nan bound"])
def test_settlement_and_resolution_revalidate_the_persisted_row_before_touching_balances(world, name):
    rt, _provider, ids = seeded(world)
    target = ids["open"] if "open" in name else ids["known"]
    con = world.db()
    con.execute(f"UPDATE budget_reservations SET amount_usd={'-0.25' if 'negative' in name else chr(39) + 'nan' + chr(39)}, state='open', actual_usd=NULL, charged_usd=NULL, breach=NULL, settled_at=NULL, resolution=NULL WHERE reservation_id='{target}'")
    con.execute(REBALANCE)
    con.commit()
    con.close()
    before = dump(world)
    store = rt.budget_store
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        store.settle(target, 0.001)
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        store.resolve(target, operator="alice", reason="refund attempt", charge_usd=0.0)  # a malformed row can never be turned into a refund
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        store.resolve(target, operator="alice", reason="default charge")
    assert dump(world) == before


@pytest.mark.parametrize("figure,value", [("spent_usd", "'nan'"), ("spent_usd", "-1"), ("reserved_usd", "-5"), ("reserved_usd", "'inf'"), ("limit_usd", "'nan'"), ("limit_usd", "-3")])
def test_a_corrupt_tenant_figure_cannot_make_reserve_or_settle_fail_open(world, figure, value):
    rt, provider, ids = seeded(world)
    con = world.db()
    con.execute(f"UPDATE tenant_budgets SET {figure}={value}")
    con.commit()
    con.close()
    before = dump(world)
    calls = provider.calls
    with pytest.raises(BudgetStateError, match="finite non-negative"):
        rt.budget_store.reserve("default", 1000.0)  # far above the cap: a NaN comparison used to admit it
    with pytest.raises(BudgetStateError, match="finite non-negative"):
        rt.budget_store.settle(ids["open"], 0.001)
    assert post(rt).status_code == 503
    assert provider.calls == calls  # no provider call was made on corrupt books
    assert dump(world) == before


@pytest.mark.parametrize("name", list(TAMPERS))
def test_every_tampered_row_is_caught_by_the_per_row_validation_itself_not_by_an_aggregate(world, name):
    rt, _provider, ids = seeded(world)
    tamper(world, ids, name)
    with pytest.raises(BudgetStateError, match=r"reservation '"):  # per-row messages name the reservation; aggregate checks do not
        rt.budget_store.verify({"default": 10.0})


@pytest.mark.parametrize("sibling_amount", ["-0.003", "'nan'", "'inf'"])
def test_a_malformed_sibling_open_row_blocks_reserve_and_settle_instead_of_being_folded_into_held_funds(world, sibling_amount):
    rt, provider, ids = seeded(world)
    con = world.db()
    con.execute(f"UPDATE budget_reservations SET amount_usd={sibling_amount} WHERE reservation_id='{ids['open2']}'")
    con.commit()
    con.close()
    before = dump(world)
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        rt.budget_store.reserve("default", 0.005)
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        rt.budget_store.settle(ids["open"], 0.001)  # a different, well-formed reservation: still refused while the tenant's books are malformed
    assert post(rt).status_code == 503 and provider.calls == 0
    assert dump(world) == before  # held funds were not recomputed, nothing repaired


def test_a_stored_held_funds_figure_that_disagrees_with_the_open_rows_is_refused_not_rewritten(world):
    rt, _provider, ids = seeded(world)
    con = world.db()
    con.execute("UPDATE tenant_budgets SET reserved_usd=0")
    con.commit()
    con.close()
    before = dump(world)
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        rt.budget_store.reserve("default", 0.005)
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        rt.budget_store.settle(ids["open"], 0.001)
    assert dump(world) == before


@pytest.mark.parametrize("value", ["'abc'", "-7", "1.5"])
def test_a_corrupt_breach_count_is_neither_rewritten_nor_trusted(world, value):
    rt, _provider, ids = seeded(world)
    con = world.db()
    con.execute(f"UPDATE tenant_budgets SET breaches={value}")
    con.commit()
    con.close()
    before = dump(world)
    with pytest.raises(BudgetStateError, match="breach count"):
        rt.budget_store.settle(ids["open"], 0.001)
    with pytest.raises(BudgetStateError, match="breach count"):
        rt.budget_store.verify({"default": 10.0})
    assert dump(world) == before


def test_a_null_tenant_row_cannot_hide_reservations_that_belong_to_no_tenant(world):
    seeded(world)
    con = world.db()
    con.execute("INSERT INTO tenant_budgets (tenant, limit_usd, spent_usd, reserved_usd, breaches, updated_at) VALUES (NULL, 1, 0, 0, 0, 0)")
    con.execute("INSERT INTO budget_reservations (reservation_id, tenant, amount_usd, state, owner, created_at) VALUES ('ghost1', 'ghost', 0.1, 'open', 'o', 0)")
    con.commit()
    con.close()
    with pytest.raises(GatewayStartupError, match="durable budget state blocks startup"):
        world.boot(limits={"default": 10.0})


def test_the_unresolved_listing_refuses_a_malformed_row_instead_of_reporting_it_as_a_figure(world):
    rt, _provider, ids = seeded(world)
    con = world.db()
    con.execute(f"UPDATE budget_reservations SET amount_usd='nan' WHERE reservation_id='{ids['open']}'")
    con.commit()
    con.close()
    with pytest.raises(BudgetStateError, match="reservation '"):
        rt.budget_store.unresolved()


def test_an_unrecognised_state_cannot_hide_a_hold_from_reserve(world):
    rt, _provider, ids = seeded(world)
    con = world.db()
    con.execute(f"UPDATE budget_reservations SET state='OPEN' WHERE reservation_id='{ids['open2']}'")
    con.execute("UPDATE tenant_budgets SET reserved_usd=0.006")  # the totals now 'agree' with the rows that are literally called open
    con.commit()
    con.close()
    before = dump(world)
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        rt.budget_store.reserve("default", 9.5)
    assert dump(world) == before


def test_the_tenant_breach_count_must_equal_the_breaching_reservations(world):
    rt, _provider, _ids = seeded(world)
    con = world.db()
    con.execute("UPDATE tenant_budgets SET breaches=breaches+1")
    con.commit()
    con.close()
    with pytest.raises(BudgetStateError, match="breach count"):
        rt.budget_store.verify({"default": 10.0})


def test_a_settlement_that_would_make_recorded_spend_non_finite_is_refused(world):
    rt, _provider, _ids = seeded(world)
    s = rt.budget_store
    s.ensure_tenant("huge", 1.7e308)
    first, second = s.reserve("huge", 0.0), s.reserve("huge", 0.0)
    s.settle(first, 1.7e308)
    before = dump(world)
    with pytest.raises(BudgetStateError, match="non-finite"):
        s.settle(second, 1.7e308)
    assert dump(world) == before
    assert s.snapshot("huge")["spent_usd"] == 1.7e308


def test_blank_tenants_and_bad_operator_inputs_are_refused_before_anything_is_written(world):
    rt, _provider, ids = seeded(world)
    s = rt.budget_store
    before = dump(world)
    with pytest.raises(BudgetStateError, match="tenant"):
        s.ensure_tenant(" ", 5.0)
    with pytest.raises(BudgetStateError, match="tenant"):
        s.reserve(" ", 1.0)
    with pytest.raises(BudgetStateError, match="operator"):
        s.resolve(ids["open"], operator=5, reason="r")  # type: ignore[arg-type]
    with pytest.raises(BudgetStateError, match="operator"):
        s.set_limit("default", 20.0, operator=None, reason="r")  # type: ignore[arg-type]
    with pytest.raises(BudgetStateError, match="non-negative"):
        s.settle(ids["open"], 10**400)
    with pytest.raises(BudgetStateError, match="configured limit"):
        s.verify({"default": float("nan")})
    assert dump(world) == before


def test_neither_a_limit_change_nor_startup_proceeds_on_books_that_are_already_malformed(world):
    rt, _provider, ids = seeded(world)
    con = world.db()
    con.execute(f"UPDATE budget_reservations SET amount_usd=-0.003 WHERE reservation_id='{ids['open']}'")
    con.commit()
    con.close()
    before = dump(world)
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        rt.budget_store.set_limit("default", 20.0, operator="alice", reason="raise the cap")
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        rt.budget_store.ensure_tenant("default", 10.0)
    assert dump(world) == before


def test_the_unresolved_listing_cannot_omit_a_row_with_an_unrecognised_state(world):
    rt, _provider, ids = seeded(world)
    con = world.db()
    con.execute(f"UPDATE budget_reservations SET state='weird' WHERE reservation_id='{ids['open2']}'")
    con.commit()
    con.close()
    with pytest.raises(BudgetStateError, match="unknown state"):
        rt.budget_store.unresolved()
    assert budget_admin.main(["unresolved"], state_factory=world.state) == 2


def test_an_overflowing_sum_cannot_satisfy_the_held_or_spend_checks(world):
    rt, _provider, _ids = seeded(world)
    s = rt.budget_store
    s.ensure_tenant("big", 1.7e308)
    a, b = s.reserve("big", 0.0), s.reserve("big", 0.0)
    con = world.db()
    con.execute(f"UPDATE budget_reservations SET amount_usd=1e308 WHERE reservation_id IN ('{a}','{b}')")
    con.execute("UPDATE tenant_budgets SET reserved_usd=1e308 WHERE tenant='big'")
    con.commit()
    con.close()
    before = dump(world)
    with pytest.raises(BudgetStateError, match="held funds"):
        s.verify({"default": 10.0, "big": 1.7e308})
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        s.reserve("big", 1.0)
    assert dump(world) == before
    con = world.db()
    con.execute(f"UPDATE budget_reservations SET state='settled', actual_usd=1e308, charged_usd=1e308, breach=0, settled_at=1 WHERE reservation_id IN ('{a}','{b}')")
    con.execute("UPDATE tenant_budgets SET reserved_usd=0, spent_usd=1.5e308, breaches=0 WHERE tenant='big'")
    con.commit()
    con.close()
    with pytest.raises(BudgetStateError, match="spend"):
        s.verify({"default": 10.0, "big": 1.7e308})


def test_a_caller_supplied_clock_must_be_a_finite_non_negative_number(world):
    rt, _provider, ids = seeded(world)
    s = rt.budget_store
    before = dump(world)
    for bad in (-5.0, float("nan"), float("inf"), True, "7"):
        with pytest.raises(BudgetStateError, match="now"):
            s.settle(ids["open"], 0.05, now=bad)
        with pytest.raises(BudgetStateError, match="now"):
            s.reserve("default", 0.001, now=bad)
    assert dump(world) == before


def test_a_new_tenant_row_does_not_silently_adopt_reservations_that_already_carry_its_name(world):
    rt, _provider, _ids = seeded(world)
    con = world.db()
    con.execute("INSERT INTO budget_reservations (reservation_id, tenant, amount_usd, state, owner, created_at) VALUES ('stray', 'newbie', 0.4, 'open', 'o', 1)")
    con.commit()
    con.close()
    before = dump(world)
    with pytest.raises(BudgetStateError, match="refusing to change balances"):
        rt.budget_store.ensure_tenant("newbie", 5.0)
    assert dump(world) == before  # the tenant row was not created


@pytest.mark.parametrize("sql,match", [
    ("UPDATE tenant_budgets SET limit_usd='nan'", "limit"),
    ("UPDATE tenant_budgets SET reserved_usd=reserved_usd+0.5", "held funds"),
    ("UPDATE tenant_budgets SET spent_usd=spent_usd+0.5", "spend"),
])
def test_verify_checks_the_tenant_figures_itself_not_only_through_ensure_tenant(world, sql, match):
    rt, _provider, _ids = seeded(world)
    con = world.db()
    con.execute(sql)
    con.commit()
    con.close()
    with pytest.raises(BudgetStateError, match=match):
        rt.budget_store.verify({"default": 10.0})


def test_money_arguments_reject_booleans_negatives_and_numeric_strings(world):
    rt, _provider, ids = seeded(world)
    s = rt.budget_store
    before = dump(world)
    for bad in (True, -1.0, "0.5", float("nan")):
        with pytest.raises(BudgetStateError):
            s.settle(ids["open"], bad)
        with pytest.raises(BudgetStateError):
            s.resolve(ids["open"], operator="alice", reason="r", charge_usd=bad)
        with pytest.raises(BudgetStateError):
            s.reserve("default", bad)
    with pytest.raises(BudgetStateError):
        s.ensure_tenant("z", True)  # type: ignore[arg-type]
    with pytest.raises(BudgetStateError):
        s.ensure_tenant("y", "0.5")  # type: ignore[arg-type]
    assert dump(world) == before


# ------------------------------------------------------------------------------------------------ native streaming x durable database budgets (end to end)
def stream_rows(world):
    con = world.db()
    try:
        return con.execute("SELECT state, actual_usd, charged_usd, breach FROM budget_reservations ORDER BY created_at").fetchall()
    finally:
        con.close()


def test_a_completed_native_stream_is_settled_from_its_final_usage_in_the_database(world):
    from test_gateway_native_streaming import StreamBackend

    backend = StreamBackend()
    rt, _p, _ = world.boot(backend, native_streaming=True)
    configure_runtime(rt)
    r = TestClient(app).post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200 and r.text.rstrip().endswith("data: [DONE]") and backend.closed and len(backend.payloads) == 1
    assert rt.budget_store.snapshot("default") == {"limit_usd": LIMIT, "spent_usd": pytest.approx(15e-6), "reserved_usd": 0.0, "breaches": 0}
    assert stream_rows(world) == [("settled", pytest.approx(15e-6), pytest.approx(15e-6), 0)]
    assert rt.budget_store.unresolved() == []


def test_a_native_stream_without_final_usage_is_charged_at_the_bound_in_the_database(world):
    from test_gateway_native_streaming import StreamBackend, chunk

    backend = StreamBackend([chunk(role=True), chunk("x"), chunk(finish="stop")])
    rt, _p, _ = world.boot(backend, native_streaming=True)
    configure_runtime(rt)
    TestClient(app).post("/v1/chat/completions", json={**BODY, "stream": True})
    ((state, actual, charged, breach),) = stream_rows(world)
    assert state == "settled" and actual is None and charged > 15e-6 and breach == 0  # usage unknown: the conservative bound, not zero
    snap = rt.budget_store.snapshot("default")
    assert snap["reserved_usd"] == 0.0 and snap["spent_usd"] == pytest.approx(charged)


def test_a_midstream_failure_is_charged_at_the_bound_and_never_calls_another_backend_in_the_database(world):
    from test_gateway_native_streaming import StreamBackend

    backend = StreamBackend(fail_after=2)
    rt, _p, _ = world.boot(backend, native_streaming=True)
    configure_runtime(rt)
    r = TestClient(app).post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200 and "connection reset" not in r.text and r.text.rstrip().endswith("data: [DONE]")
    assert len(backend.payloads) == 1 and backend.closed  # no second provider call, upstream closed
    ((state, actual, charged, _breach),) = stream_rows(world)
    assert state == "settled" and actual is None and charged > 15e-6
    assert rt.budget_store.snapshot("default")["reserved_usd"] == 0.0 and rt.budget_store.unresolved() == []


def test_a_client_that_disconnects_mid_stream_is_charged_the_bound_and_releases_admission_once_in_the_database(world):
    from gateway import streaming
    from gateway.app import ChatRequest
    from test_gateway_native_streaming import StreamBackend, chunk, usage_chunk

    backend = StreamBackend([chunk(role=True)] + [chunk("x")] * 50 + [usage_chunk()])
    rt, _p, _ = world.boot(backend, native_streaming=True)
    configure_runtime(rt)
    opened = streaming.open_stream(ChatRequest(messages=BODY["messages"], stream=True), "default", rt.budgets["default"], "chatcmpl-x")
    assert rt.budget_store.snapshot("default")["reserved_usd"] > 0  # funds are held, in the database, while the stream is open
    released = []
    events = opened.events(lambda: released.append(1))
    assert next(events).startswith("data: ")
    events.close()  # the client disconnects
    events.close()
    assert released == [1] and backend.closed and len(backend.payloads) == 1
    ((state, actual, charged, _breach),) = stream_rows(world)
    assert state == "settled" and actual is None and charged > 15e-6
    assert rt.budget_store.snapshot("default")["reserved_usd"] == 0.0
