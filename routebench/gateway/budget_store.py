"""Durable tenant budgets in the single configured database (the existing `ModelState` adapter; no second DSN, no currency cache).

Why: `TenantBudget` keeps spend in process memory, so a restart refunds every tenant and two runtimes cannot share a cap. Here the database is the only source of truth:

  * `reserve` writes a UNIQUE open reservation and raises the tenant's held funds in ONE transaction, BEFORE any provider I/O (the pre-call invariant);
  * `settle` is exact-once: it moves an open reservation to `settled`, releases the hold and books the charge in one transaction; a second settlement raises and changes nothing;
  * unknown usage is charged at the reservation bound; an actual cost above the bound is recorded in full and counted as a breach; nothing is discarded;
  * an unsettled reservation (the process died after reserving, or after the provider answered but before settling) STAYS HELD against the cap forever. Nothing resets it. Only an explicit operator call
    (`resolve`, with operator and reason) settles it, at the bound by default; a smaller charge (a refund of the difference) must be stated by the operator;
  * startup verifies the books (schema version, finite non-negative amounts, held funds equal to the open reservations, spend equal to the booked charges, persisted limit equal to the configured limit) and
    refuses to start otherwise; a configured limit that differs from the persisted one is never silently applied: `set_limit` is an explicit operator action.
Accounting figures are usage-priced estimates from provider-reported tokens at the configured prices, not invoices. SQLite serialises writers with BEGIN IMMEDIATE; PostgreSQL uses row locks (FOR UPDATE) and
was NOT exercised here (no Docker, no connection). This module never calls a provider and a settlement failure never triggers another provider call.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from .reliability import BudgetReservation
from .state import ModelState

SCHEMA_VERSION = "1"
EPS = 1e-9

DDL_SQLITE = (
    "CREATE TABLE IF NOT EXISTS budget_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS tenant_budgets (
      tenant TEXT PRIMARY KEY, limit_usd REAL NOT NULL, spent_usd REAL NOT NULL, reserved_usd REAL NOT NULL,
      breaches INTEGER NOT NULL, updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS budget_reservations (
      reservation_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, amount_usd REAL NOT NULL, state TEXT NOT NULL,
      actual_usd REAL, charged_usd REAL, breach INTEGER, owner TEXT NOT NULL, created_at REAL NOT NULL,
      settled_at REAL, resolution TEXT)""",
    "CREATE INDEX IF NOT EXISTS budget_reservations_state ON budget_reservations (state)",
    "CREATE INDEX IF NOT EXISTS budget_reservations_tenant_state ON budget_reservations (tenant, state)",
    """CREATE TABLE IF NOT EXISTS budget_events (
      sequence INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, tenant TEXT NOT NULL, kind TEXT NOT NULL,
      reservation_id TEXT, amount_usd REAL, details TEXT NOT NULL)""",
)
DDL_PG = tuple(d.replace("REAL", "DOUBLE PRECISION").replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY") for d in DDL_SQLITE)


class BudgetStateError(RuntimeError):
    """The durable budget state is unavailable, inconsistent, or the requested transition is not allowed. Fail closed."""


def _finite(value: Any, name: str) -> float:
    try:
        ok = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
    except OverflowError:  # an int too large for a float
        ok = False
    if not ok:
        raise BudgetStateError(f"{name} must be a finite non-negative number, got {value!r}")
    return float(value)


def _clock(now: Any) -> float:
    return time.time() if now is None else _finite(now, "now")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BudgetStateError(f"{name} must be a non-blank string, got {value!r}")
    return value


def _num(value: Any, what: str, *, optional: bool = False) -> float | None:
    """A persisted monetary/time figure must be a real, finite, non-negative number. Text such as 'nan' (SQLite keeps it as TEXT) or a NULL is corruption, never a value."""
    if value is None:
        if optional:
            return None
        raise BudgetStateError(f"{what} is missing")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise BudgetStateError(f"{what} is not a finite non-negative number ({value!r})")
    return float(value)


RESERVATION_COLUMNS = "reservation_id, tenant, amount_usd, state, actual_usd, charged_usd, breach, owner, created_at, settled_at, resolution"


def check_reservation_row(row: Any) -> None:
    """State-specific validation of ONE persisted reservation. Raises BudgetStateError on any malformed field; never repairs, defaults or discards."""
    rid, tenant, amount, state, actual, charged, breach, owner, created, settled, resolution = row
    ctx = f"reservation {rid!r}"
    if not isinstance(rid, str) or not rid.strip() or not isinstance(tenant, str) or not tenant.strip() or not isinstance(owner, str) or not owner.strip():
        raise BudgetStateError(f"{ctx}: identifier, tenant or owner is missing")
    bound = _num(amount, f"{ctx} bound (amount_usd)")
    _num(created, f"{ctx} created_at")
    if state == "open":
        if any(v is not None for v in (actual, charged, breach, settled, resolution)):
            raise BudgetStateError(f"{ctx} is open but carries terminal-state fields")
        return
    if state not in ("settled", "resolved"):
        raise BudgetStateError(f"{ctx} has the unknown state {state!r}")
    booked = _num(charged, f"{ctx} booked charge (charged_usd)")
    _num(settled, f"{ctx} settled_at")
    known = _num(actual, f"{ctx} actual_usd", optional=True)
    if known is None:
        if abs(booked - bound) > 1e-12:
            raise BudgetStateError(f"{ctx}: unknown usage must be charged at its bound ({bound}), but {booked} is booked")
    elif abs(known - booked) > 1e-12:
        raise BudgetStateError(f"{ctx}: the known actual cost {known} differs from the booked charge {booked}")
    if isinstance(breach, bool) or not isinstance(breach, int) or breach not in (0, 1) or breach != (1 if booked > bound + 1e-12 else 0):
        raise BudgetStateError(f"{ctx}: the breach flag {breach!r} is inconsistent with charge {booked} against bound {bound}")
    if state == "settled":
        if resolution is not None:
            raise BudgetStateError(f"{ctx} is settled but carries an operator resolution")
        return
    try:
        info = json.loads(resolution) if isinstance(resolution, str) else None
    except ValueError:
        info = None
    if (not isinstance(info, dict) or not all(isinstance(info.get(k), str) and info[k].strip() for k in ("operator", "reason"))
            or not isinstance(info.get("charge_stated_by_operator"), bool)):
        raise BudgetStateError(f"{ctx} is resolved without a valid operator, reason and charge declaration")
    if info["charge_stated_by_operator"] != (known is not None):
        raise BudgetStateError(f"{ctx}: the resolution's charge declaration contradicts the stored actual cost")


def _tenant_figures(row: Any, tenant: str) -> tuple[float, float, float]:
    limit, spent, reserved = (_num(v, f"tenant {tenant!r} {name}") for v, name in zip(row, ("limit", "spent", "reserved"), strict=True))
    return limit, spent, reserved  # type: ignore[return-value]


def _breach_count(value: Any, tenant: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BudgetStateError(f"tenant {tenant!r}: invalid breach count ({value!r})")
    return value


class BudgetStore:
    def __init__(self, state: ModelState) -> None:
        self.state = state
        self.owner = uuid.uuid4().hex  # identifies this runtime instance on the reservations it creates
        try:
            for ddl in DDL_SQLITE if state.sqlite else DDL_PG:
                state.connection.execute(ddl)
            if state.sqlite:
                state.connection.commit()
            with state.transaction() as c:
                c.execute(f"INSERT INTO budget_meta (key, value) VALUES ({self._ph}, {self._ph}) ON CONFLICT(key) DO NOTHING", ("schema_version", SCHEMA_VERSION))
        except BudgetStateError:
            raise
        except Exception as exc:
            raise BudgetStateError(f"budget schema could not be established: {exc}") from exc

    @property
    def _ph(self) -> str:
        return self.state._ph

    @property
    def _open_sum_sql(self) -> str:
        """Held funds are always the exact sum of the tenant's OPEN reservations (recomputed inside the transaction), so float add/subtract residue can never accumulate in the stored figure."""
        return f"(SELECT COALESCE(SUM(amount_usd), 0) FROM budget_reservations WHERE tenant={self._ph} AND state='open')"

    @property
    def _lock_sql(self) -> str:
        return "" if self.state.sqlite else " FOR UPDATE"

    def _locked_tenant(self, c: Any, tenant: str) -> tuple[float, float, float, int]:
        """Read the tenant's row under the write lock and validate it AND every one of its open reservations before any balance is decided or changed.
        Held funds are recomputed from the open rows afterwards, so a stored figure that disagrees with them (or a malformed open row) must be refused here, never silently corrected."""
        row = c.execute(f"SELECT limit_usd, spent_usd, reserved_usd, breaches FROM tenant_budgets WHERE tenant={self._ph}{self._lock_sql}", (tenant,)).fetchone()
        if row is None:
            raise BudgetStateError(f"unknown tenant {tenant!r}")
        try:
            limit, spent, reserved = _tenant_figures(row[:3], tenant)
            breaches = _breach_count(row[3], tenant)
            held = 0.0
            for open_row in c.execute(f"SELECT {RESERVATION_COLUMNS} FROM budget_reservations WHERE tenant={self._ph} AND state NOT IN ('settled','resolved')", (tenant,)).fetchall():
                check_reservation_row(open_row)  # an unrecognised state fails here instead of being dropped from the recomputed held funds
                held += float(open_row[2])
            if not (math.isfinite(held) and abs(held - reserved) <= EPS * max(1.0, held)):
                raise BudgetStateError(f"tenant {tenant!r}: held funds {reserved} do not equal the open reservations {held}")
        except BudgetStateError as exc:
            raise BudgetStateError(f"refusing to change balances: {exc}; nothing was repaired, refunded or discarded") from exc
        return limit, spent, reserved, breaches

    def _event(self, c: Any, tenant: str, kind: str, rid: str | None, amount: float | None, details: Mapping[str, Any], now: float) -> None:
        c.execute(f"INSERT INTO budget_events (ts, tenant, kind, reservation_id, amount_usd, details) VALUES ({', '.join([self._ph] * 6)})",
                  (now, tenant, kind, rid, amount, json.dumps(dict(details), sort_keys=True)))

    def _wrap(self, fn: Any) -> Any:
        try:
            return fn()
        except BudgetStateError:
            raise
        except Exception as exc:
            raise BudgetStateError(f"budget database error: {type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------ configuration
    def ensure_tenant(self, tenant: str, limit_usd: float, *, now: float | None = None) -> None:
        _text(tenant, "tenant")
        limit = _finite(limit_usd, "limit_usd")
        now = _clock(now)

        def go() -> None:
            with self.state.transaction() as c:
                created = c.execute(f"INSERT INTO tenant_budgets (tenant, limit_usd, spent_usd, reserved_usd, breaches, updated_at) VALUES ({', '.join([self._ph] * 6)}) "
                                    "ON CONFLICT(tenant) DO NOTHING", (tenant, limit, 0.0, 0.0, 0, now))
                row = c.execute(f"SELECT limit_usd FROM tenant_budgets WHERE tenant={self._ph}{self._lock_sql}", (tenant,)).fetchone()
                if created.rowcount == 1:
                    self._locked_tenant(c, tenant)  # reservations that already carry this name would otherwise be adopted unseen
                    self._event(c, tenant, "tenant_created", None, limit, {"limit_usd": limit}, now)
                elif abs(_num(row[0], f"tenant {tenant!r} persisted limit") - limit) > 1e-12:
                    raise BudgetStateError(f"tenant {tenant!r}: persisted limit {float(row[0])} differs from the configured limit {limit}; "
                                           "a limit change is an explicit operator action (set_limit), never applied silently")
                else:
                    self._locked_tenant(c, tenant)  # an existing tenant's books must be well-formed before startup proceeds
        self._wrap(go)

    def set_limit(self, tenant: str, limit_usd: float, *, operator: str, reason: str, now: float | None = None) -> None:
        if not isinstance(operator, str) or not isinstance(reason, str) or not operator.strip() or not reason.strip():
            raise BudgetStateError("set_limit requires a named operator and a reason")
        limit = _finite(limit_usd, "limit_usd")
        now = _clock(now)

        def go() -> None:
            with self.state.transaction() as c:
                previous = self._locked_tenant(c, tenant)[0]  # raising the cap on malformed books is refused too
                c.execute(f"UPDATE tenant_budgets SET limit_usd={self._ph}, updated_at={self._ph} WHERE tenant={self._ph}", (limit, now, tenant))
                self._event(c, tenant, "limit_set", None, limit, {"operator": operator, "reason": reason, "previous": previous}, now)
        self._wrap(go)

    def snapshot(self, tenant: str) -> dict[str, Any]:
        def go() -> dict[str, Any]:
            with self.state._lock:
                row = self.state.connection.execute(f"SELECT limit_usd, spent_usd, reserved_usd, breaches FROM tenant_budgets WHERE tenant={self._ph}", (tenant,)).fetchone()
            if row is None:
                raise BudgetStateError(f"unknown tenant {tenant!r}")
            limit, spent, reserved = _tenant_figures(row[:3], tenant)  # a corrupt figure is an error here too, never a plausible-looking number
            if isinstance(row[3], bool) or not isinstance(row[3], int) or row[3] < 0:
                raise BudgetStateError(f"tenant {tenant!r}: invalid breach count ({row[3]!r})")
            return {"limit_usd": limit, "spent_usd": spent, "reserved_usd": reserved, "breaches": row[3]}
        return self._wrap(go)

    # ------------------------------------------------------------------ reserve / settle
    def reserve(self, tenant: str, amount_usd: float, *, now: float | None = None) -> str | None:
        """Persist a unique open reservation and hold the funds, or return None if the cap would be exceeded. Raises for an unknown tenant or database failure."""
        _text(tenant, "tenant")
        amount = _finite(amount_usd, "amount")
        now = _clock(now)

        def go() -> str | None:
            with self.state.transaction() as c:
                limit, spent, reserved, _breaches = self._locked_tenant(c, tenant)
                if amount > limit - spent - reserved + 1e-12:
                    return None
                rid = uuid.uuid4().hex
                c.execute(f"INSERT INTO budget_reservations (reservation_id, tenant, amount_usd, state, owner, created_at) VALUES ({', '.join([self._ph] * 6)})",
                          (rid, tenant, amount, "open", self.owner, now))
                c.execute(f"UPDATE tenant_budgets SET reserved_usd={self._open_sum_sql}, updated_at={self._ph} WHERE tenant={self._ph}", (tenant, now, tenant))
                self._event(c, tenant, "reserved", rid, amount, {"owner": self.owner}, now)
                return rid
        return self._wrap(go)

    def _close(self, rid: str, actual: float | None, *, state: str, resolution: dict[str, Any] | None, now: float) -> tuple[bool, float]:
        """Exact-once transition open -> settled/resolved. `actual` None means usage unknown: charge the bound."""
        def go() -> tuple[bool, float]:
            with self.state.transaction() as c:
                res = c.execute(f"SELECT {RESERVATION_COLUMNS} FROM budget_reservations WHERE reservation_id={self._ph}{self._lock_sql}", (rid,)).fetchone()
                if res is None:
                    raise BudgetStateError(f"unknown reservation {rid!r}")
                if res[3] != "open" and res[3] in ("settled", "resolved"):
                    raise BudgetStateError(f"reservation {rid} is already {res[3]}; settlement is exact-once and nothing was changed")
                try:
                    check_reservation_row(res)
                except BudgetStateError as exc:
                    raise BudgetStateError(f"refusing to change balances: {exc}; nothing was repaired, refunded or discarded") from exc
                tenant, amount = res[1], float(res[2])
                charged = amount if actual is None else actual
                limit, spent, _reserved, _breaches = self._locked_tenant(c, tenant)
                breach = charged > amount + 1e-12
                new_spent = spent + charged
                if not math.isfinite(new_spent):
                    raise BudgetStateError(f"refusing to change balances: booking {charged} would make the tenant's recorded spend non-finite; nothing was changed")
                cur = c.execute(f"UPDATE budget_reservations SET state={self._ph}, actual_usd={self._ph}, charged_usd={self._ph}, breach={self._ph}, settled_at={self._ph}, resolution={self._ph} "
                                f"WHERE reservation_id={self._ph} AND state='open'",
                                (state, actual, charged, 1 if breach else 0, now, json.dumps(resolution, sort_keys=True) if resolution else None, rid))
                if cur.rowcount != 1:
                    raise BudgetStateError(f"reservation {rid} changed concurrently; nothing was applied")
                c.execute(f"UPDATE tenant_budgets SET spent_usd={self._ph}, reserved_usd={self._open_sum_sql}, breaches=breaches+{self._ph}, updated_at={self._ph} WHERE tenant={self._ph}",
                          (new_spent, tenant, 1 if breach else 0, now, tenant))
                self._event(c, tenant, state, rid, charged, {"usage_known": actual is not None, "breach": breach, **(resolution or {})}, now)
                return (not breach) and new_spent <= limit + 1e-12, charged
        return self._wrap(go)

    def settle(self, rid: str, actual_usd: float | None, *, now: float | None = None) -> bool:
        actual = None if actual_usd is None else _finite(actual_usd, "actual_usd")
        within, _ = self._close(rid, actual, state="settled", resolution=None, now=_clock(now))
        return within

    # ------------------------------------------------------------------ operator actions on unresolved reservations
    def unresolved(self) -> list[dict[str, Any]]:
        def go() -> list[dict[str, Any]]:
            with self.state._lock:
                rows = self.state.connection.execute(f"SELECT {RESERVATION_COLUMNS} FROM budget_reservations WHERE state NOT IN ('settled','resolved') ORDER BY created_at").fetchall()
            for r in rows:
                check_reservation_row(r)  # never report a malformed row as if it were a plain figure
            return [{"reservation_id": r[0], "tenant": r[1], "amount_usd": float(r[2]), "owner": r[7], "created_at": float(r[8]), "held_by_this_runtime": r[7] == self.owner} for r in rows]
        return self._wrap(go)

    def resolve(self, rid: str, *, operator: str, reason: str, charge_usd: float | None = None, now: float | None = None) -> dict[str, Any]:
        """The ONLY way to release funds held by an unsettled reservation. Default: charge the bound (nothing is refunded). A smaller `charge_usd` is an explicit, recorded refund decision."""
        if not isinstance(operator, str) or not isinstance(reason, str) or not operator.strip() or not reason.strip():
            raise BudgetStateError("resolving a reservation requires a named operator and a reason")
        charge = None if charge_usd is None else _finite(charge_usd, "charge_usd")
        _, charged = self._close(rid, charge, state="resolved", resolution={"operator": operator, "reason": reason, "charge_stated_by_operator": charge is not None}, now=_clock(now))
        return {"reservation_id": rid, "charged_usd": charged, "operator": operator}

    # ------------------------------------------------------------------ startup verification
    def verify(self, configured: Mapping[str, float]) -> list[dict[str, Any]]:
        """Raise BudgetStateError unless the persisted books are internally consistent and match the configured tenants and limits. Returns the unresolved reservations (held, never reset)."""
        def go() -> list[dict[str, Any]]:
            with self.state.transaction() as c:  # ONE transaction = one consistent snapshot even while other runtimes reserve and settle
                if not self.state.sqlite:
                    c.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                version = c.execute("SELECT value FROM budget_meta WHERE key='schema_version'").fetchone()
                tenants = c.execute("SELECT tenant, limit_usd, spent_usd, reserved_usd, breaches FROM tenant_budgets").fetchall()
                open_sum = dict(c.execute("SELECT tenant, SUM(amount_usd) FROM budget_reservations WHERE state='open' GROUP BY tenant").fetchall())
                charged = dict(c.execute("SELECT tenant, SUM(charged_usd) FROM budget_reservations WHERE state IN ('settled','resolved') GROUP BY tenant").fetchall())
                bad_state = c.execute("SELECT COUNT(*) FROM budget_reservations WHERE state NOT IN ('open','settled','resolved')").fetchone()[0]
                breach_counts = dict(c.execute("SELECT tenant, SUM(breach) FROM budget_reservations WHERE state IN ('settled','resolved') GROUP BY tenant").fetchall())
                all_rows = c.execute(f"SELECT {RESERVATION_COLUMNS} FROM budget_reservations").fetchall()
                orphans = c.execute("SELECT COUNT(*) FROM budget_reservations WHERE NOT EXISTS (SELECT 1 FROM tenant_budgets t WHERE t.tenant = budget_reservations.tenant)").fetchone()[0]
            if version is None or version[0] != SCHEMA_VERSION:
                raise BudgetStateError(f"budget schema version {None if version is None else version[0]!r} is not the supported {SCHEMA_VERSION!r}")
            for row in all_rows:  # every reservation individually, in addition to the aggregates below
                check_reservation_row(row)
            if bad_state:
                raise BudgetStateError("reservations with an unknown state exist")
            if orphans:
                raise BudgetStateError("reservations exist for a tenant that has no budget row")
            if any(not isinstance(r[0], str) or not r[0].strip() for r in tenants):
                raise BudgetStateError("a tenant budget row has a missing tenant name")
            persisted = {r[0]: r for r in tenants}
            for tenant, limit in configured.items():
                _finite(limit, f"configured limit for {tenant!r}")
                if tenant not in persisted:
                    raise BudgetStateError(f"configured tenant {tenant!r} has no persisted budget row")
                if abs(float(persisted[tenant][1]) - float(limit)) > 1e-12:
                    raise BudgetStateError(f"tenant {tenant!r}: persisted limit {persisted[tenant][1]} differs from the configured limit {limit}; use set_limit explicitly")
            for tenant, (_, lim, spent, reserved, breaches) in persisted.items():
                for name, v in (("limit", lim), ("spent", spent), ("reserved", reserved)):
                    if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                        raise BudgetStateError(f"tenant {tenant!r}: {name} is not a finite non-negative number")
                if not isinstance(breaches, int) or isinstance(breaches, bool) or breaches < 0:
                    raise BudgetStateError(f"tenant {tenant!r}: invalid breach count")
                if int(breach_counts.get(tenant) or 0) != breaches:
                    raise BudgetStateError(f"tenant {tenant!r}: breach count {breaches} does not equal the {int(breach_counts.get(tenant) or 0)} breaching reservations")
                held, booked = float(open_sum.get(tenant) or 0.0), float(charged.get(tenant) or 0.0)
                if not (math.isfinite(held) and abs(held - reserved) <= EPS * max(1.0, held)):  # written so a NaN/Inf figure fails instead of slipping through
                    raise BudgetStateError(f"tenant {tenant!r}: held funds {reserved} do not equal the open reservations {held}")
                if not (math.isfinite(booked) and abs(booked - spent) <= EPS * max(1.0, booked)):
                    raise BudgetStateError(f"tenant {tenant!r}: spend {spent} does not equal the booked charges {booked}")
            return self.unresolved()
        return self._wrap(go)


class DbReservation(BudgetReservation):
    """A claim backed by a persisted reservation row. `settle` is exact-once and raises (leaving the funds held) if the database cannot record it."""

    def __init__(self, budget: DbTenantBudget, amount: float, rid: str) -> None:
        super().__init__(budget, amount)  # type: ignore[arg-type]
        self.reservation_id = rid

    def settle(self, actual_usd: float | None = None) -> bool:
        within = self._budget.store.settle(self.reservation_id, actual_usd)  # type: ignore[attr-defined]
        self._settled = True
        return within


class DbTenantBudget:
    """Drop-in for `TenantBudget` whose every figure is read from, and every change is written to, the database."""

    def __init__(self, store: BudgetStore, tenant: str) -> None:
        self.store, self.tenant = store, tenant

    @property
    def limit_usd(self) -> float:
        return self.store.snapshot(self.tenant)["limit_usd"]

    @property
    def spent_usd(self) -> float:
        return self.store.snapshot(self.tenant)["spent_usd"]

    @property
    def reserved_usd(self) -> float:
        return self.store.snapshot(self.tenant)["reserved_usd"]

    @property
    def reservation_breaches(self) -> int:
        return self.store.snapshot(self.tenant)["breaches"]

    @property
    def cap_exceeded(self) -> bool:
        s = self.store.snapshot(self.tenant)
        return s["spent_usd"] > s["limit_usd"] + 1e-12

    def remaining(self) -> float:
        s = self.store.snapshot(self.tenant)
        return max(0.0, s["limit_usd"] - s["spent_usd"] - s["reserved_usd"])

    def reserve(self, amount: float) -> DbReservation | None:
        rid = self.store.reserve(self.tenant, amount)
        return None if rid is None else DbReservation(self, _finite(amount, "amount"), rid)

    def charge(self, amount: float) -> bool:
        reservation = self.reserve(amount)
        if reservation is None:
            return False
        reservation.settle(amount)
        return True
