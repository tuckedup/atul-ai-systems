"""Restart-safe tenant budgets: an fsynced, append-only ledger behind the existing TenantBudget interface.

`TenantBudget` keeps spend in process memory, so a restart silently refunds every tenant. `DurableBudgetLedger` records every reservation BEFORE the in-memory claim is granted and every settlement
BEFORE the in-memory books change, so the books can always be rebuilt from the file:

  * a reservation is journaled (`reserve`) and fsynced first; if that write fails the request is refused, never admitted unrecorded;
  * a settlement (`settle`) is journaled with the charged amount and whether the reservation bound was breached; actual cost is never discarded;
  * on startup a reservation with no settlement (a crash while a provider call was in flight) is settled AT ITS BOUND and counted as spent, recorded as `recovered`, so a crash can only over-count;
  * the ledger is parsed strictly: a malformed, duplicated, unknown or negative/non-finite entry stops startup and the file is left untouched (nothing is repaired or dropped);
  * exactly one process may own a ledger (an exclusive lock file); a leftover lock is refused unless the operator passes `takeover=True` after confirming the old process is gone.
This is single-host, single-worker durability, not a distributed commit: two workers need a shared store, which this does not provide (the lock refuses to let them share the file).
Limits are configuration, not ledger state: lowering a limit below recorded spend leaves the tenant exhausted (remaining 0), and raising it restores headroom.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Self

from .reliability import BudgetReservation, TenantBudget, _valid_money

VERSION = 1


class BudgetLedgerError(RuntimeError):
    """The ledger is unreadable, inconsistent, locked by another owner, or could not be written. Fail closed."""


def _pid_alive(pid: int) -> bool | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - optional
        return None
    return bool(psutil.pid_exists(pid))


class DurableBudgetLedger:
    def __init__(self, path: str | Path, *, takeover: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._io = threading.Lock()
        self._owner = False
        self._acquire(takeover)
        try:
            self._spent: dict[str, float] = {}
            self._breaches: dict[str, int] = {}
            self._open: dict[str, tuple[str, float]] = {}
            self._seen: set[str] = set()
            self.recovered: list[str] = []
            self._load()
        except BaseException:
            self.close()
            raise

    # ---------------------------------------------------------------- ownership
    def _acquire(self, takeover: bool) -> None:
        payload = json.dumps({"pid": os.getpid(), "since": time.time()})
        try:
            with self.lock_path.open("x", encoding="utf8") as fh:
                fh.write(payload)
        except FileExistsError as exc:
            info: dict[str, Any] = {}
            try:
                info = json.loads(self.lock_path.read_text(encoding="utf8"))
            except (OSError, ValueError):
                pass
            pid = info.get("pid")
            alive = _pid_alive(pid) if isinstance(pid, int) else None
            if not takeover:
                raise BudgetLedgerError(f"ledger {self.path.name} is locked by pid {pid} (process alive: {alive}); a single worker may own it; "
                                        "confirm that process is gone and pass takeover=True to reuse the ledger") from exc
            if alive and pid != os.getpid():
                raise BudgetLedgerError(f"takeover refused: pid {pid} is still running") from exc
            self.lock_path.write_text(payload, encoding="utf8")
        self._owner = True

    def close(self) -> None:
        if self._owner:
            self.lock_path.unlink(missing_ok=True)
            self._owner = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- journal
    def _append(self, entry: dict[str, Any]) -> None:
        line = json.dumps({"v": VERSION, **entry, "ts": time.time()}, sort_keys=True) + "\n"
        try:
            with self._io, self.path.open("a", encoding="utf8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise BudgetLedgerError(f"ledger write failed: {exc}") from exc

    @staticmethod
    def _money(value: Any, what: str, line: int) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise BudgetLedgerError(f"ledger line {line}: {what} is not a finite non-negative number")
        return float(value)

    def _load(self) -> None:
        if not self.path.exists():
            self.path.touch()
            return
        for n, raw in enumerate(self.path.read_text(encoding="utf8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                e = json.loads(raw)
            except ValueError as exc:
                raise BudgetLedgerError(f"ledger line {n} is unreadable (file left untouched)") from exc
            if not isinstance(e, dict) or e.get("v") != VERSION or not isinstance(e.get("tenant"), str) or not e["tenant"].strip():
                raise BudgetLedgerError(f"ledger line {n}: unexpected shape")
            op, tenant = e.get("op"), e["tenant"]
            if op == "reserve":
                rid, usd = e.get("id"), self._money(e.get("usd"), "usd", n)
                if not isinstance(rid, str) or rid in self._seen:
                    raise BudgetLedgerError(f"ledger line {n}: duplicate or invalid reservation id")
                self._seen.add(rid)
                self._open[rid] = (tenant, usd)
            elif op == "settle":
                rid, charged = e.get("id"), self._money(e.get("charged"), "charged", n)
                if rid not in self._open or self._open[rid][0] != tenant:
                    raise BudgetLedgerError(f"ledger line {n}: settlement of an unknown, foreign or already settled reservation")
                reserved = self._open.pop(rid)[1]
                self._spent[tenant] = self._spent.get(tenant, 0.0) + charged
                if charged > reserved + 1e-12:
                    self._breaches[tenant] = self._breaches.get(tenant, 0) + 1
            elif op == "charge":
                self._spent[tenant] = self._spent.get(tenant, 0.0) + self._money(e.get("usd"), "usd", n)
            else:
                raise BudgetLedgerError(f"ledger line {n}: unknown operation {op!r}")
        for rid, (tenant, usd) in sorted(self._open.items()):  # crashed in flight: charged at the bound, durably
            self._append({"op": "settle", "id": rid, "tenant": tenant, "charged": usd, "reserved": usd, "recovered": True})
            self._spent[tenant] = self._spent.get(tenant, 0.0) + usd
            self.recovered.append(rid)
        self._open.clear()

    # ---------------------------------------------------------------- operations used by the budgets
    def spent(self, tenant: str) -> float:
        return self._spent.get(tenant, 0.0)

    def breaches(self, tenant: str) -> int:
        return self._breaches.get(tenant, 0)

    def record_reserve(self, tenant: str, usd: float) -> str:
        rid = f"{tenant}:{time.time_ns()}:{len(self._seen)}"
        self._append({"op": "reserve", "id": rid, "tenant": tenant, "usd": usd})
        self._seen.add(rid)
        self._open[rid] = (tenant, usd)
        return rid

    def record_settle(self, rid: str, tenant: str, charged: float, reserved: float) -> None:
        self._append({"op": "settle", "id": rid, "tenant": tenant, "charged": charged, "reserved": reserved})
        self._open.pop(rid, None)
        self._spent[tenant] = self._spent.get(tenant, 0.0) + charged
        if charged > reserved + 1e-12:
            self._breaches[tenant] = self._breaches.get(tenant, 0) + 1

    def record_charge(self, tenant: str, usd: float) -> None:
        self._append({"op": "charge", "tenant": tenant, "usd": usd})
        self._spent[tenant] = self._spent.get(tenant, 0.0) + usd

    def budget(self, tenant: str, limit_usd: float) -> DurableTenantBudget:
        return DurableTenantBudget(self, tenant, limit_usd)


class DurableReservation(BudgetReservation):
    def __init__(self, budget: DurableTenantBudget, amount: float, rid: str) -> None:
        super().__init__(budget, amount)
        self._rid = rid
        self._journaled = False

    def settle(self, actual_usd: float | None = None) -> bool:
        actual = self.amount_usd if actual_usd is None else _valid_money(actual_usd, "actual_usd")
        budget: DurableTenantBudget = self._budget  # type: ignore[assignment]
        with budget._lock:
            if self._settled or self._journaled:
                raise RuntimeError("budget reservation already settled")
            budget.ledger.record_settle(self._rid, budget.tenant, actual, self.amount_usd)  # raises first: memory is unchanged if the write fails
            self._journaled = True
        return super().settle(actual_usd)


class DurableTenantBudget(TenantBudget):
    def __init__(self, ledger: DurableBudgetLedger, tenant: str, limit_usd: float) -> None:
        super().__init__(limit_usd=limit_usd, spent_usd=ledger.spent(tenant), reserved_usd=0.0, reservation_breaches=ledger.breaches(tenant))
        self.ledger, self.tenant = ledger, tenant

    def reserve(self, amount: float) -> DurableReservation | None:  # type: ignore[override]
        amount = _valid_money(amount, "amount")
        with self._lock:
            if amount > self.limit_usd - self.spent_usd - self.reserved_usd + 1e-12:
                return None
            rid = self.ledger.record_reserve(self.tenant, amount)  # durable BEFORE the claim exists; a write failure raises and admits nothing
            self.reserved_usd += amount
            return DurableReservation(self, amount, rid)

    def charge(self, amount: float) -> bool:
        amount = _valid_money(amount, "amount")
        with self._lock:
            if amount > self.limit_usd - self.spent_usd - self.reserved_usd + 1e-12:
                return False
            self.ledger.record_charge(self.tenant, amount)
            self.spent_usd += amount
            return True
