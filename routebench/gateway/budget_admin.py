"""Operator actions on the durable tenant budgets (the configured database, `settings.database_url`). Every state change needs a named operator and a reason and is recorded in `budget_events`.

  python -m gateway.budget_admin unresolved
  python -m gateway.budget_admin snapshot --tenant acme
  python -m gateway.budget_admin resolve --reservation <id> --operator alice --reason "process died; confirmed no provider billing" [--charge-usd 0.0004]
  python -m gateway.budget_admin set-limit --tenant acme --limit 50 --operator alice --reason "monthly cap raised"

`resolve` without --charge-usd charges the reservation's full bound (nothing is refunded). A smaller --charge-usd is an explicit refund decision of the difference, made by the operator.
Nothing here calls a provider.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import Any

from .budget_store import BudgetStateError, BudgetStore
from .state import ModelState


def _default_state() -> ModelState:
    from aisys.settings import settings

    return ModelState(settings.database_url)


def main(argv: Sequence[str] | None = None, state_factory: Callable[[], ModelState] = _default_state) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("unresolved")
    s = sub.add_parser("snapshot")
    s.add_argument("--tenant", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--reservation", required=True)
    r.add_argument("--operator", required=True)
    r.add_argument("--reason", required=True)
    r.add_argument("--charge-usd", type=float, default=None)
    sl = sub.add_parser("set-limit")
    sl.add_argument("--tenant", required=True)
    sl.add_argument("--limit", type=float, required=True)
    sl.add_argument("--operator", required=True)
    sl.add_argument("--reason", required=True)
    a = p.parse_args(argv)
    try:
        store = BudgetStore(state_factory())
        out: Any
        if a.command == "unresolved":
            out = store.unresolved()
        elif a.command == "snapshot":
            out = store.snapshot(a.tenant)
        elif a.command == "resolve":
            out = store.resolve(a.reservation, operator=a.operator, reason=a.reason, charge_usd=a.charge_usd)
        else:
            store.set_limit(a.tenant, a.limit, operator=a.operator, reason=a.reason)
            out = store.snapshot(a.tenant)
        print(json.dumps(out, indent=2))
        return 0
    except (BudgetStateError, OSError, ValueError, RuntimeError) as exc:  # a database that cannot even be opened is a refusal too
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
