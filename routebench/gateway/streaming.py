"""Native token streaming through the gateway (opt-in), with the same budget, circuit-breaker and accounting discipline as the buffered path.

Enabled by `Runtime.native_streaming` and only when every configured backend client implements `stream_chat`; otherwise the existing buffered completed-response delivery is used unchanged.

Spend safety (the hard part of streaming):
  * the call bound is RESERVED before the upstream request, exactly as in the buffered path;
  * the upstream request always asks for usage (`stream_options.include_usage`) so the final cost is known; the usage-only chunk is forwarded to the client only if the client asked for it;
  * normal end with valid usage settles the reservation at the usage-priced cost (an overshoot is recorded as a bound breach, never discarded; an HTTP 402 is impossible mid-stream);
  * a stream that ends without valid usage, fails midway, or is abandoned by the client settles at the reservation BOUND (conservative; never zero) and is recorded as an unknown-bound attempt;
  * failures BEFORE the first chunk reroute to the next backend like the buffered path; after the first chunk there is no rerouting (the client already holds partial output), the stream ends with an error event;
  * admission is released exactly once when the stream ends or is closed; streamed responses are never cached.
Time to first token is exported as `routebench_ttft_seconds`; it is a gateway-observed figure, not a provider benchmark.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException
from prometheus_client import Counter, Histogram

from . import app as gw
from .budget_store import BudgetStateError
from .reliability import BudgetReservation, TenantBudget
from .router import NoEligibleBackend, Request

TTFT = Histogram("routebench_ttft_seconds", "Gateway-observed time to first streamed chunk", ["backend"],
                 buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60))
STREAMS = Counter("routebench_streams_total", "Native streams by outcome", ["backend", "outcome"])


def native_streaming_available() -> bool:
    rt = gw.runtime
    return bool(getattr(rt, "native_streaming", False) and rt.clients and all(callable(getattr(c, "stream_chat", None)) for c in rt.clients.values()))


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def _valid_usage(usage: Any) -> tuple[int, int] | None:
    if not isinstance(usage, dict):
        return None
    p, c = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (p, c)) or p == 0:
        return None
    return p, c


@dataclass
class OpenStream:
    backend: str
    events: Callable[[Callable[[], None]], Iterator[str]]


def open_stream(body: gw.ChatRequest, tenant: str, budget: TenantBudget, request_id: str) -> OpenStream:
    """Route and reserve exactly like the buffered path, open the upstream stream and obtain its FIRST chunk (rerouting on failure before it). Raises HTTPException when nothing can serve it."""
    rt = gw.runtime
    prompt = "\n".join(message.content or "" for message in body.messages)
    classification = rt.classifier.classify(prompt)
    if not rt.router.backends:
        raise HTTPException(503, "no backends configured")
    if budget.remaining() <= 0:
        raise HTTPException(402, "tenant budget exhausted")
    forwarded = gw._forward_payload(body)
    forwarded["stream"] = True
    client_options = (body.model_extra or {}).get("stream_options") or {}
    include_usage_for_client = bool(client_options.get("include_usage"))
    forwarded["stream_options"] = {**client_options, "include_usage": True}
    prompt_tokens = max(gw._prompt_token_bound({**forwarded, "model": item.model}) for item in rt.router.backends)
    output_tokens = gw._output_cap(body)
    latency_sla = gw._metadata_number(body, "latency_sla_ms", None)
    quality_floor = gw._metadata_number(body, "quality_floor", 0.0)
    attempted: set[str] = set()
    last_error = "no available backend"
    while True:
        for metadata in rt.router.backends:
            metadata.circuit_open = rt.breakers[metadata.model].is_open
        try:
            decision = rt.router.route(Request(
                task_class=classification.task_class, difficulty=classification.difficulty, est_prompt_tokens=prompt_tokens, latency_sla_ms=latency_sla,
                tenant_budget_remaining_usd=budget.remaining(), quality_floor=quality_floor or 0.0, max_output_tokens=output_tokens,
                request_id=request_id, excluded_models=frozenset(attempted)))
        except NoEligibleBackend as error:
            if not attempted:
                affordable = False
                for item in rt.router.backends:
                    try:
                        affordable = gw._call_bound(item, prompt_tokens, output_tokens) <= budget.remaining() + 1e-12
                    except (TypeError, ValueError):
                        continue
                    if affordable:
                        break
                if not affordable:
                    raise HTTPException(402, "tenant budget exhausted") from error
            raise HTTPException(503, f"all backends failed: {last_error}" if attempted else "no eligible backend") from error
        if decision.model in attempted:
            raise HTTPException(503, f"all backends failed: {last_error}")
        attempted.add(decision.model)
        metadata = next((i for i in rt.router.backends if i.model == decision.model), None)
        client = rt.clients.get(decision.model)
        if metadata is None or client is None:
            last_error = f"backend {decision.model} is not configured"
            continue
        try:
            reserved_usd = gw._call_bound(metadata, prompt_tokens, output_tokens)
        except ValueError as error:
            last_error = str(error)
            continue
        reservation = budget.reserve(reserved_usd)
        if reservation is None:
            raise HTTPException(402, "tenant budget exhausted")
        payload = {**forwarded, "model": decision.model}
        started = time.perf_counter()
        upstream: Any = None
        try:
            upstream = client.stream_chat(payload)
            first = next(upstream)
        except StopIteration:
            _settle_unknown(reservation, decision.model)
            last_error = "empty upstream stream"
            rt.breakers[decision.model].failure()
            gw.REQUESTS.labels(decision.model, "error").inc()
            continue
        except httpx.HTTPStatusError as error:
            _settle_unknown(reservation, decision.model)
            gw.REQUESTS.labels(decision.model, "error").inc()
            status = error.response.status_code
            if 400 <= status < 500:
                raise HTTPException(status if status in {400, 404, 409, 413, 422, 429} else 400, "provider rejected request") from error
            last_error = f"provider HTTP {status}"
            rt.breakers[decision.model].failure()
            continue
        except Exception as error:  # noqa: BLE001 - a failure before the first chunk reroutes like the buffered path
            _settle_unknown(reservation, decision.model)
            last_error = str(error)
            rt.breakers[decision.model].failure()
            gw.REQUESTS.labels(decision.model, "error").inc()
            continue
        ttft = time.perf_counter() - started
        TTFT.labels(decision.model).observe(ttft)
        return OpenStream(decision.model, lambda release, _u=upstream, _f=first, _r=reservation, _m=metadata, _s=started: _events(
            _u, _f, _r, _m, request_id, _s, include_usage_for_client, release))


def _settle_unknown(reservation: BudgetReservation, model: str) -> None:
    charged = reservation.amount_usd
    reservation.settle()
    gw._record_unknown_attempt(model, charged)


def _events(upstream: Iterator[dict[str, Any]], first: dict[str, Any], reservation: BudgetReservation, metadata: Any, request_id: str,
            started: float, client_wants_usage: bool, release: Callable[[], None]) -> Iterator[str]:
    rt = gw.runtime
    model = metadata.model
    usage: Any = None
    prompt_seen = completion_seen = 0
    settled = False
    outcome = "abandoned"
    try:
        chunk: dict[str, Any] | None = first
        while chunk is not None:
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            out = dict(chunk)
            out["id"] = request_id
            if not client_wants_usage:
                out.pop("usage", None)
            usage_only = not out.get("choices")
            if not usage_only or (client_wants_usage and out.get("usage") is not None):
                yield _sse(out)
            chunk = next(upstream, None)
        valid = _valid_usage(usage)
        if valid is None:
            charge = None
        else:
            prompt_seen, completion_seen = valid
            charge = (prompt_seen * metadata.input_per_1m + completion_seen * metadata.output_per_1m) / 1e6
        reservation.settle(charge)
        settled = True
        accounted = reservation.amount_usd if charge is None else charge
        rt.router.usage.record(model, prompt_seen, completion_seen, accounted, cost_basis="unknown_bound" if charge is None else "usage_priced")
        rt.breakers[model].success()
        gw.REQUESTS.labels(model, "ok").inc()
        gw.LATENCY.labels(model).observe(time.perf_counter() - started)
        outcome = "ok" if charge is not None else "ok_usage_unknown"
        yield "data: [DONE]\n\n"
    except GeneratorExit:
        outcome = "client_closed"
        raise
    except Exception as exc:  # noqa: BLE001 - a mid-stream failure: no rerouting, conservative settlement, generic error event
        if isinstance(exc, BudgetStateError):
            outcome = "budget_state_error"  # the model is healthy; the funds stay held until an operator resolves them
        else:
            outcome = "error"
            rt.breakers[model].failure()
        gw.REQUESTS.labels(model, "error").inc()
        yield _sse({"error": {"message": "upstream stream failed", "type": "upstream_error"}})
        yield "data: [DONE]\n\n"
    finally:
        try:
            if not settled:
                try:
                    _settle_unknown(reservation, model)
                except BudgetStateError:
                    outcome = "budget_state_error"  # could not record even the bound: the reservation stays OPEN (held), never freed
        finally:  # whatever happened above, the upstream connection is closed and the admission slot is returned exactly once
            try:
                STREAMS.labels(model, outcome).inc()
                closer = getattr(upstream, "close", None)
                if callable(closer):
                    closer()
            finally:
                release()
