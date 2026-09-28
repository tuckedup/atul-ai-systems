"""Shared HTTP helpers for evalops fetch scripts.

Not a fetch script itself -- just retry/backoff plumbing used by
fetch_*.py so each one doesn't reimplement it. All network access goes
through `get_json` / `get_bytes`, which retry on 429/5xx with backoff
since the HF datasets-server endpoint rate-limits aggressively.
"""
from __future__ import annotations

import time

import httpx

DEFAULT_HEADERS = {"User-Agent": "routebench-evalops-fetch/1.0"}


def _sleep_for_attempt(attempt: int, resp: httpx.Response | None) -> None:
    wait = None
    if resp is not None:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                wait = float(ra)
            except ValueError:
                wait = None
    if wait is None:
        wait = min(60.0, 3.0 * (attempt + 1))
    time.sleep(wait)


def get_json(url: str, params: dict | None = None, max_attempts: int = 10, timeout: float = 60.0):
    """GET a URL and parse JSON, retrying on 429 / transient errors."""
    last_exc: Exception | None = None
    with httpx.Client(timeout=timeout, headers=DEFAULT_HEADERS, follow_redirects=True) as client:
        for attempt in range(max_attempts):
            try:
                r = client.get(url, params=params)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                _sleep_for_attempt(attempt, None)
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception as e:  # noqa: BLE001 - a bad-body response is retried, not fatal
                    last_exc = e
                    _sleep_for_attempt(attempt, r)
                    continue
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = RuntimeError(f"HTTP {r.status_code} for {url}")
                _sleep_for_attempt(attempt, r)
                continue
            # Non-retryable status (401/403/404/...): raise immediately with body.
            raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:500]}")
    raise RuntimeError(f"Exhausted retries for {url}: {last_exc}")


def get_bytes(url: str, max_attempts: int = 10, timeout: float = 120.0) -> bytes:
    last_exc: Exception | None = None
    with httpx.Client(timeout=timeout, headers=DEFAULT_HEADERS, follow_redirects=True) as client:
        for attempt in range(max_attempts):
            try:
                r = client.get(url)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                _sleep_for_attempt(attempt, None)
                continue
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = RuntimeError(f"HTTP {r.status_code} for {url}")
                _sleep_for_attempt(attempt, r)
                continue
            raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:500]}")
    raise RuntimeError(f"Exhausted retries for {url}: {last_exc}")


def rows_endpoint(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    return get_json(url, params=params)


def iter_rows(dataset: str, config: str, split: str, total: int | None, page_size: int = 100,
              sleep_between: float = 0.8):
    """Yield row dicts (the inner `row` field) from the datasets-server rows endpoint.

    If `total` is None, keeps paging until a page comes back empty.
    """
    offset = 0
    while total is None or offset < total:
        length = page_size if total is None else min(page_size, total - offset)
        data = rows_endpoint(dataset, config, split, offset, length)
        rows = data.get("rows", [])
        if not rows:
            break
        for r in rows:
            yield r["row"]
        offset += len(rows)
        if len(rows) < length:
            break
        time.sleep(sleep_between)
