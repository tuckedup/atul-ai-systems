"""Normalized backend contract used by the RouteBench gateway."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

MockResponse = dict[str, Any] | Iterable[dict[str, Any] | str]
MockTransport = Callable[[dict[str, Any], bool], MockResponse]


class BackendHTTPError(RuntimeError):
    """Provider-style error carrying the upstream HTTP status code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.status_code == 429 or self.status_code >= 500


@dataclass(frozen=True)
class BackendResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    attempts: int
    chunks: tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BackendAdapter(Protocol):
    name: str
    model: str

    def chat(self, payload: dict[str, Any], *, stream: bool = False) -> BackendResult: ...


class RetryingMockAdapter:
    """Base class for deterministic provider-shape adapters.

    The injected transport is the only I/O boundary. Tests can return native
    provider dictionaries or raise ``BackendHTTPError`` without network access.
    """

    name = "mock"

    def __init__(
        self,
        model: str,
        transport: MockTransport,
        *,
        input_per_1m: float,
        output_per_1m: float,
        max_retries: int = 2,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.model = model
        self.transport = transport
        self.input_per_1m = input_per_1m
        self.output_per_1m = output_per_1m
        self.max_retries = max_retries

    def _request(self, payload: dict[str, Any], stream: bool) -> tuple[MockResponse, int]:
        attempts = 0
        while True:
            attempts += 1
            try:
                return self.transport(payload, stream), attempts
            except BackendHTTPError as error:
                if not error.retryable or attempts > self.max_retries:
                    raise
            except (ConnectionError, TimeoutError) as error:
                if attempts > self.max_retries:
                    raise BackendHTTPError(503, str(error)) from error

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_1m
            + completion_tokens * self.output_per_1m
        ) / 1_000_000
