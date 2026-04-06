"""``wrk_load`` primitive: concurrent HTTP load generation (pure Python).

The bench rig's load-generation primitive. Scenarios that exercise
concerto's routing path under sustained concurrency use this primitive
to fire parallel chat-completion requests for a configurable duration,
then report per-request latency percentiles, throughput, and error
rates.

This is a pure-Python implementation using ``asyncio.gather`` — it
does **not** shell out to the ``wrk`` binary. The trade-off is lower
maximum throughput compared to native C ``wrk``, but the primitive
runs on any host without additional dependencies and its output is
structured JSON rather than text that needs parsing.

Shape at a glance
-----------------

* :class:`WrkLoadAction` — frozen pydantic model, ``extra="forbid"``.
  Configures duration, concurrency, model, prompt, and per-request
  timeout.
* :class:`WrkLoadError` — raised only on irrecoverable failures
  (zero concurrency, broken client).
* :class:`WrkLoadPrimitive` — stateless executor. Spawns ``concurrency``
  async workers that issue sequential requests for ``duration_secs``,
  then aggregates the results.

Latency tracking
----------------

Every individual request records its wall-clock latency in milliseconds.
The final result includes p50, p95, p99, max, min, and mean latencies
computed from the full distribution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WrkLoadError(RuntimeError):
    """Raised when the ``wrk_load`` primitive encounters an irrecoverable failure.

    Reserved for structurally invalid configurations (zero concurrency)
    or client-state failures. Individual request failures within the
    load test are *not* raised — they contribute to the error rate.
    """


# ---------------------------------------------------------------------------
# Action argument model
# ---------------------------------------------------------------------------


class WrkLoadAction(BaseModel):
    """Scenario YAML arguments for a single ``wrk_load`` action.

    Frozen so the runner can stash and reuse it across retries.

    Fields
    ------

    ``duration_secs``
        How long to sustain the load, in wall-clock seconds.
    ``concurrency``
        Number of concurrent async workers. Each worker sends
        sequential requests for the full duration.
    ``model``
        Model ID to target in the chat-completion request body.
    ``content``
        User message content for each request.
    ``timeout_secs``
        Per-request timeout in seconds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_secs: float = Field(
        default=60.0,
        description="Load test duration in seconds.",
    )
    concurrency: int = Field(
        default=20,
        description="Number of concurrent async workers.",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model ID to target in chat-completion requests.",
    )
    content: str = Field(
        default="Hello",
        description="User message content for each request.",
    )
    timeout_secs: float = Field(
        default=30.0,
        description="Per-request timeout in seconds.",
    )

    @field_validator("duration_secs")
    @classmethod
    def _validate_duration_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"duration_secs must be > 0, got {value}")
        return value

    @field_validator("concurrency")
    @classmethod
    def _validate_concurrency_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"concurrency must be > 0, got {value}")
        return value

    @field_validator("timeout_secs")
    @classmethod
    def _validate_timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"timeout_secs must be > 0, got {value}")
        return value


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


class WrkLoadPrimitive:
    """Stateless executor for :class:`WrkLoadAction`.

    Spawns ``concurrency`` async workers that issue sequential
    ``POST /v1/chat/completions`` requests for ``duration_secs``, then
    aggregates the latency and error metrics.

    Failure policy
    --------------

    * Individual request failures (timeout, connect error, transport
      error) are recorded as failed requests and contribute to the
      error rate. They do NOT raise.
    * A fully broken client or structurally invalid action raises
      :class:`WrkLoadError`.
    """

    _CHAT_PATH = "/v1/chat/completions"

    async def execute(
        self,
        action: WrkLoadAction,
        *,
        base_url: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """Run a load test and return aggregated metrics.

        Parameters
        ----------
        action:
            Frozen arguments for this invocation.
        base_url:
            Concerto's HTTP base URL, e.g. ``http://127.0.0.1:8000``.
        client:
            Optional pre-built :class:`httpx.AsyncClient`. When ``None``
            the primitive creates a short-lived internal client.

        Returns
        -------
        dict
            ``{"total_requests": int, "successful_requests": int,
            "failed_requests": int, "error_rate": float,
            "duration_secs": float, "rps": float,
            "latency_ms": {"p50": float, "p95": float, "p99": float,
            "max": float, "min": float, "mean": float}}``
        """
        owned_client = client is None
        active_client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(action.timeout_secs),
        )

        url = f"{base_url.rstrip('/')}{self._CHAT_PATH}"
        body = {
            "model": action.model,
            "messages": [{"role": "user", "content": action.content}],
            "stream": False,
        }

        # Shared state across workers. Lists are append-only so no lock needed.
        latencies_ms: list[float] = []
        failures: list[str] = []

        load_start = time.perf_counter()
        deadline = load_start + action.duration_secs

        async def worker(worker_id: int) -> None:
            """One concurrent worker: sends sequential requests until the deadline."""
            while time.perf_counter() < deadline:
                req_start = time.perf_counter()
                try:
                    response = await active_client.post(
                        url,
                        json=body,
                        timeout=httpx.Timeout(action.timeout_secs),
                    )
                    elapsed_ms = (time.perf_counter() - req_start) * 1000.0
                    if 200 <= response.status_code < 300:
                        latencies_ms.append(elapsed_ms)
                    else:
                        failures.append(
                            f"worker-{worker_id}: HTTP {response.status_code}"
                        )
                except Exception as exc:
                    failures.append(
                        f"worker-{worker_id}: {type(exc).__name__}: {exc}"
                    )

        try:
            workers = [worker(i) for i in range(action.concurrency)]
            await asyncio.gather(*workers)
        finally:
            if owned_client:
                await active_client.aclose()

        load_end = time.perf_counter()
        actual_duration = load_end - load_start

        total = len(latencies_ms) + len(failures)
        successful = len(latencies_ms)
        failed = len(failures)
        error_rate = failed / total if total > 0 else 0.0
        rps = total / actual_duration if actual_duration > 0 else 0.0

        latency_stats = self._compute_latency_stats(latencies_ms)

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "error_rate": error_rate,
            "duration_secs": actual_duration,
            "rps": rps,
            "latency_ms": latency_stats,
        }

    # ------------------------------------------------------------------
    # Latency statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_latency_stats(latencies: list[float]) -> dict[str, float]:
        """Compute p50/p95/p99/max/min/mean from a list of latency values."""
        if not latencies:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "min": 0.0,
                "mean": 0.0,
            }

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        mean = sum(sorted_lat) / n

        def _percentile(pct: float) -> float:
            """Nearest-rank percentile computation."""
            if n == 1:
                return sorted_lat[0]
            rank = (pct / 100.0) * (n - 1)
            lower = int(rank)
            upper = min(lower + 1, n - 1)
            frac = rank - lower
            return sorted_lat[lower] + frac * (sorted_lat[upper] - sorted_lat[lower])

        return {
            "p50": _percentile(50),
            "p95": _percentile(95),
            "p99": _percentile(99),
            "max": sorted_lat[-1],
            "min": sorted_lat[0],
            "mean": mean,
        }
