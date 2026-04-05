"""Unit tests for :class:`ConcertoStatusSampler`.

``httpx.MockTransport`` is injected via the sampler's ``client_factory``
kwarg so no real concerto process is needed. Each test drives the
sampler for a handful of ticks via the base-class loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from concerto_bench.samplers.base import SamplerResult
from concerto_bench.samplers.concerto_status import (
    ConcertoStatusSampler,
    ConcertoStatusSamplerConfig,
)
from concerto_bench.schema import TelemetrySample


BASE_URL = "http://127.0.0.1:18080"


def _config(tmp_path: Path, *, interval: float = 0.05) -> ConcertoStatusSamplerConfig:
    return ConcertoStatusSamplerConfig(
        name="concerto-status",
        interval_secs=interval,
        output_path=tmp_path / "concerto-status.jsonl",
        base_url=BASE_URL,
    )


def _make_factory(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[], httpx.AsyncClient]:
    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


def _read_rows(path: Path) -> list[TelemetrySample]:
    rows: list[TelemetrySample] = []
    with path.open("rb") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(TelemetrySample.model_validate_json(stripped))
    return rows


async def _drive(sampler: ConcertoStatusSampler, secs: float) -> SamplerResult:
    task = asyncio.create_task(sampler.run())
    try:
        await asyncio.sleep(secs)
    finally:
        task.cancel()
    return await task


def _status_body() -> dict[str, Any]:
    return {
        "gpus": [
            {
                "id": 0,
                "memory_total": "24000",
                "memory_used": "8000",
                "memory_available": "16000",
                "temperature_celsius": 55.0,
                "utilisation_percent": 42.0,
                "health": "Healthy",
                "loaded_models": [],
            }
        ],
        "registry_size": 2,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_records_status_bodies(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_status_body())

    config = _config(tmp_path)
    sampler = ConcertoStatusSampler(config, client_factory=_make_factory(handler))
    result = await _drive(sampler, 0.18)

    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(config.output_path)
    assert len(rows) == result.ticks_succeeded
    for row in rows:
        assert row.sampler == "concerto-status"
        assert row.values == _status_body()
    # Every tick hit /status on the configured base URL.
    for request in calls:
        assert request.url.path == "/status"
        assert str(request.url).startswith(BASE_URL)


# ---------------------------------------------------------------------------
# Transport and HTTP failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_error_counts_as_tick_failure(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    config = _config(tmp_path)
    sampler = ConcertoStatusSampler(config, client_factory=_make_factory(handler))
    result = await _drive(sampler, 0.15)

    assert result.ticks_succeeded == 0
    assert result.ticks_failed >= 1
    assert all("ConnectError" in reason for reason in result.failures)
    # No JSONL rows on disk.
    assert _read_rows(config.output_path) == []


@pytest.mark.asyncio
async def test_non_2xx_response_counts_as_tick_failure(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    config = _config(tmp_path)
    sampler = ConcertoStatusSampler(config, client_factory=_make_factory(handler))
    result = await _drive(sampler, 0.15)

    assert result.ticks_succeeded == 0
    assert result.ticks_failed >= 1
    # HTTPStatusError from raise_for_status
    assert any("HTTPStatusError" in reason or "Status" in reason for reason in result.failures)


@pytest.mark.asyncio
async def test_non_object_json_counts_as_tick_failure(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    config = _config(tmp_path)
    sampler = ConcertoStatusSampler(config, client_factory=_make_factory(handler))
    result = await _drive(sampler, 0.12)

    assert result.ticks_succeeded == 0
    assert result.ticks_failed >= 1


# ---------------------------------------------------------------------------
# Teardown closes the injected client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_teardown_closes_injected_client(tmp_path: Path) -> None:
    closed_clients: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    def factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        closed_clients.append(client)
        return client

    config = _config(tmp_path)
    sampler = ConcertoStatusSampler(config, client_factory=factory)
    await _drive(sampler, 0.12)

    assert len(closed_clients) == 1
    # Calling aclose() a second time is a no-op; asserting is_closed
    # confirms the sampler closed the client during teardown.
    assert closed_clients[0].is_closed


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_requires_base_url(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ConcertoStatusSamplerConfig(
            name="concerto-status",
            interval_secs=1.0,
            output_path=tmp_path / "x.jsonl",
        )
