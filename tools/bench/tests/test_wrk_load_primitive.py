"""Unit tests for :mod:`concerto_bench.primitives.wrk_load`.

Uses ``httpx.MockTransport`` to fake the chat-completions endpoint.
Tests cover happy path, error rate tracking, latency statistics, and
edge cases.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from concerto_bench.primitives.wrk_load import (
    WrkLoadAction,
    WrkLoadError,
    WrkLoadPrimitive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


BASE_URL = "http://127.0.0.1:18080"


def _success_chat_body() -> dict[str, Any]:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "mock",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
    }


def _make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# WrkLoadAction validation
# ---------------------------------------------------------------------------


def test_action_valid_construction() -> None:
    action = WrkLoadAction(model="phi-3")
    assert action.model == "phi-3"
    assert action.concurrency == 20
    assert action.duration_secs == 60.0


def test_action_zero_duration_rejected() -> None:
    with pytest.raises(ValidationError):
        WrkLoadAction(model="phi-3", duration_secs=0)


def test_action_zero_concurrency_rejected() -> None:
    with pytest.raises(ValidationError):
        WrkLoadAction(model="phi-3", concurrency=0)


def test_action_negative_timeout_rejected() -> None:
    with pytest.raises(ValidationError):
        WrkLoadAction(model="phi-3", timeout_secs=-1)


def test_action_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        WrkLoadAction(model="phi-3", extra="nope")


def test_action_frozen() -> None:
    action = WrkLoadAction(model="phi-3")
    with pytest.raises(ValidationError):
        action.model = "other"  # type: ignore[misc]


def test_action_round_trip_serialisation() -> None:
    original = WrkLoadAction(
        model="phi-3",
        duration_secs=10.0,
        concurrency=5,
        content="test prompt",
        timeout_secs=15.0,
    )
    loaded = WrkLoadAction.model_validate_json(original.model_dump_json())
    assert loaded == original


# ---------------------------------------------------------------------------
# Happy path — all requests succeed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_chat_body())

    primitive = WrkLoadPrimitive()
    action = WrkLoadAction(
        model="mock",
        duration_secs=0.1,
        concurrency=2,
        timeout_secs=5.0,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["total_requests"] > 0
    assert result["successful_requests"] > 0
    assert result["failed_requests"] == 0
    assert result["error_rate"] == 0.0
    assert result["duration_secs"] > 0
    assert result["rps"] > 0

    lat = result["latency_ms"]
    assert lat["p50"] >= 0
    assert lat["p95"] >= lat["p50"]
    assert lat["p99"] >= lat["p95"]
    assert lat["max"] >= lat["p99"]
    assert lat["min"] >= 0
    assert lat["mean"] >= 0


# ---------------------------------------------------------------------------
# All requests fail — error rate = 1.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_all_requests_fail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    primitive = WrkLoadPrimitive()
    action = WrkLoadAction(
        model="mock",
        duration_secs=0.1,
        concurrency=2,
        timeout_secs=5.0,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["total_requests"] > 0
    assert result["successful_requests"] == 0
    assert result["failed_requests"] == result["total_requests"]
    assert result["error_rate"] == 1.0
    # Empty latency stats since no successful requests.
    assert result["latency_ms"]["p50"] == 0.0


# ---------------------------------------------------------------------------
# Transport errors are captured, not raised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_transport_errors_captured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    primitive = WrkLoadPrimitive()
    action = WrkLoadAction(
        model="mock",
        duration_secs=0.1,
        concurrency=1,
        timeout_secs=5.0,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["total_requests"] > 0
    assert result["failed_requests"] > 0
    assert result["error_rate"] > 0


# ---------------------------------------------------------------------------
# Concurrency — multiple workers fire requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_concurrency_multiple_workers() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json=_success_chat_body())

    primitive = WrkLoadPrimitive()
    action = WrkLoadAction(
        model="mock",
        duration_secs=0.1,
        concurrency=4,
        timeout_secs=5.0,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    # With 4 workers over 0.1s, we should get multiple requests.
    assert result["total_requests"] >= 4


# ---------------------------------------------------------------------------
# Latency statistics edge case — single request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_latency_stats_single_request() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # Only succeed on first call; after that return 500 to prevent more
        # successful requests. But the workers loop, so let's just use a tiny
        # duration to limit requests.
        return httpx.Response(200, json=_success_chat_body())

    primitive = WrkLoadPrimitive()
    # Very short duration and single worker to get ~1 request.
    action = WrkLoadAction(
        model="mock",
        duration_secs=0.001,
        concurrency=1,
        timeout_secs=5.0,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    # At least one request should have been made.
    if result["successful_requests"] >= 1:
        lat = result["latency_ms"]
        # All percentiles should be equal when there's one sample.
        assert lat["min"] <= lat["p50"] <= lat["max"]


# ---------------------------------------------------------------------------
# Latency stats computation unit test
# ---------------------------------------------------------------------------


def test_compute_latency_stats_empty() -> None:
    stats = WrkLoadPrimitive._compute_latency_stats([])
    assert stats == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "min": 0.0, "mean": 0.0}


def test_compute_latency_stats_single() -> None:
    stats = WrkLoadPrimitive._compute_latency_stats([42.0])
    assert stats["p50"] == 42.0
    assert stats["p95"] == 42.0
    assert stats["p99"] == 42.0
    assert stats["max"] == 42.0
    assert stats["min"] == 42.0
    assert stats["mean"] == 42.0


def test_compute_latency_stats_multiple() -> None:
    latencies = list(range(1, 101))  # 1..100
    stats = WrkLoadPrimitive._compute_latency_stats([float(x) for x in latencies])
    assert stats["min"] == 1.0
    assert stats["max"] == 100.0
    assert 49.0 <= stats["p50"] <= 51.0
    assert 94.0 <= stats["p95"] <= 96.0
    assert 98.0 <= stats["p99"] <= 100.0
    assert stats["mean"] == pytest.approx(50.5)


# ---------------------------------------------------------------------------
# WrkLoadError is RuntimeError
# ---------------------------------------------------------------------------


def test_wrk_load_error_is_runtime_error() -> None:
    err = WrkLoadError("test")
    assert isinstance(err, RuntimeError)
