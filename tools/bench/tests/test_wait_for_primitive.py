"""Unit tests for :mod:`concerto_bench.primitives.wait_for`.

Uses ``httpx.MockTransport`` to fake concerto's ``/status`` endpoint.
Tests cover the three condition types, timeout behaviour, and transport
error resilience.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from concerto_bench.primitives.wait_for import (
    WaitForAction,
    WaitForError,
    WaitForPrimitive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


BASE_URL = "http://127.0.0.1:18080"


def _make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _status_with_backends(backends: list[dict[str, Any]]) -> dict[str, Any]:
    return {"backends": backends, "gpus": []}


def _loaded_backend(model: str) -> dict[str, Any]:
    return {"model": model, "status": "loaded"}


# ---------------------------------------------------------------------------
# WaitForAction validation
# ---------------------------------------------------------------------------


def test_action_valid_construction() -> None:
    action = WaitForAction(condition="model_loaded", model="phi-3")
    assert action.condition == "model_loaded"
    assert action.model == "phi-3"


def test_action_invalid_condition_rejected() -> None:
    with pytest.raises(ValidationError):
        WaitForAction(condition="bogus")


def test_action_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        WaitForAction(condition="model_loaded", extra="nope")


def test_action_timeout_non_positive_rejected() -> None:
    with pytest.raises(ValidationError):
        WaitForAction(condition="model_loaded", timeout_secs=0)


def test_action_poll_interval_non_positive_rejected() -> None:
    with pytest.raises(ValidationError):
        WaitForAction(condition="model_loaded", poll_interval_secs=-1)


def test_action_frozen() -> None:
    action = WaitForAction(condition="model_loaded")
    with pytest.raises(ValidationError):
        action.condition = "backend_count"  # type: ignore[misc]


def test_action_round_trip_serialisation() -> None:
    original = WaitForAction(
        condition="backend_count",
        expected_count=2,
        timeout_secs=30.0,
        poll_interval_secs=1.0,
    )
    loaded = WaitForAction.model_validate_json(original.model_dump_json())
    assert loaded == original


# ---------------------------------------------------------------------------
# model_loaded — immediately satisfied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_loaded_satisfied_immediately() -> None:
    status = _status_with_backends([_loaded_backend("phi-3")])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=status)

    primitive = WaitForPrimitive()
    action = WaitForAction(
        condition="model_loaded",
        model="phi-3",
        timeout_secs=1.0,
        poll_interval_secs=0.01,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["satisfied"] is True
    assert result["polls"] == 1
    assert result["elapsed_secs"] < 1.0
    assert result["final_status"] == status


# ---------------------------------------------------------------------------
# model_not_loaded — immediately satisfied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_not_loaded_satisfied_when_absent() -> None:
    status = _status_with_backends([])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=status)

    primitive = WaitForPrimitive()
    action = WaitForAction(
        condition="model_not_loaded",
        model="phi-3",
        timeout_secs=1.0,
        poll_interval_secs=0.01,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["satisfied"] is True
    assert result["polls"] == 1


# ---------------------------------------------------------------------------
# backend_count — immediately satisfied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_count_satisfied() -> None:
    status = _status_with_backends([
        _loaded_backend("a"),
        _loaded_backend("b"),
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=status)

    primitive = WaitForPrimitive()
    action = WaitForAction(
        condition="backend_count",
        expected_count=2,
        timeout_secs=1.0,
        poll_interval_secs=0.01,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["satisfied"] is True
    assert result["polls"] == 1


# ---------------------------------------------------------------------------
# Timeout — condition never satisfied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_returns_satisfied_false() -> None:
    status = _status_with_backends([])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=status)

    primitive = WaitForPrimitive()
    action = WaitForAction(
        condition="model_loaded",
        model="never-appears",
        timeout_secs=0.1,
        poll_interval_secs=0.02,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["satisfied"] is False
    assert result["polls"] >= 1
    assert result["elapsed_secs"] >= 0.09  # Allow slight timing slack


# ---------------------------------------------------------------------------
# Condition becomes true after a few polls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_condition_becomes_true_after_polls() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            return httpx.Response(200, json=_status_with_backends([_loaded_backend("phi-3")]))
        return httpx.Response(200, json=_status_with_backends([]))

    primitive = WaitForPrimitive()
    action = WaitForAction(
        condition="model_loaded",
        model="phi-3",
        timeout_secs=2.0,
        poll_interval_secs=0.01,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["satisfied"] is True
    assert result["polls"] >= 3


# ---------------------------------------------------------------------------
# Transport error resilience — retries on failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_error_retries_then_succeeds() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=_status_with_backends([_loaded_backend("phi-3")]))

    primitive = WaitForPrimitive()
    action = WaitForAction(
        condition="model_loaded",
        model="phi-3",
        timeout_secs=2.0,
        poll_interval_secs=0.01,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["satisfied"] is True
    assert result["polls"] >= 3


# ---------------------------------------------------------------------------
# model_loaded with "running" status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_loaded_with_running_status() -> None:
    status = _status_with_backends([{"model": "phi-3", "status": "running"}])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=status)

    primitive = WaitForPrimitive()
    action = WaitForAction(
        condition="model_loaded",
        model="phi-3",
        timeout_secs=1.0,
        poll_interval_secs=0.01,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["satisfied"] is True


# ---------------------------------------------------------------------------
# WaitForError exists (interface completeness)
# ---------------------------------------------------------------------------


def test_wait_for_error_is_runtime_error() -> None:
    err = WaitForError("test")
    assert isinstance(err, RuntimeError)
