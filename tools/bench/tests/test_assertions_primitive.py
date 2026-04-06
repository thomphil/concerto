"""Unit tests for :mod:`concerto_bench.primitives.assertions`.

Uses ``httpx.MockTransport`` to fake concerto's ``/status`` endpoint.
Tests cover all seven assertion types, field path navigation, and
transport error handling.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from concerto_bench.primitives.assertions import (
    AssertAction,
    AssertError,
    AssertPrimitive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


BASE_URL = "http://127.0.0.1:18080"


def _make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _status_body(backends: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "backends": backends or [],
        "gpus": [
            {"id": 0, "memory_used": 12000, "loaded_models": []},
        ],
        "registry_size": 1,
    }


# ---------------------------------------------------------------------------
# AssertAction validation
# ---------------------------------------------------------------------------


def test_action_valid_construction() -> None:
    action = AssertAction(assert_type="status_code")
    assert action.assert_type == "status_code"


def test_action_invalid_assert_type_rejected() -> None:
    with pytest.raises(ValidationError):
        AssertAction(assert_type="bogus")


def test_action_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        AssertAction(assert_type="status_code", extra="nope")


def test_action_frozen() -> None:
    action = AssertAction(assert_type="status_code")
    with pytest.raises(ValidationError):
        action.assert_type = "model_loaded"  # type: ignore[misc]


def test_action_round_trip_serialisation() -> None:
    original = AssertAction(
        assert_type="field_equals",
        field_path="backends.0.status",
        expected="loaded",
        message="custom msg",
    )
    loaded = AssertAction.model_validate_json(original.model_dump_json())
    assert loaded == original


# ---------------------------------------------------------------------------
# status_code assertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_code_pass() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="status_code")
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is True
    assert result["actual"] == 200
    assert result["expected"] == 200


@pytest.mark.asyncio
async def test_status_code_fail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="status_code")
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is False
    assert result["actual"] == 503
    assert "503" in result["message"]


# ---------------------------------------------------------------------------
# model_loaded assertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_loaded_pass() -> None:
    backends = [{"model": "phi-3", "status": "loaded"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body(backends))

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="model_loaded", model="phi-3")
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is True


@pytest.mark.asyncio
async def test_model_loaded_fail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body([]))

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="model_loaded", model="phi-3")
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is False
    assert "phi-3" in result["message"]


# ---------------------------------------------------------------------------
# model_not_loaded assertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_not_loaded_pass() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body([]))

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="model_not_loaded", model="phi-3")
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is True


@pytest.mark.asyncio
async def test_model_not_loaded_fail() -> None:
    backends = [{"model": "phi-3", "status": "loaded"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body(backends))

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="model_not_loaded", model="phi-3")
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is False
    assert "still loaded" in result["message"]


# ---------------------------------------------------------------------------
# backend_count assertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_count_pass() -> None:
    backends = [
        {"model": "a", "status": "loaded"},
        {"model": "b", "status": "loaded"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body(backends))

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="backend_count", expected=2)
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is True
    assert result["actual"] == 2


@pytest.mark.asyncio
async def test_backend_count_fail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body([]))

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="backend_count", expected=3)
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is False
    assert result["actual"] == 0
    assert result["expected"] == 3


# ---------------------------------------------------------------------------
# field_equals assertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_field_equals_pass() -> None:
    backends = [{"model": "phi-3", "status": "loaded"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body(backends))

    primitive = AssertPrimitive()
    action = AssertAction(
        assert_type="field_equals",
        field_path="backends.0.status",
        expected="loaded",
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is True
    assert result["actual"] == "loaded"


@pytest.mark.asyncio
async def test_field_equals_fail() -> None:
    backends = [{"model": "phi-3", "status": "loading"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body(backends))

    primitive = AssertPrimitive()
    action = AssertAction(
        assert_type="field_equals",
        field_path="backends.0.status",
        expected="loaded",
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is False


# ---------------------------------------------------------------------------
# field_gte / field_lte assertions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_field_gte_pass() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    primitive = AssertPrimitive()
    action = AssertAction(
        assert_type="field_gte",
        field_path="registry_size",
        expected=1,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is True
    assert result["actual"] == 1


@pytest.mark.asyncio
async def test_field_lte_pass() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    primitive = AssertPrimitive()
    action = AssertAction(
        assert_type="field_lte",
        field_path="registry_size",
        expected=5,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is True


@pytest.mark.asyncio
async def test_field_gte_fail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    primitive = AssertPrimitive()
    action = AssertAction(
        assert_type="field_gte",
        field_path="registry_size",
        expected=10,
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is False


# ---------------------------------------------------------------------------
# field_path not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_field_path_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    primitive = AssertPrimitive()
    action = AssertAction(
        assert_type="field_equals",
        field_path="nonexistent.deep.path",
        expected="anything",
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is False
    assert "not found" in result["message"]


# ---------------------------------------------------------------------------
# Missing field_path for field_* assertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_field_equals_without_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="field_equals", expected="loaded")
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is False
    assert "field_path is required" in result["message"]


# ---------------------------------------------------------------------------
# Custom message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_message_on_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body([]))

    primitive = AssertPrimitive()
    action = AssertAction(
        assert_type="backend_count",
        expected=5,
        message="expected five backends for this test",
    )
    async with _make_client(handler) as client:
        result = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert result["passed"] is False
    assert result["message"] == "expected five backends for this test"


# ---------------------------------------------------------------------------
# Transport error raises AssertError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_error_raises_assert_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    primitive = AssertPrimitive()
    action = AssertAction(assert_type="status_code")
    async with _make_client(handler) as client:
        with pytest.raises(AssertError):
            await primitive.execute(action, base_url=BASE_URL, client=client)


# ---------------------------------------------------------------------------
# AssertError is RuntimeError
# ---------------------------------------------------------------------------


def test_assert_error_is_runtime_error() -> None:
    err = AssertError("test")
    assert isinstance(err, RuntimeError)
