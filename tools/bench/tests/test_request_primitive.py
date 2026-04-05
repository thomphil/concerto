"""Unit tests for :mod:`concerto_bench.primitives.request`.

Every test uses :class:`httpx.MockTransport` so no real network IO
happens — the primitive contract (and in particular the
"transport failure becomes RequestRecord.error, not an exception"
partial-success principle) is validated end-to-end without a bound
socket anywhere.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from concerto_bench.primitives.request import (
    RequestAction,
    RequestError,
    RequestPrimitive,
)
from concerto_bench.schema import RequestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


BASE_URL = "http://127.0.0.1:18080"


def _success_chat_body(model: str = "mock-small") -> dict[str, Any]:
    """A realistic non-streaming chat-completion response body."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _sse_stream_bytes(chunks: list[dict[str, Any]]) -> bytes:
    """Encode a list of SSE event payloads into the raw bytes an axum stream would emit."""
    parts: list[bytes] = []
    for chunk in chunks:
        parts.append(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


def _make_client(handler) -> httpx.AsyncClient:
    """Build an AsyncClient bound to a mock transport handler."""
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


# ---------------------------------------------------------------------------
# RequestAction validation
# ---------------------------------------------------------------------------


def test_action_empty_content_rejected() -> None:
    with pytest.raises(ValidationError):
        RequestAction(model="mock-small", content="")


def test_action_whitespace_content_rejected() -> None:
    with pytest.raises(ValidationError):
        RequestAction(model="mock-small", content="   \n\t  ")


def test_action_timeout_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        RequestAction(model="mock-small", content="hi", timeout_secs=0)


def test_action_timeout_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        RequestAction(model="mock-small", content="hi", timeout_secs=-1.5)


def test_action_expect_status_out_of_range_low() -> None:
    with pytest.raises(ValidationError):
        RequestAction(model="mock-small", content="hi", expect_status=99)


def test_action_expect_status_out_of_range_high() -> None:
    with pytest.raises(ValidationError):
        RequestAction(model="mock-small", content="hi", expect_status=600)


def test_action_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        RequestAction(model="mock-small", content="hi", unknown_field="oops")


def test_action_round_trip_serialisation() -> None:
    original = RequestAction(
        model="mock-small",
        content="hi",
        stream=True,
        expect_status=200,
        timeout_secs=5.0,
        capture_as="first",
        max_tokens=16,
        temperature=0.2,
        system="you are a helpful assistant",
        extra_headers={"x-trace": "abc"},
    )
    loaded = RequestAction.model_validate_json(original.model_dump_json())
    assert loaded == original


# ---------------------------------------------------------------------------
# Happy paths — non-streaming and streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_non_streaming_happy_path() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_success_chat_body())

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(model="mock-small", content="hi", capture_as="first")
        record = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert isinstance(record, RequestRecord)
    assert record.status == 200
    assert record.error is None
    assert record.response_body is not None
    assert record.response_body["choices"][0]["message"]["content"] == "hello there"
    assert record.response_chunks is None
    assert record.elapsed_total_ms > 0.0
    assert record.elapsed_connect_ms is not None and record.elapsed_connect_ms >= 0.0
    assert record.elapsed_ttfb_ms is not None and record.elapsed_ttfb_ms >= 0.0
    assert record.request_body["model"] == "mock-small"
    assert record.request_body["messages"][-1]["content"] == "hi"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["stream"] is False


@pytest.mark.asyncio
async def test_execute_streaming_happy_path_preserves_chunks() -> None:
    chunks_in = [
        {"choices": [{"delta": {"content": "he"}, "index": 0}]},
        {"choices": [{"delta": {"content": "llo"}, "index": 0}]},
        {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse_stream_bytes(chunks_in),
            headers={"content-type": "text/event-stream"},
        )

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(
            model="mock-small",
            content="hi",
            stream=True,
        )
        record = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert record.status == 200
    assert record.error is None
    assert record.response_body is None
    assert record.response_chunks is not None
    assert len(record.response_chunks) == 4  # 3 deltas + [DONE]
    assert record.response_chunks[0].startswith("data: ")
    assert "he" in record.response_chunks[0]
    assert "llo" in record.response_chunks[1]
    assert record.response_chunks[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_execute_streaming_order_preserved() -> None:
    """Assert chunk order matches upstream emission order."""
    chunks_in = [
        {"seq": 1},
        {"seq": 2},
        {"seq": 3},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse_stream_bytes(chunks_in),
            headers={"content-type": "text/event-stream"},
        )

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(model="mock", content="hi", stream=True)
        record = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert record.response_chunks is not None
    # Three delta chunks plus the [DONE] sentinel.
    assert len(record.response_chunks) == 4
    assert '"seq": 1' in record.response_chunks[0]
    assert '"seq": 2' in record.response_chunks[1]
    assert '"seq": 3' in record.response_chunks[2]
    assert record.response_chunks[3] == "data: [DONE]"


# ---------------------------------------------------------------------------
# expect_status mismatch — error string, no raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_expect_status_mismatch_sets_error_does_not_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(model="mock", content="hi", expect_status=200)
        record = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert record.status == 500
    assert record.response_body == {"error": "boom"}
    assert record.error is not None
    assert "500" in record.error
    assert "200" in record.error
    assert record.elapsed_total_ms > 0.0


# ---------------------------------------------------------------------------
# Transport failures — error populated, no raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_transport_timeout_becomes_record_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("upstream too slow")

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(model="mock", content="hi", timeout_secs=1.0)
        record = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert record.status == 0
    assert record.response_body is None
    assert record.error is not None
    assert "timeout" in record.error.lower()
    assert record.elapsed_total_ms >= 0.0


@pytest.mark.asyncio
async def test_execute_connect_error_becomes_record_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(model="mock", content="hi")
        record = await primitive.execute(action, base_url=BASE_URL, client=client)

    assert record.status == 0
    assert record.error is not None
    assert "connect" in record.error.lower()


# ---------------------------------------------------------------------------
# Malformed base_url — RequestError raised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_malformed_base_url_raises_request_error() -> None:
    primitive = RequestPrimitive()
    action = RequestAction(model="mock", content="hi")
    with pytest.raises(RequestError):
        await primitive.execute(action, base_url="not-a-url")


@pytest.mark.asyncio
async def test_execute_unsupported_scheme_raises_request_error() -> None:
    primitive = RequestPrimitive()
    action = RequestAction(model="mock", content="hi")
    with pytest.raises(RequestError):
        await primitive.execute(action, base_url="ftp://example.com")


# ---------------------------------------------------------------------------
# Body construction — optional knobs propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_max_tokens_temperature_system_propagate_into_body() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_success_chat_body())

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(
            model="mock",
            content="hi",
            max_tokens=42,
            temperature=0.7,
            system="be concise",
        )
        record = await primitive.execute(action, base_url=BASE_URL, client=client)

    body = captured["body"]
    assert body["max_tokens"] == 42
    assert body["temperature"] == 0.7
    assert body["messages"][0] == {"role": "system", "content": "be concise"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert record.request_body == body


@pytest.mark.asyncio
async def test_execute_extra_headers_merged_into_request() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.headers))
        return httpx.Response(200, json=_success_chat_body())

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(
            model="mock",
            content="hi",
            extra_headers={"x-trace-id": "trace-xyz", "x-tenant": "acme"},
        )
        await primitive.execute(action, base_url=BASE_URL, client=client)

    assert captured.get("x-trace-id") == "trace-xyz"
    assert captured.get("x-tenant") == "acme"


# ---------------------------------------------------------------------------
# Client ownership — caller retains injected client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_does_not_close_injected_client() -> None:
    """An injected client must remain usable after the primitive returns."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_chat_body())

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(model="mock", content="hi")
        r1 = await primitive.execute(action, base_url=BASE_URL, client=client)
        r2 = await primitive.execute(action, base_url=BASE_URL, client=client)
        assert r1.status == 200
        assert r2.status == 200
        # Client still open — closing it ourselves must not raise.
        assert not client.is_closed
    # After the ``async with`` closes the client, further use would be
    # invalid — the assertion is simply that we reached this point
    # without the primitive having closed it mid-run.


@pytest.mark.asyncio
async def test_execute_injected_client_hooks_are_restored() -> None:
    """Injected clients' pre-existing event hooks must survive execute()."""
    marker_request: list[bool] = []
    marker_response: list[bool] = []

    async def pre_existing_request_hook(_req: httpx.Request) -> None:
        marker_request.append(True)

    async def pre_existing_response_hook(_resp: httpx.Response) -> None:
        marker_response.append(True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_chat_body())

    async with _make_client(handler) as client:
        client.event_hooks["request"] = [pre_existing_request_hook]
        client.event_hooks["response"] = [pre_existing_response_hook]
        primitive = RequestPrimitive()
        action = RequestAction(model="mock", content="hi")
        await primitive.execute(action, base_url=BASE_URL, client=client)

    # Caller's hooks fired.
    assert marker_request == [True]
    assert marker_response == [True]
    # And they are still installed (not replaced).
    assert client.event_hooks["request"] == [pre_existing_request_hook]
    assert client.event_hooks["response"] == [pre_existing_response_hook]


# ---------------------------------------------------------------------------
# capture_as surfacing
# ---------------------------------------------------------------------------


def test_capture_as_is_retained_on_action() -> None:
    """The runner reads ``action.capture_as`` to file the resulting record;
    this test pins that the field round-trips through construction."""
    action = RequestAction(
        model="mock",
        content="hi",
        capture_as="smoke-probe",
    )
    assert action.capture_as == "smoke-probe"
    dumped = action.model_dump()
    assert dumped["capture_as"] == "smoke-probe"


# ---------------------------------------------------------------------------
# Timing captures are non-None on success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_timing_fields_populated_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_chat_body())

    primitive = RequestPrimitive()
    async with _make_client(handler) as client:
        action = RequestAction(model="mock", content="hi")
        record = await primitive.execute(action, base_url=BASE_URL, client=client)

    # All three timing fields should be set on a successful request.
    assert record.elapsed_total_ms is not None
    assert record.elapsed_connect_ms is not None
    assert record.elapsed_ttfb_ms is not None
    assert record.elapsed_total_ms >= record.elapsed_connect_ms  # total >= connect
