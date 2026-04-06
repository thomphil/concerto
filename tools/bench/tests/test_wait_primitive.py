"""Unit tests for :mod:`concerto_bench.primitives.wait`.

The simplest primitive — no network, no subprocess. Tests verify
timing behaviour with short durations and pydantic validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from concerto_bench.primitives.wait import (
    WaitAction,
    WaitError,
    WaitPrimitive,
)


# ---------------------------------------------------------------------------
# WaitAction validation
# ---------------------------------------------------------------------------


def test_action_valid_construction() -> None:
    action = WaitAction(duration_secs=1.5)
    assert action.duration_secs == 1.5


def test_action_zero_duration_rejected() -> None:
    with pytest.raises(ValidationError):
        WaitAction(duration_secs=0)


def test_action_negative_duration_rejected() -> None:
    with pytest.raises(ValidationError):
        WaitAction(duration_secs=-1.0)


def test_action_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        WaitAction(duration_secs=1.0, unknown_field="oops")


def test_action_frozen() -> None:
    action = WaitAction(duration_secs=1.0)
    with pytest.raises(ValidationError):
        action.duration_secs = 2.0  # type: ignore[misc]


def test_action_round_trip_serialisation() -> None:
    original = WaitAction(duration_secs=2.5)
    loaded = WaitAction.model_validate_json(original.model_dump_json())
    assert loaded == original


# ---------------------------------------------------------------------------
# WaitPrimitive.execute — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_returns_slept_secs() -> None:
    primitive = WaitPrimitive()
    action = WaitAction(duration_secs=0.01)
    result = await primitive.execute(action, base_url="http://unused:9999")
    assert "slept_secs" in result
    assert result["slept_secs"] >= 0.01


@pytest.mark.asyncio
async def test_execute_actual_sleep_close_to_requested() -> None:
    primitive = WaitPrimitive()
    action = WaitAction(duration_secs=0.05)
    result = await primitive.execute(action, base_url="http://unused:9999")
    # Should be close to 0.05, with generous tolerance for CI.
    assert 0.04 <= result["slept_secs"] <= 0.5


@pytest.mark.asyncio
async def test_execute_accepts_base_url_and_client_without_using_them() -> None:
    """The base_url and client params exist for interface uniformity."""
    primitive = WaitPrimitive()
    action = WaitAction(duration_secs=0.01)
    result = await primitive.execute(
        action,
        base_url="http://127.0.0.1:9999",
        client=None,
    )
    assert result["slept_secs"] > 0


@pytest.mark.asyncio
async def test_execute_multiple_calls_independent() -> None:
    """A single primitive instance can be reused."""
    primitive = WaitPrimitive()
    r1 = await primitive.execute(WaitAction(duration_secs=0.01), base_url="http://x")
    r2 = await primitive.execute(WaitAction(duration_secs=0.01), base_url="http://x")
    assert r1["slept_secs"] > 0
    assert r2["slept_secs"] > 0


# ---------------------------------------------------------------------------
# WaitError exists (interface completeness)
# ---------------------------------------------------------------------------


def test_wait_error_is_runtime_error() -> None:
    err = WaitError("test")
    assert isinstance(err, RuntimeError)
    assert str(err) == "test"
