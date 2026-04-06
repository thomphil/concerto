"""Unit tests for :mod:`concerto_bench.primitives.parallel`.

Tests the meta-primitive's dispatch wiring, concurrent execution,
timeout handling, and error collection. Sub-action dispatch is faked
via simple async callables.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from concerto_bench.primitives.parallel import (
    ParallelAction,
    ParallelError,
    ParallelPrimitive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _echo_dispatch(action_spec: dict[str, Any]) -> dict[str, Any]:
    """Dispatch that returns the action spec as-is."""
    return {"echoed": action_spec}


async def _slow_dispatch(action_spec: dict[str, Any]) -> dict[str, Any]:
    """Dispatch that sleeps for the 'sleep' key, then returns."""
    sleep_secs = action_spec.get("sleep", 0.01)
    await asyncio.sleep(sleep_secs)
    return {"slept": sleep_secs}


async def _failing_dispatch(action_spec: dict[str, Any]) -> dict[str, Any]:
    """Dispatch that always raises."""
    raise RuntimeError(f"dispatch failed for {action_spec}")


# ---------------------------------------------------------------------------
# ParallelAction validation
# ---------------------------------------------------------------------------


def test_action_valid_construction() -> None:
    action = ParallelAction(actions=[{"type": "wait", "duration_secs": 1}])
    assert len(action.actions) == 1


def test_action_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        ParallelAction(actions=[], extra="nope")


def test_action_timeout_non_positive_rejected() -> None:
    with pytest.raises(ValidationError):
        ParallelAction(actions=[], timeout_secs=0)


def test_action_frozen() -> None:
    action = ParallelAction(actions=[{"type": "wait"}])
    with pytest.raises(ValidationError):
        action.timeout_secs = 999.0  # type: ignore[misc]


def test_action_round_trip_serialisation() -> None:
    original = ParallelAction(
        actions=[{"type": "wait", "duration_secs": 1.0}],
        timeout_secs=120.0,
    )
    loaded = ParallelAction.model_validate_json(original.model_dump_json())
    assert loaded == original


# ---------------------------------------------------------------------------
# Missing dispatch — ParallelError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_without_dispatch_raises() -> None:
    primitive = ParallelPrimitive()
    action = ParallelAction(actions=[{"type": "wait"}])
    with pytest.raises(ParallelError, match="dispatch"):
        await primitive.execute(action, base_url="http://unused:9999")


@pytest.mark.asyncio
async def test_execute_dispatch_none_raises() -> None:
    primitive = ParallelPrimitive()
    action = ParallelAction(actions=[{"type": "wait"}])
    with pytest.raises(ParallelError):
        await primitive.execute(action, base_url="http://unused:9999", dispatch=None)


# ---------------------------------------------------------------------------
# Happy path — all sub-actions succeed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_happy_path() -> None:
    primitive = ParallelPrimitive()
    action = ParallelAction(
        actions=[
            {"type": "wait", "id": 1},
            {"type": "wait", "id": 2},
            {"type": "wait", "id": 3},
        ],
        timeout_secs=5.0,
    )
    result = await primitive.execute(
        action,
        base_url="http://unused:9999",
        dispatch=_echo_dispatch,
    )

    assert len(result["results"]) == 3
    assert result["errors"] == []
    assert result["elapsed_secs"] > 0
    # Verify that each result echoed its input.
    ids = [r["echoed"]["id"] for r in result["results"]]
    assert sorted(ids) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Empty actions list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_empty_actions() -> None:
    primitive = ParallelPrimitive()
    action = ParallelAction(actions=[], timeout_secs=5.0)
    result = await primitive.execute(
        action,
        base_url="http://unused:9999",
        dispatch=_echo_dispatch,
    )

    assert result["results"] == []
    assert result["errors"] == []
    assert result["elapsed_secs"] >= 0


# ---------------------------------------------------------------------------
# Sub-action failures captured in errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_sub_action_failure_captured() -> None:
    primitive = ParallelPrimitive()
    action = ParallelAction(
        actions=[{"type": "fail"}],
        timeout_secs=5.0,
    )
    result = await primitive.execute(
        action,
        base_url="http://unused:9999",
        dispatch=_failing_dispatch,
    )

    assert result["results"] == []
    assert len(result["errors"]) == 1
    assert "RuntimeError" in result["errors"][0]


# ---------------------------------------------------------------------------
# Mixed success and failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_mixed_success_failure() -> None:
    call_count = 0

    async def mixed_dispatch(action_spec: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if action_spec.get("fail"):
            raise ValueError("intentional failure")
        return {"ok": True}

    primitive = ParallelPrimitive()
    action = ParallelAction(
        actions=[
            {"fail": False},
            {"fail": True},
            {"fail": False},
        ],
        timeout_secs=5.0,
    )
    result = await primitive.execute(
        action,
        base_url="http://unused:9999",
        dispatch=mixed_dispatch,
    )

    assert len(result["results"]) == 2
    assert len(result["errors"]) == 1
    assert "ValueError" in result["errors"][0]


# ---------------------------------------------------------------------------
# Concurrency — sub-actions run in parallel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_runs_in_parallel() -> None:
    """Three 100ms sleeps should complete in ~100ms, not ~300ms."""
    import time

    primitive = ParallelPrimitive()
    action = ParallelAction(
        actions=[
            {"sleep": 0.1},
            {"sleep": 0.1},
            {"sleep": 0.1},
        ],
        timeout_secs=5.0,
    )

    start = time.perf_counter()
    result = await primitive.execute(
        action,
        base_url="http://unused:9999",
        dispatch=_slow_dispatch,
    )
    elapsed = time.perf_counter() - start

    assert len(result["results"]) == 3
    assert result["errors"] == []
    # Parallel: ~100ms, not ~300ms. Allow generous headroom.
    assert elapsed < 0.4, f"expected ~0.1s, got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Timeout — overall deadline fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_timeout() -> None:
    async def forever_dispatch(action_spec: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(10.0)
        return {"done": True}

    primitive = ParallelPrimitive()
    action = ParallelAction(
        actions=[{"type": "slow"}],
        timeout_secs=0.1,
    )
    result = await primitive.execute(
        action,
        base_url="http://unused:9999",
        dispatch=forever_dispatch,
    )

    assert result["results"] == []
    assert len(result["errors"]) == 1
    assert "timed out" in result["errors"][0]
    assert result["elapsed_secs"] < 1.0


# ---------------------------------------------------------------------------
# ParallelError is RuntimeError
# ---------------------------------------------------------------------------


def test_parallel_error_is_runtime_error() -> None:
    err = ParallelError("test")
    assert isinstance(err, RuntimeError)
