"""Unit tests for :mod:`concerto_bench.primitives.kill`.

Mocks ``asyncio.create_subprocess_exec`` and ``os.kill`` so no real
processes are affected. Tests cover happy path, no-match, permission
errors, and pgrep-missing cases.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from concerto_bench.primitives.kill import (
    KillAction,
    KillError,
    KillPrimitive,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeSubprocess:
    """Minimal async test double for ``asyncio.subprocess.Process``."""

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode: Optional[int] = None
        self._returncode_on_exit = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        self.returncode = self._returncode_on_exit
        return self._stdout, self._stderr


# ---------------------------------------------------------------------------
# KillAction validation
# ---------------------------------------------------------------------------


def test_action_valid_construction() -> None:
    action = KillAction(pattern="vllm.*phi-3")
    assert action.pattern == "vllm.*phi-3"
    assert action.signal == 9
    assert action.expect_found is True


def test_action_empty_pattern_rejected() -> None:
    with pytest.raises(ValidationError):
        KillAction(pattern="")


def test_action_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        KillAction(pattern="vllm", unknown="nope")


def test_action_frozen() -> None:
    action = KillAction(pattern="vllm")
    with pytest.raises(ValidationError):
        action.pattern = "other"  # type: ignore[misc]


def test_action_custom_signal() -> None:
    action = KillAction(pattern="vllm", signal=15)
    assert action.signal == 15


def test_action_round_trip_serialisation() -> None:
    original = KillAction(pattern="vllm.*phi", signal=15, expect_found=False)
    loaded = KillAction.model_validate_json(original.model_dump_json())
    assert loaded == original


# ---------------------------------------------------------------------------
# Happy path — PIDs found and killed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_kills_matching_pids() -> None:
    fake_proc = FakeSubprocess(stdout=b"1234\n5678\n", returncode=0)

    async def fake_create_subprocess(*argv, **kwargs) -> FakeSubprocess:
        return fake_proc

    killed_pids: list[tuple[int, int]] = []

    def fake_os_kill(pid: int, sig: int) -> None:
        killed_pids.append((pid, sig))

    primitive = KillPrimitive()
    action = KillAction(pattern="vllm", signal=15)

    with patch("concerto_bench.primitives.kill.asyncio.create_subprocess_exec", fake_create_subprocess):
        with patch("concerto_bench.primitives.kill.os.kill", fake_os_kill):
            result = await primitive.execute(action, base_url="http://unused:9999")

    assert result["pids_found"] == [1234, 5678]
    assert result["pids_killed"] == [1234, 5678]
    assert result["signal"] == 15
    assert result["errors"] == []
    assert killed_pids == [(1234, 15), (5678, 15)]


# ---------------------------------------------------------------------------
# No matches — expect_found=True → error in result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_no_match_expect_found_true() -> None:
    fake_proc = FakeSubprocess(stdout=b"", returncode=1)

    async def fake_create_subprocess(*argv, **kwargs) -> FakeSubprocess:
        return fake_proc

    primitive = KillPrimitive()
    action = KillAction(pattern="nonexistent", expect_found=True)

    with patch("concerto_bench.primitives.kill.asyncio.create_subprocess_exec", fake_create_subprocess):
        result = await primitive.execute(action, base_url="http://unused:9999")

    assert result["pids_found"] == []
    assert result["pids_killed"] == []
    assert len(result["errors"]) == 1
    assert "no process matched" in result["errors"][0]


# ---------------------------------------------------------------------------
# No matches — expect_found=False → no error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_no_match_expect_found_false() -> None:
    fake_proc = FakeSubprocess(stdout=b"", returncode=1)

    async def fake_create_subprocess(*argv, **kwargs) -> FakeSubprocess:
        return fake_proc

    primitive = KillPrimitive()
    action = KillAction(pattern="nonexistent", expect_found=False)

    with patch("concerto_bench.primitives.kill.asyncio.create_subprocess_exec", fake_create_subprocess):
        result = await primitive.execute(action, base_url="http://unused:9999")

    assert result["pids_found"] == []
    assert result["pids_killed"] == []
    assert result["errors"] == []


# ---------------------------------------------------------------------------
# pgrep binary missing — expect_found=True → KillError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_pgrep_missing_expect_found_raises() -> None:
    async def fake_create_subprocess(*argv, **kwargs) -> FakeSubprocess:
        raise FileNotFoundError("pgrep not found")

    primitive = KillPrimitive()
    action = KillAction(pattern="vllm", expect_found=True)

    with patch("concerto_bench.primitives.kill.asyncio.create_subprocess_exec", fake_create_subprocess):
        with pytest.raises(KillError):
            await primitive.execute(action, base_url="http://unused:9999")


# ---------------------------------------------------------------------------
# pgrep binary missing — expect_found=False → graceful
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_pgrep_missing_expect_found_false_graceful() -> None:
    async def fake_create_subprocess(*argv, **kwargs) -> FakeSubprocess:
        raise FileNotFoundError("pgrep not found")

    primitive = KillPrimitive()
    action = KillAction(pattern="vllm", expect_found=False)

    with patch("concerto_bench.primitives.kill.asyncio.create_subprocess_exec", fake_create_subprocess):
        result = await primitive.execute(action, base_url="http://unused:9999")

    assert result["pids_found"] == []
    assert result["pids_killed"] == []
    assert "pgrep binary not found" in result["errors"][0]


# ---------------------------------------------------------------------------
# Permission error on os.kill
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_permission_error_on_kill() -> None:
    fake_proc = FakeSubprocess(stdout=b"1234\n", returncode=0)

    async def fake_create_subprocess(*argv, **kwargs) -> FakeSubprocess:
        return fake_proc

    def fake_os_kill(pid: int, sig: int) -> None:
        raise PermissionError("not allowed")

    primitive = KillPrimitive()
    action = KillAction(pattern="vllm", signal=9)

    with patch("concerto_bench.primitives.kill.asyncio.create_subprocess_exec", fake_create_subprocess):
        with patch("concerto_bench.primitives.kill.os.kill", fake_os_kill):
            result = await primitive.execute(action, base_url="http://unused:9999")

    assert result["pids_found"] == [1234]
    assert result["pids_killed"] == []
    assert len(result["errors"]) == 1
    assert "permission denied" in result["errors"][0]


# ---------------------------------------------------------------------------
# ProcessLookupError — PID already exited
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_process_already_exited() -> None:
    fake_proc = FakeSubprocess(stdout=b"9999\n", returncode=0)

    async def fake_create_subprocess(*argv, **kwargs) -> FakeSubprocess:
        return fake_proc

    def fake_os_kill(pid: int, sig: int) -> None:
        raise ProcessLookupError("no such process")

    primitive = KillPrimitive()
    action = KillAction(pattern="vllm")

    with patch("concerto_bench.primitives.kill.asyncio.create_subprocess_exec", fake_create_subprocess):
        with patch("concerto_bench.primitives.kill.os.kill", fake_os_kill):
            result = await primitive.execute(action, base_url="http://unused:9999")

    assert result["pids_found"] == [9999]
    assert result["pids_killed"] == []
    assert "already exited" in result["errors"][0]


# ---------------------------------------------------------------------------
# KillError is RuntimeError
# ---------------------------------------------------------------------------


def test_kill_error_is_runtime_error() -> None:
    err = KillError("test")
    assert isinstance(err, RuntimeError)
