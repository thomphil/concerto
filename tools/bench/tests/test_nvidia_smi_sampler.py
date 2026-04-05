"""Unit tests for :class:`NvidiaSmiSampler`.

``asyncio.create_subprocess_exec`` is patched so these tests run on any
host regardless of whether ``nvidia-smi`` is installed. The degraded-mode
path (binary missing) is exercised explicitly since it's the
SPRINT-2-PLAN §4 B.2 step 6 load-bearing requirement.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest

from concerto_bench.samplers.base import SamplerResult
from concerto_bench.samplers.nvidia_smi import (
    NvidiaSmiSampler,
    NvidiaSmiSamplerConfig,
)
from concerto_bench.schema import TelemetrySample


# ---------------------------------------------------------------------------
# FakeSubprocess + router (mirrors the pattern used by snapshot tests)
# ---------------------------------------------------------------------------


class FakeSubprocess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        communicate_delay: float = 0.0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._rc_on_exit = returncode
        self._delay = communicate_delay
        self.returncode: Optional[int] = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        self.returncode = self._rc_on_exit
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._rc_on_exit
        return self.returncode


class Router:
    def __init__(self, factory) -> None:
        self._factory = factory
        self.calls: list[tuple[str, ...]] = []

    async def __call__(self, *argv: str, **_: Any) -> FakeSubprocess:
        self.calls.append(argv)
        result = self._factory(argv)
        if isinstance(result, BaseException):
            raise result
        return result


def _config(tmp_path: Path, *, interval: float = 0.05, binary: str = "nvidia-smi") -> NvidiaSmiSamplerConfig:
    return NvidiaSmiSamplerConfig(
        name="nvidia-smi",
        interval_secs=interval,
        output_path=tmp_path / "nvidia-smi.jsonl",
        binary=binary,
    )


async def _drive(sampler: NvidiaSmiSampler, secs: float) -> SamplerResult:
    task = asyncio.create_task(sampler.run())
    try:
        await asyncio.sleep(secs)
    finally:
        task.cancel()
    return await task


def _read_rows(path: Path) -> list[TelemetrySample]:
    rows: list[TelemetrySample] = []
    with path.open("rb") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(TelemetrySample.model_validate_json(stripped))
    return rows


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_parses_two_gpu_rows(tmp_path: Path) -> None:
    stdout = (
        b"0, NVIDIA RTX A4000, 16376, 12000, 4376, 42, 58, 95.12\n"
        b"1, NVIDIA RTX A4000, 16376, 11800, 4576, 44, 60, 98.40\n"
    )
    router = Router(lambda argv: FakeSubprocess(stdout=stdout, returncode=0))

    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch(
            "concerto_bench.samplers.nvidia_smi.asyncio.create_subprocess_exec",
            router,
        ):
            sampler = NvidiaSmiSampler(_config(tmp_path))
            result = await _drive(sampler, 0.18)

    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(sampler.config.output_path)
    assert len(rows) >= 2
    for row in rows:
        gpus = row.values["gpus"]
        assert len(gpus) == 2
        # Types are coerced: integers for memory, floats for utilisation/temp/power.
        assert gpus[0]["index"] == 0
        assert gpus[0]["memory.total"] == 16376
        assert gpus[0]["memory.used"] == 12000
        assert gpus[0]["memory.free"] == 4376
        assert gpus[0]["utilization.gpu"] == pytest.approx(42.0)
        assert gpus[0]["temperature.gpu"] == pytest.approx(58.0)
        assert gpus[0]["power.draw"] == pytest.approx(95.12)
        assert gpus[0]["name"] == "NVIDIA RTX A4000"


# ---------------------------------------------------------------------------
# Degraded mode — binary missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_binary_emits_empty_degraded_rows(tmp_path: Path) -> None:
    with patch("shutil.which", return_value=None):
        sampler = NvidiaSmiSampler(_config(tmp_path))
        result = await _drive(sampler, 0.15)

    # Sampler runs normally, produces one row per tick, nothing raises.
    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(sampler.config.output_path)
    assert len(rows) == result.ticks_succeeded
    for row in rows:
        assert row.values["gpus"] == []
        assert row.values["degraded"] is True
        assert "not found" in row.values["reason"]


# ---------------------------------------------------------------------------
# Timeout handling — stuck nvidia-smi → tick failure, loop continues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_counts_as_tick_failure(tmp_path: Path) -> None:
    """A stuck nvidia-smi is killed and the tick is counted as failed."""

    # First call hangs, subsequent calls succeed so we confirm the loop
    # recovered after the timeout.
    call_count = {"n": 0}

    def factory(argv: tuple[str, ...]) -> FakeSubprocess:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return FakeSubprocess(communicate_delay=5.0, returncode=0)
        return FakeSubprocess(stdout=b"0, NVIDIA RTX A4000, 16376, 100, 16276, 5, 35, 10.0\n")

    router = Router(factory)

    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch(
            "concerto_bench.samplers.nvidia_smi.asyncio.create_subprocess_exec",
            router,
        ):
            # Interval 0.1 → timeout budget 0.08 → wait_for kills the
            # hanging call cleanly.
            sampler = NvidiaSmiSampler(_config(tmp_path, interval=0.1))
            result = await _drive(sampler, 0.35)

    assert result.ticks_failed >= 1
    assert any("timed out" in r.lower() for r in result.failures)
    # Eventually at least one tick succeeded once the hang stopped.
    assert result.ticks_succeeded >= 1


# ---------------------------------------------------------------------------
# Empty output (binary present but returns no rows) → not degraded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_csv_output_yields_empty_gpu_list(tmp_path: Path) -> None:
    router = Router(lambda argv: FakeSubprocess(stdout=b"", returncode=0))
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch(
            "concerto_bench.samplers.nvidia_smi.asyncio.create_subprocess_exec",
            router,
        ):
            sampler = NvidiaSmiSampler(_config(tmp_path))
            result = await _drive(sampler, 0.15)

    rows = _read_rows(sampler.config.output_path)
    assert len(rows) >= 2
    for row in rows:
        assert row.values == {"gpus": []}
        # Not the degraded shape.
        assert "degraded" not in row.values


# ---------------------------------------------------------------------------
# Non-zero exit code → tick failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nonzero_exit_counts_as_tick_failure(tmp_path: Path) -> None:
    router = Router(
        lambda argv: FakeSubprocess(
            stdout=b"", stderr=b"driver not loaded", returncode=9
        )
    )
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch(
            "concerto_bench.samplers.nvidia_smi.asyncio.create_subprocess_exec",
            router,
        ):
            sampler = NvidiaSmiSampler(_config(tmp_path))
            result = await _drive(sampler, 0.15)

    assert result.ticks_succeeded == 0
    assert result.ticks_failed >= 1
    assert any("rc=9" in r for r in result.failures)
