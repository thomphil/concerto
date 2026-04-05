"""Unit tests for :class:`PgrepCounterSampler`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest

from concerto_bench.samplers.base import SamplerResult
from concerto_bench.samplers.pgrep_counter import (
    PgrepCounterSampler,
    PgrepCounterSamplerConfig,
)
from concerto_bench.schema import TelemetrySample


class FakeSubprocess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._rc = returncode
        self.returncode: Optional[int] = None

    async def communicate(self) -> tuple[bytes, bytes]:
        self.returncode = self._rc
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._rc
        return self.returncode


def _make_router(per_pattern_response: dict[str, tuple[bytes, int]]):
    """Return a callable that simulates ``pgrep -c <pattern>`` per pattern."""

    calls: list[tuple[str, ...]] = []

    async def router(*argv: str, **_: Any) -> FakeSubprocess:
        calls.append(argv)
        assert argv[0] == "pgrep"
        assert argv[1] == "-c"
        pattern = argv[2]
        stdout, rc = per_pattern_response[pattern]
        return FakeSubprocess(stdout=stdout, returncode=rc)

    router.calls = calls  # type: ignore[attr-defined]
    return router


def _config(
    tmp_path: Path,
    *,
    patterns: list[str],
    interval: float = 0.05,
) -> PgrepCounterSamplerConfig:
    return PgrepCounterSamplerConfig(
        name="pgrep-count",
        interval_secs=interval,
        output_path=tmp_path / "pgrep-count.jsonl",
        patterns=patterns,
    )


async def _drive(sampler: PgrepCounterSampler, secs: float) -> SamplerResult:
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
async def test_happy_path_returns_counts_and_total(tmp_path: Path) -> None:
    router = _make_router(
        {
            "vllm": (b"2\n", 0),
            "mock-inference-backend": (b"3\n", 0),
            "llama.cpp": (b"0\n", 0),
        }
    )
    with patch("shutil.which", return_value="/usr/bin/pgrep"):
        with patch(
            "concerto_bench.samplers.pgrep_counter.asyncio.create_subprocess_exec",
            router,
        ):
            config = _config(
                tmp_path, patterns=["vllm", "mock-inference-backend", "llama.cpp"]
            )
            sampler = PgrepCounterSampler(config)
            result = await _drive(sampler, 0.18)

    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(config.output_path)
    for row in rows:
        assert row.values["counts"] == {
            "vllm": 2,
            "mock-inference-backend": 3,
            "llama.cpp": 0,
        }
        assert row.values["total"] == 5


@pytest.mark.asyncio
async def test_no_matches_exit_code_1_is_not_a_tick_failure(tmp_path: Path) -> None:
    """pgrep -c returns exit 1 with no matches; that should map to count=0."""
    router = _make_router(
        {
            "vllm": (b"0\n", 1),  # exit 1 + count 0 is the real pgrep -c
            "something-else": (b"4\n", 0),
        }
    )
    with patch("shutil.which", return_value="/usr/bin/pgrep"):
        with patch(
            "concerto_bench.samplers.pgrep_counter.asyncio.create_subprocess_exec",
            router,
        ):
            config = _config(tmp_path, patterns=["vllm", "something-else"])
            sampler = PgrepCounterSampler(config)
            result = await _drive(sampler, 0.15)

    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(config.output_path)
    for row in rows:
        assert row.values["counts"]["vllm"] == 0
        assert row.values["counts"]["something-else"] == 4
        assert row.values["total"] == 4


# ---------------------------------------------------------------------------
# Degraded mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_binary_emits_degraded_rows(tmp_path: Path) -> None:
    with patch("shutil.which", return_value=None):
        config = _config(tmp_path, patterns=["vllm", "mock-inference-backend"])
        sampler = PgrepCounterSampler(config)
        result = await _drive(sampler, 0.15)

    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(config.output_path)
    for row in rows:
        assert row.values["degraded"] is True
        assert row.values["total"] == 0
        assert row.values["counts"] == {"vllm": 0, "mock-inference-backend": 0}
        assert "not found" in row.values["reason"]


# ---------------------------------------------------------------------------
# Unexpected pgrep failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unexpected_exit_code_counts_as_tick_failure(tmp_path: Path) -> None:
    router = _make_router({"vllm": (b"", 2)})  # rc=2 is "syntax error" in pgrep
    with patch("shutil.which", return_value="/usr/bin/pgrep"):
        with patch(
            "concerto_bench.samplers.pgrep_counter.asyncio.create_subprocess_exec",
            router,
        ):
            config = _config(tmp_path, patterns=["vllm"])
            sampler = PgrepCounterSampler(config)
            result = await _drive(sampler, 0.15)

    assert result.ticks_succeeded == 0
    assert result.ticks_failed >= 1
    assert any("rc=2" in r for r in result.failures)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_requires_at_least_one_pattern(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PgrepCounterSamplerConfig(
            name="pgrep-count",
            interval_secs=1.0,
            output_path=tmp_path / "x.jsonl",
            patterns=[],
        )
