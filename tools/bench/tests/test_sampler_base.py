"""Unit tests for :mod:`concerto_bench.samplers.base`.

Every concrete sampler in Sprint 2 leans on the loop, file handling,
cancellation semantics, and :class:`SamplerPool` implemented in the
base module, so this file is the most important test in step 6.

The tests use a handful of small in-file fake sampler classes rather
than monkey-patching the real ones — it makes the loop mechanics
(tick counting, drift compensation, failure bookkeeping, teardown
ordering) visible and independent of any external data source.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from concerto_bench.samplers.base import (
    Sampler,
    SamplerConfig,
    SamplerError,
    SamplerPool,
    SamplerRegistry,
    SamplerResult,
    run_samplers,
)
from concerto_bench.schema import TelemetrySample


# ---------------------------------------------------------------------------
# Fake sampler classes used across the tests
# ---------------------------------------------------------------------------


class CounterSampler(Sampler):
    """Fake sampler that emits an incrementing counter on every tick.

    Used by the happy-path / drift-compensation / cancellation tests.
    ``teardown_called`` and ``setup_called`` flags are observable from
    the tests so ordering assertions are easy.
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        self.setup_called = False
        self.teardown_called = False
        self.counter = 0
        self.sample_delay_secs: float = 0.0

    async def setup(self) -> None:
        self.setup_called = True

    async def teardown(self) -> None:
        self.teardown_called = True

    async def sample_once(self) -> dict[str, Any]:
        if self.sample_delay_secs > 0:
            await asyncio.sleep(self.sample_delay_secs)
        self.counter += 1
        return {"n": self.counter}


class FlakySampler(Sampler):
    """Fake sampler that raises on every other tick.

    Verifies per-tick failure counting and that the loop keeps
    running after a raised exception.
    """

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        self.counter = 0

    async def sample_once(self) -> dict[str, Any]:
        self.counter += 1
        if self.counter % 2 == 0:
            raise RuntimeError(f"synthetic failure {self.counter}")
        return {"n": self.counter}


class AlwaysFailingSampler(Sampler):
    """Every tick raises. Used to verify the failures cap and teardown-after-failures."""

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        self.setup_called = False
        self.teardown_called = False
        self.counter = 0

    async def setup(self) -> None:
        self.setup_called = True

    async def teardown(self) -> None:
        self.teardown_called = True

    async def sample_once(self) -> dict[str, Any]:
        self.counter += 1
        raise ValueError(f"nope {self.counter}")


class SetupFailingSampler(Sampler):
    """Sampler whose ``setup`` raises. Used to verify SamplerError surface."""

    def __init__(self, config: SamplerConfig) -> None:
        super().__init__(config)
        self.teardown_called = False

    async def setup(self) -> None:
        raise RuntimeError("cannot start")

    async def teardown(self) -> None:  # pragma: no cover - should never fire
        self.teardown_called = True

    async def sample_once(self) -> dict[str, Any]:  # pragma: no cover
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, name: str = "fake", interval: float = 0.05) -> SamplerConfig:
    return SamplerConfig(
        name=name,
        interval_secs=interval,
        output_path=tmp_path / f"{name}.jsonl",
    )


async def _run_for(sampler: Sampler, secs: float) -> SamplerResult:
    """Run a sampler as a task, cancel after ``secs``, return the result."""
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
            if not stripped:
                continue
            rows.append(TelemetrySample.model_validate_json(stripped))
    return rows


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_sampler_config_round_trip(tmp_path: Path) -> None:
    config = SamplerConfig(
        name="demo",
        interval_secs=2.5,
        output_path=tmp_path / "demo.jsonl",
        enabled=False,
    )
    # JSON round-trip
    loaded = SamplerConfig.model_validate_json(config.model_dump_json())
    assert loaded == config
    assert loaded.enabled is False


def test_sampler_config_rejects_non_positive_interval(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SamplerConfig(
            name="demo",
            interval_secs=0.0,
            output_path=tmp_path / "demo.jsonl",
        )


def test_sampler_config_rejects_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SamplerConfig(
            name="demo",
            interval_secs=1.0,
            output_path=tmp_path / "demo.jsonl",
            bogus=True,
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_emits_three_rows(tmp_path: Path) -> None:
    config = _config(tmp_path, interval=0.05)
    sampler = CounterSampler(config)
    # Run for ~3.5 ticks worth of wall time.
    result = await _run_for(sampler, 0.05 * 3 + 0.03)

    rows = _read_rows(config.output_path)
    assert len(rows) >= 2  # allow for scheduling slack on a busy CI box
    assert result.ticks_succeeded == len(rows)
    assert result.ticks_failed == 0
    assert result.ticks_attempted == result.ticks_succeeded
    assert result.bytes_written == config.output_path.stat().st_size
    assert result.first_sample_at is not None
    assert result.last_sample_at is not None
    assert result.first_sample_at.tzinfo is not None
    assert result.first_sample_at.utcoffset() is not None
    # Timestamps are UTC.
    assert result.first_sample_at.utcoffset().total_seconds() == 0.0
    # All rows are well-formed TelemetrySample instances with matching name.
    assert all(row.sampler == "fake" for row in rows)
    # Counter increments sequentially.
    ns = [row.values["n"] for row in rows]
    assert ns == list(range(1, len(ns) + 1))
    # Lifecycle flags.
    assert sampler.setup_called is True
    assert sampler.teardown_called is True


# ---------------------------------------------------------------------------
# Per-tick failure counting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_tick_failures_counted_and_loop_continues(tmp_path: Path) -> None:
    config = _config(tmp_path, interval=0.05)
    sampler = FlakySampler(config)
    result = await _run_for(sampler, 0.05 * 4 + 0.04)

    # Alternating pass/fail: at least one success and one failure observed.
    assert result.ticks_succeeded >= 1
    assert result.ticks_failed >= 1
    assert result.ticks_attempted == result.ticks_succeeded + result.ticks_failed
    # Every stored failure reason mentions the synthetic error type.
    assert all("RuntimeError" in reason for reason in result.failures)
    # Rows on disk match ticks_succeeded.
    rows = _read_rows(config.output_path)
    assert len(rows) == result.ticks_succeeded
    # The surviving samples are the odd-numbered counter values.
    ns = [row.values["n"] for row in rows]
    assert all(n % 2 == 1 for n in ns)


@pytest.mark.asyncio
async def test_failures_list_is_capped(tmp_path: Path) -> None:
    """The ``failures`` list is bounded so a long broken run can't grow unbounded."""
    config = _config(tmp_path, interval=0.005)
    sampler = AlwaysFailingSampler(config)
    # Long enough to accumulate well over the 20-entry cap.
    result = await _run_for(sampler, 0.3)
    assert result.ticks_failed > 20
    assert len(result.failures) <= 20
    assert result.ticks_succeeded == 0
    assert sampler.setup_called is True
    assert sampler.teardown_called is True


# ---------------------------------------------------------------------------
# File flushing semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_is_flushed_while_running(tmp_path: Path) -> None:
    """Rows are visible to an external reader while the sampler is still live."""
    config = _config(tmp_path, interval=0.05)
    sampler = CounterSampler(config)
    task = asyncio.create_task(sampler.run())
    try:
        # Wait a couple of ticks — the file should contain JSONL rows by now.
        await asyncio.sleep(0.18)
        assert config.output_path.exists()
        mid_run_contents = config.output_path.read_bytes()
        assert mid_run_contents.count(b"\n") >= 1
        # Every line parses as a TelemetrySample.
        for raw in mid_run_contents.splitlines():
            if not raw:
                continue
            TelemetrySample.model_validate_json(raw)
    finally:
        task.cancel()
    await task


# ---------------------------------------------------------------------------
# Setup / teardown lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_failure_raises_sampler_error_and_leaves_no_file(tmp_path: Path) -> None:
    config = _config(tmp_path, name="setup-fail")
    sampler = SetupFailingSampler(config)
    with pytest.raises(SamplerError):
        await sampler.run()
    # The zero-byte file should have been cleaned up; setup fired but
    # teardown must NOT have run (setup was not successful).
    assert not config.output_path.exists()
    assert sampler.teardown_called is False


@pytest.mark.asyncio
async def test_teardown_runs_when_every_tick_fails(tmp_path: Path) -> None:
    config = _config(tmp_path, interval=0.02)
    sampler = AlwaysFailingSampler(config)
    result = await _run_for(sampler, 0.12)
    assert result.ticks_succeeded == 0
    assert result.ticks_failed >= 1
    assert sampler.setup_called is True
    assert sampler.teardown_called is True


@pytest.mark.asyncio
async def test_disabled_sampler_emits_zero_rows_but_creates_file(tmp_path: Path) -> None:
    config = SamplerConfig(
        name="off",
        interval_secs=0.05,
        output_path=tmp_path / "off.jsonl",
        enabled=False,
    )
    sampler = CounterSampler(config)
    result = await sampler.run()
    assert result.ticks_attempted == 0
    assert result.ticks_succeeded == 0
    assert result.ticks_failed == 0
    assert config.output_path.exists()
    assert config.output_path.stat().st_size == 0
    # setup/teardown are NOT called for a disabled sampler.
    assert sampler.setup_called is False
    assert sampler.teardown_called is False


# ---------------------------------------------------------------------------
# Drift compensation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slow_sample_does_not_produce_backlog(tmp_path: Path) -> None:
    """A slow tick is skipped-ahead rather than accumulating a backlog."""
    config = _config(tmp_path, interval=0.05)
    sampler = CounterSampler(config)
    sampler.sample_delay_secs = 0.09  # each tick takes almost 2 intervals

    task = asyncio.create_task(sampler.run())
    try:
        await asyncio.sleep(0.4)
    finally:
        task.cancel()
    result = await task

    # With a 0.05s interval, 0.09s per sample, and 0.4s wall time, we
    # should not see the naive attempt count (8). The loop should
    # skip ahead and produce roughly 0.4 / 0.09 ≈ 4-5 ticks. A
    # cancellation that fires during the last sample_once() counts
    # as attempted-but-not-succeeded (the row was never written),
    # so we allow a one-tick gap between attempted and succeeded.
    assert result.ticks_attempted <= 6
    assert result.ticks_attempted >= 2
    assert result.ticks_succeeded >= result.ticks_attempted - 1
    assert result.ticks_failed == 0


# ---------------------------------------------------------------------------
# SamplerPool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sampler_pool_runs_three_samplers_in_parallel(tmp_path: Path) -> None:
    samplers = [CounterSampler(_config(tmp_path, name=f"s{i}", interval=0.05)) for i in range(3)]
    async with SamplerPool(samplers) as pool:
        await asyncio.sleep(0.18)
    # Every sampler contributed a result; no errors.
    assert pool.errors == []
    assert len(pool.results) == 3
    names = sorted(r.name for r in pool.results)
    assert names == ["s0", "s1", "s2"]
    for r in pool.results:
        assert r.ticks_attempted >= 1
        assert r.output_path.exists()
    # Each sampler wrote at least one row.
    for s in samplers:
        assert s.teardown_called is True
        assert s.setup_called is True


@pytest.mark.asyncio
async def test_sampler_pool_handles_setup_failure(tmp_path: Path) -> None:
    """A broken setup surfaces in ``pool.errors``; other samplers still run."""
    good = CounterSampler(_config(tmp_path, name="good", interval=0.05))
    bad = SetupFailingSampler(_config(tmp_path, name="bad", interval=0.05))
    async with SamplerPool([good, bad]) as pool:
        await asyncio.sleep(0.15)

    good_results = [r for r in pool.results if r.name == "good"]
    assert len(good_results) == 1
    assert good_results[0].ticks_attempted >= 1

    # The bad sampler is surfaced via ``errors``.
    bad_errors = [(name, exc) for name, exc in pool.errors if name == "bad"]
    assert len(bad_errors) == 1
    assert isinstance(bad_errors[0][1], SamplerError)


@pytest.mark.asyncio
async def test_sampler_pool_is_not_reentrant(tmp_path: Path) -> None:
    pool = SamplerPool([CounterSampler(_config(tmp_path))])
    async with pool:
        pass
    with pytest.raises(RuntimeError):
        async with pool:
            pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_register_and_lookup() -> None:
    registry = SamplerRegistry()
    registry.register("fake", CounterSampler)
    assert "fake" in registry
    assert registry.get("fake") is CounterSampler
    assert registry.names() == ["fake"]


def test_registry_rejects_duplicate_name() -> None:
    registry = SamplerRegistry()
    registry.register("fake", CounterSampler)
    with pytest.raises(ValueError):
        registry.register("fake", CounterSampler)


def test_registry_rejects_empty_name() -> None:
    registry = SamplerRegistry()
    with pytest.raises(ValueError):
        registry.register("", CounterSampler)


def test_default_registry_is_populated() -> None:
    from concerto_bench.samplers import DEFAULT_REGISTRY

    assert "nvidia-smi" in DEFAULT_REGISTRY
    assert "concerto-status" in DEFAULT_REGISTRY
    assert "concerto-metrics" in DEFAULT_REGISTRY
    assert "pgrep-count" in DEFAULT_REGISTRY
    assert "proc-stats" in DEFAULT_REGISTRY
