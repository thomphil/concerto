"""Unit tests for :class:`ProcStatsSampler`.

Rather than mocking the filesystem we write fixture files to a
``tmp_path`` and point the config at them. Degraded mode is exercised
by passing paths that don't exist (which is what happens on macOS
when the sampler is handed the default ``/proc/loadavg``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from concerto_bench.samplers.base import SamplerResult
from concerto_bench.samplers.proc_stats import (
    ProcStatsSampler,
    ProcStatsSamplerConfig,
)
from concerto_bench.schema import TelemetrySample


LOADAVG_BODY = "0.18 0.24 0.31 3/512 12345\n"
MEMINFO_BODY = """\
MemTotal:       16400000 kB
MemFree:         1500000 kB
MemAvailable:    8200000 kB
Buffers:          250000 kB
Cached:          3000000 kB
SwapTotal:       4000000 kB
SwapFree:        3900000 kB
Extra:             10000 kB
"""


def _write_proc(tmp_path: Path) -> tuple[Path, Path]:
    loadavg = tmp_path / "loadavg"
    meminfo = tmp_path / "meminfo"
    loadavg.write_text(LOADAVG_BODY)
    meminfo.write_text(MEMINFO_BODY)
    return loadavg, meminfo


def _config(
    tmp_path: Path,
    *,
    loadavg_path: Path,
    meminfo_path: Path,
    interval: float = 0.05,
) -> ProcStatsSamplerConfig:
    return ProcStatsSamplerConfig(
        name="proc-stats",
        interval_secs=interval,
        output_path=tmp_path / "proc-stats.jsonl",
        loadavg_path=loadavg_path,
        meminfo_path=meminfo_path,
    )


async def _drive(sampler: ProcStatsSampler, secs: float) -> SamplerResult:
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
# Happy path on a simulated /proc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_parses_loadavg_and_meminfo(tmp_path: Path) -> None:
    loadavg, meminfo = _write_proc(tmp_path)
    sampler = ProcStatsSampler(
        _config(tmp_path, loadavg_path=loadavg, meminfo_path=meminfo)
    )
    result = await _drive(sampler, 0.18)

    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(sampler.config.output_path)
    for row in rows:
        la = row.values["loadavg"]
        assert la["load1"] == pytest.approx(0.18)
        assert la["load5"] == pytest.approx(0.24)
        assert la["load15"] == pytest.approx(0.31)
        mi = row.values["meminfo"]
        # kB → bytes conversion.
        assert mi["mem_total"] == 16_400_000 * 1024
        assert mi["mem_free"] == 1_500_000 * 1024
        assert mi["mem_available"] == 8_200_000 * 1024
        assert mi["buffers"] == 250_000 * 1024
        assert mi["cached"] == 3_000_000 * 1024
        assert mi["swap_total"] == 4_000_000 * 1024
        assert mi["swap_free"] == 3_900_000 * 1024
        # Non-whitelisted keys are dropped.
        assert "extra" not in mi


# ---------------------------------------------------------------------------
# Partial degradation: one sub-source missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_loadavg_still_captures_meminfo(tmp_path: Path) -> None:
    _, meminfo = _write_proc(tmp_path)
    sampler = ProcStatsSampler(
        _config(
            tmp_path,
            loadavg_path=tmp_path / "does-not-exist-loadavg",
            meminfo_path=meminfo,
        )
    )
    result = await _drive(sampler, 0.15)

    assert result.ticks_succeeded >= 2
    rows = _read_rows(sampler.config.output_path)
    for row in rows:
        assert row.values["loadavg"] is None
        assert row.values["meminfo"]["mem_total"] == 16_400_000 * 1024
        assert "degraded" not in row.values


# ---------------------------------------------------------------------------
# Full degradation: both files missing (macOS)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_paths_missing_enters_degraded_mode(tmp_path: Path) -> None:
    sampler = ProcStatsSampler(
        _config(
            tmp_path,
            loadavg_path=tmp_path / "missing-loadavg",
            meminfo_path=tmp_path / "missing-meminfo",
        )
    )
    result = await _drive(sampler, 0.15)

    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(sampler.config.output_path)
    for row in rows:
        assert row.values["degraded"] is True
        assert row.values["loadavg"] is None
        assert row.values["meminfo"] is None


# ---------------------------------------------------------------------------
# Malformed meminfo line is tolerated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_meminfo_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    loadavg = tmp_path / "loadavg"
    meminfo = tmp_path / "meminfo"
    loadavg.write_text(LOADAVG_BODY)
    # MemTotal line is malformed (no integer value); other lines are fine.
    meminfo.write_text(
        "MemTotal: notanumber kB\n"
        "MemAvailable:    8200000 kB\n"
        "no-colon-line-should-be-skipped\n"
    )
    sampler = ProcStatsSampler(
        _config(tmp_path, loadavg_path=loadavg, meminfo_path=meminfo)
    )
    result = await _drive(sampler, 0.15)

    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(sampler.config.output_path)
    for row in rows:
        mi = row.values["meminfo"]
        # Malformed MemTotal dropped; MemAvailable still present.
        assert "mem_total" not in mi
        assert mi["mem_available"] == 8_200_000 * 1024
