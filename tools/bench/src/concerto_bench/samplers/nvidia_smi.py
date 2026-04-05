"""Sampler: periodic ``nvidia-smi --query-gpu=...`` CSV capture.

On GPU hosts, shells out at 1 Hz and records per-GPU VRAM, utilisation,
temperature, and power into ``telemetry/nvidia-smi.jsonl`` as
``{"gpus": [...]}`` rows. Each GPU dict keys by the nvidia-smi column
name (e.g. ``memory.total``, ``utilization.gpu``) with values coerced to
their natural numeric types where possible.

On non-GPU hosts (Mac dev boxes, GPU-less CI runners) the sampler
enters **degraded mode** at :meth:`setup` — every subsequent
:meth:`sample_once` returns an empty payload
(``{"gpus": [], "degraded": True, "reason": ...}``) without raising,
so the JSONL still contains one row per tick and the analyzer can
distinguish "no data" from "file missing". This matches the explicit
SPRINT-2-PLAN §4 B.2 step 6 requirement: "handles 'no GPU available'
gracefully, emits empty rows".

A per-tick subprocess call is bounded by :func:`asyncio.wait_for` so a
stuck ``nvidia-smi`` cannot stall the loop; a timeout is counted as a
regular tick failure.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

from pydantic import Field

from concerto_bench.samplers.base import Sampler, SamplerConfig

logger = logging.getLogger(__name__)

_DEFAULT_QUERY_FIELDS: list[str] = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
]


class NvidiaSmiSamplerConfig(SamplerConfig):
    """Config for :class:`NvidiaSmiSampler`.

    ``query_fields`` is the list of columns passed to
    ``nvidia-smi --query-gpu=...``. The default covers the fields the
    Sprint 2 analyzer needs (VRAM breakdown, utilisation, temperature,
    power). Scenarios may override the list to capture additional
    columns, but the sampler is agnostic to which columns are
    requested — every column is round-tripped as a string-keyed entry
    on each per-GPU dict.
    """

    query_fields: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_QUERY_FIELDS),
        min_length=1,
        description="Columns passed to ``nvidia-smi --query-gpu``.",
    )
    binary: str = Field(
        default="nvidia-smi",
        description="Name / path of the nvidia-smi binary to invoke.",
    )


class NvidiaSmiSampler(Sampler):
    """1 Hz sampler for per-GPU telemetry via the ``nvidia-smi`` CLI."""

    config: NvidiaSmiSamplerConfig

    def __init__(self, config: NvidiaSmiSamplerConfig) -> None:
        super().__init__(config)
        self._degraded: bool = False
        self._degraded_reason: str = ""

    async def setup(self) -> None:
        """Probe for the nvidia-smi binary.

        Missing binary is not a setup failure — we just switch to
        degraded mode so the sampler keeps producing rows (empty
        payloads) for the lifetime of the run. This keeps the
        telemetry directory's shape consistent between GPU and
        non-GPU hosts, which the verifier and the analyzer rely on.
        """
        if shutil.which(self.config.binary) is None:
            self._degraded = True
            self._degraded_reason = f"{self.config.binary} not found on PATH"
            logger.info(
                "nvidia-smi sampler: %s; entering degraded mode",
                self._degraded_reason,
            )

    async def sample_once(self) -> dict[str, Any]:
        if self._degraded:
            return {
                "gpus": [],
                "degraded": True,
                "reason": self._degraded_reason,
            }

        query = ",".join(self.config.query_fields)
        argv = [
            self.config.binary,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        # Bound the subprocess so a wedged nvidia-smi cannot stall
        # the tick. Cap at 80% of the interval or 2 seconds,
        # whichever is shorter, so short intervals still leave
        # headroom before the next tick fires.
        timeout = min(self.config.interval_secs * 0.8, 2.0)

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            # The binary disappeared mid-run (uninstalled?) — flip
            # into degraded mode rather than failing every tick.
            self._degraded = True
            self._degraded_reason = f"{self.config.binary} vanished at runtime: {exc}"
            logger.warning("nvidia-smi sampler: %s", self._degraded_reason)
            return {
                "gpus": [],
                "degraded": True,
                "reason": self._degraded_reason,
            }

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:  # pragma: no cover - defensive
                pass
            raise RuntimeError(
                f"nvidia-smi timed out after {timeout:.2f}s"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi exited rc={proc.returncode}: "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )

        gpus = _parse_csv(
            stdout_bytes.decode("utf-8", errors="replace"),
            self.config.query_fields,
        )
        return {"gpus": gpus}


def _parse_csv(text: str, columns: list[str]) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --format=csv,noheader,nounits`` output.

    Each non-empty line is split on ``,`` into the requested number
    of columns; values are coerced to ``int`` or ``float`` when the
    string form is obviously numeric, otherwise retained as-is.
    Malformed rows (wrong column count) are logged and dropped.
    """
    rows: list[dict[str, Any]] = []
    expected = len(columns)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [cell.strip() for cell in line.split(",")]
        if len(fields) != expected:
            logger.info(
                "nvidia-smi sampler: skipping malformed row (%d fields, expected %d): %r",
                len(fields),
                expected,
                line,
            )
            continue
        row: dict[str, Any] = {}
        for column, value in zip(columns, fields):
            row[column] = _coerce_field(column, value)
        rows.append(row)
    return rows


def _coerce_field(column: str, value: str) -> Any:
    """Best-effort numeric coercion for one CSV cell.

    Integer-looking columns (``index``, ``memory.*``) become ``int``;
    float-looking columns (``utilization.*``, ``temperature.*``,
    ``power.*``) become ``float``. Non-numeric fields (``name``,
    ``uuid``) stay as strings. Fields that should be numeric but
    aren't (e.g. ``N/A`` for ``power.draw``) remain as strings so the
    analyzer can see the upstream anomaly.
    """
    if column == "index":
        try:
            return int(value)
        except ValueError:
            return value
    if column.startswith("memory."):
        try:
            return int(value)
        except ValueError:
            return value
    if (
        column.startswith("utilization.")
        or column.startswith("temperature.")
        or column.startswith("power.")
    ):
        try:
            return float(value)
        except ValueError:
            return value
    return value
