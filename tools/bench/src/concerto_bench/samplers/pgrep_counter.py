"""Sampler: periodic ``pgrep -c <pattern>`` counts per pattern.

Counts how many processes match each configured pattern at 1 Hz, writes
one row per tick into ``telemetry/pgrep-count.jsonl`` with a
``{"counts": {pattern: int, ...}, "total": int}`` payload. Useful as a
cheap orphan-backend detector — if ``total`` goes non-zero after a
shutdown step, the run has leaked a backend.

``pgrep -c`` exits with rc=1 when no matches are found. That is **not**
a failure for this sampler; it simply means the count is zero. Other
non-zero exit codes are treated as tick failures (the tick is counted
as failed but the loop continues).

On hosts without ``pgrep`` (unlikely outside Windows but cheaply
defensive), the sampler enters degraded mode and emits empty rows for
the lifetime of the run, mirroring the nvidia-smi sampler's "no data
source" behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

from pydantic import Field

from concerto_bench.samplers.base import Sampler, SamplerConfig

logger = logging.getLogger(__name__)


class PgrepCounterSamplerConfig(SamplerConfig):
    """Config for :class:`PgrepCounterSampler`.

    ``patterns`` is the list of regex patterns passed to ``pgrep``.
    Each pattern is counted independently; the sampler also records
    the sum across all patterns as ``total``.
    """

    patterns: list[str] = Field(
        ...,
        min_length=1,
        description="``pgrep`` patterns to count on every tick.",
    )
    binary: str = Field(
        default="pgrep",
        description="Name / path of the pgrep binary to invoke.",
    )


class PgrepCounterSampler(Sampler):
    """1 Hz process counter sampler."""

    config: PgrepCounterSamplerConfig

    def __init__(self, config: PgrepCounterSamplerConfig) -> None:
        super().__init__(config)
        self._degraded: bool = False
        self._degraded_reason: str = ""

    async def setup(self) -> None:
        if shutil.which(self.config.binary) is None:
            self._degraded = True
            self._degraded_reason = f"{self.config.binary} not found on PATH"
            logger.info(
                "pgrep sampler: %s; entering degraded mode",
                self._degraded_reason,
            )

    async def sample_once(self) -> dict[str, Any]:
        if self._degraded:
            return {
                "counts": {pattern: 0 for pattern in self.config.patterns},
                "total": 0,
                "degraded": True,
                "reason": self._degraded_reason,
            }

        counts: dict[str, int] = {}
        timeout = min(self.config.interval_secs * 0.8, 2.0)

        for pattern in self.config.patterns:
            try:
                proc = await asyncio.create_subprocess_exec(
                    self.config.binary,
                    "-c",
                    pattern,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                self._degraded = True
                self._degraded_reason = (
                    f"{self.config.binary} vanished at runtime: {exc}"
                )
                logger.warning("pgrep sampler: %s", self._degraded_reason)
                return {
                    "counts": {p: 0 for p in self.config.patterns},
                    "total": 0,
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
                    f"pgrep timed out after {timeout:.2f}s for pattern {pattern!r}"
                ) from exc

            rc = proc.returncode
            text = stdout_bytes.decode("utf-8", errors="replace").strip()
            if rc == 0:
                try:
                    counts[pattern] = int(text) if text else 0
                except ValueError as exc:
                    raise RuntimeError(
                        f"pgrep -c returned non-integer output {text!r} "
                        f"for pattern {pattern!r}"
                    ) from exc
            elif rc == 1:
                # "no matches" — a legitimate outcome, not an error.
                counts[pattern] = 0
            else:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"pgrep -c exited rc={rc} for pattern {pattern!r}: {stderr_text}"
                )

        total = sum(counts.values())
        return {"counts": counts, "total": total}
