"""Sampler: periodic capture of ``/proc/loadavg`` and ``/proc/meminfo``.

On Linux hosts this emits one row per tick into
``telemetry/proc-stats.jsonl`` with:

* ``loadavg``: ``{"load1": float, "load5": float, "load15": float}``
* ``meminfo``: a subset of ``/proc/meminfo`` keys converted from kB to
  bytes (``mem_total``, ``mem_available``, ``mem_free``, ``buffers``,
  ``cached``, ``swap_total``, ``swap_free``).

On non-Linux hosts (macOS developer boxes — where the Phase B tests
run locally) neither pseudofile exists. Rather than failing on every
tick the sampler enters **degraded mode** in :meth:`setup` and emits
an empty payload per tick so the JSONL shape remains consistent
across platforms.

Individual sub-captures (``loadavg`` or ``meminfo``) are allowed to
degrade independently if one file exists but the other doesn't.
Malformed ``/proc/meminfo`` lines are skipped with an INFO log rather
than crashing the tick — we always want the loadavg numbers even if
meminfo is in a weird state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from concerto_bench.samplers.base import Sampler, SamplerConfig

logger = logging.getLogger(__name__)

_DEFAULT_MEMINFO_KEYS: tuple[str, ...] = (
    "MemTotal",
    "MemAvailable",
    "MemFree",
    "Buffers",
    "Cached",
    "SwapTotal",
    "SwapFree",
)


class ProcStatsSamplerConfig(SamplerConfig):
    """Config for :class:`ProcStatsSampler`.

    The loadavg and meminfo paths are configurable so tests can point
    at fixture files in a ``tmp_path`` directory instead of the real
    ``/proc`` tree. Production scenarios leave the defaults in place.
    """

    loadavg_path: Path = Field(
        default=Path("/proc/loadavg"),
        description="Path to the loadavg pseudofile.",
    )
    meminfo_path: Path = Field(
        default=Path("/proc/meminfo"),
        description="Path to the meminfo pseudofile.",
    )


class ProcStatsSampler(Sampler):
    """1 Hz host load + memory sampler."""

    config: ProcStatsSamplerConfig

    def __init__(self, config: ProcStatsSamplerConfig) -> None:
        super().__init__(config)
        self._loadavg_available: bool = False
        self._meminfo_available: bool = False
        self._degraded: bool = False
        self._degraded_reason: str = ""

    async def setup(self) -> None:
        self._loadavg_available = self.config.loadavg_path.exists()
        self._meminfo_available = self.config.meminfo_path.exists()
        if not (self._loadavg_available or self._meminfo_available):
            self._degraded = True
            self._degraded_reason = (
                f"neither {self.config.loadavg_path} nor "
                f"{self.config.meminfo_path} exists"
            )
            logger.info(
                "proc-stats sampler: %s; entering degraded mode",
                self._degraded_reason,
            )

    async def sample_once(self) -> dict[str, Any]:
        if self._degraded:
            return {
                "loadavg": None,
                "meminfo": None,
                "degraded": True,
                "reason": self._degraded_reason,
            }

        loadavg: Optional[dict[str, float]] = None
        meminfo: Optional[dict[str, int]] = None

        if self._loadavg_available:
            loadavg = _read_loadavg(self.config.loadavg_path)
        if self._meminfo_available:
            meminfo = _read_meminfo(self.config.meminfo_path, _DEFAULT_MEMINFO_KEYS)

        payload: dict[str, Any] = {"loadavg": loadavg, "meminfo": meminfo}
        return payload


def _read_loadavg(path: Path) -> dict[str, float]:
    """Parse ``/proc/loadavg``.

    Format is ``load1 load5 load15 running/total lastpid``. We retain
    only the three load averages; a line with fewer than three fields
    raises, which the loop counts as a tick failure.
    """
    text = path.read_text(encoding="utf-8")
    parts = text.strip().split()
    if len(parts) < 3:
        raise RuntimeError(f"unexpected /proc/loadavg shape: {text!r}")
    return {
        "load1": float(parts[0]),
        "load5": float(parts[1]),
        "load15": float(parts[2]),
    }


def _read_meminfo(path: Path, wanted: tuple[str, ...]) -> dict[str, int]:
    """Parse ``/proc/meminfo`` into a ``{snake_case_key: bytes}`` dict.

    Each line is ``Key:  value kB``. Values are converted from kB to
    bytes. Keys are normalised to snake_case (``MemAvailable`` →
    ``mem_available``) so the analyzer consumes a stable shape.

    Lines that do not match the expected ``key: value unit`` shape
    are logged at INFO and skipped; a malformed line does not cause
    the tick to fail.
    """
    result: dict[str, int] = {}
    wanted_set = set(wanted)
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if ":" not in raw_line:
            logger.info("proc-stats sampler: skipping malformed meminfo line %r", raw_line)
            continue
        key, _, rest = raw_line.partition(":")
        key = key.strip()
        if key not in wanted_set:
            continue
        parts = rest.strip().split()
        if not parts:
            logger.info(
                "proc-stats sampler: empty meminfo value for %r", key
            )
            continue
        try:
            value = int(parts[0])
        except ValueError:
            logger.info(
                "proc-stats sampler: non-integer meminfo value for %r: %r",
                key,
                parts[0],
            )
            continue
        unit = parts[1].lower() if len(parts) > 1 else ""
        if unit == "kb":
            value *= 1024
        elif unit == "mb":
            value *= 1024 * 1024
        # Otherwise assume bytes.
        result[_snake(key)] = value
    return result


def _snake(name: str) -> str:
    """Convert a meminfo key (``MemAvailable``) to snake_case (``mem_available``)."""
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
