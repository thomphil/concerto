"""Background 1 Hz time-series samplers for the bench rig.

The bench rig captures telemetry during scenario execution by running
a handful of small samplers concurrently, each on its own 1 Hz tick,
each streaming rows into its own ``telemetry/<name>.jsonl`` file. This
package hosts the abstract base class that owns the loop, the
concrete samplers, and a registry that maps scenario-YAML sampler
names to their implementation class.

Concrete samplers shipped in Sprint 2:

* :class:`NvidiaSmiSampler` — per-GPU VRAM / utilisation / temperature
  / power. Degrades gracefully on non-GPU hosts (empty rows).
* :class:`ConcertoStatusSampler` — ``GET /status`` response body.
* :class:`ConcertoMetricsSampler` — ``GET /metrics`` flattened into
  ``{metric|label=value: float}`` rows.
* :class:`PgrepCounterSampler` — ``pgrep -c <pattern>`` counts per
  pattern + total. Useful as a cheap orphan-backend detector.
* :class:`ProcStatsSampler` — ``/proc/loadavg`` + ``/proc/meminfo``
  snapshot. Degrades on non-Linux hosts.

The scenario runner (Phase B.2 step 7) drives samplers via
:class:`SamplerPool`; ``run_samplers`` is a simpler one-shot helper
for tests and ad-hoc usage.
"""

from __future__ import annotations

from concerto_bench.samplers.base import (
    Sampler,
    SamplerConfig,
    SamplerError,
    SamplerPool,
    SamplerRegistry,
    SamplerResult,
    run_samplers,
)
from concerto_bench.samplers.concerto_metrics import (
    ConcertoMetricsSampler,
    ConcertoMetricsSamplerConfig,
)
from concerto_bench.samplers.concerto_status import (
    ConcertoStatusSampler,
    ConcertoStatusSamplerConfig,
)
from concerto_bench.samplers.nvidia_smi import (
    NvidiaSmiSampler,
    NvidiaSmiSamplerConfig,
)
from concerto_bench.samplers.pgrep_counter import (
    PgrepCounterSampler,
    PgrepCounterSamplerConfig,
)
from concerto_bench.samplers.proc_stats import (
    ProcStatsSampler,
    ProcStatsSamplerConfig,
)

DEFAULT_REGISTRY = SamplerRegistry()
"""Registry pre-populated with every built-in Sprint 2 sampler.

Scenario YAML uses these string keys — see SPRINT-2-PLAN §4 B.3 and
the ``sampler`` field on :class:`~concerto_bench.schema.TelemetrySample`.
"""

DEFAULT_REGISTRY.register("nvidia-smi", NvidiaSmiSampler)
DEFAULT_REGISTRY.register("concerto-status", ConcertoStatusSampler)
DEFAULT_REGISTRY.register("concerto-metrics", ConcertoMetricsSampler)
DEFAULT_REGISTRY.register("pgrep-count", PgrepCounterSampler)
DEFAULT_REGISTRY.register("proc-stats", ProcStatsSampler)


__all__ = [
    "Sampler",
    "SamplerConfig",
    "SamplerError",
    "SamplerPool",
    "SamplerRegistry",
    "SamplerResult",
    "run_samplers",
    "DEFAULT_REGISTRY",
    "ConcertoMetricsSampler",
    "ConcertoMetricsSamplerConfig",
    "ConcertoStatusSampler",
    "ConcertoStatusSamplerConfig",
    "NvidiaSmiSampler",
    "NvidiaSmiSamplerConfig",
    "PgrepCounterSampler",
    "PgrepCounterSamplerConfig",
    "ProcStatsSampler",
    "ProcStatsSamplerConfig",
]
