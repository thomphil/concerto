"""Sampler: periodic ``GET /metrics`` against the running concerto API.

Captures the Prometheus text-format scrape at 1 Hz and flattens it into
a dict of ``{metric_key: float_value}`` rows in
``telemetry/concerto-metrics.jsonl``. The key schema is:

* ``<metric_name>`` — unlabeled metric (counter or gauge).
* ``<metric_name>|label_a=value_a,label_b=value_b`` — labeled sample,
  with label pairs sorted alphabetically so the same sample always
  serialises to the same key. The label-pair separator is ``,`` with
  no spaces.
* ``<metric_name>_sum`` / ``<metric_name>_count`` — histogram and
  summary aggregates are retained at their own keys. Internal
  histogram buckets are ignored; the analyzer can reconstruct tail
  latencies from the routing-decision latency explicitly captured
  elsewhere.

The text parse uses ``prometheus_client.parser.text_string_to_metric_families``
so the sampler never has to own a Prometheus parser.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import httpx
from prometheus_client.parser import text_string_to_metric_families
from pydantic import Field

from concerto_bench.samplers.base import Sampler, SamplerConfig

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECS = 0.5

ClientFactory = Callable[[], httpx.AsyncClient]


class ConcertoMetricsSamplerConfig(SamplerConfig):
    """Config for :class:`ConcertoMetricsSampler`.

    Adds a ``base_url`` pointing at the running concerto API. Everything
    else is inherited from :class:`SamplerConfig`.
    """

    base_url: str = Field(
        ...,
        min_length=1,
        description="Base URL of the concerto API, e.g. ``http://127.0.0.1:8000``.",
    )
    timeout_secs: float = Field(
        default=_DEFAULT_TIMEOUT_SECS,
        gt=0.0,
        description="Per-request timeout in seconds.",
    )


def _flatten_metric_families(text: str) -> dict[str, float]:
    """Flatten a Prometheus scrape into ``{key: float}`` rows.

    Counter + gauge samples land at their natural name or
    ``name|label=value,...`` key. Histogram / summary ``_bucket``
    samples are dropped; ``_sum`` and ``_count`` are retained (they
    come through the parser as distinct samples with the suffix
    already baked into ``sample.name``).

    NaN values are dropped rather than round-tripped through JSON
    because the :class:`TelemetrySample` ``values`` dict is later
    round-tripped via ``model_dump_json`` which will reject them.
    """
    flattened: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name = sample.name
            if name.endswith("_bucket"):
                # Histogram bucket internals — the analyzer is not
                # interested in the full CDF at the sampler level.
                continue
            try:
                value = float(sample.value)
            except (TypeError, ValueError):
                continue
            # Drop non-finite values — JSON serialisation cannot
            # represent NaN or inf round-trippably and the analyzer
            # treats missing keys as "no data" anyway.
            if value != value or value == float("inf") or value == float("-inf"):
                continue
            labels = sample.labels or {}
            if labels:
                label_str = ",".join(
                    f"{k}={labels[k]}" for k in sorted(labels.keys())
                )
                key = f"{name}|{label_str}"
            else:
                key = name
            flattened[key] = value
    return flattened


class ConcertoMetricsSampler(Sampler):
    """1 Hz sampler that records flattened Prometheus metrics."""

    config: ConcertoMetricsSamplerConfig

    def __init__(
        self,
        config: ConcertoMetricsSamplerConfig,
        *,
        client_factory: Optional[ClientFactory] = None,
    ) -> None:
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._client_factory = client_factory

    async def setup(self) -> None:
        if self._client_factory is not None:
            self._client = self._client_factory()
        else:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_secs)

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def sample_once(self) -> dict[str, Any]:
        assert self._client is not None, "ConcertoMetricsSampler.setup was not called"
        url = f"{self.config.base_url.rstrip('/')}/metrics"
        response = await self._client.get(url, timeout=self.config.timeout_secs)
        response.raise_for_status()
        text = response.text
        flattened = _flatten_metric_families(text)
        # Wrap in a dict field so the JSONL payload structure stays
        # stable and extensible (future samplers can add a
        # "scrape_bytes" field or similar without breaking the
        # analyzer's top-level shape).
        return {"metrics": flattened, "scrape_bytes": len(text)}
