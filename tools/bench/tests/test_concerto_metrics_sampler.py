"""Unit tests for :class:`ConcertoMetricsSampler`.

Uses ``httpx.MockTransport`` injected via ``client_factory`` to serve
a realistic Prometheus text scrape; the sampler runs through a few
ticks of the base-class loop and the JSONL output is verified against
the expected flattened-metric shape.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import httpx
import pytest

from concerto_bench.samplers.base import SamplerResult
from concerto_bench.samplers.concerto_metrics import (
    ConcertoMetricsSampler,
    ConcertoMetricsSamplerConfig,
    _flatten_metric_families,
)
from concerto_bench.schema import TelemetrySample


BASE_URL = "http://127.0.0.1:18081"


SAMPLE_SCRAPE = """# HELP concerto_requests_total Requests routed by decision.
# TYPE concerto_requests_total counter
concerto_requests_total{decision="loaded"} 42
concerto_requests_total{decision="cold_start"} 3
# HELP concerto_active_backends Number of active backends.
# TYPE concerto_active_backends gauge
concerto_active_backends 2
# HELP concerto_routing_decision_duration_seconds Routing decision latency.
# TYPE concerto_routing_decision_duration_seconds histogram
concerto_routing_decision_duration_seconds_bucket{le="0.005"} 10
concerto_routing_decision_duration_seconds_bucket{le="0.01"} 20
concerto_routing_decision_duration_seconds_bucket{le="+Inf"} 25
concerto_routing_decision_duration_seconds_sum 0.18
concerto_routing_decision_duration_seconds_count 25
"""


def _config(tmp_path: Path, *, interval: float = 0.05) -> ConcertoMetricsSamplerConfig:
    return ConcertoMetricsSamplerConfig(
        name="concerto-metrics",
        interval_secs=interval,
        output_path=tmp_path / "concerto-metrics.jsonl",
        base_url=BASE_URL,
    )


def _make_factory(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[], httpx.AsyncClient]:
    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


def _read_rows(path: Path) -> list[TelemetrySample]:
    rows: list[TelemetrySample] = []
    with path.open("rb") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(TelemetrySample.model_validate_json(stripped))
    return rows


async def _drive(sampler: ConcertoMetricsSampler, secs: float) -> SamplerResult:
    task = asyncio.create_task(sampler.run())
    try:
        await asyncio.sleep(secs)
    finally:
        task.cancel()
    return await task


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_flatten_metric_families_key_shapes() -> None:
    flat = _flatten_metric_families(SAMPLE_SCRAPE)
    # Counter with a single label — key is ``name|label=value``.
    assert flat["concerto_requests_total|decision=loaded"] == 42.0
    assert flat["concerto_requests_total|decision=cold_start"] == 3.0
    # Gauge with no labels — key is just ``name``.
    assert flat["concerto_active_backends"] == 2.0
    # Histogram sum + count retained, buckets dropped.
    assert flat["concerto_routing_decision_duration_seconds_sum"] == pytest.approx(0.18)
    assert flat["concerto_routing_decision_duration_seconds_count"] == 25.0
    assert not any(
        k.endswith("_bucket") or "le=" in k for k in flat.keys()
    )


def test_flatten_metric_families_handles_multi_label_sorted_keys() -> None:
    # Gauges are preserved verbatim by prometheus_client; counters
    # are auto-suffixed with ``_total`` (which is the upstream
    # convention). Using a gauge here keeps the test focused on
    # label-key ordering without dragging in the counter rename.
    text = (
        "# TYPE demo gauge\n"
        'demo{b="two",a="one"} 7\n'
        'demo{a="one",b="two"} 7\n'
    )
    flat = _flatten_metric_families(text)
    # Labels are sorted alphabetically regardless of the order in the
    # source text, so the two lines produce the same key.
    assert flat["demo|a=one,b=two"] == 7.0
    assert len(flat) == 1


def test_flatten_metric_families_drops_nan_and_inf() -> None:
    text = (
        "# TYPE broken gauge\n"
        "broken_nan NaN\n"
        "broken_inf +Inf\n"
        "broken_ok 1.5\n"
    )
    flat = _flatten_metric_families(text)
    assert "broken_ok" in flat
    assert "broken_nan" not in flat
    assert "broken_inf" not in flat


# ---------------------------------------------------------------------------
# Sampler loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_records_flattened_metrics(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        return httpx.Response(
            200,
            text=SAMPLE_SCRAPE,
            headers={"content-type": "text/plain; version=0.0.4"},
        )

    config = _config(tmp_path)
    sampler = ConcertoMetricsSampler(config, client_factory=_make_factory(handler))
    result = await _drive(sampler, 0.18)

    assert result.ticks_succeeded >= 2
    assert result.ticks_failed == 0
    rows = _read_rows(config.output_path)
    assert len(rows) == result.ticks_succeeded
    for row in rows:
        assert row.sampler == "concerto-metrics"
        metrics = row.values["metrics"]
        assert metrics["concerto_requests_total|decision=loaded"] == 42.0
        assert metrics["concerto_active_backends"] == 2.0
        assert row.values["scrape_bytes"] == len(SAMPLE_SCRAPE)


@pytest.mark.asyncio
async def test_transport_error_counts_as_tick_failure(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("concerto not listening yet")

    config = _config(tmp_path)
    sampler = ConcertoMetricsSampler(config, client_factory=_make_factory(handler))
    result = await _drive(sampler, 0.12)

    assert result.ticks_succeeded == 0
    assert result.ticks_failed >= 1
    assert all("ConnectError" in r for r in result.failures)


@pytest.mark.asyncio
async def test_non_2xx_response_counts_as_tick_failure(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    config = _config(tmp_path)
    sampler = ConcertoMetricsSampler(config, client_factory=_make_factory(handler))
    result = await _drive(sampler, 0.12)

    assert result.ticks_succeeded == 0
    assert result.ticks_failed >= 1


@pytest.mark.asyncio
async def test_teardown_closes_injected_client(tmp_path: Path) -> None:
    clients: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=SAMPLE_SCRAPE)

    def factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    config = _config(tmp_path)
    sampler = ConcertoMetricsSampler(config, client_factory=factory)
    await _drive(sampler, 0.12)

    assert len(clients) == 1
    assert clients[0].is_closed
