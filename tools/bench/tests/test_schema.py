"""Unit tests for the versioned artifact schema.

These tests are load-bearing: every later Phase B step (artifact
builder, scenario runner, analyze/summarize, CI dry-run end-to-end test)
consumes ``concerto_bench.schema``, so the invariants asserted here —
strict-by-default validation, JSON round-trip fidelity, schema-version
pinning, timezone-aware datetimes — are what keep the rest of the rig
honest.

Test coverage spans:

1. Round-trip of every top-level model
   (``ManifestV1``, ``HostInfo``, ``TelemetrySample``, ``RequestRecord``,
   ``StateSnapshot``, ``ActionRecord``, ``StepResult``, ``SummaryV1``)
   plus representative sub-models (``LatencyHistogram``, ``ModelMetrics``,
   ``ExitCriteriaResults``).
2. ``extra="forbid"`` rejection on every top-level model.
3. Schema-version pinning — bumping ``schema_version`` to 2 must fail.
4. Forward-compat boundary — an unknown field inside a JSON blob must
   fail validation rather than being silently dropped.
5. Datetime serialisation — tz-aware UTC datetimes round-trip to ISO
   8601 strings and back without losing tzinfo.
6. Nested model validity — a fully-populated ``SummaryV1`` with
   ``ModelMetrics`` per model and a realistic ``LatencyHistogram``
   round-trips cleanly.
7. ``ManifestV1.model_json_schema()`` emits a valid JSON Schema document
   (proves the schema can be exported for documentation).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from concerto_bench import schema
from concerto_bench.schema import (
    ARTIFACT_TREE_VERSION,
    SCHEMA_VERSION,
    ActionRecord,
    ExitCriteriaResults,
    HostInfo,
    LatencyHistogram,
    ManifestV1,
    ModelMetrics,
    RequestRecord,
    StateSnapshot,
    StepResult,
    SummaryV1,
    TelemetrySample,
)

# ---------------------------------------------------------------------------
# Fixtures — realistic instances used by multiple tests
# ---------------------------------------------------------------------------


UTC = timezone.utc
T0 = datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(milliseconds=523)
T2 = T0 + timedelta(seconds=30)


def _host_info() -> HostInfo:
    return HostInfo(
        nvidia_smi_xml="<nvidia_smi_log></nvidia_smi_log>",
        lscpu_raw="Architecture: x86_64\n",
        uname={"sysname": "Linux", "release": "5.15.0"},
        vllm_version="0.5.3",
        concerto_version="concerto 0.1.0",
        concerto_git_sha="a7eb1aa",
        python_version="3.11.9",
        env_snapshot={"CUDA_VISIBLE_DEVICES": "0,1", "RUST_LOG": "info"},
        captured_at=T0,
    )


def _request_record() -> RequestRecord:
    return RequestRecord(
        status=200,
        elapsed_total_ms=123.4,
        elapsed_ttfb_ms=45.6,
        elapsed_connect_ms=7.8,
        request_body={
            "model": "qwen2.5-0.5b",
            "messages": [{"role": "user", "content": "hi"}],
        },
        response_body={
            "id": "cmpl-abc",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        },
        response_chunks=None,
        error=None,
    )


def _snapshot() -> StateSnapshot:
    return StateSnapshot(
        ts=T0,
        concerto_status={
            "loaded_models": ["qwen2.5-0.5b"],
            "gpus": [{"id": 0, "memory_used_mb": 1024}],
        },
        nvidia_smi={"gpus": [{"id": 0, "memory_used_mb": 1100, "memory_total_mb": 24576}]},
        backend_pids=[12345, 12346],
        extra={"note": "pre-step snapshot"},
    )


def _action_record() -> ActionRecord:
    return ActionRecord(
        action_type="request",
        args={"model": "qwen2.5-0.5b", "content": "hi"},
        started_at=T0,
        ended_at=T1,
        duration_ms=523.0,
        passed=True,
        failure_reason=None,
        output={"status": 200},
    )


def _step_result() -> StepResult:
    return StepResult(
        step_number=1,
        step_name="single-model-smoke",
        passed=True,
        duration_ms=523.0,
        started_at=T0,
        ended_at=T1,
        pre_state_path="steps/01-single-model-smoke/pre-state.json",
        post_state_path="steps/01-single-model-smoke/post-state.json",
        actions=[_action_record()],
        failures=[],
    )


def _telemetry_sample() -> TelemetrySample:
    return TelemetrySample(
        ts=T0,
        sampler="nvidia-smi",
        values={
            "gpu_id": 0,
            "memory_used_mb": 1100,
            "memory_total_mb": 24576,
            "utilisation_percent": 37.5,
        },
    )


def _latency_hist() -> LatencyHistogram:
    return LatencyHistogram(
        p50_ms=1.1,
        p95_ms=3.4,
        p99_ms=4.9,
        max_ms=12.0,
        count=10_000,
    )


def _model_metrics() -> ModelMetrics:
    return ModelMetrics(
        model_id="qwen2.5-0.5b",
        cold_start_ms=8_420.0,
        launch_count=1,
        eviction_count=0,
        request_count=523,
        error_count=0,
    )


def _exit_criteria() -> ExitCriteriaResults:
    return ExitCriteriaResults(
        launched_count_ok=True,
        stopped_count_ok=True,
        http_error_rate_ok=True,
        vram_drift_ok=True,
        graceful_shutdown_time_ok=True,
        orphan_processes_ok=True,
    )


def _manifest() -> ManifestV1:
    return ManifestV1(
        rig_version="0.1.0.dev0",
        concerto_version="concerto 0.1.0",
        concerto_git_sha="a7eb1aa",
        scenario_name="sprint-2-validation",
        scenario_version="1.0.0",
        started_at=T0,
        ended_at=T2,
        duration_seconds=30.0,
        exit_status="success",
        step_count=8,
        host_info_path="environment.json",
    )


def _summary() -> SummaryV1:
    return SummaryV1(
        scenario_name="sprint-2-validation",
        scenario_version="1.0.0",
        exit_status="success",
        scenario_passed=True,
        step_count=8,
        steps_passed=8,
        steps_failed=0,
        failed_step_names=[],
        launched_count=4,
        stopped_count=2,
        http_error_rate=0.0,
        vram_drift_max_percent=3.2,
        graceful_shutdown_wall_time_secs=12.5,
        orphan_processes_after_shutdown=0,
        routing_decision_latency=_latency_hist(),
        concurrent_load_throughput_rps=182.4,
        concurrent_load_error_rate=0.0,
        concurrent_load_latency=LatencyHistogram(
            p50_ms=54.0, p95_ms=120.0, p99_ms=180.0, max_ms=400.0, count=54_000
        ),
        model_metrics={
            "qwen2.5-0.5b": _model_metrics(),
            "phi-3-mini": ModelMetrics(
                model_id="phi-3-mini",
                cold_start_ms=23_100.0,
                launch_count=1,
                eviction_count=1,
                request_count=120,
                error_count=0,
            ),
            "qwen2.5-7b": ModelMetrics(
                model_id="qwen2.5-7b",
                cold_start_ms=87_500.0,
                launch_count=1,
                eviction_count=0,
                request_count=45,
                error_count=0,
            ),
        },
        exit_criteria=_exit_criteria(),
    )


TOP_LEVEL_FACTORIES: list[tuple[str, Any]] = [
    ("ManifestV1", _manifest),
    ("HostInfo", _host_info),
    ("TelemetrySample", _telemetry_sample),
    ("RequestRecord", _request_record),
    ("StateSnapshot", _snapshot),
    ("ActionRecord", _action_record),
    ("StepResult", _step_result),
    ("SummaryV1", _summary),
]


# ---------------------------------------------------------------------------
# 1. Module-level API
# ---------------------------------------------------------------------------


def test_schema_version_constant_is_one() -> None:
    assert SCHEMA_VERSION == 1


def test_artifact_tree_version_constant_is_one() -> None:
    assert ARTIFACT_TREE_VERSION == 1


def test_all_exports_are_importable_from_module() -> None:
    for name in schema.__all__:
        assert hasattr(schema, name), f"{name} missing from concerto_bench.schema"


# ---------------------------------------------------------------------------
# 2. Round-trip every top-level model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "factory"),
    TOP_LEVEL_FACTORIES,
    ids=[name for name, _ in TOP_LEVEL_FACTORIES],
)
def test_top_level_model_json_round_trip(name: str, factory: Any) -> None:
    original = factory()
    payload = original.model_dump_json()
    parsed = type(original).model_validate_json(payload)
    assert parsed == original, f"{name} failed round-trip"


def test_latency_histogram_round_trip() -> None:
    original = _latency_hist()
    parsed = LatencyHistogram.model_validate_json(original.model_dump_json())
    assert parsed == original


def test_model_metrics_round_trip() -> None:
    original = _model_metrics()
    parsed = ModelMetrics.model_validate_json(original.model_dump_json())
    assert parsed == original


def test_exit_criteria_results_round_trip() -> None:
    original = _exit_criteria()
    parsed = ExitCriteriaResults.model_validate_json(original.model_dump_json())
    assert parsed == original


# ---------------------------------------------------------------------------
# 3. extra="forbid" rejection on every top-level model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "factory"),
    TOP_LEVEL_FACTORIES,
    ids=[name for name, _ in TOP_LEVEL_FACTORIES],
)
def test_top_level_model_rejects_unknown_field(name: str, factory: Any) -> None:
    original = factory()
    as_dict = original.model_dump(mode="json")
    as_dict["this_field_does_not_exist_in_v1"] = "surprise"
    with pytest.raises(ValidationError):
        type(original).model_validate(as_dict)


# ---------------------------------------------------------------------------
# 4. Schema version pinning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "factory"),
    TOP_LEVEL_FACTORIES,
    ids=[name for name, _ in TOP_LEVEL_FACTORIES],
)
def test_schema_version_two_is_rejected(name: str, factory: Any) -> None:
    original = factory()
    as_dict = original.model_dump(mode="json")
    as_dict["schema_version"] = 2
    with pytest.raises(ValidationError):
        type(original).model_validate(as_dict)


# ---------------------------------------------------------------------------
# 5. Forward-compat boundary — unknown field in a JSON blob must fail parse
# ---------------------------------------------------------------------------


def test_manifest_unknown_field_in_json_blob_fails_to_parse() -> None:
    """Encodes the 'schema bumps are explicit' invariant."""
    blob = (
        '{"schema_version": 1, "artifact_tree_version": 1, '
        '"rig_version": "0.1.0.dev0", "concerto_version": "concerto 0.1.0", '
        '"concerto_git_sha": null, "scenario_name": "sprint-2-validation", '
        '"scenario_version": "1.0.0", '
        '"started_at": "2026-04-05T12:00:00Z", '
        '"ended_at": "2026-04-05T12:00:30Z", '
        '"duration_seconds": 30.0, "exit_status": "success", '
        '"step_count": 8, "host_info_path": "environment.json", '
        '"future_v2_field": "boom"}'
    )
    with pytest.raises(ValidationError):
        ManifestV1.model_validate_json(blob)


# ---------------------------------------------------------------------------
# 6. Datetime serialisation — tz-aware UTC round-trip
# ---------------------------------------------------------------------------


def test_naive_datetime_is_rejected() -> None:
    naive = datetime(2026, 4, 5, 12, 0, 0)  # no tzinfo
    with pytest.raises(ValidationError):
        TelemetrySample(ts=naive, sampler="nvidia-smi", values={})


def test_non_utc_tz_aware_datetime_is_normalised_to_utc() -> None:
    """A timezone-aware non-UTC datetime should be accepted and stored as UTC."""
    tokyo = timezone(timedelta(hours=9))
    t = datetime(2026, 4, 5, 21, 0, 0, tzinfo=tokyo)  # == 12:00 UTC
    sample = TelemetrySample(ts=t, sampler="nvidia-smi", values={})
    assert sample.ts.tzinfo is not None
    assert sample.ts.utcoffset() == timedelta(0)
    assert sample.ts == datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC)


def test_datetime_round_trips_preserving_tzinfo() -> None:
    sample = _telemetry_sample()
    parsed = TelemetrySample.model_validate_json(sample.model_dump_json())
    assert parsed.ts.tzinfo is not None
    assert parsed.ts.utcoffset() == timedelta(0)
    assert parsed.ts == T0


def test_step_result_datetime_round_trip() -> None:
    step = _step_result()
    parsed = StepResult.model_validate_json(step.model_dump_json())
    assert parsed.started_at == T0
    assert parsed.ended_at == T1
    assert parsed.actions[0].started_at.tzinfo is not None


# ---------------------------------------------------------------------------
# 7. Nested model validity
# ---------------------------------------------------------------------------


def test_fully_populated_summary_round_trips() -> None:
    summary = _summary()
    blob = summary.model_dump_json()
    parsed = SummaryV1.model_validate_json(blob)
    assert parsed == summary
    assert parsed.routing_decision_latency is not None
    assert parsed.routing_decision_latency.count == 10_000
    assert parsed.model_metrics["qwen2.5-7b"].cold_start_ms == 87_500.0
    assert parsed.exit_criteria.vram_drift_ok is True


def test_summary_with_partial_failure_round_trips() -> None:
    """Scenarios that abort early should still produce a round-trippable summary."""
    summary = SummaryV1(
        scenario_name="sprint-2-validation",
        scenario_version="1.0.0",
        exit_status="partial_failure",
        scenario_passed=False,
        step_count=3,
        steps_passed=2,
        steps_failed=1,
        failed_step_names=["03-eviction"],
        # every optional metric left unset — scenario aborted before the
        # concurrent-load and graceful-shutdown steps ran
    )
    parsed = SummaryV1.model_validate_json(summary.model_dump_json())
    assert parsed == summary
    assert parsed.routing_decision_latency is None
    assert parsed.concurrent_load_latency is None
    assert parsed.exit_criteria.graceful_shutdown_time_ok is None


def test_step_result_contains_nested_action_record() -> None:
    step = _step_result()
    parsed = StepResult.model_validate_json(step.model_dump_json())
    assert len(parsed.actions) == 1
    assert parsed.actions[0].action_type == "request"
    assert parsed.actions[0].passed is True


# ---------------------------------------------------------------------------
# 8. JSON Schema export — analyzer can emit schema documentation
# ---------------------------------------------------------------------------


def test_manifest_json_schema_export() -> None:
    json_schema = ManifestV1.model_json_schema()
    # pydantic v2 emits Draft 2020-12 JSON Schema.
    assert "$schema" in json_schema or "type" in json_schema
    assert "properties" in json_schema
    assert "schema_version" in json_schema["properties"]
    assert "rig_version" in json_schema["properties"]
    assert "concerto_version" in json_schema["properties"]


def test_summary_json_schema_export_lists_all_metrics() -> None:
    json_schema = SummaryV1.model_json_schema()
    assert "properties" in json_schema
    props = json_schema["properties"]
    for required_metric in [
        "launched_count",
        "stopped_count",
        "http_error_rate",
        "vram_drift_max_percent",
        "graceful_shutdown_wall_time_secs",
        "orphan_processes_after_shutdown",
        "routing_decision_latency",
        "concurrent_load_throughput_rps",
        "concurrent_load_error_rate",
        "concurrent_load_latency",
        "model_metrics",
        "exit_criteria",
    ]:
        assert required_metric in props, f"SummaryV1 schema missing {required_metric}"


# ---------------------------------------------------------------------------
# 9. Frozen-model invariant — top-level records are immutable post-capture
# ---------------------------------------------------------------------------


def test_manifest_is_frozen() -> None:
    m = _manifest()
    with pytest.raises(ValidationError):
        m.rig_version = "99.99"  # type: ignore[misc]


def test_summary_is_frozen() -> None:
    s = _summary()
    with pytest.raises(ValidationError):
        s.scenario_passed = False  # type: ignore[misc]
