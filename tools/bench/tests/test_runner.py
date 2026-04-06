"""Tests for :mod:`concerto_bench.runner`.

Split into three categories:

1. **YAML parse tests** — exercise :func:`load_scenario` against
   fabricated YAML strings, verifying correct normalisation of the
   single-key action form and rejection of structural errors. No
   network, no binary, no tempdir.
2. **Step execution tests** — exercise :func:`_execute_step` against
   an ``httpx.MockTransport``-backed concerto stub. Covers happy path,
   failure recording, and ``continue_on_failure`` semantics.
3. **Sampler construction tests** — exercise :func:`_build_samplers`
   to verify that scenario-level sampler specs produce correctly
   configured concrete samplers.

Full end-to-end integration tests (spawn real concerto + mock-inference-
backend, run a scenario, validate the tarball) are deferred to
``test_dry_run_end_to_end.py`` (Phase B.2 step 13). These unit tests
pin the runner's internal contracts so later steps can refactor
confidently.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from concerto_bench.runner import (
    ActionSpec,
    RunnerOptions,
    Scenario,
    SamplerSpec,
    ScenarioError,
    StepSpec,
    _build_samplers,
    _build_summary,
    _execute_step,
    load_scenario,
)
from concerto_bench.primitives import (
    AssertPrimitive,
    KillPrimitive,
    ParallelPrimitive,
    RequestPrimitive,
    SnapshotPrimitive,
    WaitPrimitive,
    WaitForPrimitive,
    WrkLoadPrimitive,
)

# Shared primitive keyword arguments for _execute_step calls. Every test
# that drives _execute_step must pass these so the runner's dispatch
# table has all six new primitives available.
_EXTRA_PRIMITIVES = {
    "wait_primitive": WaitPrimitive(),
    "wait_for_primitive": WaitForPrimitive(),
    "kill_primitive": KillPrimitive(),
    "assert_primitive": AssertPrimitive(),
    "wrk_load_primitive": WrkLoadPrimitive(),
    "parallel_primitive": ParallelPrimitive(),
}
from concerto_bench.schema import (
    StateSnapshot,
    StepResult,
    SummaryV1,
)

UTC = timezone.utc
T0 = datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _write_scenario(tmp_path: Path, content: str) -> Path:
    """Write a scenario YAML string into a temp file and return its path."""
    path = tmp_path / "test-scenario.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. YAML parse tests
# ---------------------------------------------------------------------------


def test_load_scenario_happy_path(tmp_path: Path) -> None:
    """A minimal well-formed scenario YAML parses into the right model."""
    path = _write_scenario(
        tmp_path,
        """\
        name: smoke
        version: "1"
        steps:
          - name: greet
            actions:
              - request:
                  model: qwen2.5-0.5b
                  content: hello
        """,
    )
    scenario = load_scenario(path)
    assert scenario.name == "smoke"
    assert scenario.version == "1"
    assert len(scenario.steps) == 1
    assert scenario.steps[0].name == "greet"
    assert len(scenario.steps[0].actions) == 1
    action = scenario.steps[0].actions[0]
    assert action.type == "request"
    assert action.args["model"] == "qwen2.5-0.5b"
    assert action.args["content"] == "hello"


def test_load_scenario_expanded_action_form(tmp_path: Path) -> None:
    """The ``{type, args}`` expanded form also parses correctly."""
    path = _write_scenario(
        tmp_path,
        """\
        name: expanded
        version: "1"
        steps:
          - name: s1
            actions:
              - type: snapshot
                args: {capture_label: pre}
        """,
    )
    scenario = load_scenario(path)
    action = scenario.steps[0].actions[0]
    assert action.type == "snapshot"
    assert action.args["capture_label"] == "pre"


def test_load_scenario_with_samplers(tmp_path: Path) -> None:
    """Samplers are parsed into SamplerSpec models."""
    path = _write_scenario(
        tmp_path,
        """\
        name: with-samplers
        version: "1"
        samplers:
          - name: nvidia-smi
            interval_secs: 2.0
            config:
              binary: my-nvidia-smi
          - name: concerto-status
        steps:
          - name: s1
            actions:
              - snapshot: {}
        """,
    )
    scenario = load_scenario(path)
    assert len(scenario.samplers) == 2
    assert scenario.samplers[0].name == "nvidia-smi"
    assert scenario.samplers[0].interval_secs == 2.0
    assert scenario.samplers[0].config["binary"] == "my-nvidia-smi"
    assert scenario.samplers[1].name == "concerto-status"
    assert scenario.samplers[1].interval_secs == 1.0


def test_load_scenario_snapshot_action_null_args(tmp_path: Path) -> None:
    """``- snapshot:`` (null value) is interpreted as ``- snapshot: {}``."""
    path = _write_scenario(
        tmp_path,
        """\
        name: null-args
        version: "1"
        steps:
          - name: s1
            actions:
              - snapshot:
        """,
    )
    scenario = load_scenario(path)
    assert scenario.steps[0].actions[0].args == {}


def test_load_scenario_rejects_unknown_action_type(tmp_path: Path) -> None:
    """An action referencing a non-existent primitive fails at parse time."""
    path = _write_scenario(
        tmp_path,
        """\
        name: bad-action
        version: "1"
        steps:
          - name: s1
            actions:
              - explode: {model: boom}
        """,
    )
    with pytest.raises(ScenarioError, match="unknown action type"):
        load_scenario(path)


def test_load_scenario_rejects_empty_steps(tmp_path: Path) -> None:
    """A scenario with an empty steps list fails validation."""
    path = _write_scenario(
        tmp_path,
        """\
        name: no-steps
        version: "1"
        steps: []
        """,
    )
    with pytest.raises(ScenarioError, match="failed validation"):
        load_scenario(path)


def test_load_scenario_rejects_multi_key_action(tmp_path: Path) -> None:
    """A single-key-form action with two keys is rejected."""
    path = _write_scenario(
        tmp_path,
        """\
        name: multi-key
        version: "1"
        steps:
          - name: s1
            actions:
              - request: {model: a, content: b}
                snapshot: {}
        """,
    )
    with pytest.raises(ScenarioError, match="exactly one key"):
        load_scenario(path)


def test_load_scenario_with_parameters_and_models(tmp_path: Path) -> None:
    """Free-form ``parameters`` and ``models`` are preserved."""
    path = _write_scenario(
        tmp_path,
        """\
        name: full
        version: "2"
        description: A rich scenario.
        parameters:
          target_rps: 50
        models:
          - id: qwen2.5-0.5b
            engine: vllm
        steps:
          - name: s1
            actions:
              - request:
                  model: qwen2.5-0.5b
                  content: hi
        exit_criteria:
          max_error_rate: 0.0
        continue_on_failure: false
        """,
    )
    scenario = load_scenario(path)
    assert scenario.parameters["target_rps"] == 50
    assert len(scenario.models) == 1
    assert scenario.models[0]["id"] == "qwen2.5-0.5b"
    assert scenario.exit_criteria["max_error_rate"] == 0.0
    assert scenario.continue_on_failure is False


def test_load_scenario_rejects_nonexistent_file(tmp_path: Path) -> None:
    """A missing file raises ScenarioError, not OSError."""
    with pytest.raises(ScenarioError, match="could not read"):
        load_scenario(tmp_path / "ghost.yaml")


def test_load_scenario_rejects_non_mapping(tmp_path: Path) -> None:
    """A YAML file containing a list (not a mapping) is rejected."""
    path = _write_scenario(tmp_path, "- one\n- two\n")
    with pytest.raises(ScenarioError, match="must be a YAML mapping"):
        load_scenario(path)


# ---------------------------------------------------------------------------
# 2. Step execution tests
# ---------------------------------------------------------------------------


def _mock_handler(request: httpx.Request) -> httpx.Response:
    """Minimal concerto stub that handles /status, /metrics, and chat completions."""
    url = str(request.url)
    if url.endswith("/status"):
        return httpx.Response(200, json={"loaded_models": [], "gpus": []})
    if url.endswith("/metrics"):
        return httpx.Response(200, text="# no metrics\n")
    if url.endswith("/v1/chat/completions"):
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "test-123",
                "choices": [
                    {"message": {"role": "assistant", "content": "hello"}, "index": 0}
                ],
                "model": body.get("model", "test"),
            },
        )
    if url.endswith("/health"):
        return httpx.Response(200, json={"status": "ok"})
    return httpx.Response(404)


async def test_execute_step_happy_path() -> None:
    """A step with one request action produces a passing StepResult with one ActionRecord."""
    step_spec = StepSpec(
        name="greet",
        actions=[ActionSpec(type="request", args={"model": "qwen2.5-0.5b", "content": "hi"})],
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))
    try:
        result, pre, post, requests = await _execute_step(
            index=1,
            step_spec=step_spec,
            base_url="http://127.0.0.1:9999",
            client=client,
            request_primitive=RequestPrimitive(),
            snapshot_primitive=SnapshotPrimitive(),
            **_EXTRA_PRIMITIVES,
        )
    finally:
        await client.aclose()

    assert result.passed
    assert result.step_number == 1
    assert result.step_name == "greet"
    assert len(result.actions) == 1
    assert result.actions[0].action_type == "request"
    assert result.actions[0].passed
    assert result.failures == []
    # pre/post snapshots should have concerto_status populated
    assert isinstance(pre.concerto_status, dict)
    assert isinstance(post.concerto_status, dict)


async def test_execute_step_with_capture_as() -> None:
    """A request with ``capture_as`` is recorded in the request_records dict."""
    step_spec = StepSpec(
        name="captured",
        actions=[
            ActionSpec(
                type="request",
                args={"model": "test", "content": "x", "capture_as": "first_greet"},
            )
        ],
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))
    try:
        result, _pre, _post, requests = await _execute_step(
            index=1,
            step_spec=step_spec,
            base_url="http://127.0.0.1:9999",
            client=client,
            request_primitive=RequestPrimitive(),
            snapshot_primitive=SnapshotPrimitive(),
            **_EXTRA_PRIMITIVES,
        )
    finally:
        await client.aclose()

    assert "first_greet" in requests
    assert requests["first_greet"].status == 200
    assert result.passed


async def test_execute_step_records_request_error_as_failure() -> None:
    """A request whose expect_status mismatches records a failure but does not raise."""
    step_spec = StepSpec(
        name="expect-fail",
        actions=[
            ActionSpec(
                type="request",
                args={"model": "test", "content": "x", "expect_status": 201},
            )
        ],
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))
    try:
        result, _pre, _post, requests = await _execute_step(
            index=1,
            step_spec=step_spec,
            base_url="http://127.0.0.1:9999",
            client=client,
            request_primitive=RequestPrimitive(),
            snapshot_primitive=SnapshotPrimitive(),
            **_EXTRA_PRIMITIVES,
        )
    finally:
        await client.aclose()

    assert not result.passed
    assert len(result.failures) == 1
    assert "unexpected status" in result.failures[0]
    assert result.actions[0].passed is False


async def test_execute_step_connect_error_records_failure() -> None:
    """A transport error is captured in the action record, not raised."""
    def _refusing_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/status"):
            return httpx.Response(200, json={"loaded_models": []})
        raise httpx.ConnectError("Connection refused")

    step_spec = StepSpec(
        name="refuse",
        actions=[
            ActionSpec(type="request", args={"model": "test", "content": "x"}),
        ],
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_refusing_handler))
    try:
        result, _pre, _post, _requests = await _execute_step(
            index=1,
            step_spec=step_spec,
            base_url="http://127.0.0.1:9999",
            client=client,
            request_primitive=RequestPrimitive(),
            snapshot_primitive=SnapshotPrimitive(),
            **_EXTRA_PRIMITIVES,
        )
    finally:
        await client.aclose()

    assert not result.passed
    assert any("connect_error" in f for f in result.failures)


async def test_execute_step_snapshot_action() -> None:
    """A step with a snapshot action succeeds and returns a passing StepResult."""
    step_spec = StepSpec(
        name="snap-test",
        actions=[ActionSpec(type="snapshot", args={})],
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))
    try:
        result, _pre, _post, _requests = await _execute_step(
            index=1,
            step_spec=step_spec,
            base_url="http://127.0.0.1:9999",
            client=client,
            request_primitive=RequestPrimitive(),
            snapshot_primitive=SnapshotPrimitive(),
            **_EXTRA_PRIMITIVES,
        )
    finally:
        await client.aclose()

    assert result.passed
    assert result.actions[0].action_type == "snapshot"
    assert result.actions[0].passed


async def test_execute_step_pre_state_path_matches_slug() -> None:
    """The StepResult.pre_state_path matches the NN-<slug>/pre-state.json convention."""
    step_spec = StepSpec(
        name="My Cool Step!",
        actions=[],
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))
    try:
        result, _pre, _post, _requests = await _execute_step(
            index=3,
            step_spec=step_spec,
            base_url="http://127.0.0.1:9999",
            client=client,
            request_primitive=RequestPrimitive(),
            snapshot_primitive=SnapshotPrimitive(),
            **_EXTRA_PRIMITIVES,
        )
    finally:
        await client.aclose()

    assert result.pre_state_path == "steps/03-my-cool-step/pre-state.json"
    assert result.post_state_path == "steps/03-my-cool-step/post-state.json"


async def test_execute_step_multiple_actions() -> None:
    """A step with multiple actions processes them all in order."""
    step_spec = StepSpec(
        name="multi",
        actions=[
            ActionSpec(type="request", args={"model": "a", "content": "first"}),
            ActionSpec(type="snapshot", args={}),
            ActionSpec(type="request", args={"model": "b", "content": "second"}),
        ],
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))
    try:
        result, _pre, _post, _requests = await _execute_step(
            index=1,
            step_spec=step_spec,
            base_url="http://127.0.0.1:9999",
            client=client,
            request_primitive=RequestPrimitive(),
            snapshot_primitive=SnapshotPrimitive(),
            **_EXTRA_PRIMITIVES,
        )
    finally:
        await client.aclose()

    assert result.passed
    assert len(result.actions) == 3
    assert result.actions[0].action_type == "request"
    assert result.actions[1].action_type == "snapshot"
    assert result.actions[2].action_type == "request"


async def test_execute_step_failed_pre_snapshot_marks_step_failed() -> None:
    """If the pre-snapshot /status 5xx's, the step records a failure for the snapshot."""
    def _broken_status_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/status"):
            return httpx.Response(500, json={"error": "internal"})
        if url.endswith("/v1/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
        return httpx.Response(404)

    step_spec = StepSpec(
        name="broken-snap",
        actions=[],
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_broken_status_handler))
    try:
        result, pre, _post, _requests = await _execute_step(
            index=1,
            step_spec=step_spec,
            base_url="http://127.0.0.1:9999",
            client=client,
            request_primitive=RequestPrimitive(),
            snapshot_primitive=SnapshotPrimitive(),
            **_EXTRA_PRIMITIVES,
        )
    finally:
        await client.aclose()

    assert not result.passed
    assert any("pre-snapshot" in f for f in result.failures)
    # Pre-snapshot still returns a placeholder with empty concerto_status
    assert pre.concerto_status == {} or "error" in pre.extra


# ---------------------------------------------------------------------------
# 3. Sampler construction tests
# ---------------------------------------------------------------------------


def test_build_samplers_produces_correct_types(tmp_path: Path) -> None:
    """_build_samplers creates the right sampler class for each spec."""
    from concerto_bench.samplers import (
        NvidiaSmiSampler,
        ConcertoStatusSampler,
        ConcertoMetricsSampler,
        PgrepCounterSampler,
        ProcStatsSampler,
    )

    specs = [
        SamplerSpec(name="nvidia-smi"),
        SamplerSpec(name="concerto-status"),
        SamplerSpec(name="concerto-metrics"),
        SamplerSpec(name="pgrep-count", config={"patterns": ["vllm"]}),
        SamplerSpec(name="proc-stats"),
    ]
    samplers = _build_samplers(
        specs=specs,
        telemetry_dir=tmp_path,
        base_url="http://127.0.0.1:8000",
    )
    assert len(samplers) == 5
    assert isinstance(samplers[0], NvidiaSmiSampler)
    assert isinstance(samplers[1], ConcertoStatusSampler)
    assert isinstance(samplers[2], ConcertoMetricsSampler)
    assert isinstance(samplers[3], PgrepCounterSampler)
    assert isinstance(samplers[4], ProcStatsSampler)


def test_build_samplers_sets_output_path_under_telemetry_dir(tmp_path: Path) -> None:
    """Each sampler writes to telemetry_dir/<name>.jsonl."""
    specs = [SamplerSpec(name="nvidia-smi")]
    samplers = _build_samplers(
        specs=specs, telemetry_dir=tmp_path, base_url="http://localhost:8000"
    )
    assert samplers[0].config.output_path == tmp_path / "nvidia-smi.jsonl"


def test_build_samplers_threads_base_url_to_concerto_samplers(tmp_path: Path) -> None:
    """concerto-status and concerto-metrics receive the runner's base_url."""
    specs = [
        SamplerSpec(name="concerto-status"),
        SamplerSpec(name="concerto-metrics"),
    ]
    samplers = _build_samplers(
        specs=specs, telemetry_dir=tmp_path, base_url="http://10.0.0.1:9000"
    )
    assert samplers[0].config.base_url == "http://10.0.0.1:9000"
    assert samplers[1].config.base_url == "http://10.0.0.1:9000"


def test_build_samplers_applies_interval_override(tmp_path: Path) -> None:
    """SamplerSpec.interval_secs flows into the sampler config."""
    specs = [SamplerSpec(name="nvidia-smi", interval_secs=5.0)]
    samplers = _build_samplers(
        specs=specs, telemetry_dir=tmp_path, base_url="http://localhost:8000"
    )
    assert samplers[0].config.interval_secs == 5.0


def test_build_samplers_rejects_unknown_sampler(tmp_path: Path) -> None:
    """A spec referencing a non-registered sampler raises ScenarioError."""
    specs = [SamplerSpec(name="quantum-telemetry")]
    with pytest.raises(ScenarioError, match="unknown sampler"):
        _build_samplers(
            specs=specs, telemetry_dir=tmp_path, base_url="http://localhost:8000"
        )


# ---------------------------------------------------------------------------
# 4. Summary construction tests
# ---------------------------------------------------------------------------


def _fake_step_result(*, number: int, passed: bool, name: str = "") -> StepResult:
    """Helper to build a minimal StepResult for summary tests."""
    return StepResult(
        step_number=number,
        step_name=name or f"step-{number}",
        passed=passed,
        duration_ms=100.0,
        started_at=T0,
        ended_at=T0 + timedelta(seconds=1),
        pre_state_path="steps/00-x/pre-state.json",
        post_state_path="steps/00-x/post-state.json",
    )


def test_summary_all_steps_passed() -> None:
    """A fully-passing run sets scenario_passed=True."""
    scenario = Scenario(
        name="test", version="1",
        steps=[StepSpec(name="s1"), StepSpec(name="s2")],
    )
    results = [
        _fake_step_result(number=1, passed=True),
        _fake_step_result(number=2, passed=True),
    ]
    summary = _build_summary(
        scenario=scenario,
        step_results=results,
        failed_step_names=[],
        exit_status="success",
    )
    assert summary.scenario_passed is True
    assert summary.steps_passed == 2
    assert summary.steps_failed == 0
    assert summary.failed_step_names == []


def test_summary_one_step_failed() -> None:
    """A partial failure sets scenario_passed=False and lists the failed step."""
    scenario = Scenario(
        name="test", version="1",
        steps=[StepSpec(name="s1"), StepSpec(name="s2")],
    )
    results = [
        _fake_step_result(number=1, passed=True),
        _fake_step_result(number=2, passed=False, name="load-test"),
    ]
    summary = _build_summary(
        scenario=scenario,
        step_results=results,
        failed_step_names=["load-test"],
        exit_status="partial_failure",
    )
    assert summary.scenario_passed is False
    assert summary.steps_failed == 1
    assert summary.failed_step_names == ["load-test"]


def test_summary_error_status_is_not_passed() -> None:
    """An exit_status='error' run is not considered passed even with 0 failures."""
    scenario = Scenario(
        name="test", version="1",
        steps=[StepSpec(name="s1")],
    )
    summary = _build_summary(
        scenario=scenario,
        step_results=[],
        failed_step_names=[],
        exit_status="error",
    )
    assert summary.scenario_passed is False
    assert summary.exit_status == "error"


# ---------------------------------------------------------------------------
# 5. ActionSpec validation tests
# ---------------------------------------------------------------------------


def test_action_spec_rejects_unknown_type() -> None:
    """ActionSpec validates the type against the supported set."""
    with pytest.raises(Exception, match="unknown action type"):
        ActionSpec(type="nonexistent_action", args={})


def test_action_spec_accepts_known_types() -> None:
    """ActionSpec accepts all supported action types."""
    for action_type in ("request", "snapshot", "wait", "wait_for", "kill", "assert", "wrk_load", "parallel"):
        assert ActionSpec(type=action_type, args={}).type == action_type
