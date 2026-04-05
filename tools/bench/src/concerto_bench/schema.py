"""Versioned output schema for the bench rig's artifact tarball.

Every persistent record the rig emits — ``manifest.json``, ``summary.json``,
per-step ``result.json`` files, pre/post state snapshots, telemetry JSONL
rows, and the host-info blob — is defined here as a pydantic v2 model.
These models are the **load-bearing type definitions** for the entire
rig: artifact builder, scenario runner, analyze/summarize, and the CI
dry-run e2e test all consume them.

Versioning
----------

Every top-level record carries a ``schema_version: Literal[1]`` field.
When the on-disk schema needs a breaking change, the Literal becomes a
``Literal[2]`` in a new model and the analyzer branches on the version
tag. Today, ``SCHEMA_VERSION = 1`` is the only shape that validates.

Strictness
----------

All models use ``ConfigDict(extra="forbid")``. An unknown field in a
candidate tarball fails fast with ``pydantic.ValidationError`` rather
than being silently dropped — that is what makes "versioned schema"
mean something operationally. Most models are also ``frozen=True``:
rig output is immutable post-capture.

Round-trip
----------

``model.model_dump_json()`` → ``Model.model_validate_json(...)`` must
round-trip every model exactly. The unit tests in
``tests/test_schema.py`` encode this as a hard invariant across every
top-level record plus representative nested shapes.

See ``SPRINT-2-PLAN.md`` §3, §4 B.1, §4 B.2 step 3, §4 B.3, §4 B.4,
§5, and §9 for the sources of truth that drove these shapes. See
``ROADMAP.md`` §7 for the metrics list that ``SummaryV1`` must be
able to represent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1
"""Current on-disk schema version for artifact JSON records.

Bumped only when a field is removed or semantically changed. Additive
field-level evolution is not permitted under v1 because every model is
``extra="forbid"`` — forward compatibility is a deliberate, explicit
schema bump, not a silent accretion.
"""

ARTIFACT_TREE_VERSION = 1
"""Current on-disk tarball layout version.

Separate from ``SCHEMA_VERSION``. The tree version describes where
files live inside the tarball (``steps/NN-<name>/``, ``telemetry/*.jsonl``,
``environment.json``, etc.); the schema version describes the contents
of those files. Either can be bumped independently.
"""

ExitStatus = Literal["success", "partial_failure", "error"]
"""Outcome tag for a scenario run.

- ``success``: every step passed and every scenario-level exit criterion
  was satisfied.
- ``partial_failure``: the run completed end-to-end but at least one
  step failed or one exit criterion was not met. The artifact is still
  valid and fully populated.
- ``error``: the rig itself failed before producing a complete artifact
  (e.g. concerto failed to start, the YAML was invalid, an unhandled
  exception escaped the runner). The tarball may be partial.
"""


def _ensure_tz_aware(value: datetime) -> datetime:
    """Field-validator helper: datetimes in the artifact schema must be UTC-aware.

    Naive datetimes cause silent bugs when artifacts are harvested from a
    remote box and analysed on a laptop in a different timezone. The rig
    captures everything as timezone-aware UTC at source and the schema
    enforces that contract at parse time.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Host / environment capture
# ---------------------------------------------------------------------------


class HostInfo(BaseModel):
    """Environment snapshot captured once at the start of a run.

    Populated by ``environment.py`` (a later Phase B step). Stored in
    the tarball as ``environment.json`` so post-hoc analysis can
    attribute observed metrics to the hardware, driver, kernel, and
    software versions they were measured on. Any field the rig could
    not capture on this host (e.g. ``nvidia-smi`` on a GPU-less dev
    laptop) is ``None`` or empty rather than fabricated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(
        default=1,
        description="Pinned to the HostInfo v1 shape.",
    )
    nvidia_smi_xml: Optional[str] = Field(
        default=None,
        description="Raw ``nvidia-smi -q -x`` XML dump, or None on non-GPU hosts.",
    )
    lscpu_raw: Optional[str] = Field(
        default=None,
        description="Raw ``lscpu`` output, or None where unavailable.",
    )
    uname: dict[str, str] = Field(
        default_factory=dict,
        description="Parsed ``uname -a`` fields (sysname, release, version, machine).",
    )
    vllm_version: Optional[str] = Field(
        default=None,
        description="Installed vLLM version string, or None if vLLM is not installed.",
    )
    concerto_version: str = Field(
        ...,
        description="Output of ``concerto --version`` captured at runtime.",
    )
    concerto_git_sha: Optional[str] = Field(
        default=None,
        description="Git SHA of the concerto build, if determinable.",
    )
    python_version: str = Field(
        ...,
        description="``sys.version`` string of the Python interpreter running the rig.",
    )
    env_snapshot: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Whitelisted subset of ``os.environ`` (e.g. CUDA_VISIBLE_DEVICES, "
            "RUST_LOG, HF_HOME). Never a full dump — secrets must not leak."
        ),
    )
    captured_at: datetime = Field(
        ...,
        description="UTC, timezone-aware wall-clock time this snapshot was taken.",
    )

    @field_validator("captured_at")
    @classmethod
    def _validate_captured_at(cls, value: datetime) -> datetime:
        return _ensure_tz_aware(value)


# ---------------------------------------------------------------------------
# Telemetry (1 Hz sampler rows)
# ---------------------------------------------------------------------------


class TelemetrySample(BaseModel):
    """A single row in a 1 Hz sampler's JSONL stream.

    The rig runs N samplers concurrently (``nvidia-smi``,
    ``concerto-status``, ``concerto-metrics``, ``pgrep-count``,
    ``proc-stats``). Each sampler writes one JSONL file under
    ``telemetry/<sampler-name>.jsonl``; every row deserialises to a
    ``TelemetrySample``.

    ``values`` is an intentionally open ``dict[str, Any]`` payload — the
    concrete per-sampler shape is defined by the sampler implementation
    (a later Phase B step). Strict typing can come later as a
    discriminated union if the per-sampler payloads stabilise; v1
    picks the looser shape because round-trippable maintenance is more
    important than structural typing on an internal boundary the
    analyzer controls.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(
        default=1,
        description="Pinned to the TelemetrySample v1 shape.",
    )
    ts: datetime = Field(
        ...,
        description="UTC, timezone-aware sample timestamp.",
    )
    sampler: str = Field(
        ...,
        description=(
            "Sampler name. Known values: ``nvidia-smi``, ``concerto-status``, "
            "``concerto-metrics``, ``pgrep-count``, ``proc-stats``. Free-form "
            "so new samplers land without a schema bump."
        ),
    )
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Sampler-specific payload. Structure is defined per-sampler.",
    )

    @field_validator("ts")
    @classmethod
    def _validate_ts(cls, value: datetime) -> datetime:
        return _ensure_tz_aware(value)


# ---------------------------------------------------------------------------
# Per-step records
# ---------------------------------------------------------------------------


class RequestRecord(BaseModel):
    """Serialised output of the ``request`` primitive.

    Written into a step's directory as ``request-<capture_as>.json``.
    Matches the ``RequestResult`` sketch in ``SPRINT-2-PLAN.md`` §4 B.4:
    every HTTP request the rig issues records its status, the full
    end-to-end timing decomposition (connect / TTFB / total), both the
    request body (as sent) and the response body (as received, or the
    streamed chunk list for SSE), and an optional error string if the
    request raised.

    Response bodies are captured in full — they are small compared to
    the rest of the tarball and silent truncation would break the
    "partial success is better than black-box crash" principle.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(
        default=1,
        description="Pinned to the RequestRecord v1 shape.",
    )
    status: int = Field(
        ...,
        description="HTTP status code, or 0 if the request never completed.",
    )
    elapsed_total_ms: float = Field(
        ...,
        ge=0.0,
        description="Total wall time from issue to last byte, milliseconds.",
    )
    elapsed_ttfb_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time to first byte in milliseconds, or None if not measured.",
    )
    elapsed_connect_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time to TCP/TLS connection established, milliseconds.",
    )
    request_body: dict[str, Any] = Field(
        ...,
        description="JSON body of the outbound request, captured as dict.",
    )
    response_body: Optional[dict[str, Any]] = Field(
        default=None,
        description="JSON body of the non-streaming response, or None for streaming.",
    )
    response_chunks: Optional[list[str]] = Field(
        default=None,
        description=(
            "Ordered list of SSE event payloads for streaming responses. "
            "``None`` on non-streaming requests."
        ),
    )
    error: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable error if the request raised (connection refused, "
            "timeout, TLS failure). ``None`` on success."
        ),
    )


class StateSnapshot(BaseModel):
    """Structured snapshot captured by the ``snapshot`` primitive.

    Taken both before and after every scenario step as ``pre-state.json``
    and ``post-state.json``. Combines concerto's own view of the cluster
    (``/status`` response) with the ground-truth operating-system view
    (``nvidia-smi``, ``pgrep``) so the analyzer can cross-check VRAM
    drift (ROADMAP M4), detect orphan backends (ROADMAP M5), and
    reconstruct state transitions across steps.

    The ``concerto_status`` and ``nvidia_smi`` payloads are typed as
    ``dict[str, Any]`` in v1; a future schema version can replace them
    with structured sub-models once the exact shapes are stable across
    upstream versions. ``extra`` is an explicit escape hatch for
    forward-compatible snapshot additions without requiring a schema
    bump for every new primitive capture.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(
        default=1,
        description="Pinned to the StateSnapshot v1 shape.",
    )
    ts: datetime = Field(
        ...,
        description="UTC, timezone-aware capture time.",
    )
    concerto_status: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parsed ``GET /status`` response from concerto. Structural "
            "typing deferred to a later schema version."
        ),
    )
    nvidia_smi: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Parsed ``nvidia-smi`` snapshot (per-GPU memory used/total, "
            "utilisation, processes). ``None`` on hosts without NVIDIA GPUs."
        ),
    )
    backend_pids: list[int] = Field(
        default_factory=list,
        description=(
            "PIDs of backend processes concerto currently owns, collected "
            "via ``pgrep`` against the known engine command patterns."
        ),
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Escape hatch for ad-hoc per-primitive additions. Keys here "
            "are not constrained by v1 and should not be relied on by "
            "the analyzer."
        ),
    )

    @field_validator("ts")
    @classmethod
    def _validate_ts(cls, value: datetime) -> datetime:
        return _ensure_tz_aware(value)


class ActionRecord(BaseModel):
    """A single action executed within a step.

    A step's ``actions`` list is the ordered record of every primitive
    invocation (``request``, ``wait``, ``snapshot``, ``kill``,
    ``wait_for``, ``wrk_load``, ``parallel``, ...) the runner performed
    inside that step, with its as-written input arguments, start/end
    wall-clock, pass/fail verdict, optional failure reason, and an
    optional primitive-specific ``output`` payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(
        default=1,
        description="Pinned to the ActionRecord v1 shape.",
    )
    action_type: str = Field(
        ...,
        description=(
            "Primitive name, e.g. ``request``, ``wait``, ``snapshot``, "
            "``kill``, ``wait_for``, ``wrk_load``, ``parallel``."
        ),
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="The action's input arguments as written in the scenario YAML.",
    )
    started_at: datetime = Field(
        ...,
        description="UTC, timezone-aware time the action began executing.",
    )
    ended_at: datetime = Field(
        ...,
        description="UTC, timezone-aware time the action finished or failed.",
    )
    duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock duration in milliseconds.",
    )
    passed: bool = Field(
        ...,
        description="``True`` if the action satisfied its own success criteria.",
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description="Human-readable failure description; ``None`` when ``passed``.",
    )
    output: Optional[dict[str, Any]] = Field(
        default=None,
        description="Primitive-specific output payload (e.g. parsed wrk histogram).",
    )

    @field_validator("started_at", "ended_at")
    @classmethod
    def _validate_datetimes(cls, value: datetime) -> datetime:
        return _ensure_tz_aware(value)


class StepResult(BaseModel):
    """Outcome of a single scenario step.

    Serialised into ``steps/NN-<name>/result.json``. A step is one named
    entry in a scenario's ``steps`` list (see
    ``SPRINT-2-PLAN.md`` §5 for the canonical 8-step Sprint 2 sequence).
    The runner records a ``StepResult`` even when the step fails —
    partial failure is a first-class outcome and the artifact stays
    fully populated.

    ``pre_state_path`` and ``post_state_path`` are relative paths inside
    the tarball (e.g. ``steps/01-single-model-smoke/pre-state.json``)
    so the analyzer can read the corresponding ``StateSnapshot`` by
    path without hard-coding the layout.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(
        default=1,
        description="Pinned to the StepResult v1 shape.",
    )
    step_number: int = Field(
        ...,
        ge=1,
        description="1-based ordinal position of this step in the scenario.",
    )
    step_name: str = Field(
        ...,
        description="Human-readable step name (e.g. ``single-model-smoke``).",
    )
    passed: bool = Field(
        ...,
        description="``True`` iff every action and every assertion in the step passed.",
    )
    duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock duration in milliseconds.",
    )
    started_at: datetime = Field(
        ...,
        description="UTC, timezone-aware time the step began.",
    )
    ended_at: datetime = Field(
        ...,
        description="UTC, timezone-aware time the step finished.",
    )
    pre_state_path: str = Field(
        ...,
        description="Relative tarball path to the pre-step ``StateSnapshot`` JSON.",
    )
    post_state_path: str = Field(
        ...,
        description="Relative tarball path to the post-step ``StateSnapshot`` JSON.",
    )
    actions: list[ActionRecord] = Field(
        default_factory=list,
        description="Ordered list of actions that ran inside this step.",
    )
    failures: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable failure descriptions. Empty when ``passed`` is "
            "``True``; non-empty when ``passed`` is ``False``."
        ),
    )

    @field_validator("started_at", "ended_at")
    @classmethod
    def _validate_datetimes(cls, value: datetime) -> datetime:
        return _ensure_tz_aware(value)


# ---------------------------------------------------------------------------
# Summary sub-models
# ---------------------------------------------------------------------------


class LatencyHistogram(BaseModel):
    """Percentile summary of a latency distribution, in milliseconds.

    Used for routing-decision latency (ROADMAP §7 metric) and for the
    concurrent-load wrk histogram (SPRINT-2-PLAN §5 step 5). Values are
    wall-clock milliseconds, not seconds, to match the way humans read
    tail latencies.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    p50_ms: float = Field(..., ge=0.0, description="50th percentile latency in ms.")
    p95_ms: float = Field(..., ge=0.0, description="95th percentile latency in ms.")
    p99_ms: float = Field(..., ge=0.0, description="99th percentile latency in ms.")
    max_ms: float = Field(..., ge=0.0, description="Maximum observed latency in ms.")
    count: int = Field(
        ...,
        ge=0,
        description="Number of samples that contributed to this histogram.",
    )


class ModelMetrics(BaseModel):
    """Per-model roll-up metrics derived from a run.

    Indexed by ``model_id`` inside ``SummaryV1.model_metrics``. Captures
    the VISION T2 "cold-start time per model" numbers plus the per-model
    request-success accounting the analyzer surfaces in
    ``docs/benchmarks.md``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(
        ...,
        description="Model identifier as registered in concerto.toml.",
    )
    cold_start_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Observed wall-clock cold-start time for this model in ms, or "
            "``None`` if the model never cold-started during the run."
        ),
    )
    launch_count: int = Field(
        default=0,
        ge=0,
        description="Number of times concerto launched this model during the run.",
    )
    eviction_count: int = Field(
        default=0,
        ge=0,
        description="Number of times concerto evicted this model during the run.",
    )
    request_count: int = Field(
        default=0,
        ge=0,
        description="Number of HTTP requests routed to this model.",
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of requests to this model that returned a non-2xx status.",
    )


class ExitCriteriaResults(BaseModel):
    """Pass/fail verdict for each scenario-level exit criterion.

    Mirrors SPRINT-2-PLAN.md §5 "Exit criteria (scenario-level)" one-to-one.
    Every field is a ``bool`` (``True`` == criterion passed) or
    ``Optional[bool]`` (``None`` == criterion could not be evaluated,
    e.g. the scenario aborted before the relevant step ran). The
    analyzer uses this to render the per-criterion status table in
    ``docs/benchmarks.md``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    launched_count_ok: Optional[bool] = Field(
        default=None,
        description="True if ``launched_count >= 4`` (SPRINT-2-PLAN §5).",
    )
    stopped_count_ok: Optional[bool] = Field(
        default=None,
        description="True if ``stopped_count >= 2`` (SPRINT-2-PLAN §5).",
    )
    http_error_rate_ok: Optional[bool] = Field(
        default=None,
        description="True if ``http_error_rate == 0.0`` during concurrent load (ROADMAP M1).",
    )
    vram_drift_ok: Optional[bool] = Field(
        default=None,
        description="True if ``vram_drift_max_percent < 10.0`` (ROADMAP M4).",
    )
    graceful_shutdown_time_ok: Optional[bool] = Field(
        default=None,
        description="True if ``graceful_shutdown_wall_time_secs < 60`` (ROADMAP M5).",
    )
    orphan_processes_ok: Optional[bool] = Field(
        default=None,
        description="True if ``orphan_processes_after_shutdown == 0`` (ROADMAP M5).",
    )


# ---------------------------------------------------------------------------
# Top-level records
# ---------------------------------------------------------------------------


class ManifestV1(BaseModel):
    """Tarball top-level metadata — written as ``manifest.json``.

    The first thing the analyzer reads. Carries enough identity to
    uniquely describe the run (concerto version, git SHA, scenario and
    its version, wall-clock window) without requiring the analyzer to
    crack open any other file in the tarball. Both a schema version
    (for the JSON shape) and an artifact tree version (for the on-disk
    layout) are pinned so the two can evolve independently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(
        default=1,
        description="Pinned to the ManifestV1 shape.",
    )
    artifact_tree_version: Literal[1] = Field(
        default=1,
        description=(
            "On-disk tarball layout version. Independent of ``schema_version`` "
            "— describes where files live, not their contents."
        ),
    )
    rig_version: str = Field(
        ...,
        description="``concerto_bench.__version__`` of the rig that produced this run.",
    )
    concerto_version: str = Field(
        ...,
        description="Output of ``concerto --version`` from the run under test.",
    )
    concerto_git_sha: Optional[str] = Field(
        default=None,
        description="Git SHA of the concerto build, if determinable at runtime.",
    )
    scenario_name: str = Field(
        ...,
        description="``name`` field from the scenario YAML.",
    )
    scenario_version: str = Field(
        ...,
        description="``version`` field from the scenario YAML.",
    )
    started_at: datetime = Field(
        ...,
        description="UTC, timezone-aware time the run started.",
    )
    ended_at: datetime = Field(
        ...,
        description="UTC, timezone-aware time the run finished (or errored out).",
    )
    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="``(ended_at - started_at).total_seconds()``, captured explicitly.",
    )
    exit_status: ExitStatus = Field(
        ...,
        description="Overall run outcome.",
    )
    step_count: int = Field(
        ...,
        ge=0,
        description="Number of steps that were executed (passed or failed).",
    )
    host_info_path: str = Field(
        default="environment.json",
        description=(
            "Relative tarball path to the ``HostInfo`` JSON blob. Defaults "
            "to ``environment.json``; stored explicitly so the analyzer "
            "never guesses the layout."
        ),
    )

    @field_validator("started_at", "ended_at")
    @classmethod
    def _validate_datetimes(cls, value: datetime) -> datetime:
        return _ensure_tz_aware(value)


class SummaryV1(BaseModel):
    """Derived summary record — written as ``summary.json``.

    The analyzer-friendly, pre-computed roll-up of a run. Every ROADMAP
    §7 Sprint 2 metric is representable here:

    - Cold-start time per model (``model_metrics[*].cold_start_ms``)
    - Routing decision latency p50/p95/p99 (``routing_decision_latency``)
    - VRAM drift max (``vram_drift_percent_max``)
    - Concurrent-load throughput / error rate / latency histogram
      (``concurrent_load_*``)
    - Graceful shutdown wall time (``graceful_shutdown_wall_time_secs``)
    - Orphan counts (``orphan_processes_after_shutdown``)

    Every SPRINT-2-PLAN §5 scenario-level exit criterion is also
    represented, both as the raw value that drove the criterion and as
    the pass/fail verdict inside ``exit_criteria``.

    Fields that would be metrically meaningful only if a particular step
    ran are ``Optional`` with a sensible default — if the scenario
    aborted early, ``summary.json`` still round-trips.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(
        default=1,
        description="Pinned to the SummaryV1 shape.",
    )
    scenario_name: str = Field(
        ...,
        description="``name`` field from the scenario YAML.",
    )
    scenario_version: str = Field(
        ...,
        description="``version`` field from the scenario YAML.",
    )
    exit_status: ExitStatus = Field(
        ...,
        description="Overall run outcome.",
    )
    scenario_passed: bool = Field(
        ...,
        description="``True`` iff every step passed and every exit criterion passed.",
    )
    step_count: int = Field(
        ...,
        ge=0,
        description="Total steps that were executed.",
    )
    steps_passed: int = Field(
        ...,
        ge=0,
        description="Count of steps whose ``passed`` was ``True``.",
    )
    steps_failed: int = Field(
        ...,
        ge=0,
        description="Count of steps whose ``passed`` was ``False``.",
    )
    failed_step_names: list[str] = Field(
        default_factory=list,
        description="Names of the steps that failed, in execution order.",
    )

    # -- SPRINT-2-PLAN §5 exit-criterion raw values --------------------

    launched_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total backend launches observed during the run.",
    )
    stopped_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total backend stops (evictions + shutdown) during the run.",
    )
    http_error_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of requests that returned non-2xx during concurrent load.",
    )
    vram_drift_max_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Maximum observed drift between concerto's ``/status`` VRAM "
            "accounting and ``nvidia-smi``'s ground-truth reading, as a "
            "percentage. ROADMAP M4."
        ),
    )
    graceful_shutdown_wall_time_secs: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Wall time from SIGTERM to process exit in seconds. ROADMAP M5.",
    )
    orphan_processes_after_shutdown: Optional[int] = Field(
        default=None,
        ge=0,
        description="Count of backend processes still alive post-shutdown. ROADMAP M5.",
    )

    # -- ROADMAP §7 metrics (additional) -------------------------------

    routing_decision_latency: Optional[LatencyHistogram] = Field(
        default=None,
        description=(
            "p50/p95/p99/max histogram of routing-decision latency observed "
            "across the run. ROADMAP §7 metric."
        ),
    )
    concurrent_load_throughput_rps: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Concurrent-load step throughput in requests/second.",
    )
    concurrent_load_error_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Concurrent-load step error rate (0.0 – 1.0).",
    )
    concurrent_load_latency: Optional[LatencyHistogram] = Field(
        default=None,
        description="p50/p95/p99/max histogram of per-request latency under concurrent load.",
    )
    model_metrics: dict[str, ModelMetrics] = Field(
        default_factory=dict,
        description=(
            "Per-model metrics keyed by ``model_id``. Covers cold-start time "
            "per model (VISION T2, ROADMAP §7) plus launch/eviction/request "
            "accounting."
        ),
    )

    exit_criteria: ExitCriteriaResults = Field(
        default_factory=ExitCriteriaResults,
        description="Pass/fail verdict for each SPRINT-2-PLAN §5 exit criterion.",
    )


__all__ = [
    "SCHEMA_VERSION",
    "ARTIFACT_TREE_VERSION",
    "ExitStatus",
    "ManifestV1",
    "HostInfo",
    "SummaryV1",
    "StepResult",
    "StateSnapshot",
    "ActionRecord",
    "RequestRecord",
    "TelemetrySample",
    "LatencyHistogram",
    "ModelMetrics",
    "ExitCriteriaResults",
]
