"""Scenario runner: the serial integrator for the bench rig.

Everything that landed in Phase B.2 waves 1 and 2 — :mod:`concerto_proc`,
:mod:`schema`, :mod:`artifact`, :mod:`primitives`, :mod:`samplers` — is
stitched together here into the one function that actually runs a
scenario end-to-end. The runner is the narrow waist every CLI
subcommand and every e2e test will eventually lean on: give it a
validated :class:`RunnerOptions`, it will spawn concerto, execute the
scenario's steps, drive the samplers, build the on-disk artifact tree,
package it into a tarball, and hand back a :class:`RunResult`.

Public surface
--------------

* :class:`Scenario` / :class:`StepSpec` / :class:`ActionSpec` /
  :class:`SamplerSpec` — strict pydantic models describing the
  scenario YAML shape. :func:`load_scenario` parses a YAML file into
  a :class:`Scenario`; validation is the same strict "extra forbidden"
  posture every other record in this package uses.
* :class:`RunnerOptions` — frozen pydantic model carrying everything
  the runner needs to know about *this* invocation (binary, output
  dir, scenario path, dry-run flags).
* :class:`RunResult` — dataclass returned by :func:`run_scenario`
  with the exit code, the finalised artifact, the assembled manifest
  and summary, and the list of per-step results.
* :func:`run_scenario` — the one-shot entry point. Owns the
  :class:`ConcertoProcess` lifecycle, the long-lived
  :class:`httpx.AsyncClient`, and the :class:`SamplerPool` window.

Execution contract
------------------

The runner honours every foundational constraint documented in
``SPRINT-2-PLAN.md`` §4 B (lines 397–415):

1. Concerto's ``--config`` flag is mandatory, even under ``--mock-gpus``.
   :func:`run_scenario` resolves a usable config path by copying
   ``concerto.example.toml`` from the repository root into the
   artifact tree. Scenarios that need a bespoke config point at one
   via :attr:`RunnerOptions.concerto_config_override`.

2. Concerto writes structured logs to **stdout**, not stderr. Startup
   failures are surfaced by reading both
   :attr:`ConcertoStartupError.stdout_tail` **and**
   :attr:`ConcertoStartupError.stderr_tail`; the runner logs both and
   still finalises a best-effort artifact.

3. :class:`SamplerPool` is the preferred entry point over raw
   :func:`run_samplers`. The runner inspects :attr:`SamplerPool.errors`
   before trusting :attr:`SamplerPool.results`.

4. Each sampler's output file is registered with
   :meth:`ArtifactBuilder.register_telemetry_file` **after**
   :meth:`SamplerPool.__aexit__` returns, so the files are flushed and
   closed before the builder reads them.

5. Exactly one long-lived :class:`httpx.AsyncClient` is constructed per
   run and passed into both :class:`RequestPrimitive.execute` and
   :class:`SnapshotPrimitive.execute`. Connection pooling applies
   across every action in every step.

6. :attr:`RunnerOptions.stable_started_at` lets callers pin
   :attr:`ManifestV1.started_at` so two runs with identical inputs
   produce byte-identical tarballs. This is the key that makes the
   regression-diff tooling in step 11 possible.

Partial-success posture
-----------------------

A step failure never aborts the run by default; every
:class:`StepResult` lands in the artifact either way. The run's
aggregate ``exit_status`` is downgraded to ``partial_failure`` when any
step fails or any sampler setup raised, and to ``error`` when the
rig itself could not reach the "ran at least one step" state (concerto
startup failure, YAML parse failure, unhandled exception). The
artifact is still built and tarred in every case except an unhandled
exception escaping :func:`run_scenario` — the goal is always to hand
the operator a tarball they can open and debug.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from concerto_bench import __version__ as _BENCH_VERSION
from concerto_bench.artifact import (
    ArtifactBuilder,
    ArtifactError,
    FinalizedArtifact,
    _slugify,
    _step_dir_name,
)
from concerto_bench.concerto_proc import (
    ConcertoProcess,
    ConcertoStartupError,
    ProcessSpec,
    pick_free_port,
)
from concerto_bench.primitives import (
    AssertAction,
    AssertPrimitive,
    KillAction,
    KillPrimitive,
    ParallelAction,
    ParallelPrimitive,
    RequestAction,
    RequestPrimitive,
    SnapshotAction,
    SnapshotPrimitive,
    WaitAction,
    WaitPrimitive,
    WaitForAction,
    WaitForPrimitive,
    WrkLoadAction,
    WrkLoadPrimitive,
)
from concerto_bench.samplers import (
    DEFAULT_REGISTRY,
    ConcertoMetricsSampler,
    ConcertoMetricsSamplerConfig,
    ConcertoStatusSampler,
    ConcertoStatusSamplerConfig,
    NvidiaSmiSampler,
    NvidiaSmiSamplerConfig,
    PgrepCounterSampler,
    PgrepCounterSamplerConfig,
    ProcStatsSampler,
    ProcStatsSamplerConfig,
    Sampler,
    SamplerPool,
    SamplerRegistry,
)
from concerto_bench.schema import (
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
)

logger = logging.getLogger(__name__)

UTC = timezone.utc

LogLevel = Literal["debug", "info", "warn", "error"]
LogFormat = Literal["pretty", "json"]

_SUPPORTED_ACTION_TYPES: frozenset[str] = frozenset({
    "request", "snapshot", "wait", "wait_for", "kill",
    "assert", "wrk_load", "parallel",
})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ScenarioError(RuntimeError):
    """Raised when a scenario YAML is structurally invalid.

    Carries the offending path (if loaded from disk) and a chained
    :class:`pydantic.ValidationError` where applicable so operators get
    a field-level diagnostic rather than a bare "bad YAML" message.
    """


class RunnerError(RuntimeError):
    """Raised when the runner itself cannot proceed.

    Distinct from :class:`ScenarioError` because the cause is an
    environment problem (missing binary, missing config file, output
    directory unwritable) rather than a malformed scenario. CLI
    callers map this onto exit code ``2``; tests assert the message
    against the specific failure class.
    """


# ---------------------------------------------------------------------------
# Scenario schema
# ---------------------------------------------------------------------------


class ActionSpec(BaseModel):
    """One action inside a scenario step.

    Scenario YAML represents an action as a single-key dict mapping a
    primitive name to its arg dict, e.g. ``- request: {model: foo,
    content: hi}``. :meth:`Scenario._parse_action` normalises that
    shape into ``ActionSpec(type="request", args={...})`` so the
    runner's dispatch table can key off ``type`` uniformly.

    The primitive name is validated against :data:`_SUPPORTED_ACTION_TYPES`
    at scenario-parse time — unknown action types fail loudly at load
    time rather than silently producing an action the runner will
    skip.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(
        ...,
        min_length=1,
        description="Primitive name — ``request`` or ``snapshot`` in v1.",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the primitive's action model.",
    )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value not in _SUPPORTED_ACTION_TYPES:
            raise ValueError(
                f"unknown action type {value!r}; supported in v1: "
                f"{sorted(_SUPPORTED_ACTION_TYPES)}"
            )
        return value


class StepSpec(BaseModel):
    """One named step inside a scenario.

    A step is a human-meaningful unit of work — "cold-start smoke",
    "concurrent load", "shutdown" — and is always bracketed by a
    pre-state and post-state snapshot by the runner. The YAML author
    supplies the step name, a short optional description, and the
    ordered list of actions that constitute the step's body.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable step name; becomes ``NN-<slug>`` on disk.",
    )
    description: str = Field(
        default="",
        description="Optional free-text description retained for provenance.",
    )
    actions: list[ActionSpec] = Field(
        default_factory=list,
        description="Ordered list of action invocations to execute.",
    )


class SamplerSpec(BaseModel):
    """Declarative sampler enablement for a scenario.

    Scenarios reference samplers by the string name registered in
    :data:`concerto_bench.samplers.DEFAULT_REGISTRY`. The runner turns
    each :class:`SamplerSpec` into a concrete :class:`Sampler`
    subclass instance with an output path under
    :meth:`ArtifactBuilder.telemetry_dir` and a per-sampler config
    merged from the built-in defaults and the scenario's
    ``config`` overrides.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        description="Registered sampler name, e.g. ``nvidia-smi``.",
    )
    interval_secs: float = Field(
        default=1.0,
        gt=0.0,
        description="Tick interval in seconds. 1 Hz is the Sprint 2 default.",
    )
    enabled: bool = Field(
        default=True,
        description="Per-scenario kill switch for this sampler.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Sampler-specific overrides merged over the runner's defaults "
            "(e.g. ``patterns`` for the pgrep sampler, ``query_fields`` "
            "for the nvidia-smi sampler)."
        ),
    )


class Scenario(BaseModel):
    """Parsed scenario YAML — the runner's single input document.

    Every field maps one-to-one to a SPRINT-2-PLAN §4 B.3 section. The
    model is frozen so the runner can stash it and reference it across
    asynchronous callbacks without worrying about in-flight mutation.

    ``parameters`` is an opaque free-form map: scenarios use it to
    describe the human intent of the run (target throughput, label
    strings) and the runner retains it verbatim in the artifact for
    post-hoc analysis. The v1 runner does not interpret it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1, description="Scenario name.")
    version: str = Field(..., min_length=1, description="Scenario version string.")
    description: str = Field(
        default="",
        description="Optional free-text scenario description.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form scenario parameters (target throughput, labels, ...).",
    )
    models: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Concerto ``[[models]]`` table entries as raw dicts. The v1 "
            "runner does not template these into the concerto config — "
            "that is the caller's responsibility via "
            "``concerto_config_override`` — but they are retained here "
            "for provenance in the manifest and future templating steps."
        ),
    )
    samplers: list[SamplerSpec] = Field(
        default_factory=list,
        description="Samplers to enable for this run.",
    )
    steps: list[StepSpec] = Field(
        ...,
        min_length=1,
        description="Ordered list of steps to execute. Must be non-empty.",
    )
    exit_criteria: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Raw scenario-level exit criteria (throughput bounds, error-"
            "rate ceilings). The v1 runner preserves these verbatim for "
            "the analyzer; it does not evaluate them itself."
        ),
    )
    continue_on_failure: bool = Field(
        default=True,
        description=(
            "When ``True``, a failed step does not abort subsequent steps. "
            "The default matches the 'partial success is better than "
            "black-box crash' principle from SPRINT-2-PLAN §4 B.3."
        ),
    )


def load_scenario(path: Path) -> Scenario:
    """Read a YAML file from disk and parse it into a :class:`Scenario`.

    The function owns the file read, the YAML parse, and the action-
    normalisation step (turning ``{request: {...}}`` single-key dicts
    into :class:`ActionSpec` records). Any failure — missing file,
    malformed YAML, invalid schema — is wrapped in a
    :class:`ScenarioError` with the path and a readable diagnostic so
    the CLI can surface it without a traceback.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"could not read scenario {path}: {exc}") from exc

    try:
        loaded = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"could not parse YAML in {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ScenarioError(
            f"scenario {path} must be a YAML mapping, got {type(loaded).__name__}"
        )

    # Normalise the actions before handing the whole doc to pydantic:
    # scenario YAML uses ``- request: {args}`` single-key dicts, but
    # ActionSpec wants ``{type, args}``. Doing the rewrite here keeps
    # the pydantic model strict and lets load errors point at the
    # specific step / action that broke.
    try:
        steps_raw = loaded.get("steps", [])
        if not isinstance(steps_raw, list):
            raise ScenarioError(
                f"scenario {path}: 'steps' must be a list, got {type(steps_raw).__name__}"
            )
        normalised_steps: list[dict[str, Any]] = []
        for step_index, step_raw in enumerate(steps_raw):
            if not isinstance(step_raw, dict):
                raise ScenarioError(
                    f"scenario {path}: step[{step_index}] must be a mapping, "
                    f"got {type(step_raw).__name__}"
                )
            actions_raw = step_raw.get("actions", [])
            if not isinstance(actions_raw, list):
                raise ScenarioError(
                    f"scenario {path}: step[{step_index}].actions must be a list, "
                    f"got {type(actions_raw).__name__}"
                )
            normalised_actions: list[dict[str, Any]] = []
            for action_index, action_raw in enumerate(actions_raw):
                normalised_actions.append(
                    _normalise_action(
                        action_raw,
                        path=path,
                        step_index=step_index,
                        action_index=action_index,
                    )
                )
            normalised_step = dict(step_raw)
            normalised_step["actions"] = normalised_actions
            normalised_steps.append(normalised_step)
        normalised_doc = dict(loaded)
        normalised_doc["steps"] = normalised_steps
    except ScenarioError:
        raise

    try:
        return Scenario.model_validate(normalised_doc)
    except ValidationError as exc:
        raise ScenarioError(f"scenario {path} failed validation: {exc}") from exc


def _normalise_action(
    action_raw: Any,
    *,
    path: Path,
    step_index: int,
    action_index: int,
) -> dict[str, Any]:
    """Turn one YAML action entry into ``{type, args}`` form.

    Scenario YAML may express an action as either::

        - request: {model: foo, content: hi}

    (a single-key mapping from primitive name to its args) or the
    already-expanded::

        - {type: request, args: {model: foo, content: hi}}

    Both forms are accepted; anything else raises
    :class:`ScenarioError` with the step / action coordinates so the
    YAML author can find the offending entry quickly.
    """
    location = f"scenario {path}: step[{step_index}].actions[{action_index}]"

    if not isinstance(action_raw, dict):
        raise ScenarioError(
            f"{location}: action must be a mapping, got {type(action_raw).__name__}"
        )

    # Already-expanded form.
    if "type" in action_raw:
        args = action_raw.get("args", {})
        if not isinstance(args, dict):
            raise ScenarioError(
                f"{location}: 'args' must be a mapping, got {type(args).__name__}"
            )
        return {"type": action_raw["type"], "args": dict(args)}

    if len(action_raw) != 1:
        raise ScenarioError(
            f"{location}: single-key action form must have exactly one key, "
            f"got {sorted(action_raw.keys())}"
        )
    (action_type, args) = next(iter(action_raw.items()))
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ScenarioError(
            f"{location}: args for {action_type!r} must be a mapping or null, "
            f"got {type(args).__name__}"
        )
    return {"type": action_type, "args": dict(args)}


# ---------------------------------------------------------------------------
# Runner options + result
# ---------------------------------------------------------------------------


class RunnerOptions(BaseModel):
    """Strict runtime configuration for a single scenario run.

    Frozen so the whole run observes a single snapshot of the inputs.
    The model intentionally captures *how* to run a scenario (binary,
    output dir, mock-gpus flag) rather than *what* to run (the
    scenario YAML is loaded separately). That split lets the CLI
    build a :class:`RunnerOptions` from its arguments and the tests
    construct one from fixtures without having to re-parse YAML.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_path: Path = Field(
        ...,
        description="Path to the scenario YAML to execute.",
    )
    output_dir: Path = Field(
        ...,
        description=(
            "Artifact root directory. Created if missing. The tarball "
            "lands at ``<output_dir>.tar.gz`` next to this directory, "
            "per :class:`ArtifactBuilder`."
        ),
    )
    concerto_bin: Path = Field(
        ...,
        description="Path to the concerto binary to spawn.",
    )
    mock_gpus: Optional[int] = Field(
        default=None,
        ge=1,
        le=8,
        description=(
            "If set, spawns concerto with ``--mock-gpus N`` for a "
            "mock-upstream dry run. ``None`` uses real GPU detection."
        ),
    )
    models_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Optional pre-downloaded models directory. Passed through as "
            "``CONCERTO_MODELS_DIR`` in the concerto child's environment "
            "for scenarios that template weight paths."
        ),
    )
    concerto_config_override: Optional[Path] = Field(
        default=None,
        description=(
            "Optional absolute path to a concerto.toml config file. When "
            "unset, the runner copies the repo's ``concerto.example.toml`` "
            "into the artifact tree and uses that."
        ),
    )
    concerto_log_level: LogLevel = Field(
        default="info",
        description="Value passed to concerto's ``--log-level`` flag.",
    )
    concerto_log_format: LogFormat = Field(
        default="json",
        description=(
            "Value passed to concerto's ``--log-format`` flag. Defaults to "
            "``json`` so downstream tooling can parse the captured stdout."
        ),
    )
    startup_timeout_secs: float = Field(
        default=60.0,
        gt=0.0,
        description="Deadline for concerto's ``/health`` gate in seconds.",
    )
    shutdown_grace_secs: float = Field(
        default=10.0,
        gt=0.0,
        description="SIGTERM → SIGKILL grace window for the concerto child.",
    )
    http_timeout_secs: float = Field(
        default=240.0,
        gt=0.0,
        description=(
            "Default timeout on the long-lived httpx client the runner "
            "threads through every request + snapshot primitive invocation. "
            "Must exceed vLLM cold-start time on real hardware."
        ),
    )
    stable_started_at: Optional[datetime] = Field(
        default=None,
        description=(
            "If set, overrides the manifest's ``started_at`` so two runs "
            "with identical inputs produce byte-identical tarballs. The "
            "regression-diff tooling in step 11 relies on this."
        ),
    )
    stable_ended_at: Optional[datetime] = Field(
        default=None,
        description=(
            "If set, pairs with ``stable_started_at`` to pin the manifest's "
            "``ended_at`` for reproducible tarballs. The pair is usually "
            "supplied together by tests; production runs leave both unset."
        ),
    )

    @field_validator("stable_started_at", "stable_ended_at")
    @classmethod
    def _require_tz_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("stable_started_at / stable_ended_at must be tz-aware UTC")
        return value.astimezone(UTC)


@dataclass(frozen=True)
class RunResult:
    """Outcome of a single :func:`run_scenario` invocation.

    ``exit_code`` follows the POSIX convention: 0 on a fully-passing
    run, 1 on a partial failure (the artifact is still valid), 2 on
    a rig-level error (concerto didn't start, YAML was invalid, ...).
    ``artifact`` is ``None`` only in the ``exit_code == 2`` path where
    the builder could not even finalise a tarball — otherwise the
    tarball path is always populated so the CLI can print it on exit.
    """

    exit_code: int
    manifest: ManifestV1
    summary: SummaryV1
    step_results: list[StepResult]
    artifact: Optional[FinalizedArtifact]
    concerto_startup_error: Optional[ConcertoStartupError] = None


# ---------------------------------------------------------------------------
# run_scenario — the one-shot entry point
# ---------------------------------------------------------------------------


async def run_scenario(options: RunnerOptions) -> RunResult:
    """Execute ``options.scenario_path`` end-to-end and return a :class:`RunResult`.

    High-level flow:

    1. Parse the scenario YAML.
    2. Initialise an :class:`ArtifactBuilder` rooted at ``options.output_dir``.
    3. Resolve a concerto config path (override, then copy example).
    4. Spawn :class:`ConcertoProcess` and wait on ``/health``.
    5. Build samplers from the scenario's :class:`SamplerSpec` list.
    6. Enter a :class:`SamplerPool` and execute every step in sequence,
       capturing pre/post snapshots and dispatching each action to its
       primitive.
    7. Exit the pool, register telemetry files, copy concerto logs,
       write manifest + summary + host info, finalise the tarball.

    Every step's :class:`StepResult` is recorded regardless of
    pass/fail status. On concerto startup failure the function still
    builds a minimal artifact (no steps, ``exit_status="error"``) so
    the operator gets a tarball with the stdout / stderr tails to
    debug against.
    """
    # Scenario parse is a rig-level error if it fails — no artifact to
    # build because we do not know the scenario's name yet.
    scenario = load_scenario(options.scenario_path)

    builder = ArtifactBuilder(options.output_dir)

    # Copy the scenario file into the artifact tree verbatim for
    # provenance. This is the easiest way to answer "what exactly ran"
    # six months later. Failure to copy is non-fatal — the manifest
    # still records the scenario name + version.
    with suppress(OSError):
        shutil.copyfile(
            options.scenario_path, builder.root_dir / options.scenario_path.name
        )

    # Resolve the concerto config path. If the user did not override it
    # we copy ``concerto.example.toml`` into the artifact tree so the
    # run is fully self-contained — future analyses can see exactly
    # which config concerto was handed.
    config_path = _resolve_concerto_config(options, builder.root_dir, scenario)

    # Concerto log files need a location that is not tarred twice.
    # We give concerto a sibling directory next to output_dir so the
    # files are not picked up by the builder's rglob during finalise;
    # we copy them into the canonical paths explicitly after the
    # child has been reaped.
    log_tmpdir = tempfile.TemporaryDirectory(prefix="concerto-proc-logs-")
    concerto_log_dir = Path(log_tmpdir.name)

    port = pick_free_port()
    spec = ProcessSpec(
        binary=options.concerto_bin,
        config_path=config_path,
        port=port,
        log_dir=concerto_log_dir,
        mock_gpus=options.mock_gpus,
        log_level=options.concerto_log_level,
        log_format=options.concerto_log_format,
        models_dir=options.models_dir,
        startup_timeout_secs=options.startup_timeout_secs,
        shutdown_grace_secs=options.shutdown_grace_secs,
    )

    started_at = options.stable_started_at or _utc_now()
    step_results: list[StepResult] = []
    exit_status: str = "success"
    failed_step_names: list[str] = []
    concerto_version = "unknown"
    concerto_git_sha: Optional[str] = None
    startup_error: Optional[ConcertoStartupError] = None
    sampler_errors: list[tuple[str, BaseException]] = []

    # Best-effort concerto --version capture. If the binary is broken
    # enough that this fails we will still attempt to spawn it below;
    # the version field on the manifest just falls back to "unknown".
    concerto_version = _query_concerto_version(options.concerto_bin)

    try:
        try:
            proc_cm = await ConcertoProcess.start(spec)
        except ConcertoStartupError as exc:
            startup_error = exc
            exit_status = "error"
            logger.error(
                "concerto failed to start (rc=%s): stdout_tail=%r stderr_tail=%r",
                exc.returncode,
                exc.stdout_tail,
                exc.stderr_tail,
            )
        else:
            async with proc_cm as proc:
                base_url = proc.base_url

                logger.info(
                    "concerto healthy at %s (pid=%d); driving %d step(s)",
                    base_url,
                    proc.pid,
                    len(scenario.steps),
                )

                # One long-lived httpx client for every request +
                # snapshot primitive call in the whole run. Connection
                # pooling applies across steps. The timeout here is the
                # per-operation default; individual actions can override
                # via their own ``timeout_secs`` fields.
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(options.http_timeout_secs),
                ) as client:
                    request_primitive = RequestPrimitive()
                    snapshot_primitive = SnapshotPrimitive()
                    wait_primitive = WaitPrimitive()
                    wait_for_primitive = WaitForPrimitive()
                    kill_primitive = KillPrimitive()
                    assert_primitive = AssertPrimitive()
                    wrk_load_primitive = WrkLoadPrimitive()
                    parallel_primitive = ParallelPrimitive()

                    samplers = _build_samplers(
                        specs=scenario.samplers,
                        telemetry_dir=builder.telemetry_dir(),
                        base_url=base_url,
                    )

                    async with SamplerPool(samplers) as pool:
                        for index, step_spec in enumerate(scenario.steps, start=1):
                            step_result, pre, post, requests = await _execute_step(
                                index=index,
                                step_spec=step_spec,
                                base_url=base_url,
                                client=client,
                                request_primitive=request_primitive,
                                snapshot_primitive=snapshot_primitive,
                                wait_primitive=wait_primitive,
                                wait_for_primitive=wait_for_primitive,
                                kill_primitive=kill_primitive,
                                assert_primitive=assert_primitive,
                                wrk_load_primitive=wrk_load_primitive,
                                parallel_primitive=parallel_primitive,
                            )
                            try:
                                builder.write_step(
                                    step_result,
                                    pre_state=pre,
                                    post_state=post,
                                    request_records=requests,
                                )
                            except ArtifactError as exc:
                                logger.error(
                                    "failed to write step %d to artifact: %s",
                                    index,
                                    exc,
                                )
                            step_results.append(step_result)
                            if not step_result.passed:
                                failed_step_names.append(step_result.step_name)
                                if not scenario.continue_on_failure:
                                    logger.warning(
                                        "step %d failed and continue_on_failure=False; "
                                        "aborting remaining steps",
                                        index,
                                    )
                                    break
                                # Smoke gate: if the first step fails and
                                # every request action got a 5xx, the
                                # backend is fundamentally broken. Abort
                                # early rather than burning through 240s
                                # wait_for timeouts on every subsequent step.
                                if index == 1 and _all_requests_5xx(requests):
                                    logger.error(
                                        "smoke gate: step 1 failed with all "
                                        "requests returning 5xx — backend is "
                                        "broken, aborting remaining steps"
                                    )
                                    exit_status = "error"
                                    break

                    # After the pool exits every sampler has been
                    # cancelled, its teardown() has run, and its file
                    # is flushed / closed. Only now is it safe to
                    # register each output file with the builder.
                    for result in pool.results:
                        try:
                            builder.register_telemetry_file(
                                result.name, result.output_path
                            )
                        except ArtifactError as exc:
                            logger.warning(
                                "failed to register telemetry file for sampler %s: %s",
                                result.name,
                                exc,
                            )
                    sampler_errors = list(pool.errors)
                    for name, exc in sampler_errors:
                        logger.error(
                            "sampler %s failed during run: %s: %s",
                            name,
                            type(exc).__name__,
                            exc,
                        )

        # Copy concerto logs into the canonical artifact paths. The
        # file names are fixed by ``ConcertoProcess._spawn`` (it writes
        # ``concerto-stdout.log`` / ``concerto-stderr.log`` under
        # ``spec.log_dir``) so we can derive them from ``concerto_log_dir``
        # even when the child crashed before the health gate opened —
        # which is exactly when those logs are most useful to the
        # operator.
        stdout_source = concerto_log_dir / "concerto-stdout.log"
        stderr_source = concerto_log_dir / "concerto-stderr.log"
        try:
            builder.copy_concerto_logs(
                stdout_path=stdout_source if stdout_source.exists() else None,
                stderr_path=stderr_source if stderr_source.exists() else None,
            )
        except ArtifactError as exc:
            logger.warning("failed to copy concerto logs: %s", exc)

        # -- Post-shutdown orphan process check ----------------------------
        # After the ConcertoProcess context manager has sent SIGTERM/SIGKILL,
        # check if any backend processes survived.
        orphan_count = _count_orphan_backend_processes()
        if orphan_count > 0:
            logger.warning(
                "%d orphan backend process(es) detected after shutdown",
                orphan_count,
            )

        if failed_step_names or sampler_errors:
            if exit_status == "success":
                exit_status = "partial_failure"

        ended_at = options.stable_ended_at or _utc_now()
        duration_seconds = max((ended_at - started_at).total_seconds(), 0.0)

        manifest = ManifestV1(
            rig_version=_BENCH_VERSION,
            concerto_version=concerto_version,
            concerto_git_sha=concerto_git_sha,
            scenario_name=scenario.name,
            scenario_version=scenario.version,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            exit_status=exit_status,  # type: ignore[arg-type]
            step_count=len(step_results),
        )

        summary = _build_summary(
            scenario=scenario,
            step_results=step_results,
            failed_step_names=failed_step_names,
            exit_status=exit_status,
            telemetry_dir=builder.telemetry_dir(),
            orphan_count=orphan_count,
        )

        host_info = _build_host_info(
            concerto_version=concerto_version,
            captured_at=started_at,
        )

        try:
            builder.write_manifest(manifest)
            builder.write_summary(summary)
            builder.write_host_info(host_info)
        except ArtifactError as exc:
            logger.error("failed to write artifact metadata: %s", exc)
            return RunResult(
                exit_code=2,
                manifest=manifest,
                summary=summary,
                step_results=step_results,
                artifact=None,
                concerto_startup_error=startup_error,
            )

        try:
            finalized = builder.finalize()
        except ArtifactError as exc:
            logger.error("failed to finalize artifact tarball: %s", exc)
            return RunResult(
                exit_code=2,
                manifest=manifest,
                summary=summary,
                step_results=step_results,
                artifact=None,
                concerto_startup_error=startup_error,
            )
    finally:
        log_tmpdir.cleanup()

    exit_code = _exit_code_for_status(exit_status)
    return RunResult(
        exit_code=exit_code,
        manifest=manifest,
        summary=summary,
        step_results=step_results,
        artifact=finalized,
        concerto_startup_error=startup_error,
    )


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------


async def _execute_step(
    *,
    index: int,
    step_spec: StepSpec,
    base_url: str,
    client: httpx.AsyncClient,
    request_primitive: RequestPrimitive,
    snapshot_primitive: SnapshotPrimitive,
    wait_primitive: WaitPrimitive,
    wait_for_primitive: WaitForPrimitive,
    kill_primitive: KillPrimitive,
    assert_primitive: AssertPrimitive,
    wrk_load_primitive: WrkLoadPrimitive,
    parallel_primitive: ParallelPrimitive,
) -> tuple[StepResult, StateSnapshot, StateSnapshot, dict[str, RequestRecord]]:
    """Run one scenario step and produce the four records it generates.

    The returned tuple is exactly what :meth:`ArtifactBuilder.write_step`
    consumes, so the caller can hand it through without reshuffling.

    A step is bracketed by two :class:`SnapshotAction` invocations
    (labels ``pre`` / ``post``) even if the scenario author did not
    ask for them — the analyzer relies on every step directory
    containing ``pre-state.json`` / ``post-state.json``. If either
    bracketing snapshot fails the step is marked failed and the
    failure reason lists the snapshot error; a missing snapshot still
    produces a placeholder :class:`StateSnapshot` (empty
    ``concerto_status``) so the artifact tree shape is preserved.
    """
    logger.info("▶ step %02d-%s: %s", index, _slugify(step_spec.name), step_spec.description or "")

    started_at = _utc_now()
    t_start = time.perf_counter()
    failures: list[str] = []
    action_records: list[ActionRecord] = []
    request_records: dict[str, RequestRecord] = {}

    pre_snapshot, pre_failure = await _capture_bracketing_snapshot(
        label="pre",
        base_url=base_url,
        client=client,
        snapshot_primitive=snapshot_primitive,
    )
    if pre_failure is not None:
        failures.append(f"pre-snapshot: {pre_failure}")

    for action_spec in step_spec.actions:
        action_started = _utc_now()
        action_t0 = time.perf_counter()
        passed: bool
        failure_reason: Optional[str]
        output: Optional[dict[str, Any]]
        try:
            if action_spec.type == "request":
                action_model = RequestAction.model_validate(action_spec.args)
                record = await request_primitive.execute(
                    action_model, base_url=base_url, client=client
                )
                output = {"request_record": record.model_dump(mode="json")}
                if record.error is not None:
                    passed = False
                    failure_reason = record.error
                else:
                    passed = True
                    failure_reason = None
                if action_model.capture_as is not None:
                    request_records[action_model.capture_as] = record
            elif action_spec.type == "snapshot":
                action_model_snap = SnapshotAction.model_validate(action_spec.args)
                snap = await snapshot_primitive.execute(
                    action_model_snap, base_url=base_url, client=client
                )
                output = {"snapshot": snap.model_dump(mode="json")}
                passed = True
                failure_reason = None
            elif action_spec.type == "wait":
                wait_action = WaitAction.model_validate(action_spec.args)
                wait_result = await wait_primitive.execute(
                    wait_action, base_url=base_url, client=client
                )
                output = wait_result
                passed = True
                failure_reason = None
            elif action_spec.type == "wait_for":
                wf_action = WaitForAction.model_validate(action_spec.args)
                wf_result = await wait_for_primitive.execute(
                    wf_action, base_url=base_url, client=client
                )
                output = wf_result
                passed = wf_result.get("satisfied", False)
                failure_reason = (
                    None if passed
                    else f"wait_for condition not satisfied after {wf_result.get('elapsed_secs', '?')}s"
                )
            elif action_spec.type == "kill":
                kill_action = KillAction.model_validate(action_spec.args)
                kill_result = await kill_primitive.execute(
                    kill_action, base_url=base_url, client=client
                )
                output = kill_result
                kill_errors = kill_result.get("errors", [])
                if kill_errors and kill_action.expect_found:
                    passed = False
                    failure_reason = "; ".join(kill_errors)
                else:
                    passed = True
                    failure_reason = None
            elif action_spec.type == "assert":
                assert_action = AssertAction.model_validate(action_spec.args)
                assert_result = await assert_primitive.execute(
                    assert_action, base_url=base_url, client=client
                )
                output = assert_result
                passed = assert_result.get("passed", False)
                failure_reason = (
                    None if passed
                    else assert_result.get("message", "assertion failed")
                )
            elif action_spec.type == "wrk_load":
                wrk_action = WrkLoadAction.model_validate(action_spec.args)
                wrk_result = await wrk_load_primitive.execute(
                    wrk_action, base_url=base_url, client=client
                )
                output = wrk_result
                error_rate = wrk_result.get("error_rate", 0.0)
                passed = error_rate == 0.0
                failure_reason = (
                    None if passed
                    else f"wrk_load error rate: {error_rate:.4f}"
                )
            elif action_spec.type == "parallel":
                par_action = ParallelAction.model_validate(action_spec.args)

                async def _dispatch(action_dict: dict) -> dict:
                    normalised = _normalise_action(
                        action_dict,
                        path=Path("<parallel-inline>"),
                        step_index=index,
                        action_index=0,
                    )
                    sub_spec = ActionSpec.model_validate(normalised)
                    sub_step, _, _, _ = await _execute_step(
                        index=index,
                        step_spec=StepSpec(name=f"parallel-sub-{sub_spec.type}", actions=[sub_spec]),
                        base_url=base_url,
                        client=client,
                        request_primitive=request_primitive,
                        snapshot_primitive=snapshot_primitive,
                        wait_primitive=wait_primitive,
                        wait_for_primitive=wait_for_primitive,
                        kill_primitive=kill_primitive,
                        assert_primitive=assert_primitive,
                        wrk_load_primitive=wrk_load_primitive,
                        parallel_primitive=parallel_primitive,
                    )
                    return {"passed": sub_step.passed, "actions": [a.model_dump(mode="json") for a in sub_step.actions]}

                par_result = await parallel_primitive.execute(
                    par_action, base_url=base_url, client=client, dispatch=_dispatch
                )
                output = par_result
                par_errors = par_result.get("errors", [])
                passed = len(par_errors) == 0
                failure_reason = (
                    None if passed
                    else f"parallel sub-action errors: {'; '.join(par_errors)}"
                )
            else:
                raise ScenarioError(
                    f"unsupported action type {action_spec.type!r} reached runner"
                )
        except ValidationError as exc:
            passed = False
            failure_reason = f"action args invalid: {exc}"
            output = None
            logger.warning(
                "step %02d %s action rejected at runtime: %s",
                index,
                action_spec.type,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive catchall
            passed = False
            failure_reason = f"{type(exc).__name__}: {exc}"
            output = None
            logger.exception(
                "step %02d %s action raised unexpectedly", index, action_spec.type
            )

        action_ended = _utc_now()
        duration_ms = max((time.perf_counter() - action_t0) * 1000.0, 0.0)

        action_records.append(
            ActionRecord(
                action_type=action_spec.type,
                args=dict(action_spec.args),
                started_at=action_started,
                ended_at=action_ended,
                duration_ms=duration_ms,
                passed=passed,
                failure_reason=failure_reason,
                output=output,
            )
        )
        if not passed:
            failures.append(f"{action_spec.type}: {failure_reason}")

    post_snapshot, post_failure = await _capture_bracketing_snapshot(
        label="post",
        base_url=base_url,
        client=client,
        snapshot_primitive=snapshot_primitive,
    )
    if post_failure is not None:
        failures.append(f"post-snapshot: {post_failure}")

    ended_at = _utc_now()
    duration_ms = max((time.perf_counter() - t_start) * 1000.0, 0.0)

    # Build the provisional step result so we can compute its canonical
    # directory name (the same ``NN-<slug>`` the builder uses) and
    # populate pre/post state path fields with the matching string.
    provisional = StepResult(
        step_number=index,
        step_name=step_spec.name,
        passed=len(failures) == 0,
        duration_ms=duration_ms,
        started_at=started_at,
        ended_at=ended_at,
        pre_state_path="placeholder",
        post_state_path="placeholder",
        actions=action_records,
        failures=failures,
    )
    dir_name = _step_dir_name(provisional)
    step_result = StepResult(
        step_number=index,
        step_name=step_spec.name,
        passed=len(failures) == 0,
        duration_ms=duration_ms,
        started_at=started_at,
        ended_at=ended_at,
        pre_state_path=f"steps/{dir_name}/pre-state.json",
        post_state_path=f"steps/{dir_name}/post-state.json",
        actions=action_records,
        failures=failures,
    )

    return step_result, pre_snapshot, post_snapshot, request_records


async def _capture_bracketing_snapshot(
    *,
    label: str,
    base_url: str,
    client: httpx.AsyncClient,
    snapshot_primitive: SnapshotPrimitive,
) -> tuple[StateSnapshot, Optional[str]]:
    """Run one snapshot and always return a usable :class:`StateSnapshot`.

    Used for the pre / post bracketing snapshots the runner takes around
    every step. A snapshot failure (``/status`` unreachable) is recorded
    in the returned ``failure_reason`` string but the caller still
    receives a populated :class:`StateSnapshot` with an empty
    ``concerto_status`` payload — the v1 artifact tree requires
    ``pre-state.json`` / ``post-state.json`` to exist regardless.
    """
    action = SnapshotAction(capture_label=label)
    try:
        snap = await snapshot_primitive.execute(
            action, base_url=base_url, client=client
        )
        return snap, None
    except Exception as exc:
        logger.warning(
            "%s snapshot failed against %s: %s: %s",
            label,
            base_url,
            type(exc).__name__,
            exc,
        )
        placeholder = StateSnapshot(
            ts=_utc_now(),
            concerto_status={},
            backend_pids=[],
            extra={
                "capture_label": label,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return placeholder, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Sampler construction
# ---------------------------------------------------------------------------


def _build_samplers(
    *,
    specs: list[SamplerSpec],
    telemetry_dir: Path,
    base_url: str,
    registry: SamplerRegistry = DEFAULT_REGISTRY,
) -> list[Sampler]:
    """Instantiate concrete samplers from a scenario's :class:`SamplerSpec` list.

    Each spec is matched against a small dispatch table that knows how
    to build the right config model for the named sampler class. Unknown
    sampler names raise :class:`ScenarioError` — the scenario would
    have parsed successfully (the spec allows any string) but the
    runner cannot honour it, so failing loudly here is better than
    silently dropping a sampler.

    ``base_url`` is threaded in so the two concerto-facing samplers
    (``concerto-status`` and ``concerto-metrics``) can point at the
    right port without the scenario author having to hard-code one.
    """
    samplers: list[Sampler] = []
    for spec in specs:
        if spec.name not in registry:
            raise ScenarioError(
                f"scenario references unknown sampler {spec.name!r}; "
                f"known: {registry.names()}"
            )
        output_path = telemetry_dir / f"{_slugify(spec.name)}.jsonl"
        overrides = dict(spec.config)
        common = {
            "name": spec.name,
            "interval_secs": spec.interval_secs,
            "enabled": spec.enabled,
            "output_path": output_path,
        }

        if spec.name == "nvidia-smi":
            config = NvidiaSmiSamplerConfig(**common, **overrides)
            samplers.append(NvidiaSmiSampler(config))
        elif spec.name == "concerto-status":
            overrides.setdefault("base_url", base_url)
            config = ConcertoStatusSamplerConfig(**common, **overrides)
            samplers.append(ConcertoStatusSampler(config))
        elif spec.name == "concerto-metrics":
            overrides.setdefault("base_url", base_url)
            config = ConcertoMetricsSamplerConfig(**common, **overrides)
            samplers.append(ConcertoMetricsSampler(config))
        elif spec.name == "pgrep-count":
            overrides.setdefault(
                "patterns", ["vllm", "python -m vllm", "mock-inference-backend"]
            )
            config = PgrepCounterSamplerConfig(**common, **overrides)
            samplers.append(PgrepCounterSampler(config))
        elif spec.name == "proc-stats":
            config = ProcStatsSamplerConfig(**common, **overrides)
            samplers.append(ProcStatsSampler(config))
        else:
            # Registered but unknown to this runner — future samplers
            # that land without a corresponding builder should still
            # fail loudly so we don't silently drop them.
            raise ScenarioError(
                f"sampler {spec.name!r} is registered but not wired into the runner"
            )

    return samplers


# ---------------------------------------------------------------------------
# Summary + HostInfo construction
# ---------------------------------------------------------------------------


def _build_summary(
    *,
    scenario: Scenario,
    step_results: list[StepResult],
    failed_step_names: list[str],
    exit_status: str,
    telemetry_dir: Optional[Path] = None,
    orphan_count: Optional[int] = None,
) -> SummaryV1:
    """Assemble the v1 :class:`SummaryV1` from captured step results.

    Computes all ROADMAP §7 and SPRINT-2-PLAN §5 metrics from the step
    results and (where available) telemetry JSONL files. Also evaluates
    exit criteria against the scenario's thresholds.
    """
    steps_passed = sum(1 for step in step_results if step.passed)
    steps_failed = len(step_results) - steps_passed

    # -- Extract metrics from step results --------------------------------
    metrics = _extract_metrics_from_steps(step_results)

    # -- Read launched/stopped counts from concerto-metrics telemetry ------
    if telemetry_dir is not None:
        telemetry_metrics = _read_telemetry_counters(telemetry_dir)
        metrics.update(telemetry_metrics)

    # -- Inject post-shutdown orphan count ---------------------------------
    if orphan_count is not None:
        metrics["orphan_processes_after_shutdown"] = orphan_count

    # -- Evaluate exit criteria -------------------------------------------
    exit_criteria_results = _evaluate_exit_criteria(
        scenario.exit_criteria, metrics
    )

    return SummaryV1(
        scenario_name=scenario.name,
        scenario_version=scenario.version,
        exit_status=exit_status,  # type: ignore[arg-type]
        scenario_passed=steps_failed == 0 and exit_status == "success",
        step_count=len(step_results),
        steps_passed=steps_passed,
        steps_failed=steps_failed,
        failed_step_names=failed_step_names,
        launched_count=metrics.get("launched_count"),
        stopped_count=metrics.get("stopped_count"),
        http_error_rate=metrics.get("http_error_rate"),
        vram_drift_max_percent=metrics.get("vram_drift_max_percent"),
        graceful_shutdown_wall_time_secs=metrics.get(
            "graceful_shutdown_wall_time_secs"
        ),
        orphan_processes_after_shutdown=metrics.get(
            "orphan_processes_after_shutdown"
        ),
        routing_decision_latency=metrics.get("routing_decision_latency"),
        concurrent_load_throughput_rps=metrics.get(
            "concurrent_load_throughput_rps"
        ),
        concurrent_load_error_rate=metrics.get("concurrent_load_error_rate"),
        concurrent_load_latency=metrics.get("concurrent_load_latency"),
        model_metrics=metrics.get("model_metrics", {}),
        exit_criteria=exit_criteria_results,
    )


def _extract_metrics_from_steps(
    step_results: list[StepResult],
) -> dict[str, Any]:
    """Walk step results and extract all computable metrics.

    Returns a dict whose keys match :class:`SummaryV1` field names.
    Values that cannot be determined are omitted (not set to None).
    """
    metrics: dict[str, Any] = {}

    # -- Per-model tracking -----------------------------------------------
    model_first_cold_start: dict[str, float] = {}  # model_id -> ms
    model_request_counts: dict[str, int] = {}
    model_error_counts: dict[str, int] = {}

    # -- HTTP error rate (individual request actions, excluding wrk_load) --
    total_requests = 0
    error_requests = 0

    for step in step_results:
        # Graceful shutdown wall time from the shutdown step
        if "shutdown" in step.step_name:
            metrics["graceful_shutdown_wall_time_secs"] = (
                step.duration_ms / 1000.0
            )

        for action in step.actions:
            atype = action.action_type
            output = action.output

            # -- request actions ------------------------------------------
            if atype == "request":
                rr = output.get("request_record", {})
                status = rr.get("status", 0)
                model_id = rr.get("request_body", {}).get("model")
                elapsed = rr.get("elapsed_total_ms", 0)

                total_requests += 1
                if status < 200 or status >= 300:
                    error_requests += 1

                if model_id:
                    model_request_counts[model_id] = (
                        model_request_counts.get(model_id, 0) + 1
                    )
                    if status < 200 or status >= 300:
                        model_error_counts[model_id] = (
                            model_error_counts.get(model_id, 0) + 1
                        )
                    # First successful request per model = cold start
                    if (
                        model_id not in model_first_cold_start
                        and 200 <= status < 300
                    ):
                        model_first_cold_start[model_id] = elapsed

            # -- wrk_load actions -----------------------------------------
            elif atype == "wrk_load":
                rps = output.get("rps")
                if rps is not None:
                    metrics["concurrent_load_throughput_rps"] = rps
                err_rate = output.get("error_rate")
                if err_rate is not None:
                    metrics["concurrent_load_error_rate"] = err_rate
                latency = output.get("latency_ms", {})
                total_reqs = output.get("total_requests", 0)
                if latency and total_reqs > 0:
                    metrics["concurrent_load_latency"] = LatencyHistogram(
                        p50_ms=latency.get("p50", 0),
                        p95_ms=latency.get("p95", 0),
                        p99_ms=latency.get("p99", 0),
                        max_ms=latency.get("max", 0),
                        count=total_reqs,
                    )

            # -- snapshot actions (VRAM drift) ----------------------------
            elif atype == "snapshot":
                snapshot = output.get("snapshot", {})
                drift = _compute_vram_drift(snapshot)
                if drift is not None:
                    prev = metrics.get("vram_drift_max_percent")
                    if prev is None or drift > prev:
                        metrics["vram_drift_max_percent"] = drift

    # -- HTTP error rate --------------------------------------------------
    if total_requests > 0:
        metrics["http_error_rate"] = error_requests / total_requests

    # -- Model metrics ----------------------------------------------------
    all_model_ids = set(model_request_counts) | set(model_first_cold_start)
    if all_model_ids:
        model_metrics = {}
        for mid in sorted(all_model_ids):
            model_metrics[mid] = ModelMetrics(
                model_id=mid,
                cold_start_ms=model_first_cold_start.get(mid),
                request_count=model_request_counts.get(mid, 0),
                error_count=model_error_counts.get(mid, 0),
            )
        metrics["model_metrics"] = model_metrics

    return metrics


def _compute_vram_drift(snapshot: dict[str, Any]) -> Optional[float]:
    """Compute max VRAM drift % between nvidia-smi and concerto /status.

    Returns None if either data source is missing.
    """
    nvidia_smi = snapshot.get("nvidia_smi")
    concerto_status = snapshot.get("concerto_status")
    if not nvidia_smi or not concerto_status:
        return None
    nvidia_gpus = nvidia_smi.get("gpus", [])
    concerto_gpus = concerto_status.get("gpus", [])
    if not nvidia_gpus or not concerto_gpus:
        return None

    max_drift = 0.0
    for nvidia_gpu in nvidia_gpus:
        gpu_idx = nvidia_gpu.get("index")
        nvidia_used_mb = nvidia_gpu.get("memory.used", 0)
        if nvidia_used_mb <= 0:
            continue

        # Find matching concerto GPU
        for c_gpu in concerto_gpus:
            if c_gpu.get("id") != gpu_idx:
                continue
            # concerto reports memory_used as a human string like "13.4 GiB"
            concerto_used_str = c_gpu.get("memory_used", "0")
            concerto_used_mb = _parse_memory_to_mb(concerto_used_str)
            if concerto_used_mb is None:
                continue
            drift = abs(concerto_used_mb - nvidia_used_mb) / nvidia_used_mb * 100
            max_drift = max(max_drift, drift)
            break

    return max_drift if max_drift > 0.0 or nvidia_gpus else None


def _parse_memory_to_mb(mem_str: str) -> Optional[float]:
    """Parse a human-readable memory string to MB.

    Handles formats like "13.4 GiB", "372.3 MiB", "2.2 GiB".
    """
    if not isinstance(mem_str, str):
        return None
    mem_str = mem_str.strip()
    try:
        parts = mem_str.split()
        if len(parts) == 2:
            value = float(parts[0])
            unit = parts[1].lower()
            if unit in ("gib", "gb"):
                return value * 1024
            elif unit in ("mib", "mb"):
                return value
            elif unit in ("kib", "kb"):
                return value / 1024
        # Try bare number (assume bytes)
        return float(mem_str) / (1024 * 1024)
    except (ValueError, IndexError):
        return None


def _read_telemetry_counters(telemetry_dir: Path) -> dict[str, Any]:
    """Read counters and histograms from concerto-metrics telemetry.

    Parses the last line of ``concerto-metrics.jsonl`` to get the final
    values of Prometheus counters (``concerto_backend_launches_total``,
    ``concerto_eviction_total``) and histograms
    (``concerto_routing_decision_seconds``).
    """
    import json as _json

    metrics_file = telemetry_dir / "concerto-metrics.jsonl"
    if not metrics_file.exists():
        return {}

    # Read the last non-empty line
    last_line = ""
    try:
        with open(metrics_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
    except OSError:
        return {}

    if not last_line:
        return {}

    try:
        row = _json.loads(last_line)
    except _json.JSONDecodeError:
        return {}

    result: dict[str, Any] = {}
    # The metrics sampler flattens Prometheus text into a dict under
    # a "metrics" key (see ConcertoMetricsSampler.sample_once).
    data = row.get("data", row)
    if "metrics" in data:
        data = data["metrics"]

    for key in ("concerto_backend_launches_total", "concerto_backend_launches"):
        val = data.get(key)
        if val is not None:
            try:
                result["launched_count"] = int(float(val))
            except (ValueError, TypeError):
                pass
            break

    for key in ("concerto_eviction_total", "concerto_eviction"):
        val = data.get(key)
        if val is not None:
            try:
                result["stopped_count"] = int(float(val))
            except (ValueError, TypeError):
                pass
            break

    # -- Routing decision latency histogram --------------------------------
    histogram = _parse_prometheus_histogram(data, "concerto_routing_decision_seconds")
    if histogram is not None:
        result["routing_decision_latency"] = histogram

    return result


def _parse_prometheus_histogram(
    data: dict[str, Any], metric_name: str
) -> Optional[LatencyHistogram]:
    """Parse a Prometheus histogram from flattened sampler data.

    The sampler stores bucket samples as keys like
    ``<metric>_bucket|le=0.005`` with cumulative counts as values,
    plus ``<metric>_sum`` and ``<metric>_count``.

    Returns a :class:`LatencyHistogram` with p50/p95/p99/max in
    milliseconds, or ``None`` if insufficient data.
    """
    # Collect bucket boundaries and their cumulative counts
    bucket_prefix = f"{metric_name}_bucket|le="
    buckets: list[tuple[float, float]] = []  # (upper_bound, cumulative_count)
    for key, value in data.items():
        if not key.startswith(bucket_prefix):
            continue
        le_str = key[len(bucket_prefix):]
        try:
            le = float(le_str)
            count = float(value)
        except (ValueError, TypeError):
            continue
        if le != float("+inf"):
            buckets.append((le, count))

    total_count_raw = data.get(f"{metric_name}_count")
    if total_count_raw is None:
        return None
    try:
        total_count = int(float(total_count_raw))
    except (ValueError, TypeError):
        return None
    if total_count == 0:
        return None

    # Sort by upper bound
    buckets.sort(key=lambda b: b[0])

    if not buckets:
        # No bucket data available — fall back to mean from sum/count
        total_sum = data.get(f"{metric_name}_sum")
        if total_sum is not None:
            try:
                mean_s = float(total_sum) / total_count
                mean_ms = mean_s * 1000.0
                return LatencyHistogram(
                    p50_ms=mean_ms, p95_ms=mean_ms, p99_ms=mean_ms,
                    max_ms=mean_ms, count=total_count,
                )
            except (ValueError, TypeError, ZeroDivisionError):
                pass
        return None

    def _percentile_from_buckets(
        sorted_buckets: list[tuple[float, float]], pct: float, total: int
    ) -> float:
        """Estimate a percentile from cumulative histogram buckets.

        Uses linear interpolation within the bucket that contains the
        target rank, converting seconds to milliseconds.
        """
        target = pct / 100.0 * total
        prev_count = 0.0
        prev_bound = 0.0
        for upper_bound, cum_count in sorted_buckets:
            if cum_count >= target:
                # Interpolate within this bucket
                bucket_width = upper_bound - prev_bound
                bucket_count = cum_count - prev_count
                if bucket_count > 0:
                    fraction = (target - prev_count) / bucket_count
                    value_s = prev_bound + fraction * bucket_width
                else:
                    value_s = upper_bound
                return value_s * 1000.0  # seconds → ms
            prev_count = cum_count
            prev_bound = upper_bound
        # Beyond the last finite bucket — use the last bucket's bound
        return sorted_buckets[-1][0] * 1000.0

    return LatencyHistogram(
        p50_ms=_percentile_from_buckets(buckets, 50, total_count),
        p95_ms=_percentile_from_buckets(buckets, 95, total_count),
        p99_ms=_percentile_from_buckets(buckets, 99, total_count),
        max_ms=sorted_buckets[-1][0] * 1000.0,
        count=total_count,
    )


def _evaluate_exit_criteria(
    criteria_spec: dict[str, Any],
    metrics: dict[str, Any],
) -> ExitCriteriaResults:
    """Compare computed metrics against scenario exit criteria thresholds.

    Each criterion is independently evaluated. If the metric required to
    evaluate a criterion is None (not computed), the result is None
    (could not evaluate).
    """
    launched = metrics.get("launched_count")
    stopped = metrics.get("stopped_count")
    http_err = metrics.get("concurrent_load_error_rate")
    vram_drift = metrics.get("vram_drift_max_percent")
    shutdown_time = metrics.get("graceful_shutdown_wall_time_secs")
    orphans = metrics.get("orphan_processes_after_shutdown")

    launched_threshold = criteria_spec.get("launched_count_gte")
    stopped_threshold = criteria_spec.get("stopped_count_gte")
    http_err_threshold = criteria_spec.get("http_error_rate_max")
    vram_threshold = criteria_spec.get("vram_drift_max_percent")
    shutdown_threshold = criteria_spec.get(
        "graceful_shutdown_wall_time_max_secs"
    )
    orphan_threshold = criteria_spec.get("orphan_processes_after_shutdown_max")

    return ExitCriteriaResults(
        launched_count_ok=(
            launched >= launched_threshold
            if launched is not None and launched_threshold is not None
            else None
        ),
        stopped_count_ok=(
            stopped >= stopped_threshold
            if stopped is not None and stopped_threshold is not None
            else None
        ),
        http_error_rate_ok=(
            http_err <= http_err_threshold
            if http_err is not None and http_err_threshold is not None
            else None
        ),
        vram_drift_ok=(
            vram_drift < vram_threshold
            if vram_drift is not None and vram_threshold is not None
            else None
        ),
        graceful_shutdown_time_ok=(
            shutdown_time < shutdown_threshold
            if shutdown_time is not None and shutdown_threshold is not None
            else None
        ),
        orphan_processes_ok=(
            orphans <= orphan_threshold
            if orphans is not None and orphan_threshold is not None
            else None
        ),
    )


def _build_host_info(*, concerto_version: str, captured_at: datetime) -> HostInfo:
    """Build the :class:`HostInfo` blob the artifact tree requires.

    A dedicated ``environment.py`` module (with ``nvidia-smi -q -x``
    dump, ``lscpu`` capture, full env whitelisting) lands in a later
    Phase B step; for step 7 we populate the minimum the schema
    requires so the artifact passes :func:`verify_artifact_tree`.
    """
    uname = os.uname()
    return HostInfo(
        concerto_version=concerto_version,
        python_version=sys.version,
        uname={
            "sysname": uname.sysname,
            "nodename": uname.nodename,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        captured_at=captured_at,
        env_snapshot={
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "RUST_LOG", "HF_HOME")
            if key in os.environ
        },
    )


# ---------------------------------------------------------------------------
# Concerto config resolution
# ---------------------------------------------------------------------------


def _resolve_concerto_config(
    options: RunnerOptions, artifact_root: Path, scenario: Scenario
) -> Path:
    """Return a usable path for concerto's ``--config`` flag.

    Order of preference:

    1. ``options.concerto_config_override`` if set — copied into the
       artifact tree for provenance, then returned.
    2. If the scenario has a ``models`` section, generate a config TOML
       from the scenario's models and the example config's ``[server]``
       / ``[routing]`` sections. This is the key enabler for dry-runs:
       the scenario author declares models in one place and the runner
       generates the config concerto needs.
    3. ``<repo>/concerto.example.toml`` discovered by walking up from
       ``options.concerto_bin`` — copied into the artifact tree,
       returned.
    4. :class:`RunnerError` — the caller must either build their own
       config file and point at it via ``concerto_config_override`` or
       run from a location where the example config is discoverable.

    Copying into the artifact tree is the load-bearing behaviour: every
    run should carry its own config file inside the tarball so a
    reviewer can see exactly what concerto was handed without having
    to cross-reference the repo at a specific commit.
    """
    target = artifact_root / "concerto.toml"

    if options.concerto_config_override is not None:
        source = options.concerto_config_override
        if not source.is_file():
            raise RunnerError(
                f"concerto_config_override points at a non-file path: {source}"
            )
        try:
            shutil.copyfile(source, target)
        except OSError as exc:
            raise RunnerError(
                f"failed to copy concerto config {source} -> {target}: {exc}"
            ) from exc
        return target

    # When the scenario declares models, generate a config that
    # includes those models so concerto recognises them. This is
    # critical for dry-runs where the example config might not list
    # all models the scenario exercises.
    if scenario.models:
        _generate_config_from_scenario(options, scenario, target)
        return target

    example = _find_concerto_example(options.concerto_bin)
    if example is None:
        raise RunnerError(
            "could not locate concerto.example.toml; either build the rig "
            "from inside the concerto repo or pass --concerto-config "
            "pointing at a valid config file"
        )
    try:
        shutil.copyfile(example, target)
    except OSError as exc:
        raise RunnerError(
            f"failed to copy concerto example config {example} -> {target}: {exc}"
        ) from exc
    return target


def _generate_config_from_scenario(
    options: RunnerOptions, scenario: Scenario, target: Path
) -> None:
    """Generate a ``concerto.toml`` from the scenario's model declarations.

    Reads the ``[server]`` and ``[routing]`` sections from the example
    config (if available) and appends ``[[models]]`` sections from the
    scenario's ``models`` list. Falls back to sensible defaults if no
    example config is found.
    """
    # Start with example config sections for [server] and [routing]
    lines: list[str] = []

    example = _find_concerto_example(options.concerto_bin)
    if example is not None:
        example_text = example.read_text(encoding="utf-8")
        # Extract [server] and [routing] sections from example
        in_models = False
        for line in example_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[[models]]") or stripped.startswith("[[gpus]]"):
                in_models = True
                continue
            if in_models and stripped.startswith("[") and not stripped.startswith("[["):
                in_models = False
            if not in_models and not stripped.startswith("[["):
                lines.append(line)
    else:
        lines.extend([
            "[server]",
            'host = "0.0.0.0"',
            "port = 8000",
            "",
            "[routing]",
            'eviction_policy = "lru"',
            "cold_start_timeout_secs = 120",
            "health_check_interval_secs = 10",
        ])

    lines.append("")

    # Emit [[models]] from scenario. Rewrite weight_path relative to
    # --models-dir when provided so the scenario can use portable paths
    # like "/models/qwen2.5-0.5b" while the actual weights live at
    # e.g. "/root/models/qwen2.5-0.5b" on the Vast.ai box.
    for model in scenario.models:
        lines.append("[[models]]")
        for key, value in model.items():
            if key == "weight_path" and isinstance(value, str) and options.models_dir is not None:
                # Rewrite: take the last path component from the scenario
                # and resolve it under --models-dir.
                model_dir_name = Path(value).name
                rewritten = str(options.models_dir / model_dir_name)
                lines.append(f'{key} = "{rewritten}"')
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, list):
                formatted_items = ", ".join(
                    f'"{item}"' if isinstance(item, str) else str(item)
                    for item in value
                )
                lines.append(f"{key} = [{formatted_items}]")
            elif isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            else:
                lines.append(f"{key} = {value}")
        lines.append("")

    # Emit [[gpus]] — concerto requires at least one.
    if options.mock_gpus:
        gpu_count = options.mock_gpus
    else:
        # Detect GPU count via nvidia-smi for real hardware.
        gpu_count = _detect_gpu_count()
    for i in range(gpu_count):
        lines.append("[[gpus]]")
        lines.append(f"id = {i}")
        lines.append("")

    target.write_text("\n".join(lines), encoding="utf-8")


def _all_requests_5xx(requests: dict[str, RequestRecord]) -> bool:
    """Return True if every captured request got a 5xx status code."""
    if not requests:
        return False
    return all(r.status >= 500 for r in requests.values())


def _detect_gpu_count() -> int:
    """Detect the number of GPUs via ``nvidia-smi``. Falls back to 2."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            count = len([l for l in result.stdout.strip().splitlines() if l.strip()])
            return max(count, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 2


def _find_concerto_example(binary: Path) -> Optional[Path]:
    """Walk upward from ``binary`` looking for ``concerto.example.toml``.

    A typical dev layout is ``<repo>/target/{debug,release}/concerto``,
    so four parent hops is usually enough. We stop at the filesystem
    root or when a ``Cargo.toml`` + ``concerto.example.toml`` pair is
    found, whichever comes first.
    """
    # Fall back to the current working directory if the binary path
    # is relative and we cannot resolve it — some test fixtures use
    # bogus binaries for scenario-level checks.
    try:
        resolved = binary.resolve()
    except OSError:
        resolved = Path.cwd()
    candidates: list[Path] = [resolved.parent, *resolved.parents, Path.cwd()]
    for candidate in candidates:
        example = candidate / "concerto.example.toml"
        if example.is_file():
            return example
    return None


def _query_concerto_version(binary: Path) -> str:
    """Best-effort ``concerto --version`` capture for the manifest.

    Any failure degrades to ``"unknown"`` rather than raising — the
    manifest's ``concerto_version`` field is diagnostic, not load-
    bearing, and we never want to fail a whole run because the version
    probe timed out.
    """
    if not binary.exists():
        return "unknown"
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = (result.stdout or b"").decode("utf-8", errors="replace").strip()
    if not text:
        text = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    return text or "unknown"


def _count_orphan_backend_processes() -> int:
    """Count backend processes still alive after concerto shutdown.

    Checks for common backend patterns (vllm, mock-inference-backend)
    using ``pgrep``. Returns 0 if ``pgrep`` is not available or finds
    no matches.
    """
    patterns = ["mock-inference-backend", "vllm.entrypoints"]
    total = 0
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
                # Filter out our own process
                our_pid = str(os.getpid())
                orphans = [p for p in pids if p != our_pid]
                total += len(orphans)
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    return total


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Tz-aware UTC wall-clock — centralised so every record uses the same clock."""
    return datetime.now(tz=UTC)


def _exit_code_for_status(exit_status: str) -> int:
    """Map an :data:`ExitStatus` tag to a POSIX exit code."""
    if exit_status == "success":
        return 0
    if exit_status == "partial_failure":
        return 1
    return 2


__all__ = [
    "ActionSpec",
    "RunResult",
    "RunnerError",
    "RunnerOptions",
    "Scenario",
    "ScenarioError",
    "SamplerSpec",
    "StepSpec",
    "load_scenario",
    "run_scenario",
]
