"""Sampler abstract base + loop runner + pool.

The bench rig captures time-series telemetry during scenario execution
by running N small samplers concurrently, each on its own 1 Hz tick,
each writing to its own ``telemetry/<name>.jsonl`` stream. This module
provides the machinery shared by every concrete sampler:

* :class:`SamplerConfig` — pydantic v2 config holding the sampler name,
  tick interval, output path, and enabled flag. Concrete samplers
  subclass this to add sampler-specific fields.
* :class:`Sampler` — abstract base class. Subclasses implement
  :meth:`sample_once` (one tick) and optionally :meth:`setup` /
  :meth:`teardown` for one-shot init and cleanup. The base class owns
  the loop, the JSONL file handle, timing, drift compensation, and
  counting of per-tick successes/failures.
* :class:`SamplerResult` — immutable pydantic record returned from
  :meth:`Sampler.run` once the sampler has stopped. Captures enough
  metadata for the runner and the analyzer to reconstruct what
  happened.
* :class:`SamplerError` — raised on unrecoverable *setup* failures.
  Per-tick failures do not raise; they are counted in
  ``SamplerResult.ticks_failed`` and logged.
* :class:`SamplerPool` — async context manager that schedules a set of
  samplers as background tasks and cancels them cleanly on exit.
  The scenario runner (Phase B.2 step 7) wraps its execution window in
  a ``SamplerPool`` so telemetry collection is bracketed by
  ``__aenter__`` / ``__aexit__``.
* :func:`run_samplers` — convenience helper that runs a set of
  samplers to completion under normal ``asyncio.gather`` semantics. The
  runner itself prefers :class:`SamplerPool`, but this is useful for
  tests and ad-hoc invocations.
* :class:`SamplerRegistry` — tiny name-based registry so scenarios can
  reference samplers by string name from YAML.

Lifecycle contract
------------------

A sampler's :meth:`Sampler.run` method runs the loop until the task is
cancelled. **On cancellation** it catches :class:`asyncio.CancelledError`
internally, runs :meth:`Sampler.teardown`, flushes and closes the JSONL
file, and **returns** a :class:`SamplerResult` rather than re-raising.

This deviates from the canonical asyncio "cooperative task re-raises
CancelledError" pattern deliberately. The runner needs structured
results from every sampler regardless of how the sampler stopped (it
always stops by cancellation in production), and surfacing the result
through the coroutine's return value is cleaner than stashing it on
``self`` just so :class:`SamplerPool` can pick it up after awaiting a
cancelled task. The deviation is contained to this class — nothing else
in the bench rig relies on the conventional pattern.

``SamplerError`` from :meth:`setup` is re-raised by :meth:`run`, so the
runner can see setup failures distinctly from normal shutdown. Per-tick
exceptions are caught, counted, and logged; they never escape.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Optional

from pydantic import BaseModel, ConfigDict, Field

from concerto_bench.schema import TelemetrySample

logger = logging.getLogger(__name__)

# Upper bound on the number of per-tick failure reasons retained on a
# :class:`SamplerResult`. Bounded so a badly-broken sampler cannot grow
# its failure list without bound over a long run.
_MAX_FAILURE_REASONS = 20


class SamplerError(RuntimeError):
    """Raised on unrecoverable sampler setup failure.

    Never raised for per-tick failures — those are caught by the loop,
    counted in :class:`SamplerResult`, and logged. Only failures that
    make the sampler impossible to run at all (cannot open the output
    file, :meth:`Sampler.setup` raises) are surfaced as
    :class:`SamplerError`.
    """


class SamplerConfig(BaseModel):
    """Base configuration shared by every concrete sampler.

    Concrete samplers subclass this to add fields specific to their
    data source (e.g. a concerto ``base_url``, an ``nvidia-smi`` query
    list). The base class keeps only the fields the loop runner itself
    needs: name, tick interval, output path, and enabled flag.

    The model is not frozen because subclasses may want mutable config
    fields for runtime overrides during tests; the analyzer only
    consumes serialised :class:`TelemetrySample` rows, not the config
    object itself.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "Sampler name. Used as the JSONL filename stem and as the "
            "``sampler`` field on every emitted :class:`TelemetrySample`. "
            "Known values in Sprint 2: ``nvidia-smi``, ``concerto-status``, "
            "``concerto-metrics``, ``pgrep-count``, ``proc-stats``."
        ),
    )
    interval_secs: float = Field(
        default=1.0,
        gt=0.0,
        description="Tick interval in seconds. 1 Hz is the Sprint 2 default.",
    )
    output_path: Path = Field(
        ...,
        description=(
            "Absolute path to the JSONL file this sampler will append to. "
            "Typically ``artifact_builder.telemetry_dir() / f'{name}.jsonl'``; "
            "samplers accept the path loosely to keep the coupling between "
            "the sampler package and :class:`ArtifactBuilder` minimal."
        ),
    )
    enabled: bool = Field(
        default=True,
        description=(
            "Per-scenario kill switch. A disabled sampler still "
            "instantiates but :meth:`Sampler.run` short-circuits."
        ),
    )


class SamplerResult(BaseModel):
    """Outcome record for a single sampler run.

    Collected by :class:`SamplerPool` / :func:`run_samplers` once the
    sampler has stopped (normally via asyncio cancellation from the
    runner at scenario end). Captures enough metadata to diagnose a
    sampler that produced no samples, dropped many ticks, or stopped
    prematurely.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., description="Sampler name — matches ``SamplerConfig.name``.")
    ticks_attempted: int = Field(
        default=0, ge=0, description="Number of tick attempts (successful + failed)."
    )
    ticks_succeeded: int = Field(
        default=0, ge=0, description="Number of ticks that produced a row."
    )
    ticks_failed: int = Field(
        default=0,
        ge=0,
        description="Number of ticks where :meth:`Sampler.sample_once` raised.",
    )
    started_at: datetime = Field(..., description="UTC, timezone-aware start time.")
    stopped_at: datetime = Field(..., description="UTC, timezone-aware stop time.")
    output_path: Path = Field(..., description="Path the sampler wrote JSONL rows to.")
    bytes_written: int = Field(
        default=0, ge=0, description="Total bytes written to ``output_path``."
    )
    first_sample_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the first emitted sample, or ``None`` if none.",
    )
    last_sample_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the last emitted sample, or ``None`` if none.",
    )
    failures: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable per-tick failure reasons, capped at "
            f"{_MAX_FAILURE_REASONS} entries. For diagnosis only; the "
            "canonical counts live in ``ticks_failed``."
        ),
    )


def _utc_now() -> datetime:
    """Current wall-clock time, tz-aware UTC.

    Centralised so every sampler records timestamps the
    :class:`TelemetrySample` validator accepts without massage.
    """
    return datetime.now(tz=timezone.utc)


class Sampler(ABC):
    """Abstract base class for all bench-rig samplers.

    Subclasses implement :meth:`sample_once` (one tick, returning the
    ``values`` dict that will land in a :class:`TelemetrySample`). They
    may optionally override :meth:`setup` for one-shot init (for
    example constructing a long-lived ``httpx.AsyncClient``) and
    :meth:`teardown` for cleanup. The base class owns the loop, the
    JSONL file, timing, counting, and lifecycle.

    Per-tick failures never escape :meth:`run`. Any exception raised
    from :meth:`sample_once` is caught, counted in
    :class:`SamplerResult`, and logged at ``DEBUG`` level (the runner
    logs aggregate counts at scenario end). This keeps a flapping
    sampler from crashing its peers or aborting the scenario.
    """

    config: SamplerConfig

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self._fh: Optional[BinaryIO] = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    async def sample_once(self) -> dict[str, Any]:
        """Take one sample and return its ``values`` dict.

        Concrete implementations shell out, hit HTTP, or read files.
        Raising from here is legal and normal: the loop catches the
        exception, counts it as a tick failure, and continues on the
        next tick.
        """

    async def setup(self) -> None:
        """One-shot init hook. Default is a no-op.

        Raising from :meth:`setup` aborts the run before any ticks
        execute and surfaces as :class:`SamplerError` from
        :meth:`run`. Subclasses use it to verify a required binary is
        on ``PATH``, construct a long-lived HTTP client, or enter a
        degraded mode on hosts that lack the data source.
        """

    async def teardown(self) -> None:
        """One-shot cleanup hook. Default is a no-op.

        Called from :meth:`run` on every exit path where
        :meth:`setup` completed successfully — normal stop,
        cancellation, or setup of a later tick failing. Must be safe
        to call even if :meth:`setup` opened nothing.
        """

    # ------------------------------------------------------------------
    # Loop runner
    # ------------------------------------------------------------------

    async def run(self) -> SamplerResult:
        """Run the sampler loop until cancelled.

        Loop shape::

            started_at = utc_now()
            setup()
            while not cancelled:
                wait until next tick
                sample_once()  # on success, write JSONL row
                               # on failure, count + log, continue
            teardown()
            return SamplerResult(...)

        On :class:`asyncio.CancelledError` (the normal shutdown path
        from the runner cancelling the sampler task), the loop exits
        cleanly, teardown runs, the file is flushed, and a populated
        :class:`SamplerResult` is **returned** — not re-raised. See
        the module docstring for the rationale.

        Raises
        ------
        SamplerError
            If :meth:`setup` raises or the output file cannot be
            opened. Both are unrecoverable — the runner surfaces the
            failure to the user rather than silently skipping the
            sampler.
        """
        started_at = _utc_now()
        ticks_attempted = 0
        ticks_succeeded = 0
        ticks_failed = 0
        bytes_written = 0
        first_sample_at: Optional[datetime] = None
        last_sample_at: Optional[datetime] = None
        failures: list[str] = []
        setup_complete = False

        # Short-circuit on an explicitly-disabled sampler. We still
        # touch the output file so the on-disk shape is consistent
        # (empty JSONL vs missing JSONL) regardless of whether a
        # sampler ran or was switched off.
        if not self.config.enabled:
            try:
                self._open_output()
                self._close_output()
            except OSError as exc:
                raise SamplerError(
                    f"sampler {self.config.name!r}: could not touch "
                    f"output file {self.config.output_path}: {exc}"
                ) from exc
            stopped_at = _utc_now()
            logger.info(
                "sampler %s disabled; emitted zero rows", self.config.name
            )
            return SamplerResult(
                name=self.config.name,
                ticks_attempted=0,
                ticks_succeeded=0,
                ticks_failed=0,
                started_at=started_at,
                stopped_at=stopped_at,
                output_path=self.config.output_path,
                bytes_written=0,
                first_sample_at=None,
                last_sample_at=None,
                failures=[],
            )

        # Open the output file before running setup: if setup() raises,
        # we want to remove the empty file to avoid leaving an
        # ambiguous zero-byte artifact behind.
        try:
            self._open_output()
        except OSError as exc:
            raise SamplerError(
                f"sampler {self.config.name!r}: could not open "
                f"output file {self.config.output_path}: {exc}"
            ) from exc

        try:
            try:
                await self.setup()
            except asyncio.CancelledError:
                # Cancellation during setup: abandon the empty file
                # and re-raise so the pool cleanly records the
                # cancellation.
                self._close_output()
                self._remove_output_if_empty()
                raise
            except Exception as exc:
                self._close_output()
                self._remove_output_if_empty()
                raise SamplerError(
                    f"sampler {self.config.name!r}: setup failed: {exc}"
                ) from exc
            setup_complete = True

            loop = asyncio.get_running_loop()
            tick_index = 0
            loop_start = loop.time()

            try:
                while True:
                    # Drift-compensated target for the next tick. Each
                    # tick is scheduled off a monotonic anchor so
                    # slow sample_once() calls don't cascade into a
                    # backlog — we skip ahead to the next valid tick
                    # if we fall behind.
                    target = loop_start + tick_index * self.config.interval_secs
                    now = loop.time()
                    delay = target - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                    else:
                        # We missed the target; skip ahead to the
                        # next un-missed tick so the cadence does not
                        # accumulate a backlog.
                        missed = int((now - target) // self.config.interval_secs) + 1
                        if missed > 1:
                            logger.warning(
                                "sampler %s fell behind by %d ticks; skipping",
                                self.config.name,
                                missed - 1,
                            )
                            tick_index += missed - 1
                            target = loop_start + tick_index * self.config.interval_secs
                            remaining = target - loop.time()
                            if remaining > 0:
                                await asyncio.sleep(remaining)

                    tick_index += 1
                    ticks_attempted += 1
                    tick_wall_start = loop.time()
                    sample_ts = _utc_now()

                    try:
                        values = await self.sample_once()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        ticks_failed += 1
                        reason = f"{type(exc).__name__}: {exc}"
                        if len(failures) < _MAX_FAILURE_REASONS:
                            failures.append(reason)
                        logger.debug(
                            "sampler %s tick %d failed: %s",
                            self.config.name,
                            tick_index,
                            reason,
                        )
                    else:
                        # Build + write the JSONL row.
                        try:
                            row = TelemetrySample(
                                ts=sample_ts,
                                sampler=self.config.name,
                                values=values,
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            ticks_failed += 1
                            reason = f"TelemetrySample validation failed: {exc}"
                            if len(failures) < _MAX_FAILURE_REASONS:
                                failures.append(reason)
                            logger.debug(
                                "sampler %s row validation failed: %s",
                                self.config.name,
                                reason,
                            )
                        else:
                            payload = row.model_dump_json().encode("utf-8") + b"\n"
                            try:
                                self._write_row(payload)
                                bytes_written += len(payload)
                                ticks_succeeded += 1
                                if first_sample_at is None:
                                    first_sample_at = sample_ts
                                last_sample_at = sample_ts
                            except OSError as exc:
                                ticks_failed += 1
                                reason = f"write failed: {exc}"
                                if len(failures) < _MAX_FAILURE_REASONS:
                                    failures.append(reason)
                                logger.warning(
                                    "sampler %s write failed: %s",
                                    self.config.name,
                                    exc,
                                )

                    tick_wall_end = loop.time()
                    tick_duration = tick_wall_end - tick_wall_start
                    if tick_duration > self.config.interval_secs * 0.5:
                        logger.warning(
                            "sampler %s tick %d took %.3fs (>50%% of %.3fs interval)",
                            self.config.name,
                            tick_index,
                            tick_duration,
                            self.config.interval_secs,
                        )
            except asyncio.CancelledError:
                # Normal shutdown path: fall through to finalisation
                # without re-raising. The returned SamplerResult is
                # the runner's record of this sampler's run.
                logger.debug(
                    "sampler %s cancelled after %d ticks",
                    self.config.name,
                    ticks_attempted,
                )
        finally:
            if setup_complete:
                try:
                    await self.teardown()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "sampler %s teardown raised: %s", self.config.name, exc
                    )
            self._close_output()

        stopped_at = _utc_now()
        return SamplerResult(
            name=self.config.name,
            ticks_attempted=ticks_attempted,
            ticks_succeeded=ticks_succeeded,
            ticks_failed=ticks_failed,
            started_at=started_at,
            stopped_at=stopped_at,
            output_path=self.config.output_path,
            bytes_written=bytes_written,
            first_sample_at=first_sample_at,
            last_sample_at=last_sample_at,
            failures=failures,
        )

    # ------------------------------------------------------------------
    # Output file plumbing
    # ------------------------------------------------------------------

    def _open_output(self) -> None:
        """Open the JSONL output file in binary append mode.

        Uses append mode so a pre-existing file is not truncated (the
        runner may tolerate re-runs in the same artifact directory).
        The parent directory is created if missing — samplers own
        their file lifecycle end-to-end, they don't rely on the
        :class:`ArtifactBuilder` having pre-created it.
        """
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.config.output_path.open("ab")

    def _write_row(self, payload: bytes) -> None:
        """Append one JSONL row and flush so the file is live-observable."""
        assert self._fh is not None, "output file not opened"
        self._fh.write(payload)
        self._fh.flush()

    def _close_output(self) -> None:
        """Flush and close the JSONL file handle if open."""
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:  # pragma: no cover - defensive
                pass
            try:
                self._fh.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._fh = None

    def _remove_output_if_empty(self) -> None:
        """Remove the output file if it exists and is zero-length.

        Used when :meth:`setup` fails before any ticks have run: we
        do not want to leave an ambiguous zero-byte JSONL behind. A
        non-empty file (e.g. because the caller re-ran into a
        pre-existing artifact directory) is preserved.
        """
        try:
            path = self.config.output_path
            if path.exists() and path.stat().st_size == 0:
                path.unlink()
        except OSError:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# Pool + bulk runner
# ---------------------------------------------------------------------------


class SamplerPool:
    """Async context manager that runs a set of samplers concurrently.

    Usage::

        async with SamplerPool(samplers) as pool:
            await run_scenario_steps()
        # pool.results is populated here; every sampler has been
        # cancelled, had teardown() run, and returned a SamplerResult.

    Enter schedules each sampler as a background task. Exit cancels
    every task, awaits its completion, and collects a
    :class:`SamplerResult` for every sampler. Samplers that exited with
    an exception (notably :class:`SamplerError` from a broken setup)
    are surfaced via :attr:`errors`; every other sampler contributes
    its :class:`SamplerResult` to :attr:`results`.

    The pool does not re-raise sampler errors from ``__aexit__`` — it
    is the caller's job to inspect :attr:`errors` after the context
    exits. This keeps a single broken sampler from masking a scenario
    failure.
    """

    def __init__(self, samplers: list[Sampler]) -> None:
        self._samplers = list(samplers)
        self._tasks: list[asyncio.Task[SamplerResult]] = []
        self._results: list[SamplerResult] = []
        self._errors: list[tuple[str, BaseException]] = []
        self._entered = False

    @property
    def results(self) -> list[SamplerResult]:
        """Collected :class:`SamplerResult` objects after exit."""
        return list(self._results)

    @property
    def errors(self) -> list[tuple[str, BaseException]]:
        """``(sampler_name, exception)`` pairs for samplers that failed."""
        return list(self._errors)

    async def __aenter__(self) -> "SamplerPool":
        if self._entered:
            raise RuntimeError("SamplerPool is not reentrant")
        self._entered = True
        for sampler in self._samplers:
            task = asyncio.create_task(
                sampler.run(), name=f"sampler:{sampler.config.name}"
            )
            self._tasks.append(task)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        for task in self._tasks:
            if not task.done():
                task.cancel()

        for sampler, task in zip(self._samplers, self._tasks):
            try:
                result = await task
            except asyncio.CancelledError:
                # Shouldn't happen under the documented Sampler.run
                # contract (it catches CancelledError internally) —
                # but if someone sub-classes Sampler and re-raises, we
                # record it as an error rather than crashing the
                # pool.
                self._errors.append((sampler.config.name, asyncio.CancelledError()))
            except BaseException as exc:  # noqa: BLE001
                self._errors.append((sampler.config.name, exc))
            else:
                self._results.append(result)

        # Never swallow an exception from the wrapped with-block.
        return None


async def run_samplers(samplers: list[Sampler]) -> list[SamplerResult]:
    """Run every sampler in ``samplers`` to normal termination.

    Convenience wrapper for tests and ad-hoc usage. Schedules every
    sampler on its own task and gathers them. Samplers normally run
    forever, so callers must arrange for cancellation externally
    (e.g. via :func:`asyncio.wait_for` or by cancelling the outer
    task); :class:`SamplerPool` is a better fit for the scenario
    runner because it scopes cancellation to an ``async with`` block.

    A sampler that raises :class:`SamplerError` from its setup is
    propagated out of this helper; the other samplers are cancelled
    before re-raising.
    """
    async with SamplerPool(samplers) as pool:
        # Wait for all tasks to complete (they only terminate on
        # cancellation in practice). This helper exists mostly for
        # symmetry with SamplerPool and for tests that want to gather
        # directly without managing a context.
        await asyncio.gather(*pool._tasks, return_exceptions=True)
    if pool.errors:
        # Re-raise the first error so callers of this helper get loud
        # feedback. Callers that want to collect errors should use
        # SamplerPool directly.
        _name, exc = pool.errors[0]
        raise exc
    return pool.results


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SamplerRegistry:
    """Name-based registry of sampler classes.

    Scenarios reference samplers by string name in YAML; this
    registry is how the runner turns a name into a class. The
    ``__init__.py`` of the :mod:`concerto_bench.samplers` package
    populates a module-level default registry with every built-in
    sampler.
    """

    def __init__(self) -> None:
        self._entries: dict[str, type[Sampler]] = {}

    def register(self, name: str, sampler_cls: type[Sampler]) -> None:
        """Register ``sampler_cls`` under ``name``.

        Re-registering the same name raises :class:`ValueError` to
        catch scenario YAML typos rather than silently overwriting.
        """
        if not name:
            raise ValueError("sampler name must be non-empty")
        if name in self._entries:
            raise ValueError(f"sampler name {name!r} already registered")
        self._entries[name] = sampler_cls

    def get(self, name: str) -> type[Sampler]:
        """Look up a sampler class by name; raises :class:`KeyError` if absent."""
        return self._entries[name]

    def names(self) -> list[str]:
        """Return the registered names in insertion order."""
        return list(self._entries.keys())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries
