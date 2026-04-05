"""``snapshot`` primitive: capture concerto + OS state in parallel.

Every scenario step begins and ends with a snapshot. The runner takes
one before any actions run (``pre-state.json``) and one after every
action has finished (``post-state.json``); the analyzer uses the delta
between the two to attribute VRAM drift (ROADMAP M4), detect orphan
backends (ROADMAP M5), and reconstruct state transitions across steps.

This primitive is the single source of truth for what "state" means to
the rig. It captures three things in parallel:

1. **Concerto's view** of the cluster, via ``GET /status``.
2. **Ground-truth VRAM / GPU state**, via ``nvidia-smi --query-gpu=...``.
3. **Ground-truth process list**, via ``pgrep -af <pattern>`` for each
   known engine command pattern.

The three sub-captures run concurrently with :func:`asyncio.gather` so
the wall-clock cost of a snapshot is the max of the three, not the sum.
Individual sub-capture failures are tolerated — a missing ``nvidia-smi``
binary on a dev laptop is expected, a concerto ``/status`` 5xx is not.
The only unrecoverable failure is the ``/status`` call itself; every
other failure degrades gracefully to ``None`` / ``[]`` with a logged
warning.

See :class:`~concerto_bench.schema.StateSnapshot` for the on-disk shape
the primitive materialises.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from concerto_bench.schema import StateSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


_DEFAULT_PGREP_PATTERNS: tuple[str, ...] = (
    "vllm",
    "python -m vllm",
    "mock-inference-backend",
)
"""Sprint 2 engine command patterns ``pgrep`` searches by default.

Covers the two engines Sprint 2 validates against: real vLLM (both the
installed-script entry point and the ``python -m vllm`` module entry)
and the bundled ``mock-inference-backend`` binary used by the CI dry
run. Scenarios can override this per-action if they spawn different
engines.
"""

_NVIDIA_SMI_COLUMNS: tuple[str, ...] = (
    "index",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "temperature.gpu",
)
"""``nvidia-smi`` CSV query columns in the exact order requested.

Chosen to match the smallest superset of fields the analyzer cares
about: GPU identity, the three memory numbers ROADMAP M4's VRAM-drift
check needs, utilisation, and temperature.
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SnapshotError(RuntimeError):
    """Raised when the snapshot primitive cannot produce a StateSnapshot.

    Reserved for the single unrecoverable failure class: concerto's
    ``/status`` endpoint is unreachable or returns a 5xx. Every other
    sub-capture (``nvidia-smi``, ``pgrep``) is allowed to fail silently
    and populates its field with ``None`` / ``[]``.
    """


# ---------------------------------------------------------------------------
# Action argument model
# ---------------------------------------------------------------------------


class SnapshotAction(BaseModel):
    """Scenario YAML arguments for a single ``snapshot`` action.

    Frozen so the same action can be stashed and replayed across steps.
    Every field has a sensible default so a scenario author writes
    ``- snapshot: {}`` most of the time; the overrides exist for
    scenarios that need to disable a sub-capture or point ``pgrep`` at
    a different engine binary.

    Fields
    ------

    ``include_nvidia_smi``
        If ``False``, the ``nvidia-smi`` sub-capture is skipped and
        ``StateSnapshot.nvidia_smi`` is ``None``.
    ``include_pgrep``
        If ``False``, the ``pgrep`` sub-capture is skipped and
        ``StateSnapshot.backend_pids`` is ``[]``.
    ``pgrep_patterns``
        List of ``pgrep`` patterns to search. Results are unioned and
        deduplicated. An empty list short-circuits to zero PIDs.
    ``timeout_secs``
        Per sub-capture wall-clock timeout. Applies independently to
        ``/status``, ``nvidia-smi``, and each ``pgrep`` invocation.
    ``capture_label``
        Optional label the runner can stash on the resulting
        :class:`~concerto_bench.schema.ActionRecord` (e.g. ``"pre"`` /
        ``"post"``). Unused by the primitive itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    include_nvidia_smi: bool = Field(
        default=True,
        description="Capture nvidia-smi alongside /status.",
    )
    include_pgrep: bool = Field(
        default=True,
        description="Capture pgrep output for the configured patterns.",
    )
    pgrep_patterns: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_PGREP_PATTERNS),
        description="pgrep patterns to union for backend_pids.",
    )
    timeout_secs: float = Field(
        default=10.0,
        description="Per sub-capture wall-clock timeout in seconds.",
    )
    capture_label: Optional[str] = Field(
        default=None,
        description="Optional runner-level label, e.g. 'pre' or 'post'.",
    )

    @field_validator("timeout_secs")
    @classmethod
    def _validate_timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"timeout_secs must be > 0, got {value}")
        return value


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


class SnapshotPrimitive:
    """Stateless executor for :class:`SnapshotAction`.

    A single instance is safe to reuse across every snapshot in a run.
    Like :class:`~concerto_bench.primitives.request.RequestPrimitive`,
    the primitive never mutates or closes an injected HTTP client —
    the caller owns its lifetime.

    Sub-capture failure policy
    --------------------------

    * ``GET /status`` failure (connect error, 5xx, transport error,
      timeout) → :class:`SnapshotError`. This is the one unrecoverable
      case: without ``/status`` there is nothing meaningful to snapshot.
    * ``nvidia-smi`` missing binary / non-zero exit / timeout →
      ``StateSnapshot.nvidia_smi = None``, logged at INFO.
    * ``pgrep`` missing binary / timeout → empty list contribution,
      logged at INFO. ``pgrep`` returning exit code 1 (no matches) is
      *not* a failure — it contributes an empty list.
    """

    async def execute(
        self,
        action: SnapshotAction,
        *,
        base_url: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> StateSnapshot:
        """Execute one snapshot and return a populated :class:`StateSnapshot`.

        Parameters
        ----------
        action:
            Frozen arguments for this snapshot.
        base_url:
            Concerto's HTTP base URL, e.g. ``http://127.0.0.1:8000``.
        client:
            Optional pre-built :class:`httpx.AsyncClient`. When supplied
            the primitive uses it verbatim and does not close it on
            exit. When ``None``, a short-lived internal client is used.

        Returns
        -------
        :class:`~concerto_bench.schema.StateSnapshot`
            Timestamp is captured at entry (not exit) to match the
            convention the runner uses across pre/post snapshots.

        Raises
        ------
        :class:`SnapshotError`
            If and only if ``GET /status`` failed. Every other
            sub-capture degrades gracefully.
        """
        # Entry-time stamp. Using ``datetime.now(timezone.utc)`` rather
        # than ``perf_counter`` because the schema requires an aware
        # wall-clock datetime; the analyzer correlates snapshots with
        # JSONL telemetry rows on wall-clock time.
        ts = datetime.now(timezone.utc)

        owned_client = client is None
        active_client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(action.timeout_secs),
        )
        try:
            # asyncio.gather with return_exceptions=True makes a failure
            # in any single sub-capture observable without aborting the
            # others. We then inspect each result and apply the
            # per-capture policy (strict for /status, lenient for the
            # two shell-outs).
            status_coro = self._capture_status(
                base_url=base_url,
                client=active_client,
                timeout_secs=action.timeout_secs,
            )
            if action.include_nvidia_smi:
                nvidia_coro: "asyncio.Future[Optional[dict[str, Any]]]" = asyncio.ensure_future(
                    self._capture_nvidia_smi(timeout_secs=action.timeout_secs)
                )
            else:
                nvidia_coro = asyncio.ensure_future(_none_future())
            if action.include_pgrep and action.pgrep_patterns:
                pgrep_coro: "asyncio.Future[tuple[list[int], dict[str, str]]]" = asyncio.ensure_future(
                    self._capture_pgrep(
                        patterns=list(action.pgrep_patterns),
                        timeout_secs=action.timeout_secs,
                    )
                )
            else:
                pgrep_coro = asyncio.ensure_future(_empty_pgrep_future())

            results = await asyncio.gather(
                asyncio.ensure_future(status_coro),
                nvidia_coro,
                pgrep_coro,
                return_exceptions=True,
            )
        finally:
            if owned_client:
                await active_client.aclose()

        status_result, nvidia_result, pgrep_result = results

        # /status is strict. Anything other than a clean dict means the
        # snapshot is not meaningful; raise.
        if isinstance(status_result, BaseException):
            raise SnapshotError(
                f"GET {base_url}/status failed: "
                f"{type(status_result).__name__}: {status_result}"
            ) from status_result
        concerto_status: dict[str, Any] = status_result  # type: ignore[assignment]

        # nvidia-smi is lenient.
        nvidia_smi: Optional[dict[str, Any]]
        if isinstance(nvidia_result, BaseException):
            logger.info(
                "snapshot: nvidia-smi sub-capture failed: %s: %s",
                type(nvidia_result).__name__,
                nvidia_result,
            )
            nvidia_smi = None
        else:
            nvidia_smi = nvidia_result  # type: ignore[assignment]

        # pgrep is lenient.
        backend_pids: list[int]
        pgrep_command_lines: dict[str, str]
        if isinstance(pgrep_result, BaseException):
            logger.info(
                "snapshot: pgrep sub-capture failed: %s: %s",
                type(pgrep_result).__name__,
                pgrep_result,
            )
            backend_pids = []
            pgrep_command_lines = {}
        else:
            backend_pids, pgrep_command_lines = pgrep_result  # type: ignore[assignment]

        extra: dict[str, Any] = {
            "pgrep_patterns": list(action.pgrep_patterns),
            "pgrep_command_lines": pgrep_command_lines,
        }
        if action.capture_label is not None:
            extra["capture_label"] = action.capture_label

        return StateSnapshot(
            ts=ts,
            concerto_status=concerto_status,
            nvidia_smi=nvidia_smi,
            backend_pids=backend_pids,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # /status capture
    # ------------------------------------------------------------------

    async def _capture_status(
        self,
        *,
        base_url: str,
        client: httpx.AsyncClient,
        timeout_secs: float,
    ) -> dict[str, Any]:
        """Fetch concerto ``/status`` and return it as a parsed dict.

        Raises any transport or status exception — the caller turns
        those into a :class:`SnapshotError`.
        """
        url = f"{base_url.rstrip('/')}/status"
        response = await asyncio.wait_for(
            client.get(url, timeout=httpx.Timeout(timeout_secs)),
            timeout=timeout_secs,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise SnapshotError(
                f"/status returned a non-object JSON body: {type(data).__name__}"
            )
        return data

    # ------------------------------------------------------------------
    # nvidia-smi capture
    # ------------------------------------------------------------------

    async def _capture_nvidia_smi(
        self,
        *,
        timeout_secs: float,
    ) -> Optional[dict[str, Any]]:
        """Shell out to ``nvidia-smi`` and parse the CSV response.

        Returns a dict with one ``gpus`` key holding a list of per-GPU
        dicts (one per row of CSV output). Returns ``None`` when the
        binary is missing, exits non-zero, or its output cannot be
        parsed — none of these are fatal to the snapshot.
        """
        query = ",".join(_NVIDIA_SMI_COLUMNS)
        argv = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.info("snapshot: nvidia-smi not on PATH; skipping")
            return None

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_secs
            )
        except asyncio.TimeoutError:
            logger.info(
                "snapshot: nvidia-smi timed out after %.2fs; killing and skipping",
                timeout_secs,
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            # Drain so we do not leak the subprocess.
            try:
                await proc.wait()
            except Exception:  # pragma: no cover - extremely defensive
                pass
            return None

        if proc.returncode != 0:
            logger.info(
                "snapshot: nvidia-smi exited non-zero (rc=%s): %s",
                proc.returncode,
                stderr_bytes.decode("utf-8", errors="replace").strip(),
            )
            return None

        try:
            parsed_rows = self._parse_nvidia_smi_csv(
                stdout_bytes.decode("utf-8", errors="replace")
            )
        except Exception as exc:  # pragma: no cover - parser is total
            logger.info("snapshot: nvidia-smi CSV parse failed: %s", exc)
            return None

        return {"gpus": parsed_rows}

    def _parse_nvidia_smi_csv(self, text: str) -> list[dict[str, Any]]:
        """Parse ``nvidia-smi --format=csv,noheader,nounits`` output.

        Each non-empty line is expected to contain exactly
        ``len(_NVIDIA_SMI_COLUMNS)`` comma-separated fields. Integer and
        float columns are coerced to their native types so the analyzer
        does not have to re-parse strings; a field that fails coercion
        is retained as a string rather than dropped, so malformed
        upstream rows remain visible.
        """
        rows: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != len(_NVIDIA_SMI_COLUMNS):
                logger.info(
                    "snapshot: skipping malformed nvidia-smi row (got %d fields, expected %d): %r",
                    len(fields),
                    len(_NVIDIA_SMI_COLUMNS),
                    line,
                )
                continue
            row: dict[str, Any] = {}
            for column, value in zip(_NVIDIA_SMI_COLUMNS, fields):
                row[column] = self._coerce_nvidia_smi_field(column, value)
            rows.append(row)
        return rows

    @staticmethod
    def _coerce_nvidia_smi_field(column: str, value: str) -> Any:
        """Best-effort numeric coercion for one ``nvidia-smi`` CSV cell."""
        if column == "index":
            try:
                return int(value)
            except ValueError:
                return value
        if column in ("memory.total", "memory.used", "memory.free"):
            try:
                return int(value)
            except ValueError:
                return value
        if column in ("utilization.gpu", "temperature.gpu"):
            try:
                return float(value)
            except ValueError:
                return value
        return value

    # ------------------------------------------------------------------
    # pgrep capture
    # ------------------------------------------------------------------

    async def _capture_pgrep(
        self,
        *,
        patterns: list[str],
        timeout_secs: float,
    ) -> tuple[list[int], dict[str, str]]:
        """Run ``pgrep -af`` for each pattern; return ``(pids, raw_output_map)``.

        The returned PID list is the sorted, deduplicated union across
        every pattern. ``raw_output_map`` maps each pattern to the
        captured stdout text so the artifact retains enough context to
        debug a pattern that is silently matching the wrong processes.

        A missing ``pgrep`` binary or a timeout on any pattern fails
        that one pattern silently; we still return whatever we
        collected from the others. ``pgrep`` exit code 1 means "no
        matches" and contributes an empty list — it is not an error.
        """
        pids_set: set[int] = set()
        command_lines: dict[str, str] = {}

        for pattern in patterns:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "pgrep",
                    "-af",
                    pattern,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                logger.info("snapshot: pgrep not on PATH; skipping all patterns")
                return [], {}

            try:
                stdout_bytes, _stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_secs
                )
            except asyncio.TimeoutError:
                logger.info(
                    "snapshot: pgrep timed out for pattern %r; skipping that pattern",
                    pattern,
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await proc.wait()
                except Exception:  # pragma: no cover - defensive
                    pass
                continue

            text = stdout_bytes.decode("utf-8", errors="replace")
            command_lines[pattern] = text

            rc = proc.returncode
            if rc == 0:
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    head, _, _tail = line.partition(" ")
                    try:
                        pids_set.add(int(head))
                    except ValueError:
                        logger.info(
                            "snapshot: could not parse pgrep line %r for pattern %r",
                            line,
                            pattern,
                        )
            elif rc == 1:
                # No matches — a legitimate outcome, not an error.
                continue
            else:
                logger.info(
                    "snapshot: pgrep pattern %r exited with rc=%s; ignoring",
                    pattern,
                    rc,
                )

        return sorted(pids_set), command_lines


# ---------------------------------------------------------------------------
# Helper coroutines for skipped sub-captures
# ---------------------------------------------------------------------------


async def _none_future() -> Optional[dict[str, Any]]:
    """Awaitable that returns ``None`` immediately.

    Used in :meth:`SnapshotPrimitive.execute` when the caller has asked
    for a sub-capture to be skipped — keeping the shape of the
    ``asyncio.gather`` call uniform regardless of which sub-captures
    are enabled.
    """
    return None


async def _empty_pgrep_future() -> tuple[list[int], dict[str, str]]:
    """Awaitable that returns an empty pgrep result immediately."""
    return [], {}
