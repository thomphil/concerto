"""``wait_for`` primitive: poll ``/status`` until a predicate is satisfied.

Many scenario steps need to wait for an asynchronous state transition to
complete before proceeding — a model finishing its cold-start, an
eviction draining, or a backend count stabilising. This primitive polls
concerto's ``GET /status`` endpoint at a configurable interval and
checks a caller-specified condition against the response JSON.

Unlike :mod:`~concerto_bench.primitives.assertions`, ``wait_for`` is
*temporal*: it retries until the condition becomes true or a timeout
expires. Assertions are instantaneous — they check the current state
exactly once and pass or fail. ``wait_for`` is the patient cousin.

Shape at a glance
-----------------

* :class:`WaitForAction` — frozen pydantic model, ``extra="forbid"``.
  Specifies the condition to check, timeout, and poll interval.
* :class:`WaitForError` — raised only on irrecoverable failures
  (malformed condition, unusable client).
* :class:`WaitForPrimitive` — stateless executor. Polls ``/status``
  in a loop, evaluates the condition, and returns a result dict.

Timeout behaviour
-----------------

On timeout the primitive returns ``{"satisfied": False, ...}`` — it
does **not** raise. The runner's assertion layer decides whether a
timed-out wait is fatal to the scenario.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WaitForError(RuntimeError):
    """Raised when the ``wait_for`` primitive encounters an irrecoverable failure.

    Reserved for structurally invalid conditions or client-state
    failures. Timeout is *not* an error — it returns
    ``{"satisfied": False}``.
    """


# ---------------------------------------------------------------------------
# Action argument model
# ---------------------------------------------------------------------------


_VALID_CONDITIONS = frozenset({"model_loaded", "model_not_loaded", "backend_count"})


class WaitForAction(BaseModel):
    """Scenario YAML arguments for a single ``wait_for`` action.

    Frozen so the runner can stash and reuse it across retries.

    Fields
    ------

    ``condition``
        One of ``"model_loaded"``, ``"model_not_loaded"``,
        ``"backend_count"``.
    ``model``
        Model ID to check. Required for ``model_loaded`` and
        ``model_not_loaded`` conditions.
    ``expected_count``
        Expected backend count. Required for the ``backend_count``
        condition.
    ``timeout_secs``
        Maximum wall-clock time to poll before giving up.
    ``poll_interval_secs``
        Delay between successive ``GET /status`` polls.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: str = Field(
        ...,
        description="Predicate to evaluate: model_loaded, model_not_loaded, backend_count.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model ID for model_loaded/model_not_loaded conditions.",
    )
    expected_count: Optional[int] = Field(
        default=None,
        description="Expected backend count for the backend_count condition.",
    )
    timeout_secs: float = Field(
        default=60.0,
        description="Maximum wall-clock seconds to poll before giving up.",
    )
    poll_interval_secs: float = Field(
        default=0.5,
        description="Seconds between successive /status polls.",
    )

    @field_validator("condition")
    @classmethod
    def _validate_condition(cls, value: str) -> str:
        if value not in _VALID_CONDITIONS:
            raise ValueError(
                f"condition must be one of {sorted(_VALID_CONDITIONS)}, got {value!r}"
            )
        return value

    @field_validator("timeout_secs")
    @classmethod
    def _validate_timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"timeout_secs must be > 0, got {value}")
        return value

    @field_validator("poll_interval_secs")
    @classmethod
    def _validate_poll_interval_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"poll_interval_secs must be > 0, got {value}")
        return value


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


class WaitForPrimitive:
    """Stateless executor for :class:`WaitForAction`.

    Polls ``GET {base_url}/status`` in a loop, evaluating the
    specified condition against the JSON response until the condition
    is satisfied or the timeout expires.

    Failure policy
    --------------

    * Timeout: returns ``{"satisfied": False, ...}`` — does NOT raise.
    * Transport error during a poll: logs a warning and retries on the
      next interval. The timeout is the hard upper bound.
    * Structurally invalid condition: raises :class:`WaitForError`
      (caught at parse time by the validator, so this is defence in
      depth).
    """

    async def execute(
        self,
        action: WaitForAction,
        *,
        base_url: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """Poll ``/status`` until the condition is satisfied or timeout.

        Parameters
        ----------
        action:
            Frozen arguments for this invocation.
        base_url:
            Concerto's HTTP base URL, e.g. ``http://127.0.0.1:8000``.
        client:
            Optional pre-built :class:`httpx.AsyncClient`. When ``None``
            the primitive creates a short-lived internal client.

        Returns
        -------
        dict
            ``{"satisfied": bool, "elapsed_secs": float, "polls": int,
            "final_status": dict}``
        """
        owned_client = client is None
        active_client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(min(action.timeout_secs, 10.0)),
        )

        url = f"{base_url.rstrip('/')}/status"
        start = time.perf_counter()
        polls = 0
        final_status: dict[str, Any] = {}

        try:
            while True:
                elapsed = time.perf_counter() - start
                if elapsed >= action.timeout_secs:
                    break

                polls += 1
                try:
                    response = await active_client.get(url)
                    status_data = response.json()
                    if not isinstance(status_data, dict):
                        status_data = {}
                except Exception as exc:
                    logger.warning(
                        "wait_for: poll %d failed: %s: %s",
                        polls,
                        type(exc).__name__,
                        exc,
                    )
                    status_data = {}

                final_status = status_data

                if self._check_condition(action, status_data):
                    elapsed = time.perf_counter() - start
                    return {
                        "satisfied": True,
                        "elapsed_secs": elapsed,
                        "polls": polls,
                        "final_status": final_status,
                    }

                # Sleep for the poll interval, but not beyond the timeout.
                remaining = action.timeout_secs - (time.perf_counter() - start)
                if remaining <= 0:
                    break
                await asyncio.sleep(min(action.poll_interval_secs, remaining))
        finally:
            if owned_client:
                await active_client.aclose()

        elapsed = time.perf_counter() - start
        return {
            "satisfied": False,
            "elapsed_secs": elapsed,
            "polls": polls,
            "final_status": final_status,
        }

    # ------------------------------------------------------------------
    # Condition evaluation
    # ------------------------------------------------------------------

    def _check_condition(
        self,
        action: WaitForAction,
        status: dict[str, Any],
    ) -> bool:
        """Evaluate the action's condition against a ``/status`` response."""
        condition = action.condition

        if condition == "model_loaded":
            return self._is_model_loaded(status, action.model)
        if condition == "model_not_loaded":
            return not self._is_model_loaded(status, action.model)
        if condition == "backend_count":
            backends = self._extract_backends(status)
            return len(backends) == action.expected_count

        # Should be unreachable thanks to the validator; defence in depth.
        logger.warning("wait_for: unknown condition %r", condition)
        return False

    @staticmethod
    def _is_model_loaded(status: dict[str, Any], model: Optional[str]) -> bool:
        """Check if a model appears in the backends list with a loaded/running status."""
        if model is None:
            return False
        # Check top-level ``backends`` list first (future-proof).
        backends = status.get("backends")
        if isinstance(backends, list) and backends:
            for backend in backends:
                if not isinstance(backend, dict):
                    continue
                backend_model = backend.get("model", backend.get("model_id", ""))
                backend_status = backend.get("status", "").lower()
                if backend_model == model and backend_status in ("loaded", "running", "ready"):
                    return True
            return False
        # Fall back to gpus[].loaded_models[] — concerto's actual
        # /status shape nests models here as objects with ``model_id``.
        gpus = status.get("gpus", [])
        if isinstance(gpus, list):
            for gpu in gpus:
                if isinstance(gpu, dict):
                    loaded = gpu.get("loaded_models", [])
                    if isinstance(loaded, list):
                        for entry in loaded:
                            if isinstance(entry, dict) and entry.get("model_id") == model:
                                return True
                            if isinstance(entry, str) and entry == model:
                                return True
        return False

    @staticmethod
    def _extract_backends(status: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract the backends list from a ``/status`` response."""
        backends = status.get("backends", [])
        if isinstance(backends, list) and backends:
            return [b for b in backends if isinstance(b, dict)]
        # Fall back to extracting from gpus[].loaded_models[].
        result: list[dict[str, Any]] = []
        gpus = status.get("gpus", [])
        if isinstance(gpus, list):
            for gpu in gpus:
                if isinstance(gpu, dict):
                    loaded = gpu.get("loaded_models", [])
                    if isinstance(loaded, list):
                        for entry in loaded:
                            if isinstance(entry, dict):
                                result.append(entry)
        return result
