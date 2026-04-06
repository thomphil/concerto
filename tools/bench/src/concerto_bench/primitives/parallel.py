"""``parallel`` primitive: run multiple actions concurrently.

A meta-primitive that wraps other actions in ``asyncio.gather`` for
concurrent execution. Used by scenarios that need to fire requests in
parallel (e.g. load testing + monitoring simultaneously) or that want
to express "all of these things should happen at the same time."

Unlike the other primitives, ``ParallelPrimitive.execute`` accepts an
extra ``dispatch`` keyword argument — a callable injected by the
runner that knows how to map an action spec dict to an awaitable
result. This keeps the primitive itself ignorant of the full action
registry; the runner wires up the dispatch at invocation time.

Shape at a glance
-----------------

* :class:`ParallelAction` — frozen pydantic model, ``extra="forbid"``.
  Holds a list of action spec dicts (same shape as scenario YAML
  actions) and an overall timeout.
* :class:`ParallelError` — raised only on irrecoverable failures
  (missing dispatch callable, structural issues).
* :class:`ParallelPrimitive` — the meta-executor. Dispatches each
  sub-action concurrently and aggregates results.

Timeout behaviour
-----------------

The overall ``timeout_secs`` is applied via ``asyncio.wait_for`` around
the ``asyncio.gather``. If the timeout fires, partial results (from
sub-actions that completed before the deadline) are included; the
remaining sub-actions contribute error strings.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ParallelError(RuntimeError):
    """Raised when the ``parallel`` primitive encounters an irrecoverable failure.

    Reserved for structural issues — a missing ``dispatch`` callable,
    an empty action list, or a timeout at the gather level. Individual
    sub-action failures are captured in the ``errors`` list of the
    result, not raised.
    """


# ---------------------------------------------------------------------------
# Action argument model
# ---------------------------------------------------------------------------


class ParallelAction(BaseModel):
    """Scenario YAML arguments for a single ``parallel`` action.

    Frozen so the runner can stash and reuse it across retries.

    Fields
    ------

    ``actions``
        List of action spec dicts, each in the same format as a single
        entry in a scenario YAML step's ``actions`` array.
    ``timeout_secs``
        Overall timeout for the parallel group. If the gather does not
        complete within this window, partial results are returned.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: list[dict[str, Any]] = Field(
        ...,
        description="List of action spec dicts (same format as scenario YAML actions).",
    )
    timeout_secs: float = Field(
        default=300.0,
        description="Overall timeout for the parallel group in seconds.",
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


class ParallelPrimitive:
    """Meta-executor for :class:`ParallelAction`.

    Unlike other primitives, this one requires a ``dispatch`` callable
    injected by the runner. The dispatch maps an action spec dict to
    an awaitable result dict.

    Failure policy
    --------------

    * Individual sub-action failures (exceptions) are captured in the
      ``errors`` list of the result. They do NOT propagate.
    * An overall timeout causes the gather to be cancelled; partial
      results are returned alongside timeout error messages.
    * A missing ``dispatch`` callable raises :class:`ParallelError`.
    """

    async def execute(
        self,
        action: ParallelAction,
        *,
        base_url: str,
        client: Optional[httpx.AsyncClient] = None,
        dispatch: Optional[Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        """Run sub-actions concurrently and return aggregated results.

        Parameters
        ----------
        action:
            Frozen arguments for this invocation.
        base_url:
            Concerto's HTTP base URL. Passed through to sub-actions
            via dispatch.
        client:
            Optional pre-built HTTP client. Passed through to
            sub-actions via dispatch.
        dispatch:
            Callable that maps an action spec dict to an awaitable
            result dict. Injected by the runner.

        Returns
        -------
        dict
            ``{"results": list[dict], "errors": list[str],
            "elapsed_secs": float}``

        Raises
        ------
        :class:`ParallelError`
            If ``dispatch`` is None.
        """
        if dispatch is None:
            raise ParallelError(
                "parallel primitive requires a dispatch callable; "
                "this is injected by the runner"
            )

        start = time.perf_counter()
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        coros = [dispatch(a) for a in action.actions]

        try:
            gathered = await asyncio.wait_for(
                asyncio.gather(*coros, return_exceptions=True),
                timeout=action.timeout_secs,
            )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start
            errors.append(
                f"parallel group timed out after {elapsed:.2f}s "
                f"(limit {action.timeout_secs}s)"
            )
            return {
                "results": results,
                "errors": errors,
                "elapsed_secs": elapsed,
            }

        for i, result in enumerate(gathered):
            if isinstance(result, BaseException):
                errors.append(
                    f"action[{i}]: {type(result).__name__}: {result}"
                )
            elif isinstance(result, dict):
                results.append(result)
            else:
                # Non-dict, non-exception result — wrap it.
                results.append({"value": result})

        elapsed = time.perf_counter() - start
        return {
            "results": results,
            "errors": errors,
            "elapsed_secs": elapsed,
        }
