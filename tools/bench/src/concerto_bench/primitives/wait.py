"""``wait`` primitive: simple sleep for inter-action delays.

The simplest primitive in the bench rig. Scenarios use it to insert
deliberate pauses between actions — for example, waiting a few seconds
after requesting a model load before probing ``/status``, or spacing
out requests to avoid overwhelming a cold backend.

The primitive has no side-effects: it does not touch the network or the
filesystem. ``base_url`` and ``client`` are accepted for interface
uniformity with the other primitives but are not used.

Shape at a glance
-----------------

* :class:`WaitAction` — frozen pydantic model, ``extra="forbid"``.
  Single required field: ``duration_secs``.
* :class:`WaitError` — raised only on irrecoverable failures (none
  expected in practice; exists for interface completeness).
* :class:`WaitPrimitive` — stateless executor. Sleeps for the
  requested duration and returns the actual elapsed time.
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


class WaitError(RuntimeError):
    """Raised when the ``wait`` primitive encounters an irrecoverable failure.

    In practice this should never fire — ``asyncio.sleep`` is about as
    reliable as it gets. The class exists so every primitive has a
    uniform ``*Error`` counterpart for the runner to catch.
    """


# ---------------------------------------------------------------------------
# Action argument model
# ---------------------------------------------------------------------------


class WaitAction(BaseModel):
    """Scenario YAML arguments for a single ``wait`` action.

    Frozen so the runner can stash and reuse it across retries.

    Fields
    ------

    ``duration_secs``
        How long to sleep, in seconds. Must be strictly positive.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_secs: float = Field(
        ...,
        gt=0,
        description="Sleep duration in seconds. Must be > 0.",
    )


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


class WaitPrimitive:
    """Stateless executor for :class:`WaitAction`.

    A single instance can be reused across every wait in a run. The
    primitive has no mutable state and performs no IO beyond
    ``asyncio.sleep``.
    """

    async def execute(
        self,
        action: WaitAction,
        *,
        base_url: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """Sleep for the requested duration and return actual elapsed time.

        Parameters
        ----------
        action:
            Frozen arguments for this invocation.
        base_url:
            Accepted for interface uniformity; not used.
        client:
            Accepted for interface uniformity; not used.

        Returns
        -------
        dict
            ``{"slept_secs": <actual_elapsed>}`` where ``actual_elapsed``
            is the wall-clock time spent sleeping, measured via
            ``time.perf_counter``.
        """
        start = time.perf_counter()
        await asyncio.sleep(action.duration_secs)
        elapsed = time.perf_counter() - start
        logger.debug("wait primitive slept %.4fs (requested %.4fs)", elapsed, action.duration_secs)
        return {"slept_secs": elapsed}
