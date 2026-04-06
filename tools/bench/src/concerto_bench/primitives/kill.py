"""``kill`` primitive: send a POSIX signal to processes matching a pattern.

Used by scenarios that need to simulate backend crashes, orphan cleanup,
or forced eviction. The primitive shells out to ``pgrep -f`` to find
matching PIDs, then sends the requested signal via :func:`os.kill`.

This is a process-level operation — it does not use ``base_url`` or
``client``. Those parameters are accepted for interface uniformity.

Shape at a glance
-----------------

* :class:`KillAction` — frozen pydantic model, ``extra="forbid"``.
  Fields: ``pattern``, ``signal``, ``expect_found``.
* :class:`KillError` — raised only on irrecoverable failures (e.g.
  ``pgrep`` binary missing when ``expect_found`` is True).
* :class:`KillPrimitive` — stateless executor. Finds PIDs via pgrep,
  sends the signal, and returns a result dict.

Signal safety
-------------

The primitive sends exactly the signal the scenario specifies. It does
not second-guess whether SIGKILL vs SIGTERM is appropriate — that is
the scenario author's responsibility.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class KillError(RuntimeError):
    """Raised when the ``kill`` primitive encounters an irrecoverable failure.

    Raised when ``expect_found`` is True but no process matched the
    pattern, or when ``pgrep`` is not available on the host.
    """


# ---------------------------------------------------------------------------
# Action argument model
# ---------------------------------------------------------------------------


class KillAction(BaseModel):
    """Scenario YAML arguments for a single ``kill`` action.

    Frozen so the runner can stash and reuse it across retries.

    Fields
    ------

    ``pattern``
        Pattern passed to ``pgrep -f`` to find matching processes.
    ``signal``
        POSIX signal number to send (default 9 = SIGKILL).
    ``expect_found``
        If True and no process matches the pattern, the result is
        treated as a failure.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: str = Field(
        ...,
        min_length=1,
        description="pgrep -f pattern to find the target process(es).",
    )
    signal: int = Field(
        default=9,
        description="POSIX signal number to send (default SIGKILL=9).",
    )
    expect_found: bool = Field(
        default=True,
        description="If True and no process matches, mark as failure.",
    )
    kill_children: bool = Field(
        default=False,
        description=(
            "If True, also kill direct child processes of each matched PID "
            "via pkill -P before killing the parent. Needed for engines like "
            "vLLM that spawn EngineCore child processes holding GPU memory."
        ),
    )


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


class KillPrimitive:
    """Stateless executor for :class:`KillAction`.

    A single instance can be reused across every kill in a run. The
    primitive has no mutable state.

    Failure policy
    --------------

    * ``pgrep`` binary missing: raises :class:`KillError` if
      ``expect_found`` is True; returns empty result otherwise.
    * No matching PIDs: returns with ``pids_found=[]``. If
      ``expect_found`` is True, the runner should treat this as a
      failure.
    * Signal delivery failure (e.g. EPERM): logged in ``errors`` list
      but does not raise. The result contains both ``pids_found`` and
      ``pids_killed`` so the caller can assess partial success.
    """

    async def execute(
        self,
        action: KillAction,
        *,
        base_url: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """Find processes matching ``pattern`` and send ``signal``.

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
            ``{"pids_found": list[int], "pids_killed": list[int],
            "signal": int, "errors": list[str]}``

        Raises
        ------
        :class:`KillError`
            If ``pgrep`` is missing and ``expect_found`` is True.
        """
        pids_found: list[int] = []
        pids_killed: list[int] = []
        errors: list[str] = []

        # Find PIDs via pgrep -f
        try:
            proc = await asyncio.create_subprocess_exec(
                "pgrep",
                "-f",
                action.pattern,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            msg = "pgrep binary not found on PATH"
            if action.expect_found:
                raise KillError(msg)
            logger.warning("kill primitive: %s", msg)
            return {
                "pids_found": [],
                "pids_killed": [],
                "signal": action.signal,
                "errors": [msg],
            }

        stdout_bytes, _stderr_bytes = await proc.communicate()

        if proc.returncode == 0:
            for line in stdout_bytes.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    pids_found.append(int(line))
                except ValueError:
                    logger.info("kill primitive: could not parse pgrep line %r", line)

        if not pids_found and action.expect_found:
            errors.append(f"no process matched pattern {action.pattern!r}")

        # Kill children first if requested (e.g. vLLM EngineCore workers
        # that hold GPU memory and won't die when the parent is killed).
        if action.kill_children and pids_found:
            for pid in pids_found:
                try:
                    child_proc = await asyncio.create_subprocess_exec(
                        "pkill",
                        f"-{action.signal}",
                        "-P",
                        str(pid),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await child_proc.communicate()
                    logger.debug(
                        "kill primitive: killed children of PID %d", pid
                    )
                except FileNotFoundError:
                    logger.warning("kill primitive: pkill not found, cannot kill children")
            # Brief pause for GPU memory teardown.
            await asyncio.sleep(1)

        # Send signal to each PID
        for pid in pids_found:
            try:
                os.kill(pid, action.signal)
                pids_killed.append(pid)
                logger.debug(
                    "kill primitive: sent signal %d to PID %d (pattern=%r)",
                    action.signal,
                    pid,
                    action.pattern,
                )
            except ProcessLookupError:
                msg = f"PID {pid} not found (already exited)"
                errors.append(msg)
                logger.info("kill primitive: %s", msg)
            except PermissionError:
                msg = f"PID {pid}: permission denied for signal {action.signal}"
                errors.append(msg)
                logger.warning("kill primitive: %s", msg)
            except OSError as exc:
                msg = f"PID {pid}: os.kill failed: {exc}"
                errors.append(msg)
                logger.warning("kill primitive: %s", msg)

        return {
            "pids_found": pids_found,
            "pids_killed": pids_killed,
            "signal": action.signal,
            "errors": errors,
        }
