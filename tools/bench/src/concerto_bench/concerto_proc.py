"""Child-process lifecycle management for the ``concerto`` binary.

The bench rig treats concerto strictly as an external process — it is
the production binary, spawned and talked to only over its public HTTP
surface. This module owns that lifecycle from the rig's point of view:

1. Build the argv from a validated :class:`ProcessSpec`.
2. Spawn the binary via :func:`asyncio.create_subprocess_exec` with
   stdout/stderr piped, so the rig can tee them into the artifact log
   directory.
3. Health-gate readiness by polling ``GET /health`` until it returns
   2xx or a timeout elapses.
4. On context-manager exit, reap the child with SIGTERM → grace period
   → SIGKILL escalation, flushing the captured output streams.

The module is deliberately small and side-effect-free at import time.
It does not configure logging, does not install signal handlers, and
does not touch global state.

Typical usage::

    async with ConcertoProcess.start(
        spec=ProcessSpec(
            binary=Path("./target/debug/concerto"),
            config_path=Path("concerto.example.toml"),
            mock_gpus=2,
            port=pick_free_port(),
            log_dir=tmp_path,
        )
    ) as proc:
        # proc.base_url, proc.pid, proc.log_dir are populated
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{proc.base_url}/health")
            assert r.status_code == 200
    # child is reaped, stdout.log / stderr.log are sealed
"""

from __future__ import annotations

import asyncio
import logging
import signal
import socket
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

LogFormat = Literal["pretty", "json"]
LogLevel = Literal["debug", "info", "warn", "error"]


class ConcertoStartupError(RuntimeError):
    """Raised when the concerto child failed to become healthy.

    Carries the exit code (if the process died before we reached the
    deadline) and a best-effort snapshot of captured stderr / stdout so
    callers can log a useful diagnostic before re-raising.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: Optional[int] = None,
        stderr_tail: str = "",
        stdout_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        self.stdout_tail = stdout_tail

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        parts = [super().__str__()]
        if self.returncode is not None:
            parts.append(f"(returncode={self.returncode})")
        if self.stderr_tail:
            parts.append(f"stderr tail:\n{self.stderr_tail}")
        if self.stdout_tail:
            parts.append(f"stdout tail:\n{self.stdout_tail}")
        return "\n".join(parts)


class ProcessSpec(BaseModel):
    """Strict configuration for spawning a concerto child process.

    Every field that ends up on the argv is captured here so a
    ``ProcessSpec`` can be round-tripped into the artifact for
    reproducibility. Frozen because the spec is an input to
    :meth:`ConcertoProcess.start` and should not drift during a run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    binary: Path = Field(
        ...,
        description="Absolute or relative path to the built concerto binary.",
    )
    config_path: Path = Field(
        ...,
        description=(
            "Path to the concerto.toml config file passed via --config. "
            "Required even in mock-gpus mode: concerto's CLI rejects "
            "missing --config regardless of GPU backend."
        ),
    )
    port: int = Field(
        ...,
        ge=1,
        le=65535,
        description="TCP port concerto should bind to (passed via --port-override).",
    )
    log_dir: Path = Field(
        ...,
        description=(
            "Directory into which the rig tees concerto's stdout and "
            "stderr. Must already exist or be creatable by the caller."
        ),
    )
    mock_gpus: Optional[int] = Field(
        default=None,
        ge=1,
        le=8,
        description=(
            "If set, passes --mock-gpus N to concerto, which rewrites "
            "the loaded config to use the bundled mock-inference-backend "
            "instead of real NVML / vLLM. Leave unset on real hardware."
        ),
    )
    log_level: LogLevel = Field(
        default="info",
        description="Value passed to concerto's --log-level flag.",
    )
    log_format: LogFormat = Field(
        default="pretty",
        description=(
            "Value passed to concerto's --log-format flag. The rig runs "
            "use 'json' so downstream tooling can parse structured logs; "
            "'pretty' is the default here to keep unit tests readable."
        ),
    )
    models_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Optional models directory. Concerto itself has no CLI flag "
            "for this today — weights paths live in the config file — so "
            "the value is exported as CONCERTO_MODELS_DIR in the child "
            "environment for scenario templates to pick up. Stored here "
            "purely for provenance in the artifact."
        ),
    )
    startup_timeout_secs: float = Field(
        default=30.0,
        gt=0.0,
        description="Max wall time the health gate will wait for /health to return 2xx.",
    )
    health_poll_interval_secs: float = Field(
        default=0.1,
        gt=0.0,
        description="How often the health gate re-polls GET /health while waiting.",
    )
    shutdown_grace_secs: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "How long to wait for concerto to exit after SIGTERM before "
            "escalating to SIGKILL."
        ),
    )
    host: str = Field(
        default="127.0.0.1",
        description="Loopback host used for the health URL. The child still binds per its config.",
    )


@dataclass(frozen=True)
class ConcertoProcessInfo:
    """Snapshot of the spawned child's identity.

    Exposed via :attr:`ConcertoProcess.info` after :meth:`start` succeeds.
    Purposely a frozen dataclass rather than a pydantic model to keep
    ``Path`` / ``int`` round-tripping unambiguous and to avoid revalidation
    overhead on every attribute read.
    """

    pid: int
    port: int
    base_url: str
    stdout_log: Path
    stderr_log: Path


def pick_free_port(host: str = "127.0.0.1") -> int:
    """Return a loopback TCP port that is currently free.

    Uses the "bind to port 0, read the OS-assigned port back out, close
    the socket" idiom. There is an inherent TOCTOU between this call and
    the child actually binding — the gap is small enough on loopback
    that the rig accepts the race, but callers running very parallel
    bench harnesses may prefer to pass explicit ports.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class ConcertoProcess:
    """Async context manager wrapping a single concerto child process.

    Construct via :meth:`start`, which spawns the binary, health-gates
    ``/health``, and returns a live instance. The instance is usable as
    an ``async with`` target; on exit (normal or exceptional) the child
    is reaped and log files are sealed.

    Instances are not reusable — once ``__aexit__`` has run, a new
    :meth:`start` call is required to get a fresh process.
    """

    def __init__(self, spec: ProcessSpec) -> None:
        self._spec = spec
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stdout_file: Optional[object] = None  # typing.IO open in text mode
        self._stderr_file: Optional[object] = None
        self._stdout_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._info: Optional[ConcertoProcessInfo] = None
        self._closed = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @classmethod
    async def start(cls, spec: ProcessSpec) -> "ConcertoProcess":
        """Spawn concerto, wait for ``/health``, and return the live handle.

        Raises :class:`ConcertoStartupError` if the child dies before
        the health gate succeeds, or if the health gate times out. On
        failure the child is killed before the exception propagates;
        no orphans are left behind.
        """
        self = cls(spec)
        try:
            await self._spawn()
            await self._wait_ready(timeout_secs=spec.startup_timeout_secs)
        except BaseException:
            # Clean up partial state — we must not leak a running child
            # or open file handles on failure.
            await self._reap(reason="startup-failure")
            raise
        return self

    async def __aenter__(self) -> "ConcertoProcess":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self._reap(reason="context-exit")
        # Let the original exception (if any) propagate.

    @property
    def info(self) -> ConcertoProcessInfo:
        """Process identity. Only valid between ``start()`` and ``__aexit__``."""
        if self._info is None:
            raise RuntimeError(
                "ConcertoProcess.info accessed before start() completed or after shutdown"
            )
        return self._info

    @property
    def pid(self) -> int:
        return self.info.pid

    @property
    def port(self) -> int:
        return self.info.port

    @property
    def base_url(self) -> str:
        return self.info.base_url

    @property
    def stdout_log(self) -> Path:
        return self.info.stdout_log

    @property
    def stderr_log(self) -> Path:
        return self.info.stderr_log

    def is_running(self) -> bool:
        """Return True if the child is still alive (returncode not yet set)."""
        return self._proc is not None and self._proc.returncode is None

    async def wait_ready(
        self, *, timeout_secs: Optional[float] = None
    ) -> None:
        """Re-poll ``/health`` until it returns 2xx or a timeout elapses.

        Intended for callers that performed a disruptive action mid-test
        (backend restart, SIGKILL of an engine worker) and want to block
        until concerto reports healthy again. Uses the same polling
        logic as the initial startup gate.
        """
        deadline = timeout_secs if timeout_secs is not None else self._spec.startup_timeout_secs
        await self._wait_ready(timeout_secs=deadline)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _argv(self) -> list[str]:
        spec = self._spec
        argv: list[str] = [
            str(spec.binary),
            "--config",
            str(spec.config_path),
            "--log-level",
            spec.log_level,
            "--log-format",
            spec.log_format,
            "--port-override",
            str(spec.port),
        ]
        if spec.mock_gpus is not None:
            argv += ["--mock-gpus", str(spec.mock_gpus)]
        return argv

    async def _spawn(self) -> None:
        spec = self._spec
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = spec.log_dir / "concerto-stdout.log"
        stderr_log = spec.log_dir / "concerto-stderr.log"

        # Text-mode files so the background reader tasks can write strings
        # decoded from the subprocess byte streams. ``utf-8`` + ``replace``
        # keeps malformed log lines from crashing the rig.
        self._stdout_file = open(stdout_log, "w", encoding="utf-8", errors="replace")
        self._stderr_file = open(stderr_log, "w", encoding="utf-8", errors="replace")

        argv = self._argv()
        logger.info(
            "spawning concerto child: %s (port=%d, log_dir=%s)",
            " ".join(argv),
            spec.port,
            spec.log_dir,
        )

        # Propagate the caller's environment and layer on extras. We deliberately
        # do not use shell=True — argv is already fully resolved.
        import os

        env = os.environ.copy()
        if spec.models_dir is not None:
            env["CONCERTO_MODELS_DIR"] = str(spec.models_dir)

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Start the tee tasks before the health gate — otherwise a chatty
        # concerto could fill and block on its pipe buffers while we're
        # still polling /health, and the health gate would time out
        # spuriously.
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._stdout_task = asyncio.create_task(
            _drain_stream(self._proc.stdout, self._stdout_file),
            name="concerto-stdout-drain",
        )
        self._stderr_task = asyncio.create_task(
            _drain_stream(self._proc.stderr, self._stderr_file),
            name="concerto-stderr-drain",
        )

        self._info = ConcertoProcessInfo(
            pid=self._proc.pid,
            port=spec.port,
            base_url=f"http://{spec.host}:{spec.port}",
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

    async def _wait_ready(self, *, timeout_secs: float) -> None:
        """Poll GET /health on a short interval until 2xx or timeout.

        Aborts early (raising :class:`ConcertoStartupError`) if the
        child process exits before we see a successful health response.
        """
        assert self._info is not None
        assert self._proc is not None

        url = f"{self._info.base_url}/health"
        interval = self._spec.health_poll_interval_secs

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_secs
        last_error: Optional[str] = None

        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            while True:
                # If concerto died, no amount of polling will help.
                if self._proc.returncode is not None:
                    tails = await self._collect_log_tails()
                    raise ConcertoStartupError(
                        f"concerto child exited before /health succeeded"
                        f" (last error: {last_error or 'none'})",
                        returncode=self._proc.returncode,
                        **tails,
                    )

                try:
                    resp = await client.get(url)
                    if 200 <= resp.status_code < 300:
                        logger.info(
                            "concerto /health ready (pid=%d, port=%d, status=%d)",
                            self._info.pid,
                            self._info.port,
                            resp.status_code,
                        )
                        return
                    last_error = f"HTTP {resp.status_code}"
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"

                remaining = deadline - loop.time()
                if remaining <= 0:
                    tails = await self._collect_log_tails()
                    raise ConcertoStartupError(
                        f"concerto /health did not return 2xx within "
                        f"{timeout_secs:.2f}s (last error: {last_error or 'none'})",
                        returncode=self._proc.returncode,
                        **tails,
                    )
                await asyncio.sleep(min(interval, remaining))

    async def _collect_log_tails(self, *, max_bytes: int = 4096) -> dict[str, str]:
        """Best-effort snapshot of the last few KB of each log file.

        Used to populate :class:`ConcertoStartupError` fields so a
        failed health gate surfaces something useful to the caller's
        traceback without them having to open the log files by hand.
        """
        info = self._info
        if info is None:
            return {"stderr_tail": "", "stdout_tail": ""}

        # Flush in-flight writes from the drain tasks before we read.
        for f in (self._stdout_file, self._stderr_file):
            if f is not None:
                with suppress(Exception):
                    f.flush()  # type: ignore[attr-defined]

        def _tail(path: Path) -> str:
            try:
                with open(path, "rb") as fh:
                    fh.seek(0, 2)
                    size = fh.tell()
                    fh.seek(max(0, size - max_bytes))
                    return fh.read().decode("utf-8", errors="replace")
            except OSError:
                return ""

        return {
            "stdout_tail": _tail(info.stdout_log),
            "stderr_tail": _tail(info.stderr_log),
        }

    async def _reap(self, *, reason: str) -> None:
        """Shut the child down cleanly.

        Idempotent: safe to call multiple times. The sequence is
        SIGTERM → wait(grace) → SIGKILL → wait → cancel drain tasks →
        close log files. Exceptions from individual steps are swallowed
        so a later step still runs; the goal is to leave no orphans.
        """
        if self._closed:
            return
        self._closed = True

        proc = self._proc
        if proc is not None and proc.returncode is None:
            logger.info(
                "reaping concerto child (pid=%s, reason=%s)",
                proc.pid,
                reason,
            )
            with suppress(ProcessLookupError):
                proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(
                    proc.wait(), timeout=self._spec.shutdown_grace_secs
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "concerto child pid=%s did not exit within %.1fs of SIGTERM; sending SIGKILL",
                    proc.pid,
                    self._spec.shutdown_grace_secs,
                )
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=5.0)

        # Drain tasks hit EOF once the pipes close on the child exiting.
        # If they haven't finished yet (shouldn't happen, but defensively)
        # cancel them so we don't hang the event loop on shutdown.
        for task in (self._stdout_task, self._stderr_task):
            if task is None:
                continue
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

        for f in (self._stdout_file, self._stderr_file):
            if f is not None:
                with suppress(Exception):
                    f.flush()  # type: ignore[attr-defined]
                    f.close()  # type: ignore[attr-defined]


async def _drain_stream(
    stream: asyncio.StreamReader,
    sink: object,
) -> None:
    """Copy every line from ``stream`` into the file-like ``sink``.

    Runs until EOF. Exceptions are logged and swallowed so a single
    malformed log line never takes the whole rig down. The function
    is module-private because its contract — in particular the line-
    oriented decoding and error recovery — is only meaningful in the
    context of :class:`ConcertoProcess`.
    """
    try:
        while True:
            chunk = await stream.readline()
            if not chunk:
                return
            try:
                sink.write(chunk.decode("utf-8", errors="replace"))  # type: ignore[attr-defined]
                sink.flush()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - extremely defensive
                logger.warning("failed to write concerto log chunk: %s", exc)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("concerto log drain task crashed: %s", exc)
