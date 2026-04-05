"""Tests for :mod:`concerto_bench.concerto_proc`.

Split into two groups:

* Pure-Python unit tests that do not require a built concerto binary.
  These run anywhere, including in CI before any Rust compilation.
* Integration tests that spawn a real ``concerto --mock-gpus 2`` child.
  These are skipped gracefully when the binary has not been built,
  rather than trying to build it inside the test suite.

The integration tests assert that:

1. After :meth:`ConcertoProcess.start`, ``/health`` returns 200, the PID
   is alive, and ``__aexit__`` leaves no orphans behind.
2. An absurdly short startup timeout raises
   :class:`ConcertoStartupError` and still reaps the child cleanly.
3. The stdout / stderr log files exist after exit and capture at least
   one of concerto's startup log lines (confirming the drain tasks
   actually ran).
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
from contextlib import suppress
from pathlib import Path
from typing import Optional

import httpx
import pytest
from pydantic import ValidationError

from concerto_bench.concerto_proc import (
    ConcertoProcess,
    ConcertoStartupError,
    ProcessSpec,
    pick_free_port,
)


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Walk up from this test file to the concerto repo root.

    Tests live at ``<repo>/tools/bench/tests/test_concerto_proc.py``, so
    the repo root is three ``parent`` hops away. Using an explicit walk
    (rather than e.g. ``git rev-parse``) keeps the test self-contained
    and avoids shelling out in pytest startup.
    """
    return Path(__file__).resolve().parents[3]


def _find_concerto_binary() -> Optional[Path]:
    """Locate a pre-built ``concerto`` binary under ``target/``.

    Prefers the debug build (what ``cargo build -p concerto-cli``
    produces by default) and falls back to the release build. Returns
    ``None`` if neither exists — tests will skip gracefully in that case.
    """
    root = _repo_root()
    for candidate in (root / "target" / "debug" / "concerto", root / "target" / "release" / "concerto"):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _example_config() -> Path:
    return _repo_root() / "concerto.example.toml"


_CONCERTO_BINARY = _find_concerto_binary()
_REQUIRES_BINARY = pytest.mark.skipif(
    _CONCERTO_BINARY is None,
    reason=(
        "concerto binary not built; run `cargo build -p concerto-cli` "
        "from the repo root to enable these tests."
    ),
)


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` still exists.

    Uses the standard ``kill(pid, 0)`` probe rather than shelling out to
    ``ps`` or ``pgrep`` — faster and with cleaner semantics across
    platforms.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it. Won't happen for a child
        # we spawned ourselves, but treat as "alive" to be safe.
        return True
    return True


# ---------------------------------------------------------------------------
# Pure unit tests (no binary needed)
# ---------------------------------------------------------------------------


def test_pick_free_port_returns_bindable_port() -> None:
    port = pick_free_port()
    assert 1 <= port <= 65535
    # The port should be immediately rebindable — this is the contract
    # pick_free_port offers callers that then hand the port to a child.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_process_spec_rejects_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ProcessSpec(  # type: ignore[call-arg]
            binary=tmp_path / "concerto",
            config_path=tmp_path / "concerto.toml",
            port=8080,
            log_dir=tmp_path,
            bogus_field="nope",
        )


def test_process_spec_validates_port_range(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ProcessSpec(
            binary=tmp_path / "concerto",
            config_path=tmp_path / "concerto.toml",
            port=0,  # below ge=1
            log_dir=tmp_path,
        )
    with pytest.raises(ValidationError):
        ProcessSpec(
            binary=tmp_path / "concerto",
            config_path=tmp_path / "concerto.toml",
            port=70000,  # above le=65535
            log_dir=tmp_path,
        )


def test_process_spec_mock_gpus_bounds(tmp_path: Path) -> None:
    # 0 rejected (ge=1), 9 rejected (le=8), 1-8 accepted.
    with pytest.raises(ValidationError):
        ProcessSpec(
            binary=tmp_path / "concerto",
            config_path=tmp_path / "concerto.toml",
            port=8080,
            log_dir=tmp_path,
            mock_gpus=0,
        )
    with pytest.raises(ValidationError):
        ProcessSpec(
            binary=tmp_path / "concerto",
            config_path=tmp_path / "concerto.toml",
            port=8080,
            log_dir=tmp_path,
            mock_gpus=9,
        )
    spec = ProcessSpec(
        binary=tmp_path / "concerto",
        config_path=tmp_path / "concerto.toml",
        port=8080,
        log_dir=tmp_path,
        mock_gpus=2,
    )
    assert spec.mock_gpus == 2


def test_process_spec_is_frozen(tmp_path: Path) -> None:
    spec = ProcessSpec(
        binary=tmp_path / "concerto",
        config_path=tmp_path / "concerto.toml",
        port=8080,
        log_dir=tmp_path,
    )
    with pytest.raises(ValidationError):
        spec.port = 9090  # type: ignore[misc]


def test_info_raises_before_start(tmp_path: Path) -> None:
    spec = ProcessSpec(
        binary=tmp_path / "concerto",
        config_path=tmp_path / "concerto.toml",
        port=8080,
        log_dir=tmp_path,
    )
    proc = ConcertoProcess(spec)
    with pytest.raises(RuntimeError):
        _ = proc.info


def test_concerto_startup_error_carries_context() -> None:
    err = ConcertoStartupError(
        "boom",
        returncode=42,
        stderr_tail="panic!",
        stdout_tail="starting...",
    )
    assert err.returncode == 42
    assert err.stderr_tail == "panic!"
    assert err.stdout_tail == "starting..."
    assert "boom" in str(err)


# ---------------------------------------------------------------------------
# Integration tests (require built binary)
# ---------------------------------------------------------------------------


@_REQUIRES_BINARY
async def test_spawn_health_gate_and_clean_shutdown(tmp_path: Path) -> None:
    """Happy path: spawn, /health returns 200, reap cleanly, no orphans."""
    assert _CONCERTO_BINARY is not None  # for type checkers; covered by skip
    port = pick_free_port()
    spec = ProcessSpec(
        binary=_CONCERTO_BINARY,
        config_path=_example_config(),
        port=port,
        log_dir=tmp_path,
        mock_gpus=2,
        startup_timeout_secs=20.0,
    )

    async with await ConcertoProcess.start(spec) as proc:
        assert proc.pid > 0
        assert _pid_is_alive(proc.pid)
        assert proc.port == port
        assert proc.base_url == f"http://127.0.0.1:{port}"
        assert proc.is_running()

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{proc.base_url}/health")
            assert resp.status_code == 200

        # Exercise wait_ready() separately — must not regress once the
        # gate has already opened.
        await proc.wait_ready(timeout_secs=5.0)

        exited_pid = proc.pid

    # Give the kernel a tick to fully reap.
    await asyncio.sleep(0.1)
    assert not _pid_is_alive(exited_pid), "concerto child should be gone after __aexit__"


@_REQUIRES_BINARY
async def test_health_gate_timeout_raises_and_reaps(tmp_path: Path) -> None:
    """An absurdly short timeout raises ConcertoStartupError, not an orphan."""
    assert _CONCERTO_BINARY is not None
    port = pick_free_port()
    spec = ProcessSpec(
        binary=_CONCERTO_BINARY,
        config_path=_example_config(),
        port=port,
        log_dir=tmp_path,
        mock_gpus=2,
        startup_timeout_secs=0.001,
        health_poll_interval_secs=0.001,
    )

    with pytest.raises(ConcertoStartupError):
        await ConcertoProcess.start(spec)

    await asyncio.sleep(0.2)
    # Nothing should be bound on the port anymore — the child is gone.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # If a leaked child still held the port, this bind would fail.
        sock.bind(("127.0.0.1", port))


@_REQUIRES_BINARY
async def test_stdout_and_stderr_logs_captured(tmp_path: Path) -> None:
    """After a successful run, log files exist and stdout is non-empty.

    Concerto's tracing subscriber writes to stdout (both pretty and JSON
    format), so stdout is the stream we expect to see content on. We
    still assert stderr exists as a file — captured but potentially
    empty — because the rig's artifact expects both files regardless.
    """
    assert _CONCERTO_BINARY is not None
    port = pick_free_port()
    spec = ProcessSpec(
        binary=_CONCERTO_BINARY,
        config_path=_example_config(),
        port=port,
        log_dir=tmp_path,
        mock_gpus=2,
        startup_timeout_secs=20.0,
        log_format="json",
    )

    async with await ConcertoProcess.start(spec) as proc:
        stdout_log = proc.stdout_log
        stderr_log = proc.stderr_log
        assert stdout_log.parent == tmp_path
        # /health hit to make sure there's some log traffic to capture.
        async with httpx.AsyncClient() as client:
            await client.get(f"{proc.base_url}/health")

    assert stdout_log.exists(), "stdout log file should exist after context exit"
    assert stderr_log.exists(), "stderr log file should exist after context exit"
    assert stdout_log.stat().st_size > 0, "concerto should have logged startup lines to stdout"
    # stderr may legitimately be empty — concerto logs to stdout by
    # default. Just asserting the file was created is the contract.


@_REQUIRES_BINARY
async def test_reap_escalates_on_unresponsive_child(tmp_path: Path) -> None:
    """If SIGTERM is ignored, the reaper escalates to SIGKILL in time.

    We simulate SIGTERM-deafness by overriding shutdown_grace_secs to a
    very small value and stopping the child's ability to react via an
    external SIGSTOP. This isn't perfect — a truly stopped process
    doesn't exit on SIGKILL either until resumed — so the test's real
    purpose is to ensure the escalation code path is exercised at all.

    We resume the child at the end to let the reap sequence finish.
    """
    assert _CONCERTO_BINARY is not None
    port = pick_free_port()
    spec = ProcessSpec(
        binary=_CONCERTO_BINARY,
        config_path=_example_config(),
        port=port,
        log_dir=tmp_path,
        mock_gpus=2,
        startup_timeout_secs=20.0,
        shutdown_grace_secs=0.2,
    )

    proc = await ConcertoProcess.start(spec)
    pid = proc.pid
    try:
        # Freeze the child so it cannot react to SIGTERM. On Linux this
        # would mean SIGKILL has to wait for SIGCONT; on Darwin SIGKILL
        # is delivered immediately even while stopped. Either way, the
        # code path we care about — the "SIGTERM grace window expired,
        # escalating to SIGKILL" branch — is exercised.
        os.kill(pid, signal.SIGSTOP)
        reap_task = asyncio.create_task(proc.__aexit__(None, None, None))
        # Give the reaper time to send SIGTERM, observe the timeout,
        # and follow up with SIGKILL.
        await asyncio.sleep(0.5)
        # Resume the child if it is somehow still around (pending
        # SIGKILL on platforms that honour SIGSTOP strictly). On Darwin
        # the process is already gone — ignore ProcessLookupError.
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGCONT)
        await reap_task
    finally:
        # Defensive: ensure nothing is left behind even if the test
        # crashed mid-way.
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGCONT)
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)

    await asyncio.sleep(0.1)
    assert not _pid_is_alive(pid), "escalation path should leave no orphan"
