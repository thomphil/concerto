"""Unit tests for :mod:`concerto_bench.primitives.snapshot`.

``httpx.MockTransport`` fakes concerto's ``/status`` endpoint, and the
three subprocess shell-outs (``nvidia-smi``, ``pgrep``) are mocked via
``unittest.mock.patch`` on ``asyncio.create_subprocess_exec``.

The fake subprocess object supports the minimal contract the primitive
relies on: ``communicate()`` and ``kill()`` plus a ``returncode`` that is
set once ``communicate`` has been awaited. This keeps the tests fast and
hermetic — no real ``nvidia-smi`` or ``pgrep`` is required on the host.
"""

from __future__ import annotations

import asyncio
import time
from datetime import timezone
from typing import Any, Optional
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from concerto_bench.primitives.snapshot import (
    SnapshotAction,
    SnapshotError,
    SnapshotPrimitive,
)
from concerto_bench.schema import StateSnapshot


# ---------------------------------------------------------------------------
# Test doubles for asyncio.create_subprocess_exec
# ---------------------------------------------------------------------------


class FakeSubprocess:
    """Minimal async test double mimicking the bits of ``asyncio.subprocess.Process`` we use."""

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        communicate_delay: float = 0.0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._returncode_on_exit = returncode
        self.returncode: Optional[int] = None
        self._communicate_delay = communicate_delay
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._communicate_delay > 0:
            await asyncio.sleep(self._communicate_delay)
        self.returncode = self._returncode_on_exit
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True
        # Mimic real behaviour: SIGKILL sets returncode to -9.
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._returncode_on_exit
        return self.returncode


class SubprocessRouter:
    """Stand-in for ``asyncio.create_subprocess_exec``.

    Routes the first argv element (the binary) to a caller-supplied
    factory so individual tests can stub ``nvidia-smi`` and ``pgrep``
    independently, including raising ``FileNotFoundError`` to simulate
    the binary not being on PATH.
    """

    def __init__(
        self,
        *,
        nvidia_smi_factory=None,
        pgrep_factory=None,
    ) -> None:
        self._nvidia = nvidia_smi_factory
        self._pgrep = pgrep_factory
        self.calls: list[tuple[str, ...]] = []

    async def __call__(self, *argv: str, **_kwargs: Any) -> FakeSubprocess:
        self.calls.append(argv)
        binary = argv[0]
        if binary == "nvidia-smi":
            if self._nvidia is None:
                raise FileNotFoundError("nvidia-smi not mocked")
            result = self._nvidia(argv)
            if isinstance(result, BaseException):
                raise result
            return result
        if binary == "pgrep":
            if self._pgrep is None:
                raise FileNotFoundError("pgrep not mocked")
            result = self._pgrep(argv)
            if isinstance(result, BaseException):
                raise result
            return result
        raise AssertionError(f"unexpected subprocess binary: {binary!r}")


BASE_URL = "http://127.0.0.1:18080"


def _status_body() -> dict[str, Any]:
    """A realistic concerto ``/status`` response body."""
    return {
        "gpus": [
            {
                "id": 0,
                "memory_total": "24000",
                "memory_used": "12000",
                "memory_available": "12000",
                "temperature_celsius": 55.0,
                "utilisation_percent": 42.0,
                "health": "Healthy",
                "loaded_models": [],
            }
        ],
        "registry_size": 1,
    }


def _make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# SnapshotAction validation
# ---------------------------------------------------------------------------


def test_action_defaults_include_sprint_2_patterns() -> None:
    action = SnapshotAction()
    assert "vllm" in action.pgrep_patterns
    assert "mock-inference-backend" in action.pgrep_patterns
    assert action.include_nvidia_smi is True
    assert action.include_pgrep is True


def test_action_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        SnapshotAction(bogus=True)


def test_action_timeout_non_positive_rejected() -> None:
    with pytest.raises(ValidationError):
        SnapshotAction(timeout_secs=0)


def test_action_round_trip_serialisation() -> None:
    original = SnapshotAction(
        include_nvidia_smi=False,
        include_pgrep=True,
        pgrep_patterns=["vllm", "custom-engine"],
        timeout_secs=3.5,
        capture_label="pre",
    )
    loaded = SnapshotAction.model_validate_json(original.model_dump_json())
    assert loaded == original


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_happy_path_populates_all_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/status"
        return httpx.Response(200, json=_status_body())

    nvidia_stdout = (
        b"0, NVIDIA RTX A4000, 16376, 12000, 4376, 42, 58\n"
        b"1, NVIDIA RTX A4000, 16376, 11800, 4576, 44, 60\n"
    )
    pgrep_outputs = {
        "vllm": b"1111 /opt/venv/bin/python -m vllm.entrypoints.openai.api_server\n",
        "python -m vllm": b"1111 /opt/venv/bin/python -m vllm.entrypoints.openai.api_server\n",  # dup
        "mock-inference-backend": b"2222 target/debug/mock-inference-backend --port 9001\n"
        b"3333 target/debug/mock-inference-backend --port 9002\n",
    }

    def nvidia_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        assert "--query-gpu=" in argv[1]
        return FakeSubprocess(stdout=nvidia_stdout, returncode=0)

    def pgrep_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        pattern = argv[-1]
        return FakeSubprocess(stdout=pgrep_outputs.get(pattern, b""), returncode=0)

    router = SubprocessRouter(
        nvidia_smi_factory=nvidia_factory,
        pgrep_factory=pgrep_factory,
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(),
                base_url=BASE_URL,
                client=client,
            )

    assert isinstance(snapshot, StateSnapshot)
    assert snapshot.ts.tzinfo is not None
    # tz-aware UTC: offset is zero.
    from datetime import timedelta as _td

    assert snapshot.ts.utcoffset() == _td(0)
    assert snapshot.concerto_status == _status_body()

    assert snapshot.nvidia_smi is not None
    gpus = snapshot.nvidia_smi["gpus"]
    assert len(gpus) == 2
    assert gpus[0]["index"] == 0
    assert gpus[0]["memory.total"] == 16376
    assert gpus[0]["memory.used"] == 12000
    assert gpus[0]["memory.free"] == 4376
    assert gpus[0]["utilization.gpu"] == pytest.approx(42.0)
    assert gpus[0]["temperature.gpu"] == pytest.approx(58.0)
    assert gpus[0]["name"] == "NVIDIA RTX A4000"

    # backend_pids: deduplicated union of {1111, 2222, 3333}, sorted.
    assert snapshot.backend_pids == [1111, 2222, 3333]

    assert snapshot.extra["pgrep_patterns"] == list(SnapshotAction().pgrep_patterns)
    assert "pgrep_command_lines" in snapshot.extra
    assert "vllm" in snapshot.extra["pgrep_command_lines"]


# ---------------------------------------------------------------------------
# /status failures — strict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_status_5xx_raises_snapshot_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    router = SubprocessRouter(
        nvidia_smi_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=0),
        pgrep_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=1),
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            with pytest.raises(SnapshotError):
                await primitive.execute(SnapshotAction(), base_url=BASE_URL, client=client)


@pytest.mark.asyncio
async def test_execute_status_connect_error_raises_snapshot_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    router = SubprocessRouter(
        nvidia_smi_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=0),
        pgrep_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=1),
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            with pytest.raises(SnapshotError):
                await primitive.execute(SnapshotAction(), base_url=BASE_URL, client=client)


@pytest.mark.asyncio
async def test_execute_status_timeout_raises_snapshot_error() -> None:
    async def slow_response(request: httpx.Request) -> httpx.Response:
        # httpx.MockTransport does not honour client timeout on its own;
        # raising httpx.ReadTimeout manually mirrors what a real slow
        # backend would produce.
        raise httpx.ReadTimeout("upstream too slow")

    router = SubprocessRouter(
        nvidia_smi_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=0),
        pgrep_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=1),
    )

    async with _make_client(slow_response) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            with pytest.raises(SnapshotError):
                await primitive.execute(
                    SnapshotAction(timeout_secs=0.5),
                    base_url=BASE_URL,
                    client=client,
                )


# ---------------------------------------------------------------------------
# nvidia-smi lenient failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_nvidia_smi_missing_binary_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    def nvidia_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        raise FileNotFoundError("nvidia-smi missing")

    router = SubprocessRouter(
        nvidia_smi_factory=nvidia_factory,
        pgrep_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=1),
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(),
                base_url=BASE_URL,
                client=client,
            )

    assert snapshot.nvidia_smi is None
    assert snapshot.concerto_status == _status_body()


@pytest.mark.asyncio
async def test_execute_nvidia_smi_nonzero_exit_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    def nvidia_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        return FakeSubprocess(stdout=b"", stderr=b"no device", returncode=2)

    router = SubprocessRouter(
        nvidia_smi_factory=nvidia_factory,
        pgrep_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=1),
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(),
                base_url=BASE_URL,
                client=client,
            )

    assert snapshot.nvidia_smi is None


@pytest.mark.asyncio
async def test_execute_nvidia_smi_timeout_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    def nvidia_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        return FakeSubprocess(stdout=b"", returncode=0, communicate_delay=5.0)

    router = SubprocessRouter(
        nvidia_smi_factory=nvidia_factory,
        pgrep_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=1),
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(timeout_secs=0.2),
                base_url=BASE_URL,
                client=client,
            )

    assert snapshot.nvidia_smi is None
    # concerto_status still populated — other sub-captures did not fail.
    assert snapshot.concerto_status == _status_body()


# ---------------------------------------------------------------------------
# pgrep lenient failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_pgrep_missing_binary_returns_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    def pgrep_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        raise FileNotFoundError("pgrep missing")

    router = SubprocessRouter(
        nvidia_smi_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=0),
        pgrep_factory=pgrep_factory,
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(),
                base_url=BASE_URL,
                client=client,
            )

    assert snapshot.backend_pids == []


@pytest.mark.asyncio
async def test_execute_pgrep_no_matches_is_not_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    def pgrep_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        # pgrep exits 1 when no process matches — a clean "nothing here"
        # signal, not a failure.
        return FakeSubprocess(stdout=b"", returncode=1)

    router = SubprocessRouter(
        nvidia_smi_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=0),
        pgrep_factory=pgrep_factory,
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(),
                base_url=BASE_URL,
                client=client,
            )

    assert snapshot.backend_pids == []


@pytest.mark.asyncio
async def test_execute_pgrep_multi_pattern_merging_and_dedup() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    pattern_outputs = {
        "a": b"1001 /usr/bin/fake-a\n1002 /usr/bin/fake-a\n",
        "b": b"1002 /usr/bin/fake-b\n1003 /usr/bin/fake-b\n",
        "c": b"1003 /usr/bin/fake-c\n1004 /usr/bin/fake-c\n",
    }

    def pgrep_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        pattern = argv[-1]
        output = pattern_outputs.get(pattern, b"")
        rc = 0 if output else 1
        return FakeSubprocess(stdout=output, returncode=rc)

    router = SubprocessRouter(
        nvidia_smi_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=0),
        pgrep_factory=pgrep_factory,
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(pgrep_patterns=["a", "b", "c"]),
                base_url=BASE_URL,
                client=client,
            )

    assert snapshot.backend_pids == [1001, 1002, 1003, 1004]


# ---------------------------------------------------------------------------
# Include toggles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_nvidia_smi_disabled_not_invoked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    def nvidia_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        raise AssertionError("nvidia-smi should not have been invoked")

    router = SubprocessRouter(
        nvidia_smi_factory=nvidia_factory,
        pgrep_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=1),
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(include_nvidia_smi=False),
                base_url=BASE_URL,
                client=client,
            )

    assert snapshot.nvidia_smi is None
    # No nvidia-smi call landed on the router.
    assert all(call[0] != "nvidia-smi" for call in router.calls)


@pytest.mark.asyncio
async def test_execute_pgrep_disabled_not_invoked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    def pgrep_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        raise AssertionError("pgrep should not have been invoked")

    router = SubprocessRouter(
        nvidia_smi_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=0),
        pgrep_factory=pgrep_factory,
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(include_pgrep=False),
                base_url=BASE_URL,
                client=client,
            )

    assert snapshot.backend_pids == []
    assert all(call[0] != "pgrep" for call in router.calls)


@pytest.mark.asyncio
async def test_execute_empty_pgrep_patterns_short_circuits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_status_body())

    def pgrep_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        raise AssertionError("pgrep should not have been invoked")

    router = SubprocessRouter(
        nvidia_smi_factory=lambda argv: FakeSubprocess(stdout=b"", returncode=0),
        pgrep_factory=pgrep_factory,
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            snapshot = await primitive.execute(
                SnapshotAction(pgrep_patterns=[]),
                base_url=BASE_URL,
                client=client,
            )

    assert snapshot.backend_pids == []


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_runs_sub_captures_in_parallel() -> None:
    """The three sub-captures should run concurrently.

    Each stub sleeps 200 ms. If they ran sequentially we would expect
    ~600 ms (one pgrep pattern runs at a time); in parallel the
    wall-clock should be closer to 200 ms. We allow generous headroom
    (<450 ms) so the assertion is not flaky on a loaded CI runner.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, json=_status_body())

    def nvidia_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        return FakeSubprocess(stdout=b"", returncode=0, communicate_delay=0.2)

    def pgrep_factory(argv: tuple[str, ...]) -> FakeSubprocess:
        return FakeSubprocess(stdout=b"", returncode=1, communicate_delay=0.2)

    router = SubprocessRouter(
        nvidia_smi_factory=nvidia_factory,
        pgrep_factory=pgrep_factory,
    )

    async with _make_client(handler) as client:
        primitive = SnapshotPrimitive()
        with patch("concerto_bench.primitives.snapshot.asyncio.create_subprocess_exec", router):
            start = time.perf_counter()
            # Single pgrep pattern so we are measuring the three-way
            # parallelism between /status, nvidia-smi, and pgrep.
            await primitive.execute(
                SnapshotAction(pgrep_patterns=["single"]),
                base_url=BASE_URL,
                client=client,
            )
            elapsed = time.perf_counter() - start

    # Three 200 ms tasks in parallel → ~200 ms wall-clock, not ~600 ms.
    assert elapsed < 0.45, f"snapshot took {elapsed:.3f}s, expected ~0.2s"


# ---------------------------------------------------------------------------
# CSV parsing sanity
# ---------------------------------------------------------------------------


def test_nvidia_smi_csv_parser_types() -> None:
    """Coercion must yield ints for memory columns and floats for util/temp."""
    primitive = SnapshotPrimitive()
    text = (
        "0, NVIDIA RTX A4000, 16376, 12000, 4376, 42, 58\n"
        "1, NVIDIA RTX A4000, 16376, 500, 15876, 3, 45\n"
    )
    rows = primitive._parse_nvidia_smi_csv(text)
    assert len(rows) == 2
    assert isinstance(rows[0]["index"], int)
    assert isinstance(rows[0]["memory.total"], int)
    assert isinstance(rows[0]["memory.used"], int)
    assert isinstance(rows[0]["memory.free"], int)
    assert isinstance(rows[0]["utilization.gpu"], float)
    assert isinstance(rows[0]["temperature.gpu"], float)
    assert rows[1]["memory.used"] == 500
    assert rows[1]["utilization.gpu"] == pytest.approx(3.0)
