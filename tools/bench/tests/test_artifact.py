"""Unit tests for the artifact builder + tree verifier.

These tests exercise :mod:`concerto_bench.artifact` end-to-end on
fabricated schema data. They are the contract the runner (step 7) and
the CI dry-run e2e test (step 13) will lean on: every invariant the
builder advertises — strict tree shape, schema validation, reproducible
tarballs, sha256 sidecar format, single-shot finalisation — is pinned
here so a later refactor can't silently break the artifact format.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from concerto_bench.artifact import (
    ArtifactBuilder,
    ArtifactError,
    FinalizedArtifact,
    verify_artifact_tree,
)
from concerto_bench.schema import (
    ActionRecord,
    HostInfo,
    ManifestV1,
    RequestRecord,
    StateSnapshot,
    StepResult,
    SummaryV1,
)

UTC = timezone.utc
T0 = datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(milliseconds=523)
T2 = T0 + timedelta(seconds=30)


# ---------------------------------------------------------------------------
# Fabrication helpers
# ---------------------------------------------------------------------------


def _manifest(step_count: int = 3) -> ManifestV1:
    return ManifestV1(
        rig_version="0.1.0.dev0",
        concerto_version="concerto 0.1.0",
        concerto_git_sha="abcdef1",
        scenario_name="unit-test",
        scenario_version="1",
        started_at=T0,
        ended_at=T2,
        duration_seconds=30.0,
        exit_status="success",
        step_count=step_count,
    )


def _summary(step_count: int = 3, steps_passed: int = 3) -> SummaryV1:
    return SummaryV1(
        scenario_name="unit-test",
        scenario_version="1",
        exit_status="success",
        scenario_passed=steps_passed == step_count,
        step_count=step_count,
        steps_passed=steps_passed,
        steps_failed=step_count - steps_passed,
    )


def _host_info() -> HostInfo:
    return HostInfo(
        concerto_version="concerto 0.1.0",
        python_version="3.11.9",
        uname={"sysname": "Linux"},
        captured_at=T0,
    )


def _snapshot(ts: datetime = T0) -> StateSnapshot:
    return StateSnapshot(
        ts=ts,
        concerto_status={"loaded_models": []},
        backend_pids=[],
    )


def _request_record() -> RequestRecord:
    return RequestRecord(
        status=200,
        elapsed_total_ms=12.5,
        elapsed_ttfb_ms=5.0,
        elapsed_connect_ms=1.0,
        request_body={"model": "qwen2.5-0.5b", "content": "hi"},
        response_body={"choices": [{"text": "hello"}]},
    )


def _action_record(
    *,
    capture_as: str | None = None,
    started_at: datetime = T0,
) -> ActionRecord:
    args: dict[str, Any] = {"model": "qwen2.5-0.5b", "content": "hi"}
    if capture_as is not None:
        args["capture_as"] = capture_as
    return ActionRecord(
        action_type="request",
        args=args,
        started_at=started_at,
        ended_at=started_at + timedelta(milliseconds=500),
        duration_ms=500.0,
        passed=True,
    )


def _step_result(
    *,
    number: int,
    name: str,
    capture_as: str | None = "greet",
) -> StepResult:
    slug = name.lower().replace(" ", "-").replace("/", "-")
    return StepResult(
        step_number=number,
        step_name=name,
        passed=True,
        duration_ms=1000.0,
        started_at=T0 + timedelta(seconds=number),
        ended_at=T0 + timedelta(seconds=number + 1),
        pre_state_path=f"steps/{number:02d}-{slug}/pre-state.json",
        post_state_path=f"steps/{number:02d}-{slug}/post-state.json",
        actions=[
            _action_record(),
            _action_record(capture_as=capture_as) if capture_as else _action_record(),
        ],
    )


def _build_happy_path(
    root: Path,
    *,
    step_count: int = 3,
    capture_names: list[str] | None = None,
) -> ArtifactBuilder:
    """Drive the full builder lifecycle against the given root.

    Returns the builder **unfinalised** so individual tests can assert
    pre-tarball state or finalise themselves.
    """
    capture_names = capture_names or ["greet"] * step_count
    builder = ArtifactBuilder(root)
    builder.write_manifest(_manifest(step_count=step_count))
    builder.write_summary(_summary(step_count=step_count, steps_passed=step_count))
    builder.write_host_info(_host_info())
    builder.write_concerto_logs(b"stdout line\n", b"stderr line\n")

    for i in range(1, step_count + 1):
        capture = capture_names[i - 1]
        step = _step_result(
            number=i,
            name=f"step-{i}",
            capture_as=capture,
        )
        builder.write_step(
            step,
            pre_state=_snapshot(ts=T0 + timedelta(seconds=i * 2)),
            post_state=_snapshot(ts=T0 + timedelta(seconds=i * 2 + 1)),
            request_records={capture: _request_record()} if capture else None,
        )

    # Register one telemetry file per canonical sampler.
    for sampler in ("nvidia-smi", "concerto-status", "concerto-metrics", "pgrep-count", "proc-stats"):
        jsonl = builder.telemetry_dir() / f"{sampler}.jsonl"
        jsonl.write_text(
            json.dumps({"schema_version": 1, "ts": T0.isoformat(), "sampler": sampler, "values": {}})
            + "\n",
            encoding="utf-8",
        )
    return builder


# ---------------------------------------------------------------------------
# 1. Happy-path build + verify + sha256sum
# ---------------------------------------------------------------------------


def test_happy_path_build_produces_expected_tree(tmp_path: Path) -> None:
    """Every canonical file lands at the right path and the verifier is clean."""
    root = tmp_path / "run-happy"
    builder = _build_happy_path(root)
    finalized = builder.finalize()

    # Top-level files
    for name in ("manifest.json", "summary.json", "environment.json", "concerto-stdout.log", "concerto-stderr.log"):
        assert (root / name).is_file(), f"missing {name}"

    # Steps
    step_dirs = sorted((root / "steps").iterdir())
    assert [p.name for p in step_dirs] == ["01-step-1", "02-step-2", "03-step-3"]
    for step_dir in step_dirs:
        for fname in ("pre-state.json", "post-state.json", "result.json", "request-greet.json"):
            assert (step_dir / fname).is_file()

    # Telemetry
    telemetry = root / "telemetry"
    assert telemetry.is_dir()
    assert len(list(telemetry.glob("*.jsonl"))) == 5

    # Verifier agrees
    assert verify_artifact_tree(root) == []

    # Tarball + sha256 sidecar placed next to the root
    assert finalized.tarball_path == tmp_path / "run-happy.tar.gz"
    assert finalized.sha256_path == tmp_path / "run-happy.tar.gz.sha256"
    assert finalized.tarball_path.is_file()
    assert finalized.sha256_path.is_file()
    assert finalized.size_bytes == finalized.tarball_path.stat().st_size
    assert finalized.file_count > 0

    # sha256 content matches the tarball
    recomputed = hashlib.sha256(finalized.tarball_path.read_bytes()).hexdigest()
    assert recomputed == finalized.sha256_hex


def test_happy_path_json_files_parse_back_into_models(tmp_path: Path) -> None:
    """Every JSON file the builder writes round-trips through its pydantic model."""
    root = tmp_path / "run-roundtrip"
    builder = _build_happy_path(root, step_count=2)
    builder.finalize()

    ManifestV1.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))
    SummaryV1.model_validate_json((root / "summary.json").read_text(encoding="utf-8"))
    HostInfo.model_validate_json((root / "environment.json").read_text(encoding="utf-8"))

    for step_dir in (root / "steps").iterdir():
        StateSnapshot.model_validate_json((step_dir / "pre-state.json").read_text(encoding="utf-8"))
        StateSnapshot.model_validate_json((step_dir / "post-state.json").read_text(encoding="utf-8"))
        StepResult.model_validate_json((step_dir / "result.json").read_text(encoding="utf-8"))
        RequestRecord.model_validate_json((step_dir / "request-greet.json").read_text(encoding="utf-8"))


def test_sha256sum_c_validates_the_bundle(tmp_path: Path) -> None:
    """Running ``sha256sum -c`` (or shasum -a 256 -c) on the sidecar succeeds.

    We shell out to whichever of ``sha256sum`` / ``shasum`` is available.
    The sidecar's ``<hex>  <basename>\\n`` format is the contract being
    asserted: a consumer in any POSIX shell can verify the tarball.
    """
    root = tmp_path / "run-checksum"
    _build_happy_path(root).finalize()

    tarball = tmp_path / "run-checksum.tar.gz"
    sha_file = tmp_path / "run-checksum.tar.gz.sha256"

    # Prefer sha256sum (Linux); fall back to `shasum -a 256 -c` (macOS).
    for cmd in (["sha256sum", "-c", sha_file.name], ["shasum", "-a", "256", "-c", sha_file.name]):
        try:
            result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            continue
        assert result.returncode == 0, f"{cmd} failed: {result.stdout} {result.stderr}"
        return
    pytest.skip("neither sha256sum nor shasum is available on this system")


# ---------------------------------------------------------------------------
# 2. Step directory naming (slugification edge cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_name,expected_suffix",
    [
        ("01 Crash Recovery / Smoke!", "01-crash-recovery-smoke"),
        ("   leading and trailing   ", "leading-and-trailing"),
        ("unicode-éclair-café", "unicode-eclair-cafe"),
        ("!!!", "step"),
        ("", "step"),
        ("simple_snake_case", "simple_snake_case"),
        ("Already-Kebab-Case", "already-kebab-case"),
        ("multi   spaces\tand\ttabs", "multi-spaces-and-tabs"),
    ],
)
def test_step_dir_naming_slugifies_edge_cases(tmp_path: Path, raw_name: str, expected_suffix: str) -> None:
    """A step's directory name is safe for every edge case the rig might see."""
    root = tmp_path / "run-slug"
    builder = ArtifactBuilder(root)
    step = StepResult(
        step_number=7,
        step_name=raw_name,
        passed=True,
        duration_ms=0.0,
        started_at=T0,
        ended_at=T1,
        pre_state_path="steps/placeholder/pre-state.json",
        post_state_path="steps/placeholder/post-state.json",
    )
    path = builder.step_dir(step)
    assert path.name == f"07-{expected_suffix}"
    assert path.is_dir()


# ---------------------------------------------------------------------------
# 3. Strict-type rejection
# ---------------------------------------------------------------------------


def test_write_manifest_rejects_raw_dict(tmp_path: Path) -> None:
    """Only ``ManifestV1`` instances are accepted — dicts raise ArtifactError."""
    builder = ArtifactBuilder(tmp_path / "run-strict")
    raw_dict = {
        "schema_version": 1,
        "rig_version": "x",
        "concerto_version": "y",
        "scenario_name": "s",
        "scenario_version": "1",
        "started_at": T0.isoformat(),
        "ended_at": T1.isoformat(),
        "duration_seconds": 0.0,
        "exit_status": "success",
        "step_count": 0,
    }
    with pytest.raises(ArtifactError, match="expected ManifestV1 instance"):
        builder.write_manifest(raw_dict)  # type: ignore[arg-type]


def test_write_summary_and_host_info_and_step_reject_raw_dict(tmp_path: Path) -> None:
    """Same strict typing contract across every top-level writer."""
    builder = ArtifactBuilder(tmp_path / "run-strict-all")
    with pytest.raises(ArtifactError, match="expected SummaryV1"):
        builder.write_summary({})  # type: ignore[arg-type]
    with pytest.raises(ArtifactError, match="expected HostInfo"):
        builder.write_host_info({})  # type: ignore[arg-type]
    with pytest.raises(ArtifactError, match="expected StepResult"):
        builder.write_step({}, _snapshot(), _snapshot())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. finalize() idempotency guard
# ---------------------------------------------------------------------------


def test_finalize_twice_raises(tmp_path: Path) -> None:
    """Double-finalising is a caller bug and must raise a clear error."""
    builder = _build_happy_path(tmp_path / "run-once")
    builder.finalize()
    with pytest.raises(ArtifactError, match="already been finalised"):
        builder.finalize()


def test_writes_after_finalize_raise(tmp_path: Path) -> None:
    """No further writes allowed once the tarball is sealed."""
    builder = _build_happy_path(tmp_path / "run-sealed")
    builder.finalize()
    with pytest.raises(ArtifactError, match="finalised"):
        builder.write_summary(_summary())


def test_finalize_without_manifest_raises(tmp_path: Path) -> None:
    """Manifest is load-bearing for reproducible mtimes; missing it fails loudly."""
    builder = ArtifactBuilder(tmp_path / "run-no-manifest")
    builder.write_summary(_summary())
    builder.write_host_info(_host_info())
    with pytest.raises(ArtifactError, match="write_manifest"):
        builder.finalize()


# ---------------------------------------------------------------------------
# 5. Missing-file detection via verifier
# ---------------------------------------------------------------------------


def test_verifier_detects_missing_result_json(tmp_path: Path) -> None:
    """Deleting a result.json between build and verify surfaces in the errors."""
    root = tmp_path / "run-missing"
    _build_happy_path(root).finalize()
    victim = root / "steps" / "02-step-2" / "result.json"
    assert victim.exists()
    victim.unlink()
    errors = verify_artifact_tree(root)
    assert any("result.json" in e for e in errors)


def test_verifier_detects_missing_manifest(tmp_path: Path) -> None:
    """Missing manifest.json is a hard error."""
    root = tmp_path / "run-nomanifest"
    _build_happy_path(root).finalize()
    (root / "manifest.json").unlink()
    errors = verify_artifact_tree(root)
    assert any("manifest.json" in e for e in errors)


# ---------------------------------------------------------------------------
# 6. Schema-version mismatch detection
# ---------------------------------------------------------------------------


def test_verifier_detects_schema_version_mismatch(tmp_path: Path) -> None:
    """Hand-rolled manifest with schema_version=2 must be rejected by the verifier."""
    root = tmp_path / "run-badversion"
    _build_happy_path(root).finalize()

    # Overwrite the manifest with a schema_version=2 payload that is
    # otherwise syntactically valid JSON. Pydantic's Literal[1] will
    # refuse to parse it, which is what we want the verifier to report.
    evil = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    evil["schema_version"] = 2
    (root / "manifest.json").write_text(json.dumps(evil, indent=2), encoding="utf-8")

    errors = verify_artifact_tree(root)
    assert any("ManifestV1" in e or "schema_version" in e for e in errors)


# ---------------------------------------------------------------------------
# 7. Telemetry file registration
# ---------------------------------------------------------------------------


def test_register_telemetry_file_copies_and_tars(tmp_path: Path) -> None:
    """A JSONL file produced outside the telemetry dir lands at the right path in the tarball."""
    external = tmp_path / "external.jsonl"
    external.write_text('{"schema_version": 1, "ts": "2026-04-05T12:00:00+00:00", "sampler": "external", "values": {}}\n', encoding="utf-8")

    root = tmp_path / "run-telreg"
    builder = ArtifactBuilder(root)
    builder.write_manifest(_manifest(step_count=0))
    builder.write_summary(_summary(step_count=0, steps_passed=0))
    builder.write_host_info(_host_info())
    builder.write_concerto_logs(b"", b"")

    target = builder.register_telemetry_file("external", external)
    assert target == root / "telemetry" / "external.jsonl"
    assert target.read_text(encoding="utf-8") == external.read_text(encoding="utf-8")

    fa = builder.finalize()
    with tarfile.open(fa.tarball_path, "r:gz") as tar:
        names = tar.getnames()
    assert "telemetry/external.jsonl" in names


def test_register_telemetry_file_rejects_missing_source(tmp_path: Path) -> None:
    """Registering a path that does not exist raises a clear error."""
    builder = ArtifactBuilder(tmp_path / "run-telmiss")
    with pytest.raises(ArtifactError, match="does not exist"):
        builder.register_telemetry_file("foo", tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# 8. Reproducible tarballs — the regression-diff invariant
# ---------------------------------------------------------------------------


def test_finalize_produces_byte_identical_tarballs(tmp_path: Path) -> None:
    """Given identical inputs, two finalised tarballs are byte-for-byte equal."""
    root_a = tmp_path / "run-a"
    root_b = tmp_path / "run-b"

    fa_a = _build_happy_path(root_a, step_count=2).finalize()
    fa_b = _build_happy_path(root_b, step_count=2).finalize()

    bytes_a = fa_a.tarball_path.read_bytes()
    bytes_b = fa_b.tarball_path.read_bytes()

    assert bytes_a == bytes_b, "tarballs are not byte-identical"
    assert fa_a.sha256_hex == fa_b.sha256_hex


# ---------------------------------------------------------------------------
# 9 + 10. Concerto log ingestion (bytes in / files in)
# ---------------------------------------------------------------------------


def test_write_concerto_logs_from_bytes(tmp_path: Path) -> None:
    """Bytes handed to the builder land verbatim in the two log files."""
    builder = ArtifactBuilder(tmp_path / "run-bytes")
    stdout_bytes = b"hello stdout\nsecond line\n"
    stderr_bytes = b"hello stderr\n"
    stdout_path, stderr_path = builder.write_concerto_logs(stdout_bytes, stderr_bytes)
    assert stdout_path.read_bytes() == stdout_bytes
    assert stderr_path.read_bytes() == stderr_bytes


def test_copy_concerto_logs_from_files(tmp_path: Path) -> None:
    """Log files from ``concerto_proc`` are copied into the canonical layout."""
    src_dir = tmp_path / "logs"
    src_dir.mkdir()
    src_stdout = src_dir / "concerto-stdout.log"
    src_stderr = src_dir / "concerto-stderr.log"
    src_stdout.write_bytes(b"outside world\n")
    src_stderr.write_bytes(b"error trace\n")

    builder = ArtifactBuilder(tmp_path / "run-copy")
    stdout_path, stderr_path = builder.copy_concerto_logs(src_stdout, src_stderr)

    assert stdout_path.read_bytes() == b"outside world\n"
    assert stderr_path.read_bytes() == b"error trace\n"


def test_copy_concerto_logs_missing_sources_become_empty(tmp_path: Path) -> None:
    """Missing source log paths yield zero-byte placeholders so the tree is stable."""
    builder = ArtifactBuilder(tmp_path / "run-empty-logs")
    stdout_path, stderr_path = builder.copy_concerto_logs(None, None)
    assert stdout_path.read_bytes() == b""
    assert stderr_path.read_bytes() == b""


# ---------------------------------------------------------------------------
# 11. Zero-byte telemetry handling
# ---------------------------------------------------------------------------


def test_empty_telemetry_included_by_default(tmp_path: Path) -> None:
    """Zero-byte telemetry files ship in the tarball unless explicitly excluded."""
    root = tmp_path / "run-emptytel-in"
    builder = ArtifactBuilder(root)
    builder.write_manifest(_manifest(step_count=0))
    builder.write_summary(_summary(step_count=0, steps_passed=0))
    builder.write_host_info(_host_info())
    builder.write_concerto_logs(b"", b"")
    (builder.telemetry_dir() / "nvidia-smi.jsonl").write_bytes(b"")

    fa = builder.finalize()
    with tarfile.open(fa.tarball_path, "r:gz") as tar:
        names = tar.getnames()
    assert "telemetry/nvidia-smi.jsonl" in names


def test_empty_telemetry_excluded_when_requested(tmp_path: Path) -> None:
    """``include_empty_telemetry=False`` drops zero-byte JSONL files from the tarball."""
    root = tmp_path / "run-emptytel-out"
    builder = ArtifactBuilder(root)
    builder.write_manifest(_manifest(step_count=0))
    builder.write_summary(_summary(step_count=0, steps_passed=0))
    builder.write_host_info(_host_info())
    builder.write_concerto_logs(b"", b"")
    (builder.telemetry_dir() / "nvidia-smi.jsonl").write_bytes(b"")
    (builder.telemetry_dir() / "proc-stats.jsonl").write_text('{"schema_version":1,"ts":"2026-04-05T12:00:00+00:00","sampler":"proc-stats","values":{}}\n', encoding="utf-8")

    fa = builder.finalize(include_empty_telemetry=False)
    with tarfile.open(fa.tarball_path, "r:gz") as tar:
        names = tar.getnames()
    assert "telemetry/nvidia-smi.jsonl" not in names
    assert "telemetry/proc-stats.jsonl" in names


# ---------------------------------------------------------------------------
# 12. Step count mismatch detection
# ---------------------------------------------------------------------------


def test_verifier_detects_step_count_mismatch(tmp_path: Path) -> None:
    """manifest.step_count must equal the number of step dirs on disk."""
    root = tmp_path / "run-countmismatch"
    builder = ArtifactBuilder(root)
    # Claim 3 steps in the manifest but only write 2.
    builder.write_manifest(_manifest(step_count=3))
    builder.write_summary(_summary(step_count=3, steps_passed=3))
    builder.write_host_info(_host_info())
    builder.write_concerto_logs(b"", b"")
    for i in (1, 2):
        builder.write_step(
            _step_result(number=i, name=f"step-{i}", capture_as=None),
            pre_state=_snapshot(),
            post_state=_snapshot(ts=T1),
            request_records=None,
        )
    errors = verify_artifact_tree(root)
    assert any("step_count" in e for e in errors)


# ---------------------------------------------------------------------------
# 13. RequestRecord capture file naming
# ---------------------------------------------------------------------------


def test_request_capture_file_naming(tmp_path: Path) -> None:
    """capture_as='hello-world' produces request-hello-world.json in the step dir."""
    root = tmp_path / "run-capname"
    builder = ArtifactBuilder(root)
    builder.write_manifest(_manifest(step_count=1))
    builder.write_summary(_summary(step_count=1, steps_passed=1))
    builder.write_host_info(_host_info())
    builder.write_concerto_logs(b"", b"")

    step = _step_result(number=1, name="caps", capture_as="hello-world")
    builder.write_step(
        step,
        pre_state=_snapshot(),
        post_state=_snapshot(ts=T1),
        request_records={"hello-world": _request_record()},
    )
    assert (root / "steps" / "01-caps" / "request-hello-world.json").is_file()


# ---------------------------------------------------------------------------
# 14. Missing request-capture file detection
# ---------------------------------------------------------------------------


def test_verifier_detects_missing_request_capture_file(tmp_path: Path) -> None:
    """An action that references capture_as without a matching file is reported."""
    root = tmp_path / "run-cap-missing"
    builder = _build_happy_path(root, step_count=1)
    builder.finalize()
    # Delete the request-greet.json file; the action record still references it.
    (root / "steps" / "01-step-1" / "request-greet.json").unlink()
    errors = verify_artifact_tree(root)
    assert any("capture_as='greet'" in e or "request-greet.json" in e for e in errors)


def test_verifier_detects_orphan_capture_file(tmp_path: Path) -> None:
    """A request-*.json without a backing action is flagged as orphan."""
    root = tmp_path / "run-cap-orphan"
    builder = _build_happy_path(root, step_count=1, capture_names=[None])  # type: ignore[list-item]
    builder.finalize()
    # Plant a rogue capture file with no action referencing it.
    rogue = root / "steps" / "01-step-1" / "request-rogue.json"
    rogue.write_text(_request_record().model_dump_json(indent=2), encoding="utf-8")
    errors = verify_artifact_tree(root)
    assert any("rogue" in e for e in errors)


# ---------------------------------------------------------------------------
# 15. SHA-256 sidecar file format
# ---------------------------------------------------------------------------


def test_sha256_sidecar_file_format(tmp_path: Path) -> None:
    """The .sha256 file is exactly ``<hex>  <basename>\\n`` (two spaces)."""
    root = tmp_path / "run-sha-fmt"
    fa = _build_happy_path(root).finalize()
    content = fa.sha256_path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    line = content.rstrip("\n")
    # Single line, two-space separator
    assert "  " in line
    hex_part, name_part = line.split("  ", 1)
    assert len(hex_part) == 64
    assert all(c in "0123456789abcdef" for c in hex_part)
    assert name_part == fa.tarball_path.name


# ---------------------------------------------------------------------------
# 16. Tar member ordering is sorted-by-path
# ---------------------------------------------------------------------------


def test_tar_members_are_sorted_by_path(tmp_path: Path) -> None:
    """Tar member ordering is the key guarantee behind reproducible builds."""
    root = tmp_path / "run-order"
    _build_happy_path(root, step_count=2).finalize()
    tarball = tmp_path / "run-order.tar.gz"
    with tarfile.open(tarball, "r:gz") as tar:
        names = tar.getnames()
    assert names == sorted(names), f"tar members are not sorted: {names}"


# ---------------------------------------------------------------------------
# Extra — accessors + FinalizedArtifact dataclass shape
# ---------------------------------------------------------------------------


def test_finalized_artifact_is_frozen_dataclass(tmp_path: Path) -> None:
    """``FinalizedArtifact`` is immutable and exposes the documented fields."""
    fa = _build_happy_path(tmp_path / "run-fa").finalize()
    assert isinstance(fa, FinalizedArtifact)
    with pytest.raises((AttributeError, TypeError)):
        fa.sha256_hex = "overwritten"  # type: ignore[misc]


def test_telemetry_dir_accessor(tmp_path: Path) -> None:
    """``telemetry_dir()`` returns an existing directory under the root."""
    builder = ArtifactBuilder(tmp_path / "run-tdir")
    assert builder.telemetry_dir() == tmp_path / "run-tdir" / "telemetry"
    assert builder.telemetry_dir().is_dir()


def test_duplicate_step_number_rejected(tmp_path: Path) -> None:
    """Writing two steps with the same step_number raises ArtifactError."""
    builder = ArtifactBuilder(tmp_path / "run-dupstep")
    step = _step_result(number=1, name="dup", capture_as=None)
    builder.write_step(step, _snapshot(), _snapshot(ts=T1))
    with pytest.raises(ArtifactError, match="duplicate step_number"):
        builder.write_step(step, _snapshot(), _snapshot(ts=T1))
