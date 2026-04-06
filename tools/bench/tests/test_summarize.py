"""Tests for :mod:`concerto_bench.analyze.summarize`.

Exercises the summarize module against fabricated artifact directories
and tarballs. The tests cover the full public surface:

1. Happy-path summarization from a directory with all files present.
2. Summarization from a ``.tar.gz`` bundle.
3. Tolerance of missing telemetry files.
4. Failed-step detail rendering.
5. Graceful handling when no request records exist.
6. ``SummarizeError`` on an invalid ``manifest.json``.
7. ``SummarizeError`` on a missing ``manifest.json``.
8. Writing the markdown to an output file.
9. Percentile computation edge cases.
10. Tolerance of missing ``summary.json``.

All artifact data is fabricated inline using pydantic models from
:mod:`concerto_bench.schema`, serialised with ``.model_dump_json()``,
and written to ``tmp_path`` fixture directories.
"""

from __future__ import annotations

import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from concerto_bench.analyze import SummarizeError, summarize_artifact
from concerto_bench.analyze.summarize import _percentile
from concerto_bench.schema import (
    ActionRecord,
    ManifestV1,
    RequestRecord,
    StateSnapshot,
    StepResult,
    SummaryV1,
    TelemetrySample,
)

UTC = timezone.utc
T0 = datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=30)


# ---------------------------------------------------------------------------
# Fabrication helpers
# ---------------------------------------------------------------------------


def _manifest(
    *,
    step_count: int = 2,
    exit_status: str = "success",
) -> ManifestV1:
    return ManifestV1(
        rig_version="0.1.0.dev0",
        concerto_version="concerto 0.1.0",
        concerto_git_sha="abcdef1",
        scenario_name="summarize-test",
        scenario_version="1",
        started_at=T0,
        ended_at=T1,
        duration_seconds=30.0,
        exit_status=exit_status,
        step_count=step_count,
    )


def _summary(
    *,
    step_count: int = 2,
    steps_passed: int = 2,
    failed_step_names: list[str] | None = None,
) -> SummaryV1:
    return SummaryV1(
        scenario_name="summarize-test",
        scenario_version="1",
        exit_status="success" if steps_passed == step_count else "partial_failure",
        scenario_passed=steps_passed == step_count,
        step_count=step_count,
        steps_passed=steps_passed,
        steps_failed=step_count - steps_passed,
        failed_step_names=failed_step_names or [],
    )


def _action_record(
    *,
    passed: bool = True,
    capture_as: str | None = None,
) -> ActionRecord:
    args: dict[str, Any] = {"model": "qwen2.5-0.5b", "content": "hi"}
    if capture_as is not None:
        args["capture_as"] = capture_as
    return ActionRecord(
        action_type="request",
        args=args,
        started_at=T0,
        ended_at=T0 + timedelta(milliseconds=500),
        duration_ms=500.0,
        passed=passed,
    )


def _step_result(
    *,
    number: int,
    name: str,
    passed: bool = True,
    failures: list[str] | None = None,
    capture_as: str | None = "smoke",
) -> StepResult:
    slug = name.lower().replace(" ", "-")
    return StepResult(
        step_number=number,
        step_name=name,
        passed=passed,
        duration_ms=1500.0,
        started_at=T0 + timedelta(seconds=number),
        ended_at=T0 + timedelta(seconds=number + 1, milliseconds=500),
        pre_state_path=f"steps/{number:02d}-{slug}/pre-state.json",
        post_state_path=f"steps/{number:02d}-{slug}/post-state.json",
        actions=[_action_record(capture_as=capture_as)],
        failures=failures or [],
    )


def _request_record(
    *,
    status: int = 200,
    elapsed_total_ms: float = 234.5,
    elapsed_ttfb_ms: float | None = 189.2,
) -> RequestRecord:
    return RequestRecord(
        status=status,
        elapsed_total_ms=elapsed_total_ms,
        elapsed_ttfb_ms=elapsed_ttfb_ms,
        request_body={"model": "qwen2.5-0.5b", "prompt": "hello"},
        response_body={"choices": [{"text": "world"}]},
    )


def _snapshot(ts: datetime = T0) -> StateSnapshot:
    return StateSnapshot(
        ts=ts,
        concerto_status={"loaded_models": []},
        backend_pids=[],
    )


def _telemetry_row(sampler: str, ts: datetime = T0) -> str:
    sample = TelemetrySample(ts=ts, sampler=sampler, values={"dummy": 1})
    return sample.model_dump_json() + "\n"


def _build_artifact(
    root: Path,
    *,
    step_count: int = 2,
    include_summary: bool = True,
    include_telemetry: bool = True,
    include_requests: bool = True,
    failed_steps: list[int] | None = None,
) -> Path:
    """Build a fabricated artifact directory at *root* and return *root*.

    This is a minimal builder that writes the files the summarizer needs
    without going through ``ArtifactBuilder`` (to keep the test isolated).
    """
    failed_steps = failed_steps or []
    root.mkdir(parents=True, exist_ok=True)

    # manifest.json
    exit_status = "partial_failure" if failed_steps else "success"
    manifest = _manifest(step_count=step_count, exit_status=exit_status)
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    # summary.json
    if include_summary:
        steps_passed = step_count - len(failed_steps)
        failed_names = [f"step-{i}" for i in failed_steps]
        summary = _summary(
            step_count=step_count,
            steps_passed=steps_passed,
            failed_step_names=failed_names,
        )
        (root / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    # steps/
    steps_dir = root / "steps"
    for i in range(1, step_count + 1):
        name = f"step-{i}"
        is_failed = i in failed_steps
        failures = [f"assertion failed: expected 200 got 500"] if is_failed else []
        sr = _step_result(
            number=i,
            name=name,
            passed=not is_failed,
            failures=failures,
            capture_as="smoke" if include_requests else None,
        )
        slug = name.lower().replace(" ", "-")
        step_dir = steps_dir / f"{i:02d}-{slug}"
        step_dir.mkdir(parents=True, exist_ok=True)

        (step_dir / "result.json").write_text(sr.model_dump_json(indent=2), encoding="utf-8")
        (step_dir / "pre-state.json").write_text(
            _snapshot(ts=T0 + timedelta(seconds=i)).model_dump_json(indent=2),
            encoding="utf-8",
        )
        (step_dir / "post-state.json").write_text(
            _snapshot(ts=T0 + timedelta(seconds=i + 1)).model_dump_json(indent=2),
            encoding="utf-8",
        )

        if include_requests:
            rr = _request_record(
                elapsed_total_ms=100.0 + i * 50.0,
                status=500 if is_failed else 200,
            )
            (step_dir / "request-smoke.json").write_text(
                rr.model_dump_json(indent=2), encoding="utf-8"
            )

    # telemetry/
    if include_telemetry:
        tel_dir = root / "telemetry"
        tel_dir.mkdir(parents=True, exist_ok=True)
        for sampler in ("nvidia-smi", "concerto-status"):
            rows = ""
            for j in range(3):
                rows += _telemetry_row(sampler, ts=T0 + timedelta(seconds=j))
            (tel_dir / f"{sampler}.jsonl").write_text(rows, encoding="utf-8")

    return root


# ---------------------------------------------------------------------------
# 1. Happy-path: full artifact directory
# ---------------------------------------------------------------------------


def test_summarize_full_artifact_directory(tmp_path: Path) -> None:
    """A complete fabricated artifact produces markdown with all sections."""
    root = _build_artifact(tmp_path / "run-happy")
    md = summarize_artifact(root)

    assert "# Concerto Benchmark: summarize-test v1" in md
    assert "**Duration:** 30.0s" in md
    assert "**Concerto version:** concerto 0.1.0 (abcdef1)" in md
    assert "**Rig version:** 0.1.0.dev0" in md
    assert "**Exit status:** success" in md

    # Summary table
    assert "Steps passed | 2/2" in md
    assert "Steps failed | 0" in md

    # Step results table
    assert "step-1" in md
    assert "step-2" in md
    assert "\u2705 PASS" in md

    # Request latencies
    assert "## Request Latencies" in md
    assert "smoke" in md
    assert "200" in md

    # Telemetry
    assert "## Telemetry Summary" in md
    assert "nvidia-smi" in md
    assert "concerto-status" in md

    # Footer
    assert "*Generated by concerto-bench 0.1.0.dev0*" in md


# ---------------------------------------------------------------------------
# 2. Summarize from a .tar.gz
# ---------------------------------------------------------------------------


def test_summarize_from_tarball(tmp_path: Path) -> None:
    """Summarize works when given a .tar.gz path instead of a directory."""
    root = _build_artifact(tmp_path / "run-tar")

    # Package into a tarball
    tarball_path = tmp_path / "run-tar.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(root, arcname=root.name)

    md = summarize_artifact(tarball_path)

    assert "# Concerto Benchmark: summarize-test v1" in md
    assert "**Duration:** 30.0s" in md
    assert "step-1" in md
    assert "step-2" in md


# ---------------------------------------------------------------------------
# 3. Missing telemetry (should still produce markdown)
# ---------------------------------------------------------------------------


def test_summarize_missing_telemetry(tmp_path: Path) -> None:
    """Missing telemetry directory does not prevent markdown generation."""
    root = _build_artifact(tmp_path / "run-notel", include_telemetry=False)
    md = summarize_artifact(root)

    assert "# Concerto Benchmark:" in md
    assert "step-1" in md
    # Telemetry section should be absent
    assert "## Telemetry Summary" not in md


# ---------------------------------------------------------------------------
# 4. Failed steps show failure details
# ---------------------------------------------------------------------------


def test_summarize_failed_steps(tmp_path: Path) -> None:
    """Failed steps render with FAIL icon and failure detail block."""
    root = _build_artifact(tmp_path / "run-fail", step_count=3, failed_steps=[2])
    md = summarize_artifact(root)

    # Summary table shows failures
    assert "Steps passed | 2/3" in md
    assert "Steps failed | 1" in md
    assert "step-2" in md

    # Step results table has FAIL icon
    assert "\u274c FAIL" in md

    # Failure detail block
    assert "### Step 2" in md
    assert "assertion failed" in md


# ---------------------------------------------------------------------------
# 5. No request records => skip latency table
# ---------------------------------------------------------------------------


def test_summarize_no_request_records(tmp_path: Path) -> None:
    """When no request-*.json files exist, the latency section is omitted."""
    root = _build_artifact(tmp_path / "run-noreq", include_requests=False)
    md = summarize_artifact(root)

    assert "# Concerto Benchmark:" in md
    # Request latencies section should be absent
    assert "## Request Latencies" not in md


# ---------------------------------------------------------------------------
# 6. SummarizeError on invalid manifest
# ---------------------------------------------------------------------------


def test_summarize_error_on_invalid_manifest(tmp_path: Path) -> None:
    """An invalid manifest.json (bad schema) raises SummarizeError."""
    root = tmp_path / "run-badmanifest"
    root.mkdir(parents=True)
    # Write a manifest with schema_version=2 which Literal[1] rejects.
    bad_manifest = {
        "schema_version": 2,
        "rig_version": "x",
        "concerto_version": "y",
        "scenario_name": "bad",
        "scenario_version": "1",
        "started_at": T0.isoformat(),
        "ended_at": T1.isoformat(),
        "duration_seconds": 30.0,
        "exit_status": "success",
        "step_count": 0,
    }
    (root / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")

    with pytest.raises(SummarizeError, match="invalid manifest.json"):
        summarize_artifact(root)


# ---------------------------------------------------------------------------
# 7. SummarizeError on missing manifest
# ---------------------------------------------------------------------------


def test_summarize_error_on_missing_manifest(tmp_path: Path) -> None:
    """A directory without manifest.json raises SummarizeError."""
    root = tmp_path / "run-nomanifest"
    root.mkdir(parents=True)

    with pytest.raises(SummarizeError, match="missing manifest.json"):
        summarize_artifact(root)


# ---------------------------------------------------------------------------
# 8. Output written to file
# ---------------------------------------------------------------------------


def test_summarize_writes_output_file(tmp_path: Path) -> None:
    """When output= is provided, the markdown is written to that path."""
    root = _build_artifact(tmp_path / "run-output")
    output_path = tmp_path / "out" / "summary.md"

    md = summarize_artifact(root, output=output_path)

    assert output_path.is_file()
    file_content = output_path.read_text(encoding="utf-8")
    assert file_content == md
    assert "# Concerto Benchmark:" in file_content


# ---------------------------------------------------------------------------
# 9. Percentile edge cases
# ---------------------------------------------------------------------------


def test_percentile_empty_list() -> None:
    """Percentile of an empty list is 0.0."""
    assert _percentile([], 50) == 0.0


def test_percentile_single_value() -> None:
    """Percentile of a single-element list is that element."""
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([42.0], 99) == 42.0


def test_percentile_known_values() -> None:
    """Percentile of a known distribution matches expected values."""
    values = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
    p50 = _percentile(values, 50)
    assert p50 == 30.0
    p0 = _percentile(values, 0)
    assert p0 == 10.0
    p100 = _percentile(values, 100)
    assert p100 == 50.0


# ---------------------------------------------------------------------------
# 10. Missing summary.json is tolerated
# ---------------------------------------------------------------------------


def test_summarize_missing_summary_json(tmp_path: Path) -> None:
    """A missing summary.json does not crash; the markdown notes its absence."""
    root = _build_artifact(tmp_path / "run-nosummary", include_summary=False)
    md = summarize_artifact(root)

    assert "# Concerto Benchmark:" in md
    assert "summary.json not available" in md


# ---------------------------------------------------------------------------
# 11. Non-existent path raises SummarizeError
# ---------------------------------------------------------------------------


def test_summarize_error_on_nonexistent_path(tmp_path: Path) -> None:
    """A path that is neither a directory nor a file raises SummarizeError."""
    bogus = tmp_path / "does-not-exist.tar.gz"
    with pytest.raises(SummarizeError, match="neither a directory nor"):
        summarize_artifact(bogus)
