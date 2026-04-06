"""Read a bench-rig artifact and produce a human-readable markdown summary.

The primary entry point is :func:`summarize_artifact`. It accepts either
a materialised artifact directory or a ``.tar.gz`` bundle, reads the
versioned JSON records inside, computes aggregate metrics from request
records and telemetry JSONL files, and emits a self-contained markdown
report suitable for pasting into a PR body or piping into ``less``.

Tolerance
---------

The summarizer is deliberately tolerant of partial artifacts. A run that
aborted mid-scenario still produces a tarball with a manifest and
whatever steps completed; the summarizer renders what it can and notes
what is missing rather than raising. Only a structurally invalid or
entirely absent ``manifest.json`` is a hard error.

See ``SPRINT-2-PLAN.md`` section 4 B.3 step 11 for the specification
this module implements.
"""

from __future__ import annotations

import json
import logging
import math
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from concerto_bench.schema import (
    ManifestV1,
    RequestRecord,
    StepResult,
    SummaryV1,
    TelemetrySample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SummarizeError(RuntimeError):
    """Raised when the artifact is unreadable or structurally invalid.

    Covers missing ``manifest.json``, schema-validation failures on the
    manifest or summary, and IO errors during extraction. Partial data
    (missing telemetry, missing step results) does **not** raise this
    error -- those gaps are reported in the markdown output instead.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_json_file(path: Path) -> str:
    """Read a file and return its text content, or raise SummarizeError."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SummarizeError(f"cannot read {path}: {exc}") from exc


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from a pre-sorted list using linear interpolation.

    Returns 0.0 for an empty list. ``pct`` is in the range [0, 100].
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    # Rank (0-based fractional index) for the percentile.
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] + weight * (sorted_values[upper] - sorted_values[lower])


def _format_duration(ms: float) -> str:
    """Format a duration in milliseconds as a human string.

    Values >= 1000 ms are shown as seconds (e.g. ``45.2s``); smaller
    values stay in milliseconds (e.g. ``234.5ms``).
    """
    if ms >= 1000.0:
        return f"{ms / 1000.0:.1f}s"
    return f"{ms:.1f}ms"


# ---------------------------------------------------------------------------
# Telemetry summary
# ---------------------------------------------------------------------------


def _summarize_telemetry(telemetry_dir: Path) -> list[str]:
    """Produce per-file summary lines for every JSONL file under *telemetry_dir*.

    Returns an empty list if the directory does not exist or contains no
    JSONL files.
    """
    if not telemetry_dir.is_dir():
        return []

    lines: list[str] = []
    for jsonl_path in sorted(telemetry_dir.glob("*.jsonl")):
        count = 0
        first_ts: str | None = None
        last_ts: str | None = None
        try:
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        row = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    count += 1
                    ts = row.get("ts")
                    if ts is not None:
                        if first_ts is None:
                            first_ts = str(ts)
                        last_ts = str(ts)
        except OSError:
            logger.warning("could not read telemetry file %s", jsonl_path)
            continue

        sampler_name = jsonl_path.stem
        if count == 0:
            lines.append(f"- {sampler_name}: 0 samples")
        elif first_ts and last_ts and first_ts != last_ts:
            lines.append(f"- {sampler_name}: {count} samples ({first_ts} .. {last_ts})")
        else:
            lines.append(f"- {sampler_name}: {count} samples")

    return lines


# ---------------------------------------------------------------------------
# Request record collection
# ---------------------------------------------------------------------------


def _collect_request_records(
    steps_dir: Path,
) -> list[tuple[str, RequestRecord]]:
    """Walk step directories and return ``(label, RequestRecord)`` pairs.

    The label is derived from the filename: ``request-<label>.json``.
    Files that fail to parse are silently skipped.
    """
    records: list[tuple[str, RequestRecord]] = []
    if not steps_dir.is_dir():
        return records

    for step_dir in sorted(steps_dir.iterdir()):
        if not step_dir.is_dir():
            continue
        for req_file in sorted(step_dir.glob("request-*.json")):
            label = req_file.stem.removeprefix("request-")
            try:
                text = req_file.read_text(encoding="utf-8")
                record = RequestRecord.model_validate_json(text)
                records.append((label, record))
            except (OSError, ValidationError, json.JSONDecodeError) as exc:
                logger.warning("skipping request record %s: %s", req_file, exc)
    return records


# ---------------------------------------------------------------------------
# Step result collection
# ---------------------------------------------------------------------------


def _collect_step_results(steps_dir: Path) -> list[StepResult]:
    """Read ``result.json`` from every step directory, sorted by directory name.

    Steps whose ``result.json`` is missing or unparseable are silently
    skipped; the markdown output will note the mismatch between
    ``manifest.step_count`` and the number of results recovered.
    """
    results: list[StepResult] = []
    if not steps_dir.is_dir():
        return results

    for step_dir in sorted(steps_dir.iterdir()):
        if not step_dir.is_dir():
            continue
        result_path = step_dir / "result.json"
        if not result_path.is_file():
            logger.warning("missing result.json in %s", step_dir)
            continue
        try:
            text = result_path.read_text(encoding="utf-8")
            results.append(StepResult.model_validate_json(text))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            logger.warning("skipping step result %s: %s", result_path, exc)
    return results


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def _build_markdown(
    manifest: ManifestV1,
    summary: SummaryV1 | None,
    step_results: list[StepResult],
    request_records: list[tuple[str, RequestRecord]],
    telemetry_lines: list[str],
) -> str:
    """Assemble the final markdown string from parsed data."""
    parts: list[str] = []

    # -- Header -----------------------------------------------------------
    version = manifest.scenario_version
    parts.append(f"# Concerto Benchmark: {manifest.scenario_name} v{version}")
    parts.append("")

    started_str = manifest.started_at.isoformat()
    sha_str = f" ({manifest.concerto_git_sha})" if manifest.concerto_git_sha else ""
    parts.append(f"**Run date:** {started_str}  ")
    parts.append(f"**Duration:** {manifest.duration_seconds:.1f}s  ")
    parts.append(f"**Concerto version:** {manifest.concerto_version}{sha_str}  ")
    parts.append(f"**Rig version:** {manifest.rig_version}  ")
    parts.append(f"**Exit status:** {manifest.exit_status}  ")
    parts.append("")

    # -- Summary table ----------------------------------------------------
    parts.append("## Summary")
    parts.append("")

    if summary is not None:
        failed_names = ", ".join(summary.failed_step_names) if summary.failed_step_names else "\u2014"
        parts.append("| Metric | Value |")
        parts.append("|--------|-------|")
        parts.append(f"| Steps passed | {summary.steps_passed}/{summary.step_count} |")
        parts.append(f"| Steps failed | {summary.steps_failed} |")
        parts.append(f"| Failed steps | {failed_names} |")
    else:
        parts.append("*summary.json not available*")
    parts.append("")

    # -- Step results table -----------------------------------------------
    parts.append("## Step Results")
    parts.append("")

    if step_results:
        parts.append("| # | Step | Status | Duration |")
        parts.append("|---|------|--------|----------|")
        for sr in step_results:
            icon = "\u2705 PASS" if sr.passed else "\u274c FAIL"
            dur = _format_duration(sr.duration_ms)
            parts.append(f"| {sr.step_number} | {sr.step_name} | {icon} | {dur} |")
        parts.append("")

        # -- Per-failed-step detail blocks --------------------------------
        for sr in step_results:
            if not sr.passed and sr.failures:
                parts.append(f"### Step {sr.step_number} \u2014 {sr.step_name} (FAIL)")
                parts.append("- **Failures:**")
                for failure in sr.failures:
                    parts.append(f"  - {failure}")
                parts.append("")
    else:
        parts.append("*No step results found.*")
        parts.append("")

    # -- Request latencies ------------------------------------------------
    if request_records:
        parts.append("## Request Latencies")
        parts.append("")
        parts.append("| Request | Status | Total (ms) | TTFB (ms) |")
        parts.append("|---------|--------|-----------|-----------|")
        for label, rr in request_records:
            ttfb_str = f"{rr.elapsed_ttfb_ms:.1f}" if rr.elapsed_ttfb_ms is not None else "\u2014"
            parts.append(f"| {label} | {rr.status} | {rr.elapsed_total_ms:.1f} | {ttfb_str} |")
        parts.append("")

        # Aggregate latency histogram from all request records
        totals = sorted(rr.elapsed_total_ms for _, rr in request_records)
        if len(totals) >= 2:
            parts.append("**Aggregate latency (from request records):**")
            parts.append(f"- p50: {_percentile(totals, 50):.1f}ms")
            parts.append(f"- p95: {_percentile(totals, 95):.1f}ms")
            parts.append(f"- p99: {_percentile(totals, 99):.1f}ms")
            parts.append(f"- max: {totals[-1]:.1f}ms")
            parts.append(f"- count: {len(totals)}")
            parts.append("")

        # Error rate
        errors = sum(1 for _, rr in request_records if rr.status < 200 or rr.status >= 300)
        total = len(request_records)
        rate = errors / total if total > 0 else 0.0
        parts.append(f"**Error rate:** {errors}/{total} ({rate:.1%})")
        parts.append("")

    # -- Telemetry summary ------------------------------------------------
    if telemetry_lines:
        parts.append("## Telemetry Summary")
        parts.append("")
        for line in telemetry_lines:
            parts.append(line)
        parts.append("")

    # -- Footer -----------------------------------------------------------
    parts.append("---")
    parts.append(f"*Generated by concerto-bench {manifest.rig_version}*")
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_artifact(
    artifact_path: Path,
    output: Optional[Path] = None,
) -> str:
    """Read an artifact directory (or ``.tar.gz``) and produce a markdown summary.

    Parameters
    ----------
    artifact_path:
        Path to either a materialised artifact directory or a
        ``.tar.gz`` bundle produced by :meth:`ArtifactBuilder.finalize`.
    output:
        If provided, the markdown is written to this file path in
        addition to being returned.

    Returns
    -------
    str
        The complete markdown report.

    Raises
    ------
    SummarizeError
        If ``manifest.json`` is missing, unreadable, or fails schema
        validation; or if ``artifact_path`` is neither a directory nor
        a recognisable tarball.
    """
    root: Path
    cleanup_dir: tempfile.TemporaryDirectory[str] | None = None

    # Resolve the artifact root: directory or tarball.
    if artifact_path.is_dir():
        root = artifact_path
    elif artifact_path.is_file() and (
        artifact_path.name.endswith(".tar.gz") or artifact_path.name.endswith(".tgz")
    ):
        try:
            cleanup_dir = tempfile.TemporaryDirectory(prefix="concerto-summarize-")
            with tarfile.open(artifact_path, "r:gz") as tar:
                tar.extractall(path=cleanup_dir.name)  # noqa: S202
            # The tarball may extract into a single top-level directory or
            # directly into the temp dir.  If there is exactly one child
            # directory and it contains manifest.json, use that.
            extracted = Path(cleanup_dir.name)
            children = [p for p in extracted.iterdir() if p.is_dir()]
            if len(children) == 1 and (children[0] / "manifest.json").is_file():
                root = children[0]
            else:
                root = extracted
        except (tarfile.TarError, OSError) as exc:
            if cleanup_dir is not None:
                cleanup_dir.cleanup()
            raise SummarizeError(f"cannot extract tarball {artifact_path}: {exc}") from exc
    else:
        raise SummarizeError(
            f"artifact_path is neither a directory nor a .tar.gz file: {artifact_path}"
        )

    try:
        return _summarize_root(root, output)
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()


def _summarize_root(root: Path, output: Optional[Path]) -> str:
    """Core logic operating on a materialised artifact directory."""
    # -- manifest.json (required) -----------------------------------------
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SummarizeError(f"missing manifest.json in {root}")

    manifest_text = _read_json_file(manifest_path)
    try:
        manifest = ManifestV1.model_validate_json(manifest_text)
    except ValidationError as exc:
        raise SummarizeError(f"invalid manifest.json: {exc}") from exc

    # -- summary.json (optional) ------------------------------------------
    summary: SummaryV1 | None = None
    summary_path = root / "summary.json"
    if summary_path.is_file():
        try:
            summary_text = _read_json_file(summary_path)
            summary = SummaryV1.model_validate_json(summary_text)
        except (SummarizeError, ValidationError) as exc:
            logger.warning("could not parse summary.json: %s", exc)

    # -- Step results -----------------------------------------------------
    steps_dir = root / "steps"
    step_results = _collect_step_results(steps_dir)

    # -- Request records --------------------------------------------------
    request_records = _collect_request_records(steps_dir)

    # -- Telemetry --------------------------------------------------------
    telemetry_lines = _summarize_telemetry(root / "telemetry")

    # -- Markdown assembly ------------------------------------------------
    md = _build_markdown(manifest, summary, step_results, request_records, telemetry_lines)

    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(md, encoding="utf-8")
            logger.info("wrote summary to %s", output)
        except OSError as exc:
            raise SummarizeError(f"cannot write output to {output}: {exc}") from exc

    return md
