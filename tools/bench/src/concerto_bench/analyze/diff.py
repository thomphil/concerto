"""Compare two bench-rig artifacts and report regressions.

The primary entry point is :func:`diff_artifacts`. It loads the
``summary.json`` from each artifact, compares numeric metrics
side-by-side, and returns a human-readable report plus a boolean
indicating whether any regression was detected.

Regression thresholds are intentionally generous (10% worse) to
avoid false positives from run-to-run variance on shared hardware.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from concerto_bench.schema import SummaryV1


class DiffError(RuntimeError):
    """Raised when an artifact cannot be loaded for comparison."""


# Metrics where lower is better (latency, error rates, counts).
_LOWER_IS_BETTER = {
    "http_error_rate",
    "concurrent_load_error_rate",
    "graceful_shutdown_wall_time_secs",
    "orphan_processes_after_shutdown",
    "vram_drift_max_percent",
}

# Metrics where higher is better (throughput).
_HIGHER_IS_BETTER = {
    "concurrent_load_throughput_rps",
    "steps_passed",
}

# Regression threshold: candidate must be >10% worse to flag.
_REGRESSION_THRESHOLD = 0.10


@dataclass(frozen=True)
class MetricComparison:
    name: str
    baseline_value: Optional[float]
    candidate_value: Optional[float]
    delta_pct: Optional[float]
    regression: bool


def _load_summary(artifact_path: Path) -> SummaryV1:
    """Load ``summary.json`` from an artifact directory or tarball."""
    if artifact_path.is_dir():
        summary_path = artifact_path / "summary.json"
        if not summary_path.is_file():
            raise DiffError(f"missing summary.json in {artifact_path}")
        text = summary_path.read_text(encoding="utf-8")
    elif artifact_path.is_file() and (
        artifact_path.name.endswith(".tar.gz") or artifact_path.name.endswith(".tgz")
    ):
        text = _extract_summary_from_tarball(artifact_path)
    else:
        raise DiffError(
            f"artifact_path is neither a directory nor a .tar.gz file: {artifact_path}"
        )
    try:
        return SummaryV1.model_validate_json(text)
    except ValidationError as exc:
        raise DiffError(f"invalid summary.json in {artifact_path}: {exc}") from exc


def _extract_summary_from_tarball(tarball_path: Path) -> str:
    """Extract and return ``summary.json`` text from a tarball."""
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            # Try top-level summary.json first, then one level deep
            for member in tar.getmembers():
                if member.name.endswith("summary.json"):
                    f = tar.extractfile(member)
                    if f is not None:
                        return f.read().decode("utf-8")
    except (tarfile.TarError, OSError) as exc:
        raise DiffError(f"cannot read tarball {tarball_path}: {exc}") from exc
    raise DiffError(f"no summary.json found in {tarball_path}")


def _compare_metric(
    name: str,
    baseline: Optional[float],
    candidate: Optional[float],
) -> MetricComparison:
    """Compare a single metric value between baseline and candidate."""
    if baseline is None or candidate is None:
        return MetricComparison(
            name=name,
            baseline_value=baseline,
            candidate_value=candidate,
            delta_pct=None,
            regression=False,
        )

    if baseline == 0:
        delta_pct = 0.0 if candidate == 0 else 100.0
    else:
        delta_pct = ((candidate - baseline) / abs(baseline)) * 100.0

    regression = False
    if name in _LOWER_IS_BETTER:
        # Candidate higher = worse
        regression = candidate > baseline * (1 + _REGRESSION_THRESHOLD)
    elif name in _HIGHER_IS_BETTER:
        # Candidate lower = worse
        regression = candidate < baseline * (1 - _REGRESSION_THRESHOLD)

    return MetricComparison(
        name=name,
        baseline_value=baseline,
        candidate_value=candidate,
        delta_pct=delta_pct,
        regression=regression,
    )


def _extract_scalar_metrics(summary: SummaryV1) -> dict[str, Optional[float]]:
    """Pull all comparable scalar metrics from a summary."""
    metrics: dict[str, Optional[float]] = {
        "steps_passed": float(summary.steps_passed),
        "steps_failed": float(summary.steps_failed),
        "http_error_rate": summary.http_error_rate,
        "vram_drift_max_percent": summary.vram_drift_max_percent,
        "graceful_shutdown_wall_time_secs": summary.graceful_shutdown_wall_time_secs,
        "orphan_processes_after_shutdown": (
            float(summary.orphan_processes_after_shutdown)
            if summary.orphan_processes_after_shutdown is not None
            else None
        ),
        "concurrent_load_throughput_rps": summary.concurrent_load_throughput_rps,
        "concurrent_load_error_rate": summary.concurrent_load_error_rate,
    }

    # Latency histograms
    if summary.routing_decision_latency is not None:
        rdl = summary.routing_decision_latency
        metrics["routing_decision_p50_ms"] = rdl.p50_ms
        metrics["routing_decision_p95_ms"] = rdl.p95_ms
        metrics["routing_decision_p99_ms"] = rdl.p99_ms

    if summary.concurrent_load_latency is not None:
        cll = summary.concurrent_load_latency
        metrics["concurrent_load_p50_ms"] = cll.p50_ms
        metrics["concurrent_load_p95_ms"] = cll.p95_ms
        metrics["concurrent_load_p99_ms"] = cll.p99_ms

    # Per-model cold start times
    for model_id, mm in summary.model_metrics.items():
        if mm.cold_start_ms is not None:
            metrics[f"cold_start_{model_id}_ms"] = mm.cold_start_ms

    return metrics


def _format_value(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if v == int(v) and abs(v) < 1e9:
        return str(int(v))
    return f"{v:.3f}"


def diff_artifacts(
    baseline_path: Path,
    candidate_path: Path,
) -> tuple[str, bool]:
    """Compare two artifacts and return ``(report_text, has_regression)``.

    Parameters
    ----------
    baseline_path:
        Path to the baseline artifact (directory or ``.tar.gz``).
    candidate_path:
        Path to the candidate artifact to compare against the baseline.

    Returns
    -------
    tuple[str, bool]
        A human-readable comparison report and whether any metric
        regressed beyond the threshold.

    Raises
    ------
    DiffError
        If either artifact cannot be loaded.
    """
    baseline = _load_summary(baseline_path)
    candidate = _load_summary(candidate_path)

    baseline_metrics = _extract_scalar_metrics(baseline)
    candidate_metrics = _extract_scalar_metrics(candidate)

    all_keys = sorted(set(baseline_metrics) | set(candidate_metrics))

    comparisons = []
    for key in all_keys:
        cmp = _compare_metric(key, baseline_metrics.get(key), candidate_metrics.get(key))
        comparisons.append(cmp)

    # Build report
    lines: list[str] = []
    lines.append(f"# Regression Report: {baseline.scenario_name}")
    lines.append("")
    lines.append(
        f"Baseline: {baseline.scenario_name} v{baseline.scenario_version} "
        f"({baseline.exit_status})"
    )
    lines.append(
        f"Candidate: {candidate.scenario_name} v{candidate.scenario_version} "
        f"({candidate.exit_status})"
    )
    lines.append("")
    lines.append(f"| {'Metric':<42} | {'Baseline':>12} | {'Candidate':>12} | {'Delta':>8} | {'Status':<12} |")
    lines.append(f"|{'-' * 44}|{'-' * 14}|{'-' * 14}|{'-' * 10}|{'-' * 14}|")

    has_regression = False
    for cmp in comparisons:
        if cmp.baseline_value is None and cmp.candidate_value is None:
            continue
        bv = _format_value(cmp.baseline_value)
        cv = _format_value(cmp.candidate_value)
        if cmp.delta_pct is not None:
            delta = f"{cmp.delta_pct:+.1f}%"
        else:
            delta = "n/a"
        if cmp.regression:
            status = "[REGRESSION]"
            has_regression = True
        else:
            status = "ok"
        lines.append(f"| {cmp.name:<42} | {bv:>12} | {cv:>12} | {delta:>8} | {status:<12} |")

    lines.append("")
    if has_regression:
        lines.append("**REGRESSIONS DETECTED** — review the flagged metrics above.")
    else:
        lines.append("No regressions detected.")
    lines.append("")

    return "\n".join(lines), has_regression
