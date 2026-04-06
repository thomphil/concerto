"""Typer CLI surface for the Concerto bench rig.

Subcommands: ``run``, ``dry-run``, ``summarize``, ``verify-weights``,
``estimate``. Each drives a specific workflow documented in
SPRINT-2-PLAN §4 B.2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from concerto_bench import __version__

app = typer.Typer(
    name="concerto-bench",
    help=(
        "Concerto benchmark and validation rig. Drives the real concerto "
        "binary through scripted scenarios, captures telemetry, produces "
        "a versioned artifact tarball."
    ),
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

logger = logging.getLogger(__name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"concerto-bench {__version__}")
        raise typer.Exit()


def _configure_logging(level: str) -> None:
    """Set up basic logging at the requested verbosity."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print concerto-bench version and exit.",
    ),
) -> None:
    """Root callback. All real work happens in subcommands."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command("run")
def run_cmd(
    scenario: Path = typer.Option(
        ...,
        "--scenario",
        exists=True,
        help="Path to the scenario YAML to execute.",
    ),
    concerto_bin: Path = typer.Option(
        ...,
        "--concerto-bin",
        exists=True,
        help="Path to the concerto release binary to exercise.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Directory to build the artifact tree in before tarring.",
    ),
    models_dir: Optional[Path] = typer.Option(
        None,
        "--models-dir",
        help="Directory containing pre-downloaded model weights.",
    ),
    config_override: Optional[Path] = typer.Option(
        None,
        "--concerto-config",
        help="Override path to concerto.toml config file.",
    ),
    http_timeout: float = typer.Option(
        240.0,
        "--http-timeout",
        help="HTTP request timeout in seconds. Must exceed cold-start time (default 240s).",
    ),
    startup_timeout: float = typer.Option(
        60.0,
        "--startup-timeout",
        help="Concerto /health gate timeout in seconds.",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Rig log verbosity (debug|info|warn|error).",
    ),
) -> None:
    """Execute a scenario against a pre-built concerto binary.

    This is the Vast.ai code path: concerto is assumed to already be
    built locally on the host, weights pre-downloaded, and the rig's
    job is to drive the scenario end-to-end and package the artifact.
    """
    _configure_logging(log_level)

    from concerto_bench.runner import RunnerOptions, run_scenario

    options = RunnerOptions(
        scenario_path=scenario.resolve(),
        output_dir=output.resolve(),
        concerto_bin=concerto_bin.resolve(),
        models_dir=models_dir.resolve() if models_dir else None,
        concerto_config_override=config_override.resolve() if config_override else None,
        concerto_log_level=log_level if log_level in ("debug", "info", "warn", "error") else "info",
        concerto_log_format="json",
        http_timeout_secs=http_timeout,
        startup_timeout_secs=startup_timeout,
    )

    result = asyncio.run(run_scenario(options))

    if result.artifact is not None:
        typer.echo(f"Artifact: {result.artifact.tarball_path}")
    typer.echo(f"Exit status: {result.manifest.exit_status}")

    if result.summary.failed_step_names:
        typer.echo(f"Failed steps: {', '.join(result.summary.failed_step_names)}", err=True)

    raise typer.Exit(code=result.exit_code)


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------


@app.command("dry-run")
def dry_run_cmd(
    scenario: Path = typer.Option(
        ...,
        "--scenario",
        exists=True,
        help="Path to the scenario YAML to execute.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Directory to build the artifact tree in before tarring.",
    ),
    mock_gpus: int = typer.Option(
        2,
        "--mock-gpus",
        min=1,
        max=8,
        help="Number of mock GPUs to spawn concerto with (mock upstream).",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Rig log verbosity (debug|info|warn|error).",
    ),
) -> None:
    """Execute a scenario locally against mock-inference-backend.

    This is the CI regression gate -- it spawns a local concerto with
    ``--mock-gpus`` pointed at the bundled mock upstream, runs the
    scenario, and produces an artifact. The tarball shape is what the
    CI test asserts on.
    """
    _configure_logging(log_level)

    from concerto_bench.runner import RunnerOptions, run_scenario

    # Locate the concerto binary from a cargo build.
    concerto_bin = _find_concerto_binary()
    if concerto_bin is None:
        typer.echo(
            "Could not locate concerto binary. Run `cargo build` first, "
            "or use `--concerto-bin` via the `run` subcommand.",
            err=True,
        )
        raise typer.Exit(code=2)

    options = RunnerOptions(
        scenario_path=scenario.resolve(),
        output_dir=output.resolve(),
        concerto_bin=concerto_bin,
        mock_gpus=mock_gpus,
        concerto_log_level=log_level if log_level in ("debug", "info", "warn", "error") else "info",
        concerto_log_format="json",
    )

    result = asyncio.run(run_scenario(options))

    if result.artifact is not None:
        typer.echo(f"Artifact: {result.artifact.tarball_path}")
    typer.echo(f"Exit status: {result.manifest.exit_status}")

    if result.summary.failed_step_names:
        typer.echo(f"Failed steps: {', '.join(result.summary.failed_step_names)}", err=True)

    raise typer.Exit(code=result.exit_code)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


@app.command("summarize")
def summarize_cmd(
    artifact: Path = typer.Argument(
        ...,
        help="Path to a run tarball (or an unpacked artifact directory).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help=(
            "Output markdown path. Defaults to stdout if omitted. "
            "Typical target: docs/benchmarks.md."
        ),
    ),
) -> None:
    """Render an artifact tarball into a human-readable markdown summary."""
    from concerto_bench.analyze.summarize import SummarizeError, summarize_artifact

    try:
        markdown = summarize_artifact(artifact, output=output)
    except SummarizeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if output is None:
        typer.echo(markdown)
    else:
        typer.echo(f"Summary written to {output}")


# ---------------------------------------------------------------------------
# verify-weights
# ---------------------------------------------------------------------------


@app.command("verify-weights")
def verify_weights_cmd(
    models_dir: Path = typer.Option(
        ...,
        "--models-dir",
        exists=True,
        help="Directory containing the pre-downloaded model weights.",
    ),
    checksums: Optional[Path] = typer.Option(
        None,
        "--checksums",
        help=(
            "Override path to the weight-checksums manifest. Defaults to "
            "the bundled tools/bench/fixtures/weight-checksums.json."
        ),
    ),
) -> None:
    """Verify SHA-256 checksums of downloaded model weight files.

    Guards against silent HuggingFace re-uploads drifting the numbers
    between Sprint 2 runs. Used by the remote bootstrap script and
    locally when adding a new model to a scenario.
    """
    import hashlib

    checksums_path = checksums
    if checksums_path is None:
        checksums_path = _find_fixtures_dir() / "weight-checksums.json"
    if not checksums_path.is_file():
        typer.echo(f"Checksums file not found: {checksums_path}", err=True)
        raise typer.Exit(code=2)

    manifest = json.loads(checksums_path.read_text(encoding="utf-8"))
    models = manifest.get("models", {})
    if not models:
        typer.echo("No models in checksums manifest.", err=True)
        raise typer.Exit(code=2)

    all_ok = True
    for model_id, file_checksums in models.items():
        model_path = models_dir / model_id
        if not model_path.is_dir():
            typer.echo(f"  MISSING  {model_id}/ — directory not found", err=True)
            all_ok = False
            continue
        for relative_file, expected_sha in file_checksums.items():
            file_path = model_path / relative_file
            if not file_path.is_file():
                typer.echo(f"  MISSING  {model_id}/{relative_file}", err=True)
                all_ok = False
                continue
            actual_sha = _sha256_file(file_path)
            if actual_sha == expected_sha:
                typer.echo(f"  OK       {model_id}/{relative_file}")
            else:
                typer.echo(
                    f"  MISMATCH {model_id}/{relative_file}\n"
                    f"           expected: {expected_sha}\n"
                    f"           actual:   {actual_sha}",
                    err=True,
                )
                all_ok = False

    if all_ok:
        typer.echo("All weight checksums verified.")
        raise typer.Exit(code=0)
    else:
        typer.echo("Some checksums failed or files are missing.", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


@app.command("estimate")
def estimate_cmd(
    scenario: Path = typer.Option(
        ...,
        "--scenario",
        exists=True,
        help="Path to the scenario YAML to estimate.",
    ),
    gpu_profile: str = typer.Option(
        "2xRTX_A4000",
        "--gpu-profile",
        help="GPU profile identifier for cost/wall-clock heuristics.",
    ),
    hourly_rate_gbp: float = typer.Option(
        0.80,
        "--hourly-rate-gbp",
        help="Hourly rental rate in GBP for the chosen profile.",
    ),
) -> None:
    """Forecast wall-clock time and cost for a scenario.

    Local-only; does not hit the Vast.ai API. Heuristics are refined
    from measured dry-run wall times and GPU-type multipliers.
    """
    from concerto_bench.runner import load_scenario

    scenario_obj = load_scenario(scenario)

    # Heuristic time estimates per action type (seconds). These are
    # rough estimates based on dry-run observations with mock backends;
    # real vLLM times are multiplied by the gpu_multiplier below.
    action_time: dict[str, float] = {
        "request": 5.0,
        "snapshot": 1.0,
        "wait": 0.0,  # uses its own duration_secs
        "wait_for": 30.0,
        "kill": 2.0,
        "assert": 1.0,
        "wrk_load": 0.0,  # uses its own duration_secs
        "parallel": 30.0,
    }
    # GPU multiplier: real hardware is slower than mock for cold starts
    gpu_multipliers: dict[str, float] = {
        "2xRTX_A4000": 3.0,
        "1xRTX_A4000": 4.0,
        "2xRTX_3090": 2.5,
        "1xA100_40GB": 1.5,
    }
    multiplier = gpu_multipliers.get(gpu_profile, 3.0)

    estimated_secs = 0.0
    for step in scenario_obj.steps:
        for action in step.actions:
            base = action_time.get(action.type, 5.0)
            # Use the action's own duration if it specifies one
            if action.type == "wait":
                base = action.args.get("duration_secs", 1.0)
            elif action.type == "wrk_load":
                base = action.args.get("duration_secs", 60.0)
            elif action.type == "wait_for":
                base = min(action.args.get("timeout_secs", 60.0), 60.0)
            estimated_secs += base

    # Apply GPU multiplier for cold-start-heavy actions
    estimated_secs *= multiplier
    # Add bootstrap overhead (15 min for first run)
    bootstrap_secs = 15 * 60
    total_secs = estimated_secs + bootstrap_secs
    hours = total_secs / 3600
    cost_gbp = hours * hourly_rate_gbp

    typer.echo(f"Scenario: {scenario_obj.name} v{scenario_obj.version}")
    typer.echo(f"Steps: {len(scenario_obj.steps)}")
    typer.echo(f"GPU profile: {gpu_profile} (multiplier: {multiplier:.1f}x)")
    typer.echo(f"Estimated scenario time: {estimated_secs / 60:.1f} min")
    typer.echo(f"Estimated total (incl. bootstrap): {total_secs / 60:.1f} min")
    typer.echo(f"Estimated cost: £{cost_gbp:.2f} (at £{hourly_rate_gbp:.2f}/hr)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_concerto_binary() -> Optional[Path]:
    """Locate the concerto binary from a cargo build.

    Searches debug then release target directories, walking up from cwd
    to find the repo root.
    """
    for parent in [Path.cwd(), *Path.cwd().parents]:
        for profile in ("debug", "release"):
            candidate = parent / "target" / profile / "concerto"
            if candidate.is_file():
                return candidate
        # Stop at repo root
        if (parent / "Cargo.toml").is_file():
            break
    return None


def _find_fixtures_dir() -> Path:
    """Locate the fixtures/ directory relative to this package."""
    # Walk up from the package source to find tools/bench/fixtures/
    pkg_dir = Path(__file__).resolve().parent
    for parent in [pkg_dir, *pkg_dir.parents]:
        candidate = parent / "fixtures"
        if candidate.is_dir() and (candidate / "weight-checksums.json").exists():
            return candidate
        # Also check tools/bench/fixtures from the repo root
        if (parent / "Cargo.toml").is_file():
            bench_fixtures = parent / "tools" / "bench" / "fixtures"
            if bench_fixtures.is_dir():
                return bench_fixtures
            break
    return pkg_dir.parent.parent.parent / "fixtures"


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Module-level ``main`` used by the console script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m`
    sys.exit(app())
