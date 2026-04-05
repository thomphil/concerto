"""Typer CLI surface for the Concerto bench rig.

Phase B.1 scope: define the full subcommand list with the option
surfaces documented in SPRINT-2-PLAN §4 B.1 / B.2, but leave each
command body as a stub that exits with a clear ``not implemented``
message. This gives subsequent Phase B steps a stable entry point to
hang real behaviour off without having to reshape the CLI.

Subcommand stubs are intentionally not ``NotImplementedError`` raises —
they return exit code 2 (``EX_USAGE``-adjacent) with a message naming
the Phase B sub-step that will land the behaviour. This keeps
``--help`` honest while making accidental invocations obviously
incomplete rather than crashing with a traceback.
"""

from __future__ import annotations

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


# Exit code used by stubs to signal "this command is declared but not
# yet implemented". Distinct from 0 (success) and 1 (usage/runtime
# error) so CI can tell the difference if we ever accidentally wire a
# stub into a real code path.
_EXIT_NOT_IMPLEMENTED = 64


def _stub(command: str, phase: str) -> None:
    typer.echo(
        f"concerto-bench {command}: not implemented yet (lands in {phase}).",
        err=True,
    )
    raise typer.Exit(code=_EXIT_NOT_IMPLEMENTED)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"concerto-bench {__version__}")
        raise typer.Exit()


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


@app.command("run")
def run_cmd(
    scenario: Path = typer.Option(
        ...,
        "--scenario",
        exists=False,  # validation happens once the runner lands
        help="Path to the scenario YAML to execute.",
    ),
    concerto_bin: Path = typer.Option(
        ...,
        "--concerto-bin",
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
    _stub("run", "Phase B.10")


@app.command("dry-run")
def dry_run_cmd(
    scenario: Path = typer.Option(
        ...,
        "--scenario",
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
) -> None:
    """Execute a scenario locally against mock-inference-backend.

    This is the CI regression gate — it spawns a local concerto with
    ``--mock-gpus`` pointed at the bundled mock upstream, runs the
    scenario, and produces an artifact. The tarball shape is what the
    CI test asserts on.
    """
    _stub("dry-run", "Phase B.12")


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
    _stub("summarize", "Phase B.11")


@app.command("verify-weights")
def verify_weights_cmd(
    models_dir: Path = typer.Option(
        ...,
        "--models-dir",
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
    _stub("verify-weights", "Phase B.15")


@app.command("estimate")
def estimate_cmd(
    scenario: Path = typer.Option(
        ...,
        "--scenario",
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

    Local-only; does not hit the Vast.ai API. Heuristics live in the
    rig and are refined from measured dry-run wall times.
    """
    _stub("estimate", "Phase B.16")


def main() -> None:
    """Module-level ``main`` used by the console script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m`
    sys.exit(app())
