"""Smoke tests for the CLI surface.

These tests verify that the typer app loads, ``--help`` lists every
declared subcommand, version flag works, and subcommands with missing
required args produce usage errors. Real behavioural tests for each
subcommand land alongside their implementations.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from concerto_bench import __version__
from concerto_bench.cli import app

runner = CliRunner()


EXPECTED_SUBCOMMANDS = (
    "run",
    "dry-run",
    "summarize",
    "verify-weights",
    "estimate",
)


def test_root_help_exits_cleanly() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("subcommand", EXPECTED_SUBCOMMANDS)
def test_help_lists_subcommand(subcommand: str) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert subcommand in result.output


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


@pytest.mark.parametrize("subcommand", EXPECTED_SUBCOMMANDS)
def test_subcommand_help_exits_cleanly(subcommand: str) -> None:
    """Each subcommand's --help should work."""
    result = runner.invoke(app, [subcommand, "--help"])
    assert result.exit_code == 0, result.output


def test_run_requires_scenario_and_bin() -> None:
    """run without required options exits with error."""
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_dry_run_requires_scenario() -> None:
    """dry-run without --scenario exits with error."""
    result = runner.invoke(app, ["dry-run"])
    assert result.exit_code != 0


def test_estimate_requires_scenario() -> None:
    """estimate without --scenario exits with error."""
    result = runner.invoke(app, ["estimate"])
    assert result.exit_code != 0


def test_rig_config_model_importable() -> None:
    """Phase B.1 only ships the skeleton, but the pydantic model must
    at least parse valid input and reject unknown fields."""
    from concerto_bench.config import RigConfig

    cfg = RigConfig(scenario_path="scenario.yaml", output_dir="/tmp/out")  # type: ignore[arg-type]
    assert cfg.log_level == "info"
    assert cfg.concerto_bin is None

    with pytest.raises(Exception):  # pydantic.ValidationError
        RigConfig(
            scenario_path="scenario.yaml",
            output_dir="/tmp/out",
            bogus_field="nope",  # type: ignore[call-arg]
        )
