"""Smoke tests for the Phase B.1 CLI skeleton.

These tests verify only the scaffolding: that the package imports,
the typer app loads, ``--help`` lists every declared subcommand, and
each stub exits with the documented ``not implemented`` exit code when
invoked. Real behavioural tests land alongside the implementations in
later Phase B sub-steps.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from concerto_bench import __version__
from concerto_bench.cli import _EXIT_NOT_IMPLEMENTED, app

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


@pytest.mark.parametrize(
    ("subcommand", "extra_args"),
    [
        (
            "run",
            [
                "--scenario",
                "scenario.yaml",
                "--concerto-bin",
                "/tmp/concerto",
                "--output",
                "/tmp/out",
            ],
        ),
        (
            "dry-run",
            ["--scenario", "scenario.yaml", "--output", "/tmp/out"],
        ),
        ("summarize", ["/tmp/run.tar.gz"]),
        ("verify-weights", ["--models-dir", "/tmp/models"]),
        ("estimate", ["--scenario", "scenario.yaml"]),
    ],
)
def test_stub_commands_exit_with_not_implemented_code(
    subcommand: str, extra_args: list[str]
) -> None:
    result = runner.invoke(app, [subcommand, *extra_args])
    assert result.exit_code == _EXIT_NOT_IMPLEMENTED, result.output
    # The error message is written to stderr; CliRunner captures both in
    # result.output unless mix_stderr=False is passed, which it is not.
    assert "not implemented" in result.output.lower()


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
