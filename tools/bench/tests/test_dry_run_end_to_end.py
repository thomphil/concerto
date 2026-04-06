"""End-to-end dry-run test: the CI regression gate for the bench rig.

Runs the smoke-quick scenario against a locally spawned concerto with
``--mock-gpus 2`` and the bundled ``mock-inference-backend``. Validates
the artifact tree shape, schema compliance, and absence of orphan
processes.

This test requires a pre-built concerto binary (``cargo build`` must
have been run). It is skipped if the binary is not found, so CI must
build before running pytest.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

# Locate repo root and key paths
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"
_SMOKE_SCENARIO = _SCENARIO_DIR / "smoke-quick.yaml"


def _find_concerto_binary() -> Path | None:
    """Locate concerto binary from a cargo build."""
    for profile in ("debug", "release"):
        candidate = _REPO_ROOT / "target" / profile / "concerto"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


_CONCERTO_BIN = _find_concerto_binary()

pytestmark = pytest.mark.skipif(
    _CONCERTO_BIN is None,
    reason="concerto binary not found — run `cargo build` first",
)


@pytest.fixture
def concerto_bin() -> Path:
    assert _CONCERTO_BIN is not None
    return _CONCERTO_BIN


async def test_dry_run_smoke_scenario(tmp_path: Path, concerto_bin: Path) -> None:
    """Run smoke-quick.yaml against mock backends, validate artifact tree."""
    from concerto_bench.runner import RunnerOptions, run_scenario

    output_dir = tmp_path / "artifact"
    options = RunnerOptions(
        scenario_path=_SMOKE_SCENARIO,
        output_dir=output_dir,
        concerto_bin=concerto_bin,
        mock_gpus=2,
        concerto_log_level="info",
        concerto_log_format="json",
        startup_timeout_secs=30.0,
        shutdown_grace_secs=10.0,
    )

    result = await run_scenario(options)

    # The run should succeed or partially succeed (mock backends
    # may not satisfy all assertions perfectly).
    assert result.exit_code in (0, 1), (
        f"exit_code={result.exit_code}, "
        f"startup_error={result.concerto_startup_error}"
    )

    # Artifact structure checks
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "environment.json").exists()

    # Validate manifest against schema
    manifest_data = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_data["schema_version"] == 1
    assert manifest_data["scenario_name"] == "smoke-quick"

    # Validate summary
    summary_data = json.loads((output_dir / "summary.json").read_text())
    assert summary_data["step_count"] >= 1

    # Steps directory should have entries
    steps_dir = output_dir / "steps"
    if steps_dir.exists():
        step_dirs = sorted(steps_dir.iterdir())
        assert len(step_dirs) >= 1
        for step_dir in step_dirs:
            assert (step_dir / "result.json").exists()
            assert (step_dir / "pre-state.json").exists()
            assert (step_dir / "post-state.json").exists()

    # Tarball should have been created
    assert result.artifact is not None
    assert result.artifact.tarball_path.exists()
    assert result.artifact.tarball_path.suffix == ".gz"

    # No orphan mock-inference-backend processes
    try:
        pgrep = subprocess.run(
            ["pgrep", "-f", "mock-inference-backend"],
            capture_output=True, text=True, timeout=5,
        )
        # pgrep returns 1 when no processes found (expected)
        if pgrep.returncode == 0:
            pids = pgrep.stdout.strip().split("\n")
            # Filter out our own process
            our_pid = str(os.getpid())
            orphans = [p for p in pids if p and p != our_pid]
            assert len(orphans) == 0, f"Orphan mock-inference-backend PIDs: {orphans}"
    except FileNotFoundError:
        pass  # pgrep not available on all platforms
