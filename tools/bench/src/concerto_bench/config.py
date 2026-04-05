"""Rig-level runtime configuration models.

This is the configuration of *the rig itself* as it runs — things like
where the concerto binary lives, where to put artifacts, how verbose to
be. Scenario YAML parsing and its own schema live in ``schema.py``
(added in a later Phase B step).

Phase B.1 ships the minimal shape so later steps have a stable anchor.
Fields will be added as the primitives + runner land.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LogLevel = Literal["debug", "info", "warn", "error"]


class RigConfig(BaseModel):
    """Top-level runtime configuration for a single bench rig invocation.

    Populated from CLI arguments (and, later, an optional TOML override
    file). Validation happens once at startup — if this model fails to
    build, the rig refuses to proceed rather than running partially
    configured.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_path: Path = Field(
        ...,
        description="Path to the scenario YAML to execute.",
    )
    output_dir: Path = Field(
        ...,
        description=(
            "Directory where the on-disk artifact tree is built and, at "
            "the end of the run, tar.gz'd."
        ),
    )
    concerto_bin: Optional[Path] = Field(
        default=None,
        description=(
            "Path to the concerto binary. Required for ``run``; ``dry-run`` "
            "will build or locate one on demand."
        ),
    )
    models_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Directory containing pre-downloaded model weights. Scenario "
            "``models`` entries are resolved relative to this path."
        ),
    )
    log_level: LogLevel = Field(
        default="info",
        description="Structured-log verbosity for the rig's own output.",
    )
