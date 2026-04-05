"""Scenario action primitives: the smallest useful units the runner invokes.

Every scenario YAML step lists an ``actions:`` array; each entry in that
array names a primitive and supplies its arguments. This package hosts
the strictly-typed primitive definitions that the runner (Phase B.2
step 7) wires together.

Phase B.2 step 5 ships the two smallest useful action primitives:

* :class:`RequestPrimitive` — POSTs ``/v1/chat/completions`` with full
  connect / TTFB / total timing capture.
* :class:`SnapshotPrimitive` — captures ``GET /status`` + ``nvidia-smi``
  + ``pgrep`` in parallel into a :class:`~concerto_bench.schema.StateSnapshot`.

Later Phase B.2 steps will add ``wait``, ``wait_for``, ``kill``,
``parallel``, ``assertions``, and ``wrk_load`` primitives. All primitives
share the same broad contract: stateless execution, strict pydantic
``*Action`` argument model, and a rich result that the runner wraps in a
:class:`~concerto_bench.schema.ActionRecord`.
"""

from __future__ import annotations

from concerto_bench.primitives.request import (
    RequestAction,
    RequestError,
    RequestPrimitive,
)
from concerto_bench.primitives.snapshot import (
    SnapshotAction,
    SnapshotError,
    SnapshotPrimitive,
)

__all__ = [
    "RequestAction",
    "RequestError",
    "RequestPrimitive",
    "SnapshotAction",
    "SnapshotError",
    "SnapshotPrimitive",
]
