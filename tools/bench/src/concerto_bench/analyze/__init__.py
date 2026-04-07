"""Post-run analysis tools for concerto-bench artifacts.

Re-exports the public surface of the analyze sub-package so callers
can write ``from concerto_bench.analyze import summarize_artifact``.
"""

from concerto_bench.analyze.diff import DiffError, diff_artifacts
from concerto_bench.analyze.summarize import SummarizeError, summarize_artifact

__all__ = ["summarize_artifact", "SummarizeError", "diff_artifacts", "DiffError"]
