# concerto-bench

Benchmark and validation rig for the [Concerto](https://github.com/thomphil/concerto)
inference multiplexer.

`concerto-bench` drives the real production `concerto` binary through
data-driven scenarios, captures excessive telemetry (nvidia-smi,
`/status`, `/metrics`, process counts, raw request/response bodies),
and produces a versioned tarball artifact for offline analysis. It is
used both locally against `mock-inference-backend` (as a CI regression
gate) and against real vLLM on rented GPUs (Sprint 2 validation runs).

## Status

**Phase B.1 — skeleton only.** Subcommands are declared and document
themselves through `--help`, but all command bodies are stubs that
exit with code 64 (`not implemented`) and name the Phase B sub-step
that will land the behaviour. See the sprint plan for the full
implementation order.

## Install (editable)

```sh
cd tools/bench
pip install -e .
# ...or, with the dev dependencies (pytest etc.)
pip install -e '.[dev]'
```

Python 3.10 or newer is required.

## Usage

```sh
python -m concerto_bench --help
# or, after install, the console script:
concerto-bench --help
```

Subcommands (all currently stubbed):

| Command | Purpose |
|---|---|
| `run` | Execute a scenario against a pre-built `concerto` binary (Vast.ai path). |
| `dry-run` | Execute a scenario locally against `mock-inference-backend` (CI path). |
| `summarize` | Render an artifact tarball into a markdown summary. |
| `verify-weights` | Check SHA-256 of downloaded model weights against a pinned manifest. |
| `estimate` | Forecast wall-clock and cost for a scenario on a given GPU profile. |

## Development

```sh
cd tools/bench
pip install -e '.[dev]'
pytest
```

The test suite currently exercises the CLI skeleton only. It grows
with each Phase B sub-step; the eventual gate is an end-to-end
dry-run test that spawns a real `concerto --mock-gpus 2`, drives the
full Sprint 2 validation scenario, and asserts on the shape of the
produced tarball.
