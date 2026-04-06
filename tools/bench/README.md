# concerto-bench

Benchmark and validation rig for Concerto. Drives the real production
`concerto` binary through scripted scenarios, captures comprehensive
telemetry, and produces structured, versioned artifact tarballs.

Used both locally against `mock-inference-backend` (as a CI regression
gate) and against real vLLM on rented GPUs (Sprint 2 validation runs).

## Quick Start

### Prerequisites

- Python 3.10+
- A built concerto binary (`cargo build` from the repo root)

### Install

```sh
pip install -e tools/bench
```

### Local dry-run (against mock backends)

```sh
python -m concerto_bench dry-run \
    --scenario tools/bench/scenarios/smoke-quick.yaml \
    --output /tmp/bench-dry-run
```

### Full validation run (against pre-built binary)

```sh
python -m concerto_bench run \
    --scenario tools/bench/scenarios/sprint-2-validation.yaml \
    --concerto-bin ./target/release/concerto \
    --models-dir /root/models \
    --output /tmp/bench-run
```

## Subcommands

### `run`

Execute a scenario against a pre-built concerto binary. This is the
Vast.ai path: concerto is already built, weights pre-downloaded, and
the rig drives the scenario end-to-end.

```sh
python -m concerto_bench run \
    --scenario <path.yaml> \
    --concerto-bin <path> \
    --output <dir> \
    [--models-dir <dir>] \
    [--concerto-config <path>] \
    [--log-level debug|info|warn|error]
```

### `dry-run`

Execute a scenario locally against `mock-inference-backend`. This is
the CI path. The rig locates a `concerto` binary from `target/debug`
or `target/release` automatically and spawns it with `--mock-gpus`.

```sh
python -m concerto_bench dry-run \
    --scenario <path.yaml> \
    --output <dir> \
    [--mock-gpus 2] \
    [--log-level debug|info|warn|error]
```

### `summarize`

Render an artifact tarball (or unpacked directory) into human-readable
markdown. Typical target is `docs/benchmarks.md`.

```sh
python -m concerto_bench summarize /tmp/bench-run.tar.gz --output docs/benchmarks.md
# or print to stdout:
python -m concerto_bench summarize /tmp/bench-run.tar.gz
```

### `verify-weights`

Verify SHA-256 checksums of model weight files against a pinned
manifest. Guards against silent HuggingFace re-uploads drifting
numbers between runs.

```sh
python -m concerto_bench verify-weights \
    --models-dir /root/models \
    [--checksums tools/bench/fixtures/weight-checksums.json]
```

### `estimate`

Forecast wall-clock time and rental cost for a scenario on a given GPU
profile. Local-only; does not hit the Vast.ai API.

```sh
python -m concerto_bench estimate \
    --scenario tools/bench/scenarios/sprint-2-validation.yaml \
    [--gpu-profile 2xRTX_A4000] \
    [--hourly-rate-gbp 0.80]
```

## Scenarios

Scenarios are YAML files in `scenarios/`. Two are included:

| Scenario | Steps | Target time | Purpose |
|---|---|---|---|
| `sprint-2-validation.yaml` | 8 | ~15 min on GPU, ~3 min mock | Full lifecycle: cold start, eviction, concurrent load, crash recovery, orphan check, shutdown |
| `smoke-quick.yaml` | 3 | ~30 sec mock, ~60 sec GPU | Quick sanity check for rig plumbing |

### Writing custom scenarios

A scenario YAML has five top-level keys:

```yaml
name: my-scenario
version: "1"
description: What this scenario validates.

parameters:
  concurrent_clients: 10
  cold_start_timeout_secs: 120

models:
  - id: qwen2.5-0.5b
    name: "Qwen 2.5 0.5B"
    weight_path: "/models/qwen2.5-0.5b"
    vram_required: "2GB"
    engine: vllm
    engine_args: ["--dtype", "float16"]

samplers:
  - name: concerto-status
    interval_secs: 1.0
  - name: pgrep-count
    interval_secs: 1.0
    config:
      patterns: ["vllm", "mock-inference-backend"]

steps:
  - name: load-model
    description: Cold-start the model and send a request.
    actions:
      - request:
          model: qwen2.5-0.5b
          content: "hello"
          capture_as: smoke
      - snapshot: {}
      - assert:
          condition: backend_count
          operator: gte
          value: 1

exit_criteria:
  launched_count_gte: 1

continue_on_failure: true
```

**Supported action types:** `request`, `snapshot`, `wait`, `wait_for`,
`kill`, `assert`, `wrk_load`, `parallel`.

**Available samplers:** `nvidia-smi`, `concerto-status`,
`concerto-metrics`, `pgrep-count`, `proc-stats`. All run at the
configured interval and write JSONL into the artifact's `telemetry/`
directory.

The runner validates the YAML schema at load time. Unknown action types
or extra fields fail fast with a diagnostic error.

## Artifact Structure

Every run produces a tarball with this layout:

```
<run-name>/
  manifest.json              # Run identity, versions, timing, exit status
  summary.json               # Pre-computed metrics roll-up for analysis
  environment.json           # Host info: GPU, CPU, driver, software versions
  concerto-stdout.log        # Concerto's structured log output
  concerto-stderr.log        # Concerto's stderr
  steps/
    01-single-model-smoke/
      pre-state.json         # StateSnapshot before the step
      post-state.json        # StateSnapshot after the step
      result.json            # StepResult with pass/fail, timing, actions
      request-smoke.json     # RequestRecord per captured request
    02-multi-model-cold-start/
      ...
  telemetry/
    nvidia-smi.jsonl         # 1 Hz GPU memory/utilisation samples
    concerto-status.jsonl    # 1 Hz /status endpoint snapshots
    concerto-metrics.jsonl   # 1 Hz /metrics endpoint scrapes
    pgrep-count.jsonl        # 1 Hz process-count samples
    proc-stats.jsonl         # 1 Hz process CPU/RSS samples
```

A sibling `.tar.gz.sha256` checksum file is generated alongside the
tarball. Both the schema version (JSON shape) and the artifact tree
version (file layout) are pinned in `manifest.json` so they can evolve
independently.

Tarballs are reproducible: member ordering is sorted-by-path and mtimes
are pinned to `manifest.started_at`.

## Running on Vast.ai

See `SPRINT-2-PLAN.md` Phase C for the full remote-execution workflow.
The short version:

1. Rent a 2x RTX A4000 (16 GB each) instance.
2. Clone the repo, `cargo build --release`, download model weights.
3. `pip install tools/bench` on the instance.
4. `concerto-bench verify-weights --models-dir /root/models`
5. `concerto-bench run --scenario tools/bench/scenarios/sprint-2-validation.yaml --concerto-bin ./target/release/concerto --models-dir /root/models --output /tmp/bench-run`
6. `scp` the tarball back, then `concerto-bench summarize` locally.

Use `concerto-bench estimate` beforehand to forecast wall time and cost.

## Development

```sh
pip install -e 'tools/bench[dev]'
pytest tools/bench/tests/
```

The test suite covers the full stack: schema round-trip, artifact
builder, primitives, samplers, runner, CLI smoke tests, and summarizer.

## FAQ

**Q: Why Python, not Rust?**
A: The rig interacts with concerto exclusively over HTTP. Python's
ecosystem (httpx, pydantic, typer, rich) is faster to iterate on for
tooling that is not on the hot path. The concerto binary under test is
always real Rust.

**Q: Can I add a new scenario?**
A: Yes. Copy `smoke-quick.yaml`, add steps. The runner validates the
YAML schema at load time -- unknown action types or extra fields fail
fast with a diagnostic error.

**Q: Where do artifacts go?**
A: `bench-artifacts/` at the repo root (gitignored). The committed
deliverable is `docs/benchmarks.md`, produced by the `summarize`
subcommand.

**Q: What does "partial failure" mean?**
A: The run completed end-to-end but at least one step failed or one
exit criterion was not met. The artifact is still fully populated and
valid for analysis. The rig never silently drops data.
