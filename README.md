# concerto

**A Rust inference multiplexer for self-hosted LLMs.**

Concerto sits in front of 1–8 GPUs on a single node and orchestrates inference engine processes (vLLM, llama.cpp, SGLang) — loading, unloading, and routing requests automatically based on demand and GPU health. It does not replace inference engines; it manages them.

## Why

Running multiple self-hosted models today means one inference engine process per model, each permanently reserving VRAM. On a 2× GPU box with four models, 50–70% of VRAM sits idle holding weights nobody is currently using. Concerto dynamically loads, unloads, and places models across the GPUs you already have.

## Good fit / bad fit

Concerto multiplexes multiple inference models onto fewer GPUs by loading and unloading them dynamically. The honest tradeoff: **the first request to a model after it goes idle eats a cold-start penalty** — 30–90 seconds for a 7B model on an RTX A4000, plus whatever your inference engine needs to warm its KV caches. In exchange, a 2-GPU box can serve 4–6 models that would otherwise need 4–6 dedicated GPUs.

**Good fit for:**

- Internal developer tools and staging environments where a ~minute of first-request latency is fine
- Batch pipelines and scheduled jobs that process work in chunks
- Multi-tenant fine-tune inference where each tenant gets their own weights
- Feature-gated AI — AI features used by a minority of users or behind paywalls, where keeping every model warm 24/7 is wasteful

**Not a good fit for:**

- Sub-second consumer chatbots where first-request latency is a visible UX failure
- Real-time SLAs and latency-sensitive production traffic
- Workloads where first-request latency must be predictable across every model

A v0.2 warm pool — keeping idle model processes resident in CPU RAM and resuming them to the GPU on demand — will reduce the 7B cold-start from ~60s to ~5–10s; see the roadmap.

## Features

- Pure-logic routing core: no I/O, takes cluster state in, returns decisions out
- Pluggable eviction policies: LRU, LFU, size-weighted LRU
- GPU health classification from temperature, utilisation, and ECC errors
- NVML-backed telemetry with a deterministic mock for tests (feature-gated)
- Process lifecycle management for vLLM, llama.cpp, and SGLang backends
- TOML configuration with a model registry and per-GPU overrides

## Roadmap

- [x] **Sprint 1** — runnable server: routing core, GPU telemetry (NVML + mock), backend process management (vLLM, llama.cpp, SGLang), TOML config, OpenAI-compatible HTTP API with SSE streaming, CLI binary with graceful shutdown, end-to-end integration scenarios
- [x] **Sprint 2** — production hardening: Prometheus `/metrics` endpoint, Python bench rig, real-hardware validation on 2× RTX A4000 (8/8 scenarios passing under sustained concurrent load)
- [ ] **Sprint 3** *(in progress)* — v0.1.0 release prep: orchestrator hardening, public quickstart and deployment docs, container image
- [ ] **Sprint 4** — launch: tagged v0.1.0, technical writeup, public announcement

## Architecture

```
          Client (OpenAI-compatible requests)
                      │
                      ▼
              ┌───────────────┐
              │ concerto-api  │  HTTP server, OpenAI-compatible surface
              └──────┬────────┘
                     │
              ┌──────┴────────┐
              │ concerto-core │  pure routing, eviction, memory accounting
              └──────┬────────┘
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
┌────────────────┐         ┌────────────────┐
│ concerto-      │         │ concerto-gpu   │
│ backend        │         │ NVML / mock    │
│ process mgmt   │         │ telemetry      │
└────────────────┘         └────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for detail.

## Install

Linux with NVIDIA GPUs (recommended):

```sh
cargo install concerto-cli --features nvml
concerto --config concerto.toml
```

Container:

```sh
docker run --gpus all -v $PWD/concerto.toml:/etc/concerto.toml -p 8080:8080 ghcr.io/thomphil/concerto:latest
```

For development on macOS or any host without NVIDIA GPUs, omit the `nvml` feature and run with `--mock-gpus N` for a self-contained dev mode. Full walkthrough — config layout, model registry, systemd unit, troubleshooting — in [`docs/quickstart.md`](docs/quickstart.md).

## Benchmarks

Validated end-to-end on 2× NVIDIA RTX A4000 (Vast.ai) running real vLLM backends across three models (qwen2.5-0.5b, phi-3-mini, qwen2.5-7b). The Sprint 2 validation scenario covers cold start, multi-model routing, LRU eviction, 5-minute sustained concurrent load (20 clients, 379/379 successful, 0% error rate), backend crash recovery via SIGKILL + auto re-launch, orphan detection, and graceful shutdown — 8/8 steps passed.

See [`docs/benchmarks.md`](docs/benchmarks.md) for the full run record, latencies, and configuration.

## Building

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```

NVML telemetry is feature-gated so default builds stay portable:

```sh
cargo build --features nvml
```

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.
