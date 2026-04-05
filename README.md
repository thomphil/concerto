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

- [x] Routing core with LRU / LFU / size-weighted eviction
- [x] GPU telemetry (NVML + mock) with health classification
- [x] Backend process management for vLLM, llama.cpp, SGLang
- [x] TOML configuration
- [x] OpenAI-compatible HTTP API with SSE streaming
- [x] CLI binary with graceful shutdown
- [x] Prometheus metrics endpoint
- [x] End-to-end integration scenarios
- [ ] Real-hardware validation on rented GPUs (in flight)
- [ ] Published container image and tagged v0.1.0 release

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
