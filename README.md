# concerto

**A Rust inference multiplexer for self-hosted LLMs.**

Concerto sits in front of 1–8 GPUs on a single node and orchestrates inference engine processes (vLLM, llama.cpp, SGLang) — loading, unloading, and routing requests automatically based on demand and GPU health. It does not replace inference engines; it manages them.

## Why

Running multiple self-hosted models today means one inference engine process per model, each permanently reserving VRAM. On a 2× GPU box with four models, 50–70% of VRAM sits idle holding weights nobody is currently using. Concerto dynamically loads, unloads, and places models across the GPUs you already have.

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
- [ ] OpenAI-compatible HTTP API with SSE streaming
- [ ] CLI binary with graceful shutdown
- [ ] Prometheus metrics endpoint
- [ ] End-to-end integration scenarios

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
