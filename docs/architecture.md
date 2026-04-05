# Concerto Architecture

This document is a detailed architectural reference for Concerto, a Rust-based
inference multiplexer for self-hosted LLMs. It expands on the summary in
`CLAUDE.md` and is intended for contributors working on the codebase.

Concerto sits in front of 1-8 GPUs on a single node and manages multiple AI
models on shared hardware. It does not implement inference itself — it
orchestrates existing engines such as vLLM and llama.cpp by loading,
unloading, and routing requests to them based on demand and GPU health.

## High-Level Architecture

```
[Client requests — OpenAI-compatible API]
       │
       ▼
   ┌──────────────┐
   │ concerto-api  │  ← Rust HTTP server (axum + tokio)
   │              │  ← Single OpenAI-compatible endpoint
   └──────┬───────┘
          │
   ┌──────┴───────┐
   │ concerto-core │  ← Pure routing logic, eviction, memory accounting
   │              │  ← NO IO, NO side effects, 100% unit testable
   └──────┬───────┘
          │
   ┌──────┴──────────────────────────┐
   │                                 │
   ▼                                 ▼
┌───────────────┐           ┌───────────────┐
│ concerto-backend│          │ concerto-gpu    │
│ Process mgmt  │          │ GPU telemetry  │
│ for inference │          │ via NVML or    │
│ engines       │          │ mock           │
└───────────────┘          └───────────────┘
```

The hard split between pure logic (`concerto-core`) and side-effectful I/O
(`concerto-api`, `concerto-backend`, `concerto-gpu`) is the single most important
architectural decision. Every routing, eviction, and placement rule is a pure
function on in-memory state, which makes the interesting logic trivially
testable without GPUs, processes, or sockets.

## Crate Responsibilities

### concerto-core

Pure, side-effect-free domain logic. This crate holds the definitions of
`ClusterState`, `GpuState`, `LoadedModel`, `RoutingDecision`, `RoutingConfig`,
and the functions that operate on them. It has no dependency on tokio, axum,
reqwest, NVML, or the standard library's process APIs.

Responsibilities:

- Model the cluster as data: GPUs, their memory, temperature, health, and the
  models currently resident on each GPU.
- Decide where to route an incoming request via the `route_request` function.
- Decide which models to evict when memory is tight, per the configured
  eviction policy.
- Classify GPU health from telemetry snapshots.
- Perform memory-aware bin packing of models to GPUs.

### concerto-config

Parses `concerto.toml` into strongly typed configuration structs and validates
it. Human-readable fields such as `vram_required = "14GB"` are converted into
byte-exact numeric values using the `bytesize` crate.

Responsibilities:

- Define serde-derived structs that mirror the TOML schema.
- Parse and validate configuration at startup (fail fast on bad inputs).
- Expose a `ModelRegistry` lookup keyed by `ModelId`.
- Normalise values: expand `~`, canonicalise paths, coerce sizes into `Bytes`.

### concerto-gpu

Defines the `GpuMonitor` trait and ships two implementations: `NvmlMonitor`
(feature-gated behind the `nvml` feature) and `MockGpuMonitor`. The trait
returns `GpuSnapshot` values that `concerto-core` consumes to update cluster
state.

Responsibilities:

- Abstract all GPU telemetry behind `GpuMonitor`.
- Provide a configurable mock for unit and integration tests.
- Keep NVML bindings strictly optional so the project builds on any machine.

### concerto-backend

Defines the `BackendManager` trait and ships two implementations:
`ProcessBackendManager` (spawns real vLLM and llama.cpp processes) and
`MockBackendManager` (spawns the mock HTTP backend used in tests). It owns the
lifecycle of every child process and maps `BackendHandle`s to live processes.

Responsibilities:

- Launch inference engines on specific GPUs with model-specific arguments.
- Supervise child processes: detect crashes, kill on shutdown, tear down on
  eviction.
- Poll `GET /health` on each backend on a configurable interval.
- Expose a port allocator so launched backends do not collide.

### concerto-api

The HTTP front door. Built on axum and tokio, it exposes the OpenAI-compatible
API that clients talk to and proxies traffic to the appropriate backend after
consulting `concerto-core`.

Responsibilities:

- Implement the HTTP surface: `/v1/chat/completions`, `/v1/models`,
  `/health`, `/metrics`, `/status`.
- Stream Server-Sent Events from the backend back to the client unchanged.
- Maintain shared state (`Arc<ClusterState>` and friends) across handlers.
- Orchestrate the load-then-route flow by calling `concerto-core` and then
  `concerto-backend`.

### concerto-cli

The binary entry point. A thin shell built on clap that parses command-line
arguments, loads the configuration via `concerto-config`, wires up the chosen
implementations of `GpuMonitor` and `BackendManager`, and starts the
`concerto-api` server.

Responsibilities:

- Parse CLI arguments (`--config`, `--mock-gpus`, `--log-level`, etc.).
- Select mock or real implementations based on flags and features.
- Initialise the `tracing` subscriber.
- Install signal handlers for graceful shutdown.

## Request Data Flow

The canonical happy path for a chat completion request:

1. The client sends `POST /v1/chat/completions` with `"model": "qwen2.5-7b"`.
2. `concerto-api` deserialises the request and extracts the model id.
3. `concerto-api` reads the current `ClusterState` snapshot and calls
   `concerto_core::route_request(&model_id, &state, &config)`.
4. The router returns a `RoutingDecision`:
   - `RouteToLoaded { gpu_id, port }` if the model is already resident.
   - `LoadModel { gpu_id, evict }` if the model must be brought up first.
   - `Reject { reason }` if no GPU can serve the request.
5. On `RouteToLoaded`, `concerto-api` forwards the request body to
   `http://127.0.0.1:{port}/v1/chat/completions` via reqwest, streaming the
   response back to the client.
6. On `LoadModel`, `concerto-api` asks `concerto-backend` to stop each model in
   `evict` and then launch the new one on `gpu_id`. It waits for
   `health_check` to succeed (bounded by `cold_start_timeout_secs`), updates
   the `ClusterState`, and then falls through to the routing step above.
7. On `Reject`, `concerto-api` returns `503 Service Unavailable` with the
   provided reason.

For streaming requests, step 5 opens a reqwest stream and pipes chunks
directly into an axum `Sse` response. The proxy layer is content-agnostic —
it never parses SSE events, only forwards byte chunks.

## Testing Strategy

Concerto is developed without GPU hardware. The testing pyramid is:

1. **Unit tests (inside `concerto-core`).** Every routing rule, eviction
   policy, and memory accounting edge case is exercised with direct calls to
   the relevant pure function. These tests need no runtime, no sockets, and
   no mocks because `concerto-core` has no external dependencies to mock.
2. **Integration tests (`tests/scenarios/`).** A harness (`TestEnv`) spins up
   a full `concerto-api` server with `MockGpuMonitor` and `MockBackendManager`,
   fires real HTTP requests, and asserts behaviour end-to-end. Each scenario
   tests a user-observable story: cold start, eviction under pressure,
   backend crash recovery.
3. **Property tests (`tests/proptest/`).** Random sequences of routing events
   are applied to a state machine, and invariants such as "GPU memory is
   never overcommitted" and "a `RouteToLoaded` decision always names a GPU
   that currently hosts the requested model" are asserted.
4. **Mock inference backend (`tests/mock_backend/`).** A ~100-line axum app
   that speaks enough of the OpenAI API to satisfy the proxy layer. It
   supports configurable latency, canned responses, SSE streaming, a
   `--startup-delay` flag to simulate model load time, and on-demand failure
   injection.
5. **Real hardware validation.** Deferred to periodic Vast.ai or RunPod
   sessions on rented GPUs; not part of the CI cycle.

## Process Lifecycle

A backend process moves through the following states:

1. **Spawning.** `BackendManager::launch` fork-execs the engine binary with
   model-specific arguments and a pre-allocated free port. The process is
   registered in the manager's handle table immediately.
2. **Starting.** The manager polls `GET /health` until it returns 200 or the
   `cold_start_timeout_secs` timer expires. During this window the backend is
   considered "loading" and no traffic is routed to it.
3. **Ready.** The `ClusterState` is updated to reflect that the model is
   resident on a specific GPU and accepting traffic. The API layer may now
   forward requests.
4. **Degraded.** If a health check starts failing, the manager marks the
   backend as degraded and the router stops selecting it for new requests.
   In-flight requests are allowed to complete.
5. **Stopping.** On eviction or shutdown, the manager sends SIGTERM, waits
   for a grace period, then sends SIGKILL if still running. The handle is
   removed from the table and the GPU's memory accounting is updated.
6. **Crashed.** If a process exits unexpectedly, the supervisor observes the
   wait-status, logs the failure, removes the handle, and updates cluster
   state so the router no longer considers the model loaded.

## Eviction Policies

When a request targets a model that is not resident and no GPU has enough
free memory, the router must choose victims to evict. The policy is
configurable via `routing.eviction_policy`:

- **LRU (least recently used).** The default. Each `LoadedModel` tracks a
  `last_request_at` `Instant`. Victims are chosen in ascending order of this
  timestamp until enough memory is freed. Best when request patterns have
  good temporal locality.
- **LFU (least frequently used).** Victims are chosen in ascending order of
  `request_count`. Better than LRU for workloads with a stable "hot set" that
  must not be disturbed by occasional cold requests. Vulnerable to
  cache-pollution by historically hot models that are no longer used.
- **Size-weighted LRU.** Combines recency with footprint: victims are scored
  as `age * vram_usage`, and the highest-scoring models are evicted first.
  Tends to favour keeping small models resident even when they are slightly
  stale, which minimises future reload cost.

All policies share a precondition: the total freed VRAM must be sufficient
to fit the incoming model, and the router must never evict a model that is
currently serving in-flight requests.

## GPU Health Classification

`concerto-core` classifies each GPU into one of three bands based on the
latest `GpuSnapshot`:

- **Healthy.** Temperature below the configured `max_temperature`, zero
  uncorrected ECC errors since startup, and utilisation within normal bounds.
  The router is free to place new models here.
- **Degraded.** Temperature above `max_temperature` but below a hard cutoff,
  or a non-zero but stable ECC error count. The router will continue to serve
  requests from models already resident, but will not place new models on
  this GPU until it recovers.
- **Unhealthy.** Temperature above the hard cutoff, rapidly increasing ECC
  errors, or the GPU has disappeared from telemetry snapshots. The router
  refuses to place work here and will attempt to evacuate resident models
  onto other GPUs if possible.

Thresholds are configurable per GPU via the `[[gpus]]` table in
`concerto.toml`.

## Configuration Model

Configuration is parsed from `concerto.toml` (see `concerto.example.toml` for a
reference) into the following internal types:

- `[server]` maps to a `ServerConfig` struct containing `host` and `port`.
- `[routing]` maps to a `RoutingConfig` containing the eviction policy,
  cold-start timeout, and health check interval.
- Each `[[models]]` entry maps to a `ModelSpec`, with `vram_required` parsed
  from human-readable strings like `"14GB"` into a `Bytes` newtype.
- Each `[[gpus]]` entry maps to a `GpuConfig` containing the numeric id and
  any per-GPU overrides (for example, `max_temperature`).

All configuration is validated at startup. Missing weight paths, unknown
engine names, duplicate model ids, and GPU ids that do not exist on the
machine are rejected before the server begins listening.

## Failure Modes and Recovery

Concerto is expected to survive common failure modes without operator
intervention:

- **Backend crash.** The supervisor in `concerto-backend` detects the exit,
  removes the handle, and updates cluster state. Subsequent requests for the
  same model trigger a normal cold start.
- **Backend hang.** Health checks time out. The backend is marked degraded,
  no new traffic is routed to it, and after a grace window it is SIGKILLed
  and restarted.
- **GPU disappearance.** `NvmlMonitor` returns a shrunken GPU list. The
  missing GPU is marked `Unhealthy`, resident models are considered lost, and
  the router will re-load them on surviving GPUs if capacity allows.
- **Request timeout.** The client can be served a 504 while the backend work
  continues in the background; no cluster state changes.
- **Config reload failure.** Invalid reloaded configuration is rejected and
  the previously validated configuration remains in effect; the reload
  attempt is logged at warning level.
- **Evict-then-load race.** The router serialises evict-then-load sequences
  for a given GPU so that two concurrent requests cannot both try to place a
  new model on the same partially-freed GPU.

## Concurrency Model

Concerto is built on tokio from top to bottom. The general pattern:

- A single multi-threaded tokio runtime hosts all async work.
- Shared cluster state is stored in an `Arc<RwLock<ClusterState>>`. Reads
  vastly outnumber writes (every request reads state; only loads, evictions,
  and periodic telemetry ticks write), so a read-write lock is a better fit
  than a mutex.
- Fast-changing keyed state, such as the map from `ModelId` to
  `BackendHandle`, is stored in an `Arc<DashMap<_, _>>` so that lookups can
  proceed without coordinating with unrelated writers.
- Per-model "load in progress" serialisation is implemented with a
  `Mutex<HashMap<ModelId, Arc<Notify>>>` so that concurrent requests for a
  cold model share a single load attempt rather than racing.
- The telemetry poller and the health checker each run on their own
  `tokio::spawn` tasks with jittered intervals.
- Graceful shutdown is driven by a `tokio::sync::Notify`: a SIGTERM handler
  flips the notify, running tasks drain in-flight requests, and the main
  task awaits all backends stopping before exiting.

`Arc<T>` is preferred over `Rc<T>` throughout, and there are no `unsafe`
blocks anywhere in the codebase.

## Non-Goals for MVP

The following are explicitly out of scope for the first release and should
not influence architectural decisions:

- Multi-node clustering. Concerto is single-node for the MVP.
- Implementing our own inference engine. Concerto orchestrates existing
  engines and will never contain CUDA kernels or ML code.
- A web UI. The product surface is the CLI and the HTTP API.
- Authentication and authorization. These will be added in a later phase.
- Model downloading. Operators pre-stage weights on local disk.
- Quantisation or model conversion. Out of scope entirely.
