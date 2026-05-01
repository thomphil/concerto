//! End-to-end scenario harness.
//!
//! Each scenario spins up a full `concerto-api::serve` HTTP server on an
//! ephemeral loopback port, backed by:
//!
//! - `MockGpuMonitor::with_healthy_gpus(...)` for deterministic GPU state
//! - `ProcessBackendManager` — the real production backend manager —
//!   spawning the `mock-inference-backend` binary as a child process per
//!   model via `EngineType::Custom`
//!
//! This exercises the production code path end-to-end (HTTP → orchestrator
//! → process spawn → health probe → reverse proxy) the way ROADMAP §6.3
//! originally specified. It's slower than the old `MockBackendManager`-only
//! harness (real subprocess startup takes 200–800ms), so scenarios prefer
//! small numbers of backends and short chains.
//!
//! Each harness gets its own TCP port range via a global atomic counter, so
//! parallel `cargo test` runs don't collide on backend ports.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytesize::ByteSize;
use concerto_api::state_file::StateRecorder;
use concerto_api::{serve, AppState};
use concerto_backend::{
    BackendError, BackendHandle, BackendManager, PortAllocator, ProcessBackendManager,
};
use concerto_config::{
    ConcertoConfig, GpuConfigEntry, ModelConfigEntry, RoutingSection, ServerConfig,
};
use concerto_core::{ClusterState, EngineType, GpuHealth, GpuState, ModelSpec};
use concerto_gpu::{GpuMonitor, MockGpuMonitor};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;

/// Where a scenario can find the `mock-inference-backend` binary. Resolved
/// once per process: walks up from the current test exe (under
/// `target/debug/deps/`) to `target/debug/` and looks for the binary there.
fn mock_backend_binary() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    // target/debug/deps/<testbin>-<hash>  → target/debug/
    let debug_dir = exe
        .parent()
        .and_then(|deps| deps.parent())
        .expect("test binary should live under target/<profile>/deps/");
    let candidate = debug_dir.join("mock-inference-backend");
    assert!(
        candidate.exists(),
        "mock-inference-backend not found at {}. \
         Run `cargo build -p mock-inference-backend` first, or depend on \
         the crate as a [dev-dependencies] so cargo builds it as part of \
         the scenarios test target.",
        candidate.display()
    );
    candidate
}

/// No-op [`StateRecorder`] for scenarios — keeps the production recorder's
/// home-directory file from being touched by tests.
struct NoopStateRecorder;

#[async_trait]
impl StateRecorder for NoopStateRecorder {
    async fn record_launch(&self, _handle: &concerto_backend::BackendHandle) {}
    async fn record_stop(&self, _handle: &concerto_backend::BackendHandle) {}
    async fn clear(&self) {}
}

/// Hands out a unique 100-port range to every scenario harness so parallel
/// `cargo test` runs don't contend on backend ports. Starts at a process-
/// unique base port so two test binaries running concurrently land in
/// different windows (each `cargo test` binary is its own process; each
/// resets the in-process `NEXT` counter to 0).
///
/// Base port = 19100 + (pid mod 50) * 200; window = 100 ports. That gives
/// up to 50 concurrent binaries before the windows overlap, which is far
/// more than `cargo test`'s default test-threads parallelism. This fix
/// also lands in A.2 (PR #16); they are independent and will converge.
fn next_port_range() -> std::ops::Range<u16> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    static BASE: std::sync::OnceLock<u16> = std::sync::OnceLock::new();
    let base = *BASE.get_or_init(|| 19100 + ((std::process::id() % 50) as u16) * 200);
    let idx = NEXT.fetch_add(1, Ordering::SeqCst);
    let start = base + (idx as u16) * 100;
    start..(start + 100)
}

/// Test-only [`BackendManager`] wrapper that counts launch / stop calls.
///
/// This keeps production `ProcessBackendManager` free of observability hooks
/// that are only useful in tests. In real deployments the equivalent
/// information comes from the Prometheus metrics landing in Sprint 3.
pub struct CountingBackendManager {
    inner: Arc<dyn BackendManager>,
    launched: AtomicUsize,
    stopped: AtomicUsize,
}

impl CountingBackendManager {
    pub fn new(inner: Arc<dyn BackendManager>) -> Self {
        Self {
            inner,
            launched: AtomicUsize::new(0),
            stopped: AtomicUsize::new(0),
        }
    }

    pub fn launched_count(&self) -> usize {
        self.launched.load(Ordering::SeqCst)
    }

    pub fn stopped_count(&self) -> usize {
        self.stopped.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl BackendManager for CountingBackendManager {
    async fn launch(
        &self,
        spec: &ModelSpec,
        gpu_id: concerto_core::GpuId,
    ) -> Result<BackendHandle, BackendError> {
        let result = self.inner.launch(spec, gpu_id).await;
        if result.is_ok() {
            self.launched.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    async fn stop(&self, handle: &BackendHandle) -> Result<(), BackendError> {
        let result = self.inner.stop(handle).await;
        // Stop counts the intent even if the underlying call errored — the
        // process is gone either way and the test assertion should reflect
        // that intent.
        self.stopped.fetch_add(1, Ordering::SeqCst);
        result
    }

    async fn health_check(&self, handle: &BackendHandle) -> bool {
        self.inner.health_check(handle).await
    }
}

/// Extra per-model args passed to the `mock-inference-backend` binary.
#[derive(Debug, Clone, Default)]
pub struct ModelMockArgs {
    /// Add `--startup-delay-secs <n>` so backend startup has a long window
    /// during which concurrent requests can pile up on the dedup channel.
    pub startup_delay_secs: Option<u64>,
    /// Add `--crash-after <n>` so the backend exits after handling N
    /// chat-completion requests. Used by the crash-recovery scenario.
    pub crash_after: Option<usize>,
    /// Override `--response-latency-ms <n>` (default 10ms). Used by the
    /// request-timeout scenario to make a chat completion take longer than
    /// `request_timeout_secs`.
    pub response_latency_ms: Option<u64>,
    /// Override `--stream-chunk-delay-ms <n>` (default 2ms). Used by the
    /// streaming-exemption scenario to make the SSE body outlive
    /// `request_timeout_secs` without delaying response headers.
    pub stream_chunk_delay_ms: Option<u64>,
}

/// Configuration for a scenario server.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    pub gpu_count: usize,
    pub memory_per_gpu_gb: u64,
    /// `(id, vram_gb, mock_args)` tuples. Order matters for LRU-style tests.
    pub models: Vec<(String, u64, ModelMockArgs)>,
    /// Override the default health-check loop interval (10s is far too slow
    /// for tests; scenarios usually want something on the order of 200ms).
    pub health_check_interval_secs: u64,
    /// Override `routing.request_timeout_secs`. Default `0` (disabled),
    /// matching production. Set to a small positive value to exercise the
    /// per-request timeout middleware.
    pub request_timeout_secs: u64,
}

impl ScenarioConfig {
    pub fn new(gpu_count: usize, memory_per_gpu_gb: u64) -> Self {
        Self {
            gpu_count,
            memory_per_gpu_gb,
            models: vec![],
            health_check_interval_secs: 1,
            request_timeout_secs: 0,
        }
    }

    pub fn with_model(mut self, id: &str, vram_gb: u64) -> Self {
        self.models
            .push((id.into(), vram_gb, ModelMockArgs::default()));
        self
    }

    pub fn with_model_args(mut self, id: &str, vram_gb: u64, args: ModelMockArgs) -> Self {
        self.models.push((id.into(), vram_gb, args));
        self
    }

    pub fn with_request_timeout_secs(mut self, secs: u64) -> Self {
        self.request_timeout_secs = secs;
        self
    }
}

/// A running scenario server. Always call [`ServerHandle::shutdown`] — the
/// `Drop` impl is a best-effort safety net, but explicit shutdown guarantees
/// every spawned `mock-inference-backend` child is reaped before the test
/// returns.
pub struct ServerHandle {
    pub base_url: String,
    pub backend: Arc<CountingBackendManager>,
    pub gpu: Arc<MockGpuMonitor>,
    pub state: AppState,
    pub client: reqwest::Client,
    shutdown: Arc<Notify>,
    serve_task: Option<JoinHandle<()>>,
    shut_down: AtomicBool,
}

impl ServerHandle {
    /// Trigger graceful shutdown and await completion of the serve task.
    /// Idempotent — calling twice is a no-op on the second call.
    pub async fn shutdown(mut self) {
        self.shut_down.store(true, Ordering::SeqCst);
        self.shutdown.notify_waiters();
        if let Some(task) = self.serve_task.take() {
            let _ = tokio::time::timeout(Duration::from_secs(10), task).await;
        }
    }

    /// POST a non-streaming chat-completion to the server.
    pub async fn post_chat(&self, model: &str, content: &str) -> reqwest::Response {
        self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": content}]
            }))
            .send()
            .await
            .expect("chat POST should complete")
    }

    /// POST a streaming chat-completion and return the raw response for the
    /// caller to consume the SSE stream from.
    pub async fn post_chat_stream(&self, model: &str, content: &str) -> reqwest::Response {
        self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&serde_json::json!({
                "model": model,
                "stream": true,
                "messages": [{"role": "user", "content": content}]
            }))
            .send()
            .await
            .expect("streaming chat POST should complete")
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if !self.shut_down.load(Ordering::SeqCst) {
            // Best-effort safety net for scenarios that forget to call
            // shutdown explicitly. The spawn task will abort and the OS
            // will reap the children eventually, but the `.shutdown()`
            // path is strictly better.
            self.shutdown.notify_waiters();
            if let Some(task) = self.serve_task.take() {
                task.abort();
            }
        }
    }
}

/// Spin up a scenario server. Returns once the HTTP listener is ready to
/// accept connections.
pub async fn spawn_scenario(cfg: ScenarioConfig) -> ServerHandle {
    let binary = mock_backend_binary();
    let binary_str = binary
        .to_str()
        .expect("mock-inference-backend path must be UTF-8")
        .to_string();

    // Build engine args per model, so the crash/delay scenarios can drive
    // the mock-backend binary with whatever flags they need.
    let models: Vec<ModelConfigEntry> = cfg
        .models
        .iter()
        .map(|(id, vram_gb, mock_args)| {
            let mut args = vec!["--port".to_string(), "{port}".to_string()];
            if let Some(delay) = mock_args.startup_delay_secs {
                args.push("--startup-delay-secs".to_string());
                args.push(delay.to_string());
            }
            if let Some(crash_after) = mock_args.crash_after {
                args.push("--crash-after".to_string());
                args.push(crash_after.to_string());
            }
            if let Some(latency_ms) = mock_args.response_latency_ms {
                args.push("--response-latency-ms".to_string());
                args.push(latency_ms.to_string());
            }
            if let Some(chunk_delay_ms) = mock_args.stream_chunk_delay_ms {
                args.push("--stream-chunk-delay-ms".to_string());
                args.push(chunk_delay_ms.to_string());
            }
            ModelConfigEntry {
                id: id.clone(),
                name: id.clone(),
                weight_path: format!("/models/{id}"),
                vram_required: ByteSize::gb(*vram_gb),
                engine: EngineType::Custom {
                    command: binary_str.clone(),
                    args,
                    health_endpoint: "/health".to_string(),
                },
                engine_args: vec![],
                pin: false,
                max_vram_fraction: None,
            }
        })
        .collect();

    let gpu_entries: Vec<GpuConfigEntry> = (0..cfg.gpu_count)
        .map(|i| GpuConfigEntry {
            id: i,
            max_temperature: None,
        })
        .collect();

    // The server bind port comes from the ephemeral listener below; the
    // `port_range_*` fields in RoutingSection drive the backend-child port
    // allocator, so those need the scenario's unique range.
    let port_range = next_port_range();
    let routing = RoutingSection {
        health_check_interval_secs: cfg.health_check_interval_secs,
        port_range_start: port_range.start,
        port_range_end: port_range.end,
        cold_start_timeout_secs: 15, // keep tests fast when a spawn is wrong
        request_timeout_secs: cfg.request_timeout_secs,
        ..RoutingSection::default()
    };

    let config = Arc::new(ConcertoConfig {
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
        },
        routing,
        models,
        gpus: gpu_entries,
    });

    // Mocks: GPU telemetry is fake, but the backend manager spawns real
    // subprocesses.
    let gpu = Arc::new(MockGpuMonitor::with_healthy_gpus(
        cfg.gpu_count,
        cfg.memory_per_gpu_gb,
    ));
    let inner_backend: Arc<dyn BackendManager> = Arc::new(
        ProcessBackendManager::with_port_allocator(PortAllocator::with_range(port_range)),
    );
    let backend = Arc::new(CountingBackendManager::new(inner_backend));

    // Seed cluster state from the GPU snapshot.
    let snapshots = gpu.snapshot().await;
    let gpu_states: Vec<GpuState> = snapshots
        .into_iter()
        .map(|snap| GpuState {
            id: snap.id,
            memory_total: snap.memory_total,
            memory_used: snap.memory_used,
            memory_available: ByteSize::b(
                snap.memory_total
                    .as_u64()
                    .saturating_sub(snap.memory_used.as_u64()),
            ),
            temperature_celsius: snap.temperature_celsius,
            utilisation_percent: snap.utilisation_percent,
            health: GpuHealth::Healthy,
            loaded_models: vec![],
        })
        .collect();
    let cluster = ClusterState::new(gpu_states, config.model_registry());

    let shutdown = Arc::new(Notify::new());
    // Install the Prometheus recorder once per process. Scenarios spawn
    // multiple harnesses in parallel within a single test binary; the
    // install function is idempotent and returns a shared handle.
    let prometheus = concerto_api::metrics::install().expect("installing Prometheus recorder");
    // Scenarios use a no-op state recorder — they don't need to persist
    // anything across runs and the production recorder writes to
    // `~/.local/share/concerto/state.json`, which would create
    // cross-test interference.
    let state_recorder: Arc<dyn StateRecorder> = Arc::new(NoopStateRecorder);
    let state = AppState {
        cluster: Arc::new(Mutex::new(cluster)),
        gpu: gpu.clone() as Arc<dyn GpuMonitor>,
        backend: backend.clone() as Arc<dyn BackendManager>,
        config: config.clone(),
        loading: Arc::new(Mutex::new(HashMap::new())),
        backends: Arc::new(Mutex::new(HashMap::new())),
        shutdown: shutdown.clone(),
        state_recorder,
        prometheus,
    };

    // Grab an ephemeral loopback port the serve loop can bind. Binding and
    // dropping leaks the port briefly; the serve() rebind race-window is
    // short enough that this is fine in practice on macOS and Linux.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral");
    let addr: SocketAddr = listener.local_addr().expect("local_addr");
    drop(listener);

    let base_url = format!("http://{addr}");

    // Spawn the serve loop.
    let shutdown_for_serve = shutdown.clone();
    let state_for_serve = state.clone();
    let serve_task = tokio::spawn(async move {
        let shutdown_future = async move {
            shutdown_for_serve.notified().await;
        };
        let _ = serve(state_for_serve, addr, shutdown_future).await;
    });

    // Give serve() a moment to rebind the port before we return. 100ms is
    // comfortably longer than any realistic bind latency on dev machines.
    tokio::time::sleep(Duration::from_millis(100)).await;

    ServerHandle {
        base_url,
        backend,
        gpu,
        state,
        client: reqwest::Client::new(),
        shutdown,
        serve_task: Some(serve_task),
        shut_down: AtomicBool::new(false),
    }
}

/// Helper for scenarios that want to assert on the chat-completion JSON
/// shape.
pub async fn chat_json(resp: reqwest::Response) -> serde_json::Value {
    assert!(
        resp.status().is_success(),
        "expected 2xx, got {}",
        resp.status()
    );
    resp.json().await.expect("chat response is valid JSON")
}
