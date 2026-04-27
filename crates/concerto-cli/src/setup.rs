//! Wire up tracing, monitors, backend manager, and [`AppState`] from CLI args.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use bytesize::ByteSize;
use concerto_api::metrics as api_metrics;
use concerto_api::state_file::{
    default_state_file_path, JsonStateRecorder, RecordingBackendManager, StateRecorder,
};
use concerto_api::AppState;
use concerto_backend::{BackendManager, PortAllocator, ProcessBackendManager};
use concerto_config::ConcertoConfig;
use concerto_core::{
    ClusterState, EngineType, GpuHealth, GpuId, GpuState, LoadedModel, ModelId, ModelSpec,
};
use concerto_gpu::{classify_health, GpuMonitor, HealthThresholds, MockGpuMonitor};
use tokio::sync::{Mutex, Notify};
use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::cli::{Cli, LogFormat};

/// Install the `tracing_subscriber` global default based on `args.log_level` +
/// `args.log_format`.
pub fn init_tracing(args: &Cli) -> Result<()> {
    let filter = args
        .log_level
        .as_deref()
        .map(EnvFilter::new)
        .unwrap_or_else(|| {
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
        });

    let registry = tracing_subscriber::registry().with(filter);
    match args.log_format {
        LogFormat::Pretty => registry.with(fmt::layer().pretty()).try_init(),
        LogFormat::Json => registry.with(fmt::layer().json()).try_init(),
    }
    .map_err(|e| anyhow!("failed to install tracing subscriber: {e}"))
}

/// Parse the configuration file, build every required piece of shared state,
/// and return an assembled [`AppState`] along with the bind [`SocketAddr`].
pub async fn build_app_state(args: &Cli) -> Result<(AppState, SocketAddr)> {
    let mut config = ConcertoConfig::from_path(&args.config)
        .with_context(|| format!("loading config from {}", args.config.display()))?;

    // --- mock GPU path: rewrite config for a fully self-contained dev run ---
    if let Some(n) = args.mock_gpus {
        apply_mock_gpu_overrides(&mut config, n)?;
    }

    let gpu: Arc<dyn GpuMonitor> = build_gpu_monitor(args).await?;

    // Sprint 3 §A.1: SIGTERM-then-SIGKILL grace is wired from
    // routing.eviction_grace_period_secs so the backend's
    // process-group kill matches the eviction policy.
    let inner_backend: Arc<dyn BackendManager> = Arc::new(
        ProcessBackendManager::with_port_allocator(PortAllocator::with_range(
            config.routing.port_range_start..config.routing.port_range_end,
        ))
        .with_termination_grace(std::time::Duration::from_secs(
            config.routing.eviction_grace_period_secs,
        )),
    );

    // Sprint 3 §A.3: every launch / stop is recorded to a state file
    // so a startup reconcile after a Concerto crash can clean up.
    let state_path = default_state_file_path()
        .context("resolving state-file path for the recording backend manager")?;
    let state_recorder: Arc<dyn StateRecorder> = Arc::new(JsonStateRecorder::new(state_path));
    let backend: Arc<dyn BackendManager> = Arc::new(RecordingBackendManager::new(
        inner_backend,
        state_recorder.clone(),
    ));
    let config = Arc::new(config);

    let cluster = build_initial_cluster_state(&gpu, &config).await?;

    // Install the process-global Prometheus recorder before any metric is
    // emitted. Idempotent: subsequent calls in the same process (e.g. the
    // scenario harness also calls this) return the same cached handle.
    let prometheus = api_metrics::install().context("installing Prometheus recorder")?;

    let state = AppState {
        cluster: Arc::new(Mutex::new(cluster)),
        gpu,
        backend,
        config: config.clone(),
        loading: Arc::new(Mutex::new(HashMap::new())),
        backends: Arc::new(Mutex::new(HashMap::new())),
        shutdown: Arc::new(Notify::new()),
        state_recorder,
        prometheus,
    };

    let port = args.port_override.unwrap_or(config.server.port);
    let addr: SocketAddr = format!("{}:{}", config.server.host, port)
        .parse()
        .with_context(|| format!("parsing bind address {}:{port}", config.server.host))?;

    info!(
        model_count = config.models.len(),
        gpu_count = config.gpus.len(),
        %addr,
        "concerto ready to serve"
    );
    Ok((state, addr))
}

/// When `--mock-gpus N` is set, replace the config's `[[gpus]]` with N synthetic
/// entries and rewrite every model to use the bundled `mock-inference-backend`
/// binary via the `EngineType::Custom` extension hatch. The result is a fully
/// self-contained dev run: no NVML, no vLLM, just in-process orchestration
/// against a real HTTP backend that produces OpenAI-shaped responses.
fn apply_mock_gpu_overrides(config: &mut ConcertoConfig, n: usize) -> Result<()> {
    if n == 0 {
        return Err(anyhow!("--mock-gpus must be at least 1"));
    }
    config.gpus = (0..n)
        .map(|i| concerto_config::GpuConfigEntry {
            id: i,
            max_temperature: None,
        })
        .collect();

    let mock_bin = locate_mock_backend_binary()?;
    info!(binary = %mock_bin, "using mock inference backend");

    for model in &mut config.models {
        model.engine = EngineType::Custom {
            command: mock_bin.clone(),
            args: vec!["--port".into(), "{port}".into()],
            health_endpoint: "/health".into(),
        };
        // Engine-specific args from the real config (e.g. vLLM's --dtype,
        // --max-model-len) don't apply to the mock backend. Drop them so
        // `build_command` doesn't forward invalid flags.
        model.engine_args.clear();
    }
    Ok(())
}

/// Resolve the absolute path of the `mock-inference-backend` binary by
/// looking next to the currently-running `concerto` executable (the common
/// case when running from `target/debug/`). Falls back to bare
/// `"mock-inference-backend"` for a PATH lookup.
fn locate_mock_backend_binary() -> Result<String> {
    let exe = std::env::current_exe().context("determining current exe path")?;
    if let Some(dir) = exe.parent() {
        let candidate = dir.join("mock-inference-backend");
        if candidate.exists() {
            return Ok(candidate.to_string_lossy().into_owned());
        }
    }
    Ok("mock-inference-backend".to_string())
}

async fn build_gpu_monitor(args: &Cli) -> Result<Arc<dyn GpuMonitor>> {
    if let Some(n) = args.mock_gpus {
        return Ok(Arc::new(MockGpuMonitor::with_healthy_gpus(n, 24)));
    }

    // Real NVML monitor path. Gated on both the `nvml` feature (opt-in at
    // build time) and `target_os = "linux"` because `nvml-wrapper` only
    // links on Linux. Exactly one of the two `#[cfg]` blocks below is
    // compiled in any given build and becomes the function's tail
    // expression — avoiding an explicit `return` keeps clippy's
    // `needless_return` happy on Linux.
    #[cfg(all(feature = "nvml", target_os = "linux"))]
    {
        let monitor = concerto_gpu::NvmlMonitor::new().context("initialising NVML GPU monitor")?;
        Ok(Arc::new(monitor))
    }

    #[cfg(not(all(feature = "nvml", target_os = "linux")))]
    {
        Err(anyhow!(
            "--mock-gpus is required unless concerto-cli is built with the \
             `nvml` feature on Linux. On a Linux host with NVIDIA drivers \
             installed: cargo build --release -p concerto-cli --features nvml"
        ))
    }
}

/// Build a fresh [`ClusterState`] from a GPU snapshot and the config's model
/// registry. No models are loaded initially — every request starts cold.
async fn build_initial_cluster_state(
    gpu: &Arc<dyn GpuMonitor>,
    config: &ConcertoConfig,
) -> Result<ClusterState> {
    let snapshots = gpu.snapshot().await;
    if snapshots.is_empty() {
        return Err(anyhow!(
            "GPU monitor returned no snapshots; concerto cannot start with zero GPUs"
        ));
    }

    let thresholds = HealthThresholds {
        max_healthy_temperature: config.routing.max_healthy_temperature,
        max_degraded_temperature: config.routing.max_degraded_temperature,
        max_tolerated_ecc: 0,
    };

    let gpus: Vec<GpuState> = snapshots
        .into_iter()
        .map(|snap| {
            let health = classify_health(&snap, &thresholds);
            let memory_available = ByteSize::b(
                snap.memory_total
                    .as_u64()
                    .saturating_sub(snap.memory_used.as_u64()),
            );
            GpuState {
                id: snap.id,
                memory_total: snap.memory_total,
                memory_used: snap.memory_used,
                memory_available,
                temperature_celsius: snap.temperature_celsius,
                utilisation_percent: snap.utilisation_percent,
                health,
                loaded_models: Vec::new(),
            }
        })
        .collect();

    Ok(ClusterState::new(gpus, config.model_registry()))
}

// Silence dead-code warnings on helper types exported only for the tests that
// would otherwise exercise them.
#[allow(dead_code)]
fn _assert_handle_shape(_: LoadedModel, _: ModelId, _: ModelSpec, _: GpuId, _: GpuHealth) {}
