//! Shared harness for the orchestrator scenario suite.
//!
//! Each scenario drives [`concerto_api::orchestrator::route_and_dispatch`]
//! directly against a freshly constructed [`AppState`] — this is what
//! actually tests the state machine from ROADMAP §3 (concurrent cold-start
//! dedup, eviction-then-launch races, failed-load rollback, backend crash
//! recovery).
//!
//! For v0.1 the backend is [`MockBackendManager`], which is configurable
//! (launch latency, crash injection, launch failure) and exposes counters
//! (`launched_count`, `stopped_count`) the scenarios can assert on.
//!
//! This deliberately deviates from ROADMAP §6.3's "production code path"
//! goal: §6.3 asks scenarios to exercise `ProcessBackendManager` spawning
//! real `mock-inference-backend` subprocesses. That migration happens in
//! Sprint 2, which is already the forcing function for the process-spawn
//! code path. The HTTP layer is covered by the interactive verification
//! in the U3 commit message; the scenarios here protect the orchestrator
//! state machine — which is what we actually need to keep correct during
//! the rest of Sprint 1.

use std::collections::HashMap;
use std::sync::Arc;

use bytesize::ByteSize;
use concerto_api::AppState;
use concerto_backend::{BackendManager, MockBackendManager};
use concerto_config::{
    ConcertoConfig, GpuConfigEntry, ModelConfigEntry, RoutingSection, ServerConfig,
};
use concerto_core::{ClusterState, EngineType, GpuHealth, GpuState};
use concerto_gpu::{GpuMonitor, MockGpuMonitor};
use tokio::sync::{Mutex, Notify};

/// Everything a scenario test needs to drive the orchestrator.
pub struct ScenarioHarness {
    pub state: AppState,
    pub backend: Arc<MockBackendManager>,
    pub gpu: Arc<MockGpuMonitor>,
}

/// Configuration for a scenario.
pub struct ScenarioConfig {
    pub gpu_count: usize,
    pub memory_per_gpu_gb: u64,
    /// `(id, vram_gb)` pairs. Order matters for `lru_eviction`-style tests
    /// because the model registry is iterated for placement decisions.
    pub models: Vec<(String, u64)>,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            gpu_count: 2,
            memory_per_gpu_gb: 24,
            models: vec![("model-a".into(), 8)],
        }
    }
}

/// Build a [`ScenarioHarness`] with in-memory mocks and an empty cluster.
pub async fn build_harness(cfg: ScenarioConfig) -> ScenarioHarness {
    let models: Vec<ModelConfigEntry> = cfg
        .models
        .iter()
        .map(|(id, vram_gb)| ModelConfigEntry {
            id: id.clone(),
            name: id.clone(),
            weight_path: format!("/models/{id}"),
            vram_required: ByteSize::gb(*vram_gb),
            engine: EngineType::Mock,
            engine_args: vec![],
            pin: false,
        })
        .collect();

    let gpu_entries: Vec<GpuConfigEntry> = (0..cfg.gpu_count)
        .map(|i| GpuConfigEntry {
            id: i,
            max_temperature: None,
        })
        .collect();

    let config = Arc::new(ConcertoConfig {
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 8000,
        },
        routing: RoutingSection::default(),
        models,
        gpus: gpu_entries,
    });

    let gpu = Arc::new(MockGpuMonitor::with_healthy_gpus(
        cfg.gpu_count,
        cfg.memory_per_gpu_gb,
    ));
    let backend = Arc::new(MockBackendManager::new());

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

    let state = AppState {
        cluster: Arc::new(Mutex::new(cluster)),
        gpu: gpu.clone() as Arc<dyn GpuMonitor>,
        backend: backend.clone() as Arc<dyn BackendManager>,
        config,
        loading: Arc::new(Mutex::new(HashMap::new())),
        backends: Arc::new(Mutex::new(HashMap::new())),
        shutdown: Arc::new(Notify::new()),
    };

    ScenarioHarness {
        state,
        backend,
        gpu,
    }
}
