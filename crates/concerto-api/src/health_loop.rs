//! Background health-check loop (ROADMAP §5 T7).
//!
//! A single `tokio::task` periodically probes every known backend via
//! [`BackendManager::health_check`]. When a backend is found unhealthy, it's
//! removed from both `state.backends` and `state.cluster.loaded_models`, and
//! a warning is logged. Subsequent requests for that model will trigger a
//! fresh cold-start through the orchestrator.

use std::time::Duration;

use concerto_backend::BackendHandle;
use concerto_core::ModelId;
use metrics::counter;
use tracing::{info, warn};

use crate::app::AppState;
use crate::metrics::BACKEND_HEALTH_CHECK_FAILURES_TOTAL;

/// Run the health loop until shutdown is signalled via `state.shutdown`.
pub async fn run(state: AppState) {
    let interval = Duration::from_secs(state.config.routing.health_check_interval_secs.max(1));
    info!(?interval, "health loop started");

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = state.shutdown.notified() => {
                info!("health loop stopping");
                return;
            }
        }

        let snapshot: Vec<(ModelId, BackendHandle)> = {
            let backends = state.backends.lock().await;
            backends
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };

        for (model_id, handle) in snapshot {
            if !state.backend.health_check(&handle).await {
                warn!(%model_id, pid = handle.pid, port = handle.port, "backend unhealthy; dropping");
                counter!(BACKEND_HEALTH_CHECK_FAILURES_TOTAL).increment(1);
                state.backends.lock().await.remove(&model_id);
                let mut cluster = state.cluster.lock().await;
                for gpu in &mut cluster.gpus {
                    let freed: u64 = gpu
                        .loaded_models
                        .iter()
                        .filter(|m| m.model_id == model_id)
                        .map(|m| m.vram_usage.as_u64())
                        .sum();
                    gpu.loaded_models.retain(|m| m.model_id != model_id);
                    if freed > 0 {
                        gpu.memory_used =
                            bytesize::ByteSize::b(gpu.memory_used.as_u64().saturating_sub(freed));
                        gpu.memory_available =
                            bytesize::ByteSize::b(gpu.memory_available.as_u64() + freed);
                    }
                }
            }
        }
    }
}
