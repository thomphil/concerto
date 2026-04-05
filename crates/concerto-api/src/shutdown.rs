//! Graceful shutdown logic.
//!
//! On shutdown:
//! 1. Notify background tasks (health loop) via `state.shutdown`.
//! 2. Give in-flight requests a brief window to drain.
//! 3. Stop every known backend.
//!
//! In v0.1 the drain is a simple fixed sleep — there is no in-flight request
//! tracker. Sprint 3 will replace this with a proper drain tracker that
//! counts active requests and waits for them with a bounded deadline.

use std::time::Duration;

use concerto_backend::BackendHandle;
use concerto_core::ModelId;
use tracing::{info, warn};

use crate::app::AppState;

/// Best-effort graceful shutdown.
pub async fn graceful_shutdown(state: AppState) {
    info!("graceful shutdown: notifying background tasks");
    state.shutdown.notify_waiters();

    // Give in-flight requests a moment to drain. This is intentionally
    // conservative — most mock-backend requests complete in < 50ms, and a
    // real deployment will tune this via the systemd TimeoutStopSec or the
    // docker-compose stop_grace_period.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let handles: Vec<(ModelId, BackendHandle)> = {
        let mut backends = state.backends.lock().await;
        backends.drain().collect()
    };
    info!(count = handles.len(), "stopping backends");
    for (model_id, handle) in handles {
        if let Err(e) = state.backend.stop(&handle).await {
            warn!(%model_id, error = %e, "backend stop returned error during shutdown");
        }
    }
    info!("graceful shutdown complete");
}
