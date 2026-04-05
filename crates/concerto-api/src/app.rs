//! Shared application state handed to every request and background task.

use std::collections::HashMap;
use std::sync::Arc;

use concerto_backend::{BackendHandle, BackendManager};
use concerto_config::ConcertoConfig;
use concerto_core::{ClusterState, ModelId};
use concerto_gpu::GpuMonitor;
use tokio::sync::{broadcast, Mutex, Notify};

/// Outcome of a single model-load operation, broadcast to every waiter.
///
/// When a cold-start is in progress, concurrent requests for the same model
/// subscribe to a [`broadcast::Sender<LoadResult>`] stored in
/// [`AppState::loading`] instead of triggering a second launch. The first
/// requester performs the launch and sends the outcome; every subscriber
/// receives it and either proceeds (on success) or propagates the error.
#[derive(Debug, Clone)]
pub enum LoadResult {
    Ok(BackendHandle),
    Err(String),
}

/// Shared state handed to every request handler and background task.
///
/// Every field is cheaply cloneable ([`Arc`] or [`Clone`]) so the state can
/// be moved into spawned tasks without contention.
#[derive(Clone)]
pub struct AppState {
    /// Authoritative snapshot of the cluster: loaded models, available VRAM,
    /// GPU health. Mutated under the lock in short critical sections by the
    /// orchestrator and the background health loop.
    pub cluster: Arc<Mutex<ClusterState>>,
    /// GPU telemetry source.
    pub gpu: Arc<dyn GpuMonitor>,
    /// Process lifecycle manager for backends.
    pub backend: Arc<dyn BackendManager>,
    /// Parsed, validated configuration (immutable for the lifetime of the
    /// process).
    pub config: Arc<ConcertoConfig>,
    /// Cold-start dedup table. Key: model being loaded. Value: broadcast
    /// sender used to notify every concurrent waiter of the load outcome.
    pub loading: Arc<Mutex<HashMap<ModelId, broadcast::Sender<LoadResult>>>>,
    /// Map from model id to the live backend handle serving it. Maintained
    /// alongside `cluster.loaded_models` so the orchestrator and the health
    /// loop can call `BackendManager::stop` / `health_check` without having
    /// to re-derive the handle from the cluster snapshot.
    pub backends: Arc<Mutex<HashMap<ModelId, BackendHandle>>>,
    /// Notifier used to signal graceful shutdown to background tasks.
    pub shutdown: Arc<Notify>,
}
