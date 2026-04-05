//! The [`BackendManager`] trait and its associated handle and error types.

use async_trait::async_trait;
use concerto_core::{GpuId, ModelId, ModelSpec};
use serde::{Deserialize, Serialize};

/// A handle to a running inference backend.
///
/// This is what [`BackendManager::launch`] returns once a backend has been
/// started and is responding to health checks. The handle carries just enough
/// information to route requests to the backend and to later stop it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendHandle {
    /// Operating system process ID of the backend.
    pub pid: u32,
    /// TCP port that the backend is listening on (on localhost).
    pub port: u16,
    /// The model this backend is serving.
    pub model_id: ModelId,
    /// The GPU this backend was placed on.
    pub gpu_id: GpuId,
    /// HTTP path the backend exposes for health probes (e.g. `/health` for
    /// built-in engines, or a user-supplied path for `EngineType::Custom`).
    #[serde(default = "default_health_path")]
    pub health_path: String,
}

fn default_health_path() -> String {
    "/health".to_string()
}

/// Abstraction over the lifecycle of an inference engine backend.
///
/// Implementations are responsible for starting a backend for a given model on
/// a given GPU, stopping it again, and reporting whether it is currently
/// healthy. Implementors must be `Send + Sync` so the manager can be shared
/// across async tasks behind an `Arc`.
#[async_trait]
pub trait BackendManager: Send + Sync {
    /// Launch a new inference backend for `spec` on `gpu_id`.
    ///
    /// Returns a handle once the backend is responding to health checks. If
    /// the backend fails to start or does not become healthy in time, an
    /// appropriate [`BackendError`] is returned.
    async fn launch(&self, spec: &ModelSpec, gpu_id: GpuId) -> Result<BackendHandle, BackendError>;

    /// Stop a previously launched backend.
    ///
    /// Implementations should attempt a graceful shutdown first where
    /// possible, and release any resources (ports, PIDs) associated with the
    /// handle.
    async fn stop(&self, handle: &BackendHandle) -> Result<(), BackendError>;

    /// Check whether a backend is currently healthy.
    ///
    /// Returns `true` if the backend is reachable and reports itself healthy,
    /// `false` otherwise. This method must not return an error — callers use
    /// the boolean result to drive routing decisions.
    async fn health_check(&self, handle: &BackendHandle) -> bool;
}

/// Errors that can occur when managing backend processes.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The backend process failed to launch (spawn failed, binary missing,
    /// invalid args, etc.).
    #[error("failed to launch backend: {0}")]
    LaunchFailed(String),

    /// A low-level IO error occurred while managing a backend.
    #[error("backend IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The backend did not become healthy within the configured timeout.
    #[error("backend did not become healthy within the startup timeout")]
    HealthCheckTimeout,

    /// No free port is available in the configured allocation range.
    #[error("no free port available for new backend")]
    NoFreePort,

    /// A running backend process exited unexpectedly.
    #[error("backend process (pid {pid}) exited: {status}")]
    ProcessExited { pid: u32, status: String },

    /// An HTTP request to the backend failed.
    #[error("backend HTTP request failed: {0}")]
    Reqwest(#[from] reqwest::Error),
}
