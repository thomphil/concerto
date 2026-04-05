//! In-memory mock implementation of [`BackendManager`].
//!
//! `MockBackendManager` pretends to launch and stop backends without spawning
//! any real processes. It hands out fake PIDs and real ports from a
//! [`PortAllocator`], and keeps a map of "running" handles so that
//! [`BackendManager::health_check`] can return truthful answers.
//!
//! The mock is deliberately configurable: tests can set a launch latency,
//! force launches to fail, or force health checks to fail, in order to
//! exercise the orchestration logic in `concerto-core` and the API layer
//! without depending on real inference engines.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use concerto_core::{GpuId, ModelSpec};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::manager::{BackendError, BackendHandle, BackendManager};
use crate::port_alloc::PortAllocator;

/// First PID handed out by the mock. Deliberately high so mock PIDs are
/// visually distinct from real low-numbered system PIDs in test output.
const FIRST_MOCK_PID: u32 = 10_000;

/// An in-memory mock [`BackendManager`] for tests and development.
pub struct MockBackendManager {
    handles: RwLock<HashMap<u16, BackendHandle>>,
    ports: PortAllocator,
    next_pid: AtomicU32,
    launch_latency_nanos: AtomicU64,
    fail_launch: AtomicBool,
    fail_health_check: AtomicBool,
    launched_count: AtomicUsize,
    stopped_count: AtomicUsize,
}

impl MockBackendManager {
    /// Create a new mock manager with the default port allocator.
    pub fn new() -> Self {
        Self::with_port_allocator(PortAllocator::new())
    }

    /// Create a new mock manager using a custom [`PortAllocator`].
    pub fn with_port_allocator(ports: PortAllocator) -> Self {
        Self {
            handles: RwLock::new(HashMap::new()),
            ports,
            next_pid: AtomicU32::new(FIRST_MOCK_PID),
            launch_latency_nanos: AtomicU64::new(0),
            fail_launch: AtomicBool::new(false),
            fail_health_check: AtomicBool::new(false),
            launched_count: AtomicUsize::new(0),
            stopped_count: AtomicUsize::new(0),
        }
    }

    /// Configure the launch latency — how long [`BackendManager::launch`]
    /// sleeps before returning, used to simulate slow model loads.
    pub fn with_launch_latency(self, latency: Duration) -> Self {
        self.set_launch_latency(latency);
        self
    }

    /// Toggle on launch failures: subsequent calls to `launch` will return
    /// [`BackendError::LaunchFailed`].
    pub fn with_launch_failure(self) -> Self {
        self.set_launch_failure(true);
        self
    }

    /// Toggle on forced health-check failures: subsequent calls to
    /// `health_check` will always return `false`, even for handles that are
    /// "running".
    pub fn with_health_check_failure(self) -> Self {
        self.set_health_check_failure(true);
        self
    }

    /// Enable or disable launch failures at runtime. Useful for tests that
    /// want to simulate a transient failure and then recover.
    pub fn set_launch_failure(&self, fail: bool) {
        self.fail_launch.store(fail, Ordering::SeqCst);
    }

    /// Enable or disable forced health-check failures at runtime.
    pub fn set_health_check_failure(&self, fail: bool) {
        self.fail_health_check.store(fail, Ordering::SeqCst);
    }

    /// Set the launch latency at runtime.
    pub fn set_launch_latency(&self, latency: Duration) {
        self.launch_latency_nanos
            .store(latency.as_nanos() as u64, Ordering::SeqCst);
    }

    /// Number of successful `launch` calls since construction.
    pub fn launched_count(&self) -> usize {
        self.launched_count.load(Ordering::SeqCst)
    }

    /// Number of successful `stop` calls since construction.
    pub fn stopped_count(&self) -> usize {
        self.stopped_count.load(Ordering::SeqCst)
    }

    /// Number of handles currently considered "running" by this mock.
    pub async fn running_count(&self) -> usize {
        self.handles.read().await.len()
    }
}

impl Default for MockBackendManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BackendManager for MockBackendManager {
    async fn launch(&self, spec: &ModelSpec, gpu_id: GpuId) -> Result<BackendHandle, BackendError> {
        let latency_nanos = self.launch_latency_nanos.load(Ordering::SeqCst);
        if latency_nanos > 0 {
            tokio::time::sleep(Duration::from_nanos(latency_nanos)).await;
        }

        if self.fail_launch.load(Ordering::SeqCst) {
            warn!(model_id = %spec.id, %gpu_id, "mock backend: forced launch failure");
            return Err(BackendError::LaunchFailed(format!(
                "mock backend configured to fail for model {}",
                spec.id
            )));
        }

        let port = self.ports.allocate().ok_or(BackendError::NoFreePort)?;
        let handle = BackendHandle {
            pid: self.next_pid.fetch_add(1, Ordering::SeqCst),
            port,
            model_id: spec.id.clone(),
            gpu_id,
        };

        self.handles.write().await.insert(port, handle.clone());
        self.launched_count.fetch_add(1, Ordering::SeqCst);
        info!(model_id = %spec.id, %gpu_id, pid = handle.pid, port, "mock backend launched");
        Ok(handle)
    }

    async fn stop(&self, handle: &BackendHandle) -> Result<(), BackendError> {
        if self.handles.write().await.remove(&handle.port).is_none() {
            debug!(
                port = handle.port,
                "mock backend stop called on unknown handle; ignoring"
            );
            return Ok(());
        }
        self.ports.release(handle.port);
        self.stopped_count.fetch_add(1, Ordering::SeqCst);
        info!(
            model_id = %handle.model_id,
            pid = handle.pid,
            port = handle.port,
            "mock backend stopped"
        );
        Ok(())
    }

    async fn health_check(&self, handle: &BackendHandle) -> bool {
        if self.fail_health_check.load(Ordering::SeqCst) {
            return false;
        }
        self.handles.read().await.contains_key(&handle.port)
    }
}
