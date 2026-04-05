//! An in-memory, configurable [`GpuMonitor`] used for testing and development.

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use async_trait::async_trait;
use bytesize::ByteSize;
use concerto_core::GpuId;

use crate::monitor::{GpuMonitor, GpuSnapshot};

/// A [`GpuMonitor`] backed by an in-memory `Vec<GpuSnapshot>`.
///
/// `MockGpuMonitor` is the workhorse of Concerto's test suite: it lets tests
/// declare the exact fleet they want to see (including edge cases like
/// overheating GPUs, GPUs with ECC errors, or GPUs that disappear mid-run) and
/// then drive the system under test through those scenarios.
///
/// Cloning is cheap — the monitor is backed by an `Arc<RwLock<_>>` so clones
/// share the same underlying state. This lets a test hold a handle for
/// mutation while also passing the monitor into the system under test.
///
/// The lock is a `std::sync::RwLock` (not `tokio::sync::RwLock`) so that
/// [`GpuMonitor::gpu_count`], which is synchronous, can read the length
/// without entering the async runtime. Critical sections are always tiny
/// (a clone, a field write, or a `retain`) so blocking here is harmless.
#[derive(Debug, Clone, Default)]
pub struct MockGpuMonitor {
    snapshots: Arc<RwLock<Vec<GpuSnapshot>>>,
}

impl MockGpuMonitor {
    /// Create a mock monitor from an explicit list of snapshots.
    pub fn new(snapshots: Vec<GpuSnapshot>) -> Self {
        Self {
            snapshots: Arc::new(RwLock::new(snapshots)),
        }
    }

    /// Create a mock monitor reporting `count` healthy GPUs, each with
    /// `memory_per_gpu_gb` gigabytes of VRAM, zero utilisation, zero memory
    /// used, 40 degrees Celsius, and no ECC errors.
    pub fn with_healthy_gpus(count: usize, memory_per_gpu_gb: u64) -> Self {
        let snapshots = (0..count)
            .map(|i| GpuSnapshot {
                id: GpuId(i),
                memory_total: ByteSize::gb(memory_per_gpu_gb),
                memory_used: ByteSize::b(0),
                temperature_celsius: 40,
                utilisation_percent: 0,
                ecc_errors_uncorrected: 0,
            })
            .collect();
        Self::new(snapshots)
    }

    /// Overwrite the `memory_used` field of the GPU with the given id.
    ///
    /// No-op if the GPU is not present (e.g. it has been removed via
    /// [`MockGpuMonitor::remove_gpu`]).
    pub async fn set_memory_used(&self, gpu_id: GpuId, bytes: ByteSize) {
        self.update(gpu_id, |snap| snap.memory_used = bytes);
    }

    /// Overwrite the `temperature_celsius` field of the GPU with the given id.
    pub async fn set_temperature(&self, gpu_id: GpuId, celsius: u32) {
        self.update(gpu_id, |snap| snap.temperature_celsius = celsius);
    }

    /// Increment the uncorrected ECC error count of the GPU with the given id
    /// by one.
    pub async fn inject_ecc_error(&self, gpu_id: GpuId) {
        self.update(gpu_id, |snap| {
            snap.ecc_errors_uncorrected = snap.ecc_errors_uncorrected.saturating_add(1);
        });
    }

    /// Remove a GPU from the monitor's view, simulating a GPU that has
    /// dropped off the bus (driver crash, hardware fault, hot-unplug).
    pub async fn remove_gpu(&self, gpu_id: GpuId) {
        if let Some(mut guard) = self.write_guard("remove_gpu") {
            guard.retain(|s| s.id != gpu_id);
        }
    }

    /// Apply `f` to the snapshot with the given id, if one exists. No-op
    /// otherwise — callers use this to drive specific GPUs from tests without
    /// first checking whether the GPU is still present.
    fn update(&self, gpu_id: GpuId, f: impl FnOnce(&mut GpuSnapshot)) {
        let Some(mut guard) = self.write_guard("update") else {
            return;
        };
        if let Some(snap) = guard.iter_mut().find(|s| s.id == gpu_id) {
            f(snap);
        }
    }

    fn read_guard(&self, op: &'static str) -> Option<RwLockReadGuard<'_, Vec<GpuSnapshot>>> {
        match self.snapshots.read() {
            Ok(guard) => Some(guard),
            Err(_) => {
                tracing::error!(op, "MockGpuMonitor lock poisoned");
                None
            }
        }
    }

    fn write_guard(&self, op: &'static str) -> Option<RwLockWriteGuard<'_, Vec<GpuSnapshot>>> {
        match self.snapshots.write() {
            Ok(guard) => Some(guard),
            Err(_) => {
                tracing::error!(op, "MockGpuMonitor lock poisoned");
                None
            }
        }
    }
}

#[async_trait]
impl GpuMonitor for MockGpuMonitor {
    fn gpu_count(&self) -> usize {
        self.read_guard("gpu_count").map_or(0, |g| g.len())
    }

    async fn snapshot(&self) -> Vec<GpuSnapshot> {
        self.read_guard("snapshot")
            .map_or_else(Vec::new, |g| g.clone())
    }
}
