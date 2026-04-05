//! The [`GpuMonitor`] trait and its associated snapshot and error types.

use async_trait::async_trait;
use bytesize::ByteSize;
use concerto_core::GpuId;
use serde::{Deserialize, Serialize};

/// A point-in-time view of a single GPU's telemetry.
///
/// `GpuSnapshot` is the lowest common denominator of what every [`GpuMonitor`]
/// implementation must be able to report. Higher layers (e.g. `concerto-core`)
/// turn these snapshots into richer state such as `GpuState`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuSnapshot {
    /// The GPU's index on the node.
    pub id: GpuId,
    /// Total physical VRAM on the device.
    pub memory_total: ByteSize,
    /// VRAM currently in use (from all processes on the device, not just ours).
    pub memory_used: ByteSize,
    /// Core temperature in degrees Celsius.
    pub temperature_celsius: u32,
    /// GPU core utilisation as a percentage (0-100).
    pub utilisation_percent: u32,
    /// Cumulative count of uncorrectable ECC errors reported by the driver.
    pub ecc_errors_uncorrected: u64,
}

/// Abstraction over a source of GPU telemetry.
///
/// Implementors must be `Send + Sync` so that the monitor can be shared across
/// tasks behind an `Arc`.
#[async_trait]
pub trait GpuMonitor: Send + Sync {
    /// The number of GPUs this monitor is tracking.
    ///
    /// This is the value at construction time; implementations that support
    /// hot-plug / hot-unplug (such as `MockGpuMonitor` for test scenarios) may
    /// return a different value after the monitor has been mutated.
    fn gpu_count(&self) -> usize;

    /// Take a snapshot of all GPUs currently visible to this monitor.
    async fn snapshot(&self) -> Vec<GpuSnapshot>;
}

/// Errors that can occur when constructing or using a [`GpuMonitor`].
#[derive(Debug, thiserror::Error)]
pub enum GpuMonitorError {
    /// NVML could not be initialised (driver missing, permission denied, etc.).
    #[error("failed to initialise NVML: {0}")]
    NvmlInit(String),

    /// A query against the NVML backend failed.
    #[error("NVML query failed: {0}")]
    NvmlQuery(String),

    /// A GPU index was requested that does not exist on this node.
    #[error("GPU index {0} is out of range")]
    GpuOutOfRange(usize),

    /// The requested feature is not supported on this platform.
    #[error("feature not supported on this platform: {0}")]
    Unsupported(&'static str),
}
