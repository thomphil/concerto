//! # concerto-gpu
//!
//! GPU monitoring abstractions for Concerto.
//!
//! This crate provides the [`GpuMonitor`] trait, which abstracts over the source
//! of GPU telemetry (NVML, mock, or anything else), and two implementations:
//!
//! - [`MockGpuMonitor`] — a configurable in-memory monitor used in all tests and
//!   for development without real GPU hardware.
//! - `NvmlMonitor` — a real NVML-backed monitor, gated behind the `nvml`
//!   feature and only available on Linux. See the `nvml` module for platform
//!   caveats.
//!
//! The trait is deliberately minimal: callers drive snapshots from the outside,
//! which keeps the implementation trivially testable and leaves caching/interval
//! policy to higher layers.

pub mod health;
pub mod mock;
pub mod monitor;

#[cfg(feature = "nvml")]
pub mod nvml;

pub use health::{classify_health, HealthThresholds};
pub use mock::MockGpuMonitor;
pub use monitor::{GpuMonitor, GpuMonitorError, GpuSnapshot};

#[cfg(all(feature = "nvml", target_os = "linux"))]
pub use nvml::NvmlMonitor;
