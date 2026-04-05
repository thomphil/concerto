//! NVML-backed [`GpuMonitor`] implementation.
//!
//! This module is gated behind both the `nvml` feature flag **and**
//! `target_os = "linux"`. `nvml-wrapper` links against `libnvidia-ml`, which
//! only ships with NVIDIA's Linux drivers; building this module on macOS or
//! Windows CI machines would fail to link even though we never actually run
//! the code there.
//!
//! On non-Linux platforms, enabling the `nvml` feature is therefore a no-op:
//! the crate still compiles, but this module is absent and `NvmlMonitor` is
//! not re-exported from the crate root.

#![cfg(all(feature = "nvml", target_os = "linux"))]

use async_trait::async_trait;
use bytesize::ByteSize;
use concerto_core::GpuId;
use nvml_wrapper::{enum_wrappers::device::TemperatureSensor, error::NvmlError, Nvml};

use crate::monitor::{GpuMonitor, GpuMonitorError, GpuSnapshot};

/// A [`GpuMonitor`] that queries NVIDIA's NVML library for real GPU telemetry.
///
/// This monitor is only available on Linux with the `nvml` feature enabled and
/// a working NVIDIA driver installed. All NVML calls are synchronous; we wrap
/// them in [`tokio::task::spawn_blocking`] so snapshotting does not stall the
/// async runtime.
pub struct NvmlMonitor {
    nvml: std::sync::Arc<Nvml>,
    device_count: usize,
}

impl NvmlMonitor {
    /// Initialise NVML and count the visible devices.
    pub fn new() -> Result<Self, GpuMonitorError> {
        let nvml = Nvml::init().map_err(|e| GpuMonitorError::NvmlInit(e.to_string()))?;
        let device_count =
            nvml.device_count()
                .map_err(|e| GpuMonitorError::NvmlQuery(e.to_string()))? as usize;
        Ok(Self {
            nvml: std::sync::Arc::new(nvml),
            device_count,
        })
    }

    fn snapshot_blocking(nvml: &Nvml, device_count: usize) -> Vec<GpuSnapshot> {
        let mut out = Vec::with_capacity(device_count);
        for idx in 0..device_count {
            match Self::snapshot_one(nvml, idx) {
                Ok(snap) => out.push(snap),
                Err(err) => {
                    tracing::warn!(
                        gpu = idx,
                        error = %err,
                        "failed to read NVML snapshot for device; skipping"
                    );
                }
            }
        }
        out
    }

    fn snapshot_one(nvml: &Nvml, idx: usize) -> Result<GpuSnapshot, NvmlError> {
        let device = nvml.device_by_index(idx as u32)?;
        let memory = device.memory_info()?;
        let temperature = device.temperature(TemperatureSensor::Gpu)?;
        let utilisation = device.utilization_rates()?;
        // ECC error counts are only meaningful on datacentre GPUs; if the
        // device doesn't support ECC we treat it as zero errors rather than
        // failing the whole snapshot.
        let ecc = device
            .total_ecc_errors(
                nvml_wrapper::enum_wrappers::device::MemoryError::Uncorrected,
                nvml_wrapper::enum_wrappers::device::EccCounter::Aggregate,
            )
            .unwrap_or(0);

        Ok(GpuSnapshot {
            id: GpuId(idx),
            memory_total: ByteSize::b(memory.total),
            memory_used: ByteSize::b(memory.used),
            temperature_celsius: temperature,
            utilisation_percent: utilisation.gpu,
            ecc_errors_uncorrected: ecc,
        })
    }
}

#[async_trait]
impl GpuMonitor for NvmlMonitor {
    fn gpu_count(&self) -> usize {
        self.device_count
    }

    async fn snapshot(&self) -> Vec<GpuSnapshot> {
        let nvml = self.nvml.clone();
        let device_count = self.device_count;
        tokio::task::spawn_blocking(move || Self::snapshot_blocking(&nvml, device_count))
            .await
            .unwrap_or_else(|err| {
                tracing::error!(error = %err, "NVML snapshot task panicked");
                Vec::new()
            })
    }
}
