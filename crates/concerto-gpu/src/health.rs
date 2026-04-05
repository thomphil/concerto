//! Pure health classification from a raw [`GpuSnapshot`].
//!
//! This lives in `concerto-gpu` rather than `concerto-core` because the inputs
//! (temperature, ECC) are GPU telemetry concerns. The output ([`GpuHealth`])
//! feeds back into `concerto-core` routing decisions.

use concerto_core::GpuHealth;

use crate::monitor::GpuSnapshot;

/// Thresholds at which a GPU transitions between [`GpuHealth`] states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HealthThresholds {
    /// The maximum temperature (in Celsius) at which a GPU is still considered
    /// fully healthy. Above this value, the GPU is marked `Degraded`.
    pub max_healthy_temperature: u32,
    /// The maximum temperature (in Celsius) at which a GPU is still considered
    /// merely degraded. Above this value, the GPU is marked `Unhealthy` and
    /// should not receive new workloads.
    pub max_degraded_temperature: u32,
    /// The maximum number of cumulative uncorrected ECC errors that will be
    /// tolerated before the GPU is marked `Unhealthy`.
    pub max_tolerated_ecc: u64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            max_healthy_temperature: 75,
            max_degraded_temperature: 85,
            max_tolerated_ecc: 0,
        }
    }
}

/// Classify a [`GpuSnapshot`] into a [`GpuHealth`] state using the given
/// thresholds.
///
/// The rules, evaluated in order:
/// 1. If temperature exceeds `max_degraded_temperature`, or uncorrected ECC
///    errors exceed `max_tolerated_ecc`, the GPU is `Unhealthy`.
/// 2. Otherwise, if temperature exceeds `max_healthy_temperature`, the GPU is
///    `Degraded`.
/// 3. Otherwise, `Healthy`.
pub fn classify_health(snapshot: &GpuSnapshot, thresholds: &HealthThresholds) -> GpuHealth {
    if snapshot.temperature_celsius > thresholds.max_degraded_temperature
        || snapshot.ecc_errors_uncorrected > thresholds.max_tolerated_ecc
    {
        GpuHealth::Unhealthy
    } else if snapshot.temperature_celsius > thresholds.max_healthy_temperature {
        GpuHealth::Degraded
    } else {
        GpuHealth::Healthy
    }
}
