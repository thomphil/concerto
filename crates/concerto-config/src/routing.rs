//! The `[routing]` section of the configuration.

use bytesize::ByteSize;
use concerto_core::EvictionPolicy;
use serde::{Deserialize, Serialize};

/// Routing/eviction tuning parameters.
///
/// All fields have sensible defaults, so a completely empty `[routing]`
/// section (or an omitted section altogether) produces a fully populated
/// [`RoutingSection`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingSection {
    /// Which eviction policy to apply when a GPU is full.
    #[serde(default = "default_eviction_policy")]
    pub eviction_policy: EvictionPolicy,

    /// Maximum time to wait for a model to become ready after issuing a load.
    #[serde(default = "default_cold_start_timeout_secs")]
    pub cold_start_timeout_secs: u64,

    /// How often to poll backend health checks.
    #[serde(default = "default_health_check_interval_secs")]
    pub health_check_interval_secs: u64,

    /// Temperature (in celsius) above which a GPU is considered `Degraded`.
    #[serde(default = "default_max_healthy_temperature")]
    pub max_healthy_temperature: u32,

    /// Temperature (in celsius) above which a GPU is considered `Unhealthy`.
    #[serde(default = "default_max_degraded_temperature")]
    pub max_degraded_temperature: u32,

    /// VRAM headroom to reserve on every GPU (for KV cache growth, etc.).
    #[serde(default = "default_vram_headroom")]
    pub vram_headroom: ByteSize,

    /// Grace period (seconds) given to an in-flight stream before a backend
    /// targeted for eviction is forcibly stopped.
    #[serde(default = "default_eviction_grace_period_secs")]
    pub eviction_grace_period_secs: u64,

    /// Lower bound of the TCP port range used to allocate backend listen
    /// ports. On startup, Concerto scans this range for orphan processes and
    /// kills them with a warning before starting.
    #[serde(default = "default_port_range_start")]
    pub port_range_start: u16,

    /// Exclusive upper bound of the TCP port range used for backend listen
    /// ports. Must be strictly greater than `port_range_start`.
    #[serde(default = "default_port_range_end")]
    pub port_range_end: u16,

    /// Per-request timeout applied as an axum middleware. `0` disables the
    /// timeout (intended default for v0.1 — the TimeoutLayer ships with
    /// Sprint 3, the config field is here so users can prepare).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
}

impl Default for RoutingSection {
    fn default() -> Self {
        Self {
            eviction_policy: default_eviction_policy(),
            cold_start_timeout_secs: default_cold_start_timeout_secs(),
            health_check_interval_secs: default_health_check_interval_secs(),
            max_healthy_temperature: default_max_healthy_temperature(),
            max_degraded_temperature: default_max_degraded_temperature(),
            vram_headroom: default_vram_headroom(),
            eviction_grace_period_secs: default_eviction_grace_period_secs(),
            port_range_start: default_port_range_start(),
            port_range_end: default_port_range_end(),
            request_timeout_secs: default_request_timeout_secs(),
        }
    }
}

fn default_eviction_policy() -> EvictionPolicy {
    EvictionPolicy::Lru
}

fn default_cold_start_timeout_secs() -> u64 {
    120
}

fn default_health_check_interval_secs() -> u64 {
    10
}

fn default_max_healthy_temperature() -> u32 {
    75
}

fn default_max_degraded_temperature() -> u32 {
    85
}

fn default_vram_headroom() -> ByteSize {
    ByteSize::gb(1)
}

fn default_eviction_grace_period_secs() -> u64 {
    30
}

fn default_port_range_start() -> u16 {
    8100
}

fn default_port_range_end() -> u16 {
    9000
}

fn default_request_timeout_secs() -> u64 {
    0
}
