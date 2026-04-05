use bytesize::ByteSize;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a GPU (index on the node, 0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GpuId(pub usize);

impl fmt::Display for GpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu:{}", self.0)
    }
}

/// Unique identifier for a model, matching the `model` field in OpenAI API requests.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        ModelId(s.to_string())
    }
}

/// The health classification of a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuHealth {
    /// Operating normally
    Healthy,
    /// Showing warning signs (high temp, rising ECC errors) but still functional
    Degraded,
    /// Should not receive new workloads
    Unhealthy,
}

/// Snapshot of a single GPU's current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuState {
    pub id: GpuId,
    pub memory_total: ByteSize,
    pub memory_used: ByteSize,
    pub memory_available: ByteSize,
    pub temperature_celsius: u32,
    pub utilisation_percent: u32,
    pub health: GpuHealth,
    pub loaded_models: Vec<LoadedModel>,
}

impl GpuState {
    /// Can this GPU fit an additional model of the given size?
    pub fn can_fit(&self, vram_required: ByteSize) -> bool {
        self.health != GpuHealth::Unhealthy && self.memory_available >= vram_required
    }

    /// How much VRAM would be available if we evicted the given models?
    pub fn available_after_eviction(&self, models_to_evict: &[ModelId]) -> ByteSize {
        let freed: u64 = self
            .loaded_models
            .iter()
            .filter(|m| models_to_evict.contains(&m.model_id))
            .map(|m| m.vram_usage.as_u64())
            .sum();
        ByteSize::b(self.memory_available.as_u64() + freed)
    }
}

/// A model currently loaded on a GPU with an active backend process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModel {
    pub model_id: ModelId,
    pub vram_usage: ByteSize,
    pub last_request_at: DateTime<Utc>,
    pub request_count: u64,
    pub backend_port: u16,
}

/// Specification for a model that Concerto can serve (from config).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: ModelId,
    pub name: String,
    pub weight_path: String,
    pub vram_required: ByteSize,
    pub engine: EngineType,
    pub engine_args: Vec<String>,
}

/// Which inference engine to use for a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineType {
    Vllm,
    LlamaCpp,
    Sglang,
    /// For testing — uses our mock backend
    Mock,
}

/// Configuration for routing behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Which eviction policy to use when GPU memory is full
    pub eviction_policy: EvictionPolicy,
    /// Maximum temperature before a GPU is considered degraded
    pub max_healthy_temperature: u32,
    /// Maximum temperature before a GPU is considered unhealthy
    pub max_degraded_temperature: u32,
    /// How many bytes of VRAM headroom to maintain (for KV cache growth)
    pub vram_headroom: ByteSize,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            eviction_policy: EvictionPolicy::Lru,
            max_healthy_temperature: 75,
            max_degraded_temperature: 85,
            vram_headroom: ByteSize::gb(1),
        }
    }
}

/// Available eviction policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    /// Least Recently Used — evict the model with the oldest last_request_at
    Lru,
    /// Least Frequently Used — evict the model with the fewest requests
    Lfu,
    /// Size-weighted LRU — prefer evicting larger models when recency is similar
    SizeWeightedLru,
}

/// The output of the routing function. Describes what action Concerto should take.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingDecision {
    /// Model is already loaded — route the request to this backend.
    RouteToLoaded { gpu_id: GpuId, port: u16 },
    /// Model needs to be loaded. Optionally evict other models first.
    LoadModel { gpu_id: GpuId, evict: Vec<ModelId> },
    /// Cannot serve this request.
    Reject { reason: String },
}

/// Errors that can occur in core logic.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("Model not found in registry: {0}")]
    ModelNotFound(ModelId),
    #[error("No GPU available with sufficient memory for model {0} (requires {1})")]
    InsufficientMemory(ModelId, ByteSize),
    #[error("All GPUs are unhealthy")]
    AllGpusUnhealthy,
}
