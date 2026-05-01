//! The `[[models]]` section of the configuration.

use bytesize::ByteSize;
use concerto_core::{EngineType, ModelId, ModelSpec};
use serde::{Deserialize, Serialize};

/// One entry in the `[[models]]` table.
///
/// This mirrors the on-disk TOML shape; the canonical in-memory representation
/// used by [`concerto_core`] is [`ModelSpec`], which you can obtain via `From`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfigEntry {
    /// Stable identifier used in API requests (matches OpenAI's `model` field).
    pub id: String,

    /// Human-friendly display name.
    pub name: String,

    /// Filesystem path to the model weights.
    pub weight_path: String,

    /// Estimated VRAM footprint, e.g. `"14GB"`.
    pub vram_required: ByteSize,

    /// Which inference engine to use.
    pub engine: EngineType,

    /// Extra command-line arguments to pass to the engine.
    #[serde(default)]
    pub engine_args: Vec<String>,

    /// Protect this model from eviction. Pinned models are never selected as
    /// eviction candidates under any policy. If the only way to make room for
    /// a new request would be to evict a pinned model, the request is rejected
    /// with a clear reason instead.
    #[serde(default)]
    pub pin: bool,

    /// Per-model VRAM budget as a fraction of the GPU's total memory, in the
    /// open range `(0.0, 1.0]`.
    ///
    /// **vLLM-specific.** Translated to `--gpu-memory-utilization <x>` when
    /// the engine is `vllm`. Silently ignored on other engines (with a
    /// load-time warning). If `engine_args` already specifies
    /// `--gpu-memory-utilization`, the explicit value wins and a warning is
    /// logged at config load.
    #[serde(default)]
    pub max_vram_fraction: Option<f64>,
}

impl From<&ModelConfigEntry> for ModelSpec {
    fn from(entry: &ModelConfigEntry) -> Self {
        ModelSpec {
            id: ModelId(entry.id.clone()),
            name: entry.name.clone(),
            weight_path: entry.weight_path.clone(),
            vram_required: entry.vram_required,
            engine: entry.engine.clone(),
            engine_args: entry.engine_args.clone(),
            pin: entry.pin,
            max_vram_fraction: entry.max_vram_fraction,
        }
    }
}
