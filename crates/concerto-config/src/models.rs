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
        }
    }
}
