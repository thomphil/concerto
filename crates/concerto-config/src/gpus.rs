//! The `[[gpus]]` section of the configuration.

use serde::{Deserialize, Serialize};

/// One entry in the `[[gpus]]` table.
///
/// GPUs are enumerated by index on the local node. All fields beyond `id` are
/// optional per-GPU overrides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuConfigEntry {
    /// 0-based GPU index on the node.
    pub id: usize,

    /// Optional per-GPU override for the maximum operating temperature.
    #[serde(default)]
    pub max_temperature: Option<u32>,
}
