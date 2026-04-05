//! Top-level Concerto configuration.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use concerto_core::{ModelId, ModelSpec, RoutingConfig};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::error::ConfigError;
use crate::gpus::GpuConfigEntry;
use crate::models::ModelConfigEntry;
use crate::routing::RoutingSection;
use crate::server::ServerConfig;

/// Fully parsed and validated Concerto configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConcertoConfig {
    #[serde(default)]
    pub server: ServerConfig,

    #[serde(default)]
    pub routing: RoutingSection,

    #[serde(default)]
    pub models: Vec<ModelConfigEntry>,

    #[serde(default)]
    pub gpus: Vec<GpuConfigEntry>,
}

impl ConcertoConfig {
    /// Parse a configuration from an in-memory TOML string.
    ///
    /// The returned config is guaranteed to have passed [`Self::validate`].
    pub fn from_toml_str(input: &str) -> Result<Self, ConfigError> {
        let config: ConcertoConfig = toml::from_str(input)?;
        config.validate()?;
        debug!(
            models = config.models.len(),
            gpus = config.gpus.len(),
            "loaded concerto config"
        );
        Ok(config)
    }

    /// Load a configuration from a TOML file on disk.
    pub fn from_path(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_toml_str(&contents)
    }

    /// Validate semantic constraints that cannot be expressed in the TOML
    /// schema itself (unique IDs, non-empty sections, valid port).
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.models.is_empty() {
            return Err(ConfigError::EmptyModels);
        }
        if self.gpus.is_empty() {
            return Err(ConfigError::EmptyGpus);
        }
        if self.server.port == 0 {
            return Err(ConfigError::InvalidPort(self.server.port));
        }

        let mut seen_models = HashSet::with_capacity(self.models.len());
        for model in &self.models {
            if !seen_models.insert(model.id.as_str()) {
                return Err(ConfigError::DuplicateModelId(model.id.clone()));
            }
        }

        let mut seen_gpus = HashSet::with_capacity(self.gpus.len());
        for gpu in &self.gpus {
            if !seen_gpus.insert(gpu.id) {
                return Err(ConfigError::DuplicateGpuId(gpu.id));
            }
        }

        Ok(())
    }

    /// Build a model registry keyed by [`ModelId`] for use by `concerto-core`.
    pub fn model_registry(&self) -> HashMap<ModelId, ModelSpec> {
        self.models
            .iter()
            .map(|entry| (ModelId(entry.id.clone()), ModelSpec::from(entry)))
            .collect()
    }

    /// Convert the `[routing]` section into the core [`RoutingConfig`].
    ///
    /// Fields that have no analogue on [`RoutingConfig`] (such as
    /// `cold_start_timeout_secs`) are consumed elsewhere in the system.
    pub fn routing_config(&self) -> RoutingConfig {
        RoutingConfig {
            eviction_policy: self.routing.eviction_policy,
            max_healthy_temperature: self.routing.max_healthy_temperature,
            max_degraded_temperature: self.routing.max_degraded_temperature,
            vram_headroom: self.routing.vram_headroom,
        }
    }
}
