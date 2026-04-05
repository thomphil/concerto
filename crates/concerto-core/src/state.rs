use crate::types::*;
use bytesize::ByteSize;
use std::collections::HashMap;

/// The complete state of the GPU cluster at a point in time.
/// This is the input to all routing decisions.
#[derive(Debug, Clone)]
pub struct ClusterState {
    pub gpus: Vec<GpuState>,
    pub model_registry: HashMap<ModelId, ModelSpec>,
}

impl ClusterState {
    pub fn new(gpus: Vec<GpuState>, model_registry: HashMap<ModelId, ModelSpec>) -> Self {
        Self {
            gpus,
            model_registry,
        }
    }

    /// Find which GPU (if any) currently has the given model loaded.
    pub fn find_loaded_model(&self, model_id: &ModelId) -> Option<(GpuId, u16)> {
        for gpu in &self.gpus {
            for model in &gpu.loaded_models {
                if &model.model_id == model_id {
                    return Some((gpu.id, model.backend_port));
                }
            }
        }
        None
    }

    /// Get the spec for a model from the registry.
    pub fn get_model_spec(&self, model_id: &ModelId) -> Option<&ModelSpec> {
        self.model_registry.get(model_id)
    }

    /// Get all healthy GPUs, sorted by available memory (most available first).
    pub fn healthy_gpus_by_available_memory(&self) -> Vec<&GpuState> {
        let mut gpus: Vec<&GpuState> = self
            .gpus
            .iter()
            .filter(|g| g.health != GpuHealth::Unhealthy)
            .collect();
        gpus.sort_by(|a, b| b.memory_available.cmp(&a.memory_available));
        gpus
    }

    /// Get all GPUs that could fit a model of the given size (with headroom).
    pub fn gpus_with_space_for(
        &self,
        vram_required: ByteSize,
        headroom: ByteSize,
    ) -> Vec<&GpuState> {
        let total_needed = ByteSize::b(vram_required.as_u64() + headroom.as_u64());
        self.gpus
            .iter()
            .filter(|g| g.health != GpuHealth::Unhealthy && g.memory_available >= total_needed)
            .collect()
    }
}

#[cfg(test)]
pub mod test_helpers {
    use super::*;
    use chrono::{DateTime, Utc};

    /// Builder for creating test GPU states.
    pub struct GpuStateBuilder {
        id: GpuId,
        memory_total: ByteSize,
        memory_used: ByteSize,
        temperature: u32,
        utilisation: u32,
        health: GpuHealth,
        loaded_models: Vec<LoadedModel>,
    }

    impl GpuStateBuilder {
        pub fn new(id: usize) -> Self {
            Self {
                id: GpuId(id),
                memory_total: ByteSize::gb(24),
                memory_used: ByteSize::b(0),
                temperature: 45,
                utilisation: 0,
                health: GpuHealth::Healthy,
                loaded_models: Vec::new(),
            }
        }

        pub fn memory_total_gb(mut self, gb: u64) -> Self {
            self.memory_total = ByteSize::gb(gb);
            self
        }

        pub fn temperature(mut self, celsius: u32) -> Self {
            self.temperature = celsius;
            self
        }

        pub fn health(mut self, health: GpuHealth) -> Self {
            self.health = health;
            self
        }

        pub fn with_model(mut self, model_id: &str, vram_gb: u64, port: u16) -> Self {
            let vram = ByteSize::gb(vram_gb);
            self.memory_used = ByteSize::b(self.memory_used.as_u64() + vram.as_u64());
            self.loaded_models.push(LoadedModel {
                model_id: ModelId(model_id.to_string()),
                vram_usage: vram,
                last_request_at: Utc::now(),
                request_count: 0,
                backend_port: port,
            });
            self
        }

        /// Add a model with a specific last_request_at for LRU testing.
        pub fn with_model_last_used(
            mut self,
            model_id: &str,
            vram_gb: u64,
            port: u16,
            last_request_at: DateTime<Utc>,
        ) -> Self {
            let vram = ByteSize::gb(vram_gb);
            self.memory_used = ByteSize::b(self.memory_used.as_u64() + vram.as_u64());
            self.loaded_models.push(LoadedModel {
                model_id: ModelId(model_id.to_string()),
                vram_usage: vram,
                last_request_at,
                request_count: 0,
                backend_port: port,
            });
            self
        }

        pub fn build(self) -> GpuState {
            let memory_available = ByteSize::b(
                self.memory_total
                    .as_u64()
                    .saturating_sub(self.memory_used.as_u64()),
            );
            GpuState {
                id: self.id,
                memory_total: self.memory_total,
                memory_used: self.memory_used,
                memory_available,
                temperature_celsius: self.temperature,
                utilisation_percent: self.utilisation,
                health: self.health,
                loaded_models: self.loaded_models,
            }
        }
    }

    /// Create a simple model spec for testing.
    pub fn test_model_spec(id: &str, vram_gb: u64) -> (ModelId, ModelSpec) {
        let model_id = ModelId(id.to_string());
        let spec = ModelSpec {
            id: model_id.clone(),
            name: id.to_string(),
            weight_path: format!("/models/{}", id),
            vram_required: ByteSize::gb(vram_gb),
            engine: EngineType::Mock,
            engine_args: vec![],
            pin: false,
        };
        (model_id, spec)
    }
}
