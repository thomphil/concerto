use crate::eviction::select_evictions;
use crate::state::ClusterState;
use crate::types::*;

/// The core routing function. Pure logic, no IO.
///
/// Given a model ID and the current cluster state, decide what to do:
/// - Route to an already-loaded backend
/// - Load the model (possibly evicting others first)
/// - Reject the request
pub fn route_request(
    model_id: &ModelId,
    cluster: &ClusterState,
    config: &RoutingConfig,
) -> RoutingDecision {
    // 1. Is the model already loaded somewhere?
    if let Some((gpu_id, port)) = cluster.find_loaded_model(model_id) {
        // Check if the GPU it's on is still healthy enough to serve
        if let Some(gpu) = cluster.gpus.iter().find(|g| g.id == gpu_id) {
            if gpu.health != GpuHealth::Unhealthy {
                return RoutingDecision::RouteToLoaded { gpu_id, port };
            }
            // GPU is unhealthy — fall through to find another placement
        }
    }

    // 2. Look up the model spec
    let spec = match cluster.get_model_spec(model_id) {
        Some(spec) => spec,
        None => {
            return RoutingDecision::Reject {
                reason: format!("Model '{}' not found in registry", model_id),
            };
        }
    };

    // 3. Find a GPU that can fit this model without eviction
    let candidates = cluster.gpus_with_space_for(spec.vram_required, config.vram_headroom);
    if let Some(best_gpu) = select_best_gpu(&candidates, config) {
        return RoutingDecision::LoadModel {
            gpu_id: best_gpu.id,
            evict: vec![],
        };
    }

    // 4. No GPU has enough free space — try eviction on each GPU
    let healthy_gpus = cluster.healthy_gpus_by_available_memory();
    for gpu in healthy_gpus {
        if let Some(evictions) = select_evictions(
            gpu,
            spec.vram_required,
            config.vram_headroom,
            config.eviction_policy,
        ) {
            if !evictions.is_empty() {
                return RoutingDecision::LoadModel {
                    gpu_id: gpu.id,
                    evict: evictions,
                };
            }
        }
    }

    // 5. Cannot serve — not enough memory even after eviction
    RoutingDecision::Reject {
        reason: format!(
            "No GPU can fit model '{}' (requires {}), even after eviction",
            model_id, spec.vram_required
        ),
    }
}

/// Select the best GPU from candidates that already have enough free space.
/// Prefers: healthy > degraded, then most available memory.
fn select_best_gpu<'a>(
    candidates: &[&'a GpuState],
    _config: &RoutingConfig,
) -> Option<&'a GpuState> {
    if candidates.is_empty() {
        return None;
    }

    // Sort: healthy first, then by most available memory
    let mut sorted: Vec<&&GpuState> = candidates.iter().collect();
    sorted.sort_by(|a, b| {
        let health_ord = health_priority(a.health).cmp(&health_priority(b.health));
        if health_ord != std::cmp::Ordering::Equal {
            return health_ord;
        }
        b.memory_available.cmp(&a.memory_available)
    });

    sorted.first().copied().copied()
}

fn health_priority(health: GpuHealth) -> u8 {
    match health {
        GpuHealth::Healthy => 0,
        GpuHealth::Degraded => 1,
        GpuHealth::Unhealthy => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::test_helpers::*;
    use bytesize::ByteSize;
    use chrono::{Duration, Utc};
    use std::collections::HashMap;

    fn make_registry(specs: Vec<(&str, u64)>) -> HashMap<ModelId, ModelSpec> {
        specs
            .into_iter()
            .map(|(id, vram_gb)| test_model_spec(id, vram_gb))
            .collect()
    }

    fn default_config() -> RoutingConfig {
        RoutingConfig {
            vram_headroom: ByteSize::gb(1),
            ..Default::default()
        }
    }

    #[test]
    fn routes_to_already_loaded_model() {
        let gpus = vec![GpuStateBuilder::new(0)
            .memory_total_gb(24)
            .with_model("model-a", 8, 8001)
            .build()];
        let registry = make_registry(vec![("model-a", 8)]);
        let cluster = ClusterState::new(gpus, registry);

        let decision = route_request(&"model-a".into(), &cluster, &default_config());

        assert!(matches!(
            decision,
            RoutingDecision::RouteToLoaded {
                gpu_id: GpuId(0),
                port: 8001
            }
        ));
    }

    #[test]
    fn loads_model_on_gpu_with_space() {
        let gpus = vec![
            GpuStateBuilder::new(0).memory_total_gb(24).build(),
            GpuStateBuilder::new(1).memory_total_gb(24).build(),
        ];
        let registry = make_registry(vec![("model-a", 8)]);
        let cluster = ClusterState::new(gpus, registry);

        let decision = route_request(&"model-a".into(), &cluster, &default_config());

        assert!(matches!(
            decision,
            RoutingDecision::LoadModel { evict, .. } if evict.is_empty()
        ));
    }

    #[test]
    fn evicts_when_no_space_available() {
        let old = Utc::now() - Duration::hours(2);
        let recent = Utc::now() - Duration::minutes(5);

        let gpus = vec![GpuStateBuilder::new(0)
            .memory_total_gb(24)
            .with_model_last_used("old-model", 12, 8001, old)
            .with_model_last_used("recent-model", 10, 8002, recent)
            .build()];
        let registry = make_registry(vec![
            ("old-model", 12),
            ("recent-model", 10),
            ("new-model", 10),
        ]);
        let cluster = ClusterState::new(gpus, registry);

        let decision = route_request(&"new-model".into(), &cluster, &default_config());

        match decision {
            RoutingDecision::LoadModel { gpu_id, evict } => {
                assert_eq!(gpu_id, GpuId(0));
                assert!(evict.contains(&ModelId("old-model".into())));
            }
            other => panic!("Expected LoadModel, got {:?}", other),
        }
    }

    #[test]
    fn rejects_unknown_model() {
        let gpus = vec![GpuStateBuilder::new(0).memory_total_gb(24).build()];
        let registry = make_registry(vec![]);
        let cluster = ClusterState::new(gpus, registry);

        let decision = route_request(&"nonexistent".into(), &cluster, &default_config());

        assert!(matches!(decision, RoutingDecision::Reject { .. }));
    }

    #[test]
    fn rejects_model_too_large_for_any_gpu() {
        let gpus = vec![
            GpuStateBuilder::new(0).memory_total_gb(24).build(),
            GpuStateBuilder::new(1).memory_total_gb(24).build(),
        ];
        let registry = make_registry(vec![("huge-model", 48)]);
        let cluster = ClusterState::new(gpus, registry);

        let decision = route_request(&"huge-model".into(), &cluster, &default_config());

        assert!(matches!(decision, RoutingDecision::Reject { .. }));
    }

    #[test]
    fn avoids_unhealthy_gpu() {
        let gpus = vec![
            GpuStateBuilder::new(0)
                .memory_total_gb(24)
                .health(GpuHealth::Unhealthy)
                .build(),
            GpuStateBuilder::new(1)
                .memory_total_gb(24)
                .health(GpuHealth::Healthy)
                .build(),
        ];
        let registry = make_registry(vec![("model-a", 8)]);
        let cluster = ClusterState::new(gpus, registry);

        let decision = route_request(&"model-a".into(), &cluster, &default_config());

        match decision {
            RoutingDecision::LoadModel { gpu_id, .. } => {
                assert_eq!(gpu_id, GpuId(1), "Should route to healthy GPU");
            }
            other => panic!("Expected LoadModel, got {:?}", other),
        }
    }

    #[test]
    fn prefers_healthy_over_degraded() {
        let gpus = vec![
            GpuStateBuilder::new(0)
                .memory_total_gb(48)
                .health(GpuHealth::Degraded)
                .build(),
            GpuStateBuilder::new(1)
                .memory_total_gb(24)
                .health(GpuHealth::Healthy)
                .build(),
        ];
        let registry = make_registry(vec![("model-a", 8)]);
        let cluster = ClusterState::new(gpus, registry);

        let decision = route_request(&"model-a".into(), &cluster, &default_config());

        match decision {
            RoutingDecision::LoadModel { gpu_id, .. } => {
                assert_eq!(
                    gpu_id,
                    GpuId(1),
                    "Should prefer healthy GPU even with less memory"
                );
            }
            other => panic!("Expected LoadModel, got {:?}", other),
        }
    }

    #[test]
    fn does_not_route_to_unhealthy_gpu_even_if_model_loaded_there() {
        let gpus = vec![
            GpuStateBuilder::new(0)
                .memory_total_gb(24)
                .health(GpuHealth::Unhealthy)
                .with_model("model-a", 8, 8001)
                .build(),
            GpuStateBuilder::new(1)
                .memory_total_gb(24)
                .health(GpuHealth::Healthy)
                .build(),
        ];
        let registry = make_registry(vec![("model-a", 8)]);
        let cluster = ClusterState::new(gpus, registry);

        let decision = route_request(&"model-a".into(), &cluster, &default_config());

        // Should NOT route to GPU 0 even though model-a is loaded there
        assert!(
            !matches!(
                decision,
                RoutingDecision::RouteToLoaded {
                    gpu_id: GpuId(0),
                    ..
                }
            ),
            "Should not route to unhealthy GPU"
        );
    }
}
