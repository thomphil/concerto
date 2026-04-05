//! Property-based tests for the `concerto-core` routing function.
//!
//! These tests generate random cluster states and model specs, call
//! `route_request`, and assert invariants that must hold for *every*
//! valid input. Invariants tested:
//!
//! 1. `no_overcommit_after_load` — applying a decision never leaves a GPU
//!    with `memory_used > memory_total`.
//! 2. `determinism` — two calls on identical state produce identical decisions.
//! 3. `unknown_model_rejected` — a model not in the registry is always rejected.
//! 4. `lru_evicts_oldest_first` — under LRU, evicted models are at least as
//!    stale as any non-evicted model on the same GPU.
//! 5. `never_routes_to_unhealthy_gpu` — decisions never target an unhealthy GPU.
//! 6. `routed_model_is_actually_loaded` — RouteToLoaded must reference a
//!    GPU that really has the model with the matching port.
//! 7. `eviction_frees_enough_space` — when a decision evicts, the freed VRAM
//!    plus existing headroom covers the new model's requirement.

use bytesize::ByteSize;
use chrono::{DateTime, Duration, Utc};
use concerto_core::{
    route_request, ClusterState, EngineType, EvictionPolicy, GpuHealth, GpuId, GpuState,
    LoadedModel, ModelId, ModelSpec, RoutingConfig, RoutingDecision,
};
use proptest::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper constructors built directly from the public API.
// The crate-private `test_helpers` module in state.rs is `#[cfg(test)]`-gated
// and therefore not visible from integration tests.
// ---------------------------------------------------------------------------

fn make_loaded_model(id: &str, vram_gb: u64, port: u16, age_secs: i64) -> LoadedModel {
    LoadedModel {
        model_id: ModelId(id.to_string()),
        vram_usage: ByteSize::gb(vram_gb),
        last_request_at: Utc::now() - Duration::seconds(age_secs),
        request_count: 0,
        backend_port: port,
    }
}

fn make_gpu_state(
    id: usize,
    total_gb: u64,
    models: Vec<LoadedModel>,
    health: GpuHealth,
) -> GpuState {
    let memory_total = ByteSize::gb(total_gb);
    let used: u64 = models.iter().map(|m| m.vram_usage.as_u64()).sum();
    let memory_used = ByteSize::b(used.min(memory_total.as_u64()));
    let memory_available =
        ByteSize::b(memory_total.as_u64().saturating_sub(memory_used.as_u64()));
    GpuState {
        id: GpuId(id),
        memory_total,
        memory_used,
        memory_available,
        temperature_celsius: 45,
        utilisation_percent: 0,
        health,
        loaded_models: models,
    }
}

fn make_model_spec(id: &str, vram_gb: u64) -> (ModelId, ModelSpec) {
    let model_id = ModelId(id.to_string());
    let spec = ModelSpec {
        id: model_id.clone(),
        name: id.to_string(),
        weight_path: format!("/models/{id}"),
        vram_required: ByteSize::gb(vram_gb),
        engine: EngineType::Mock,
        engine_args: vec![],
    };
    (model_id, spec)
}

/// Mutate a ClusterState as the orchestrator would, given a routing decision.
/// - `RouteToLoaded` leaves the state unchanged (we only serve from an existing backend).
/// - `LoadModel { gpu_id, evict }` removes the listed models from the GPU, then
///   appends the newly-loaded model with its required VRAM footprint.
/// - `Reject` leaves the state unchanged.
fn apply_decision(
    state: &mut ClusterState,
    model_id: &ModelId,
    decision: &RoutingDecision,
) {
    match decision {
        RoutingDecision::RouteToLoaded { .. } | RoutingDecision::Reject { .. } => {}
        RoutingDecision::LoadModel { gpu_id, evict } => {
            let spec_vram = state
                .get_model_spec(model_id)
                .map_or(ByteSize::b(0), |s| s.vram_required);
            if let Some(gpu) = state.gpus.iter_mut().find(|g| g.id == *gpu_id) {
                gpu.loaded_models.retain(|m| !evict.contains(&m.model_id));
                if !gpu.loaded_models.iter().any(|m| &m.model_id == model_id) {
                    gpu.loaded_models.push(LoadedModel {
                        model_id: model_id.clone(),
                        vram_usage: spec_vram,
                        last_request_at: Utc::now(),
                        request_count: 0,
                        backend_port: 9000,
                    });
                }
                let used: u64 = gpu.loaded_models.iter().map(|m| m.vram_usage.as_u64()).sum();
                gpu.memory_used = ByteSize::b(used);
                gpu.memory_available =
                    ByteSize::b(gpu.memory_total.as_u64().saturating_sub(used));
            }
        }
    }
}

fn default_config() -> RoutingConfig {
    RoutingConfig {
        vram_headroom: ByteSize::gb(1),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Proptest strategies
// ---------------------------------------------------------------------------

/// Strategy producing a `GpuHealth`, weighted heavily toward `Healthy` so that
/// most generated scenarios can actually make progress.
fn arb_health() -> impl Strategy<Value = GpuHealth> {
    prop_oneof![
        8 => Just(GpuHealth::Healthy),
        2 => Just(GpuHealth::Degraded),
        1 => Just(GpuHealth::Unhealthy),
    ]
}

/// Strategy producing a `LoadedModel` with up to `max_vram_gb` of VRAM usage.
/// `slot` is used to make unique model ids and ports per GPU.
fn arb_loaded_model(gpu_id: usize, slot: usize, max_vram_gb: u64) -> impl Strategy<Value = LoadedModel> {
    (
        1u64..=max_vram_gb.max(1),
        0i64..=86_400, // up to 1 day of staleness
    )
        .prop_map(move |(vram_gb, age_secs)| {
            let id = format!("gpu{gpu_id}-model{slot}");
            let port = 8000 + (gpu_id * 100 + slot) as u16;
            make_loaded_model(&id, vram_gb, port, age_secs)
        })
}

/// Strategy producing a `GpuState`. The generator guarantees:
/// - `1 <= memory_total <= 80 GB`
/// - `sum(loaded_models.vram_usage) <= memory_used <= memory_total`
/// - 0..=5 loaded models
/// - temperature in 30..=100, utilisation in 0..=100
fn arb_gpu_state(id: usize) -> impl Strategy<Value = GpuState> {
    (1u64..=80, 30u32..=100, 0u32..=100, arb_health(), 0usize..=5).prop_flat_map(
        move |(total_gb, temp, util, health, model_count)| {
            let models_strategy: Vec<_> = (0..model_count)
                .map(|slot| arb_loaded_model(id, slot, total_gb))
                .collect();
            models_strategy.prop_map(move |raw_models| {
                // Drop any models that would exceed the GPU's capacity.
                let cap = ByteSize::gb(total_gb).as_u64();
                let mut used: u64 = 0;
                let accepted: Vec<LoadedModel> = raw_models
                    .into_iter()
                    .filter(|m| {
                        let v = m.vram_usage.as_u64();
                        if used + v <= cap {
                            used += v;
                            true
                        } else {
                            false
                        }
                    })
                    .collect();
                let mut gpu = make_gpu_state(id, total_gb, accepted, health);
                gpu.temperature_celsius = temp;
                gpu.utilisation_percent = util;
                gpu
            })
        },
    )
}

/// Strategy producing a ClusterState with 1..=4 GPUs and 1..=10 registered models.
/// The registry always contains the model ids that were placed on GPUs, plus
/// some extras that may or may not be loaded anywhere.
fn arb_cluster_state() -> impl Strategy<Value = ClusterState> {
    (1usize..=4, 1usize..=10).prop_flat_map(|(gpu_count, extra_model_count)| {
        let gpu_strategies: Vec<_> = (0..gpu_count).map(arb_gpu_state).collect();
        gpu_strategies.prop_map(move |gpus| {
            let mut registry: HashMap<ModelId, ModelSpec> = HashMap::new();
            for gpu in &gpus {
                for m in &gpu.loaded_models {
                    // vram_usage is generated in whole GB so truncation is lossless.
                    let vram_gb = m.vram_usage.as_u64() / 1_000_000_000;
                    let (id, spec) = make_model_spec(&m.model_id.0, vram_gb);
                    registry.entry(id).or_insert(spec);
                }
            }
            for i in 0..extra_model_count {
                let name = format!("extra-model-{i}");
                let size_gb = 1 + (i as u64 % 16);
                let (id, spec) = make_model_spec(&name, size_gb);
                registry.insert(id, spec);
            }
            ClusterState::new(gpus, registry)
        })
    })
}

/// Strategy producing a `(ClusterState, ModelId)` pair where the model id is
/// usually (but not always) present in the registry. We generate the pair
/// together because the model id's distribution depends on the state.
fn arb_cluster_and_request() -> impl Strategy<Value = (ClusterState, ModelId)> {
    arb_cluster_state().prop_flat_map(|state| {
        let registered: Vec<ModelId> = state.model_registry.keys().cloned().collect();
        let model_strategy: BoxedStrategy<ModelId> = if registered.is_empty() {
            "[a-z]{3,10}".prop_map(ModelId).boxed()
        } else {
            prop_oneof![
                5 => proptest::sample::select(registered),
                1 => "[a-z]{3,10}".prop_map(ModelId),
            ]
            .boxed()
        };
        (Just(state), model_strategy)
    })
}

// ---------------------------------------------------------------------------
// Proptests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Invariant 1: after applying a routing decision, no GPU is over-committed.
    #[test]
    fn no_overcommit_after_load((state, request_model) in arb_cluster_and_request()) {
        let config = default_config();
        let decision = route_request(&request_model, &state, &config);

        let mut mutated = state.clone();
        apply_decision(&mut mutated, &request_model, &decision);

        for gpu in &mutated.gpus {
            prop_assert!(
                gpu.memory_used <= gpu.memory_total,
                "gpu {} over-committed: used={} total={}",
                gpu.id.0,
                gpu.memory_used,
                gpu.memory_total,
            );
        }
    }

    /// Invariant 2: `route_request` is a pure function of its inputs.
    #[test]
    fn determinism((state, model_id) in arb_cluster_and_request()) {
        let config = default_config();
        let a = route_request(&model_id, &state, &config);
        let b = route_request(&model_id, &state.clone(), &config);
        prop_assert_eq!(a, b);
    }

    /// Invariant 3: unknown models are always rejected.
    #[test]
    fn unknown_model_rejected(state in arb_cluster_state(), unknown in "[A-Z]{20,30}") {
        // Use an uppercase id that definitely isn't in the registry (all registry ids are lowercase).
        let model_id = ModelId(unknown);
        prop_assume!(!state.model_registry.contains_key(&model_id));
        // Also ensure no loaded model has this id.
        prop_assume!(!state
            .gpus
            .iter()
            .any(|g| g.loaded_models.iter().any(|m| m.model_id == model_id)));

        let decision = route_request(&model_id, &state, &default_config());
        prop_assert!(
            matches!(decision, RoutingDecision::Reject { .. }),
            "expected Reject for unknown model, got {:?}",
            decision,
        );
    }

    /// Invariant 4: under LRU, every evicted model is at least as stale as
    /// every model left behind on the same GPU.
    #[test]
    fn lru_evicts_oldest_first(state in arb_cluster_state()) {
        let config = RoutingConfig {
            eviction_policy: EvictionPolicy::Lru,
            ..default_config()
        };
        // Try every registered model id — we want to maximise the chance of
        // triggering an eviction path.
        for model_id in state.model_registry.keys() {
            let decision = route_request(model_id, &state, &config);
            if let RoutingDecision::LoadModel { gpu_id, evict } = &decision {
                if evict.is_empty() {
                    continue;
                }
                let gpu = state.gpus.iter().find(|g| g.id == *gpu_id).unwrap();
                let max_evicted_ts: DateTime<Utc> = gpu
                    .loaded_models
                    .iter()
                    .filter(|m| evict.contains(&m.model_id))
                    .map(|m| m.last_request_at)
                    .max()
                    .expect("non-empty eviction set must match models on GPU");
                for m in &gpu.loaded_models {
                    if !evict.contains(&m.model_id) {
                        prop_assert!(
                            m.last_request_at >= max_evicted_ts,
                            "LRU evicted a newer model than one it kept on gpu {}",
                            gpu.id.0,
                        );
                    }
                }
            }
        }
    }

    /// Invariant 5: routing decisions never target an unhealthy GPU.
    #[test]
    fn never_routes_to_unhealthy_gpu(state in arb_cluster_state()) {
        let config = default_config();
        for model_id in state.model_registry.keys() {
            let decision = route_request(model_id, &state, &config);
            let gpu_id = match &decision {
                RoutingDecision::RouteToLoaded { gpu_id, .. } => Some(*gpu_id),
                RoutingDecision::LoadModel { gpu_id, .. } => Some(*gpu_id),
                RoutingDecision::Reject { .. } => None,
            };
            if let Some(gid) = gpu_id {
                let gpu = state.gpus.iter().find(|g| g.id == gid).unwrap();
                prop_assert!(
                    gpu.health != GpuHealth::Unhealthy,
                    "decision {:?} targeted unhealthy gpu {}",
                    decision,
                    gid.0,
                );
            }
        }
    }

    /// Invariant 6: `RouteToLoaded` must point at a GPU that really has the
    /// requested model with the matching backend port.
    #[test]
    fn routed_model_is_actually_loaded(state in arb_cluster_state()) {
        let config = default_config();
        for model_id in state.model_registry.keys() {
            let decision = route_request(model_id, &state, &config);
            if let RoutingDecision::RouteToLoaded { gpu_id, port } = decision {
                let gpu = state.gpus.iter().find(|g| g.id == gpu_id).unwrap();
                let found = gpu
                    .loaded_models
                    .iter()
                    .any(|m| &m.model_id == model_id && m.backend_port == port);
                prop_assert!(
                    found,
                    "RouteToLoaded pointed at gpu {} port {} but no matching model found",
                    gpu_id.0,
                    port,
                );
            }
        }
    }

    /// Invariant 7 (bonus): when a decision evicts models, the freed VRAM
    /// plus the GPU's current available memory must be enough to fit the new
    /// model (excluding headroom, which the router may choose to waive).
    #[test]
    fn eviction_frees_enough_space(state in arb_cluster_state()) {
        let config = default_config();
        for model_id in state.model_registry.keys() {
            let decision = route_request(model_id, &state, &config);
            if let RoutingDecision::LoadModel { gpu_id, evict } = &decision {
                if evict.is_empty() {
                    continue;
                }
                let gpu = state.gpus.iter().find(|g| g.id == *gpu_id).unwrap();
                let spec = state.get_model_spec(model_id).unwrap();
                let freed: u64 = gpu
                    .loaded_models
                    .iter()
                    .filter(|m| evict.contains(&m.model_id))
                    .map(|m| m.vram_usage.as_u64())
                    .sum();
                let total_available = gpu.memory_available.as_u64() + freed;
                prop_assert!(
                    total_available >= spec.vram_required.as_u64(),
                    "eviction insufficient on gpu {}: have {} after eviction, need {}",
                    gpu.id.0,
                    total_available,
                    spec.vram_required.as_u64(),
                );
            }
        }
    }
}
