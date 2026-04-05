//! ROADMAP §6.3 scenario 2: a second request for the same model is served
//! from the existing backend — no new launch, request counter bumps.

use concerto_api::orchestrator::route_and_dispatch;
use concerto_core::ModelId;
use concerto_scenarios::{build_harness, ScenarioConfig};

#[tokio::test]
async fn cached_request_reuses_backend() {
    let h = build_harness(ScenarioConfig {
        gpu_count: 1,
        memory_per_gpu_gb: 24,
        models: vec![("model-a".into(), 8)],
    })
    .await;

    let first = route_and_dispatch(&h.state, ModelId("model-a".into()))
        .await
        .expect("first request loads");
    let second = route_and_dispatch(&h.state, ModelId("model-a".into()))
        .await
        .expect("second request reuses");

    // Same port both times.
    assert_eq!(first.port, second.port);
    // Exactly one launch.
    assert_eq!(h.backend.launched_count(), 1);
    assert_eq!(h.backend.stopped_count(), 0);

    // The second request bumped the request_count on the loaded model.
    let cluster = h.state.cluster.lock().await;
    let loaded = cluster
        .gpus
        .iter()
        .flat_map(|g| g.loaded_models.iter())
        .find(|m| m.model_id == ModelId("model-a".into()))
        .expect("model is loaded");
    assert_eq!(
        loaded.request_count, 2,
        "both requests should have touched the counter"
    );
}
