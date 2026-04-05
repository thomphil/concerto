//! ROADMAP §6.3 scenario 1: one request for a cold model launches exactly
//! one backend and returns a valid dispatch target.

use concerto_api::orchestrator::route_and_dispatch;
use concerto_core::ModelId;
use concerto_scenarios::{build_harness, ScenarioConfig};

#[tokio::test]
async fn cold_start_single_request() {
    let h = build_harness(ScenarioConfig {
        gpu_count: 2,
        memory_per_gpu_gb: 24,
        models: vec![("model-a".into(), 8)],
    })
    .await;

    let target = route_and_dispatch(&h.state, ModelId("model-a".into()))
        .await
        .expect("cold start should succeed");

    // Exactly one launch, no stops, and the target port matches the handle
    // we just recorded in state.backends.
    assert_eq!(h.backend.launched_count(), 1);
    assert_eq!(h.backend.stopped_count(), 0);

    let backends = h.state.backends.lock().await;
    let handle = backends
        .get(&ModelId("model-a".into()))
        .expect("handle recorded in state.backends");
    assert_eq!(target.port, handle.port);
}
