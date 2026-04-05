//! ROADMAP §6.3 scenario 10: when every GPU is marked Unhealthy, routing
//! rejects with a clear reason rather than blindly attempting to place.

use concerto_api::orchestrator::route_and_dispatch;
use concerto_api::ApiError;
use concerto_core::{GpuHealth, ModelId};
use concerto_scenarios::{build_harness, ScenarioConfig};

#[tokio::test]
async fn all_gpus_unhealthy_rejects_with_503() {
    let h = build_harness(ScenarioConfig {
        gpu_count: 2,
        memory_per_gpu_gb: 24,
        models: vec![("model-a".into(), 8)],
    })
    .await;

    // Mark every GPU in the cluster snapshot Unhealthy. (The GPU monitor
    // snapshot is separate from the cluster snapshot; the orchestrator
    // decision is made against cluster state, so that's what we mutate.)
    {
        let mut cluster = h.state.cluster.lock().await;
        for gpu in &mut cluster.gpus {
            gpu.health = GpuHealth::Unhealthy;
        }
    }

    let err = route_and_dispatch(&h.state, ModelId("model-a".into()))
        .await
        .expect_err("must reject when no GPU is usable");
    assert!(
        matches!(err, ApiError::BackendUnavailable(_)),
        "got {err:?}"
    );
    assert_eq!(h.backend.launched_count(), 0);
}
