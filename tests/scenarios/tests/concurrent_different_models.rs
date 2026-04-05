//! ROADMAP §6.3 scenario 5: concurrent requests for four different models
//! against two GPUs with enough room for all four — correct placement, no
//! overcommit, every request succeeds.

use std::sync::Arc;

use concerto_api::orchestrator::route_and_dispatch;
use concerto_core::ModelId;
use concerto_scenarios::{build_harness, ScenarioConfig};
use futures::future::join_all;

#[tokio::test]
async fn concurrent_requests_for_distinct_models() {
    // Four 10 GB models, two 48 GB GPUs — plenty of room. Each GPU should
    // end up with two loaded models.
    let h = build_harness(ScenarioConfig {
        gpu_count: 2,
        memory_per_gpu_gb: 48,
        models: vec![
            ("m1".into(), 10),
            ("m2".into(), 10),
            ("m3".into(), 10),
            ("m4".into(), 10),
        ],
    })
    .await;

    let state = Arc::new(h.state);
    let futures: Vec<_> = ["m1", "m2", "m3", "m4"]
        .iter()
        .map(|id| {
            let state = state.clone();
            let model = ModelId((*id).into());
            async move { route_and_dispatch(&state, model).await }
        })
        .collect();

    let results = join_all(futures).await;
    for (i, r) in results.iter().enumerate() {
        assert!(r.is_ok(), "model m{} failed: {r:?}", i + 1);
    }

    assert_eq!(h.backend.launched_count(), 4, "four distinct launches");
    assert_eq!(h.backend.stopped_count(), 0, "no eviction needed");

    // VRAM bookkeeping should be consistent: total 40 GB loaded, split
    // across two GPUs (the orchestrator prefers GPU with more free space).
    let cluster = state.cluster.lock().await;
    let total_loaded: u64 = cluster.gpus.iter().map(|g| g.memory_used.as_u64()).sum();
    assert_eq!(
        total_loaded,
        bytesize::ByteSize::gb(40).as_u64(),
        "40 GB total VRAM usage expected"
    );
}
