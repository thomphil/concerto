//! ROADMAP §6.3 scenario 5: four concurrent requests for distinct models on
//! two 48 GB GPUs. Each GPU ends up with two loaded models; no eviction.

use std::sync::Arc;

use concerto_scenarios::{spawn_scenario, ScenarioConfig};
use futures::future::join_all;

#[tokio::test]
async fn concurrent_requests_for_distinct_models() {
    let server = Arc::new(
        spawn_scenario(
            ScenarioConfig::new(2, 48)
                .with_model("m1", 10)
                .with_model("m2", 10)
                .with_model("m3", 10)
                .with_model("m4", 10),
        )
        .await,
    );

    let futures: Vec<_> = ["m1", "m2", "m3", "m4"]
        .iter()
        .map(|id| {
            let server = server.clone();
            let id = (*id).to_string();
            async move { (id.clone(), server.post_chat(&id, "hi").await) }
        })
        .collect();

    for (id, resp) in join_all(futures).await {
        assert_eq!(resp.status(), 200, "model {id} failed");
    }

    assert_eq!(server.backend.launched_count(), 4);
    assert_eq!(server.backend.stopped_count(), 0);

    // 40 GB total (4 × 10 GB) split across two GPUs with no overcommit.
    let cluster = server.state.cluster.lock().await;
    let total_loaded: u64 = cluster.gpus.iter().map(|g| g.memory_used.as_u64()).sum();
    assert_eq!(total_loaded, bytesize::ByteSize::gb(40).as_u64());
    for gpu in &cluster.gpus {
        assert!(
            gpu.memory_used.as_u64() <= bytesize::ByteSize::gb(48).as_u64(),
            "GPU {} overcommitted: {}",
            gpu.id.0,
            gpu.memory_used
        );
    }
    drop(cluster);

    Arc::try_unwrap(server)
        .ok()
        .expect("no outstanding refs")
        .shutdown()
        .await;
}
