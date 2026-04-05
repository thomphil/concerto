//! ROADMAP §6.3 scenario 2: second request for the same model reuses the
//! existing backend. Asserts no new launch + request_count bump in the
//! cluster state.

use concerto_core::ModelId;
use concerto_scenarios::{spawn_scenario, ScenarioConfig};

#[tokio::test]
async fn cached_request_reuses_backend() {
    let server = spawn_scenario(ScenarioConfig::new(1, 24).with_model("model-a", 8)).await;

    let first = server.post_chat("model-a", "hi").await;
    assert_eq!(first.status(), 200);
    let second = server.post_chat("model-a", "again").await;
    assert_eq!(second.status(), 200);

    // Only one launch, no stops.
    assert_eq!(server.backend.launched_count(), 1);
    assert_eq!(server.backend.stopped_count(), 0);

    // The loaded model's request_count should reflect both touches.
    let cluster = server.state.cluster.lock().await;
    let loaded = cluster
        .gpus
        .iter()
        .flat_map(|g| g.loaded_models.iter())
        .find(|m| m.model_id == ModelId("model-a".into()))
        .expect("model is loaded");
    assert_eq!(loaded.request_count, 2);
    drop(cluster);

    server.shutdown().await;
}
