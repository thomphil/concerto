//! ROADMAP §6.3 scenario 10: when every GPU is marked Unhealthy, requests
//! are rejected with 503 rather than attempting to place anywhere.

use concerto_core::GpuHealth;
use concerto_scenarios::{spawn_scenario, ScenarioConfig};

#[tokio::test]
async fn all_gpus_unhealthy_rejects_with_503() {
    let server = spawn_scenario(ScenarioConfig::new(2, 24).with_model("model-a", 8)).await;

    // Mark every GPU in the cluster snapshot Unhealthy. The cluster
    // snapshot drives the routing decision; we mutate it directly so the
    // next POST sees an unhealthy fleet without waiting for a health-loop
    // tick.
    {
        let mut cluster = server.state.cluster.lock().await;
        for gpu in &mut cluster.gpus {
            gpu.health = GpuHealth::Unhealthy;
        }
    }

    let resp = server.post_chat("model-a", "hi").await;
    assert_eq!(resp.status(), 503);

    // No backend should have been launched.
    assert_eq!(server.backend.launched_count(), 0);

    server.shutdown().await;
}
