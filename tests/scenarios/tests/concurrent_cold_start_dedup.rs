//! ROADMAP §6.3 scenario 3 (§3 hard problem 1): ten concurrent requests
//! for a cold model must trigger exactly one launch. The remaining nine
//! subscribe to the in-flight load via the broadcast dedup channel.
//!
//! We force a window for the concurrency to observe by configuring the
//! `mock-inference-backend` binary with `--startup-delay-secs 2`, so the
//! first launch takes long enough that the other requests are guaranteed
//! to hit the dedup map before the launch commits.

use std::sync::Arc;

use concerto_scenarios::{spawn_scenario, ModelMockArgs, ScenarioConfig};
use futures::future::join_all;

#[tokio::test]
async fn concurrent_cold_start_deduplicates() {
    let cfg = ScenarioConfig::new(2, 24).with_model_args(
        "model-a",
        8,
        ModelMockArgs {
            startup_delay_secs: Some(2),
            ..Default::default()
        },
    );
    let server = Arc::new(spawn_scenario(cfg).await);

    let futures: Vec<_> = (0..10)
        .map(|i| {
            let server = server.clone();
            async move { server.post_chat("model-a", &format!("req {i}")).await }
        })
        .collect();

    let results = join_all(futures).await;
    for (i, resp) in results.iter().enumerate() {
        assert_eq!(resp.status(), 200, "request {i} failed: {:?}", resp);
    }

    // Exactly one launch — dedup must prevent duplicates even with a 2s
    // startup delay.
    assert_eq!(server.backend.launched_count(), 1);
    assert_eq!(server.backend.stopped_count(), 0);

    Arc::try_unwrap(server)
        .ok()
        .expect("no outstanding refs")
        .shutdown()
        .await;
}
