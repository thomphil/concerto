//! ROADMAP §6.3 scenario 3 (ROADMAP §3 hard problem 1): ten concurrent
//! requests for a cold model must trigger exactly one launch. The other
//! nine requests subscribe to the in-flight load via the broadcast dedup
//! channel and receive the same handle.

use std::sync::Arc;
use std::time::Duration;

use concerto_api::orchestrator::route_and_dispatch;
use concerto_core::ModelId;
use concerto_scenarios::{build_harness, ScenarioConfig};
use futures::future::join_all;

#[tokio::test]
async fn concurrent_cold_start_deduplicates() {
    let h = build_harness(ScenarioConfig {
        gpu_count: 2,
        memory_per_gpu_gb: 24,
        models: vec![("model-a".into(), 8)],
    })
    .await;

    // Inject latency on launch so we're guaranteed to observe concurrency —
    // without it the first launch may finish before later requests even
    // reach the dedup lookup.
    h.backend.set_launch_latency(Duration::from_millis(150));

    let state = Arc::new(h.state);

    let futures: Vec<_> = (0..10)
        .map(|_| {
            let state = state.clone();
            async move { route_and_dispatch(&state, ModelId("model-a".into())).await }
        })
        .collect();

    let results = join_all(futures).await;
    let successes: Vec<_> = results.into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(
        successes.len(),
        10,
        "all 10 concurrent requests should succeed"
    );

    // Exactly one launch regardless of concurrency.
    assert_eq!(
        h.backend.launched_count(),
        1,
        "dedup must prevent duplicate launches"
    );
    assert_eq!(h.backend.stopped_count(), 0);

    // Every success should have resolved to the same target port.
    let first_port = successes[0].port;
    for target in &successes {
        assert_eq!(target.port, first_port);
    }
}
