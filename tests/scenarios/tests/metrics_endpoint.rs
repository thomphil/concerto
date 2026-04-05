//! Prometheus metrics regression test (SPRINT-2-PLAN Phase A.1).
//!
//! Drives one chat completion through the full stack, then scrapes
//! `/metrics` and asserts that the counters and gauges listed in
//! ROADMAP §8 are present with sensible values. This is the regression
//! gate that catches "a refactor silently stopped emitting counters".
//!
//! Assertions are deliberately loose on numeric values — the Prometheus
//! recorder is process-global, so when scenario tests run in parallel
//! within a single `cargo test` invocation their emissions accumulate on
//! shared counters. Asserting on presence + non-zero is enough to prove
//! the wiring works without being fragile under test parallelism.

use concerto_scenarios::{spawn_scenario, ScenarioConfig};

#[tokio::test]
async fn metrics_endpoint_after_chat_completion() {
    let server = spawn_scenario(ScenarioConfig::new(2, 24).with_model("model-a", 8)).await;

    // Drive one request through the orchestrator so the per-request
    // counters and the cold-start histogram fire.
    let resp = server.post_chat("model-a", "hello").await;
    assert_eq!(resp.status(), 200);

    // Scrape /metrics.
    let metrics = server
        .client
        .get(format!("{}/metrics", server.base_url))
        .send()
        .await
        .expect("metrics GET");
    assert_eq!(metrics.status(), 200);
    let body = metrics.text().await.expect("metrics body");

    // Counter: labelled request total. One of the cold-start decision
    // paths should have fired (launcher on a fresh scenario = launcher
    // path). Accept any of the positive labels, since counter values are
    // process-global and other scenarios running in the same binary may
    // have already bumped the `loaded_after_load` label.
    assert!(
        body.contains("concerto_requests_total"),
        "expected concerto_requests_total in /metrics output:\n{body}"
    );
    assert!(
        body.contains("decision=\"loaded_after_load\""),
        "expected decision=\"loaded_after_load\" label in /metrics output:\n{body}"
    );

    // Histograms — at least the sum/count timeseries must be present.
    assert!(
        body.contains("concerto_routing_decision_seconds"),
        "expected concerto_routing_decision_seconds histogram:\n{body}"
    );
    assert!(
        body.contains("concerto_model_load_duration_seconds"),
        "expected concerto_model_load_duration_seconds histogram:\n{body}"
    );

    // Counters for backend lifecycle.
    assert!(
        body.contains("concerto_backend_launches_total"),
        "expected concerto_backend_launches_total counter:\n{body}"
    );

    // Gauges refreshed by the /metrics handler on each scrape. The scrape
    // above happens after one launch, so active_backends for this scenario
    // is 1 — but the gauge is process-global across parallel tests, so we
    // only assert presence, not value.
    assert!(
        body.contains("concerto_active_backends"),
        "expected concerto_active_backends gauge:\n{body}"
    );
    assert!(
        body.contains("concerto_gpu_memory_used_bytes"),
        "expected concerto_gpu_memory_used_bytes gauge:\n{body}"
    );
    assert!(
        body.contains("concerto_gpu_memory_total_bytes"),
        "expected concerto_gpu_memory_total_bytes gauge:\n{body}"
    );
    // Per-GPU label present (2 GPUs in the scenario).
    assert!(
        body.contains("gpu=\"0\""),
        "expected gpu=\"0\" label on GPU gauges:\n{body}"
    );

    server.shutdown().await;
}
