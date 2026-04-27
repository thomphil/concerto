//! Sprint 3 A.4: `routing.request_timeout_secs` enforcement.
//!
//! Two scenarios in this file:
//!
//! 1. **Positive** — non-streaming completion against a slow backend exceeds
//!    the configured budget; concerto returns 504 Gateway Timeout with the
//!    canonical `ErrorBody` envelope (kind = `"request_timeout"`).
//! 2. **Negative** — streaming completion under the same configuration
//!    completes successfully even though the upstream takes longer than the
//!    timeout, because the middleware bounds the response *future* (the
//!    handler returns a streaming body quickly), not the body lifetime.

use std::time::{Duration, Instant};

use concerto_scenarios::{spawn_scenario, ModelMockArgs, ScenarioConfig};

#[tokio::test]
async fn request_timeout_returns_504_for_slow_non_streaming() {
    let server = spawn_scenario(
        ScenarioConfig::new(1, 24)
            .with_model_args(
                "model-a",
                8,
                ModelMockArgs {
                    response_latency_ms: Some(5_000),
                    ..ModelMockArgs::default()
                },
            )
            .with_request_timeout_secs(1),
    )
    .await;

    let started = Instant::now();
    let resp = server.post_chat("model-a", "hello").await;
    let elapsed = started.elapsed();

    assert_eq!(resp.status(), 504, "expected 504 Gateway Timeout");

    // The middleware should fire ~1s in, well before the 5s upstream latency.
    assert!(
        elapsed < Duration::from_millis(2_500),
        "response should arrive within ~timeout + slack, took {elapsed:?}"
    );
    assert!(
        elapsed >= Duration::from_millis(950),
        "response should not return before the configured timeout, took {elapsed:?}"
    );

    let body: serde_json::Value = resp.json().await.expect("error body parses as JSON");
    assert_eq!(
        body.get("kind").and_then(|v| v.as_str()),
        Some("request_timeout"),
        "error envelope kind should be request_timeout, got {body:?}"
    );

    server.shutdown().await;
}

#[tokio::test]
async fn request_timeout_does_not_apply_to_streaming() {
    let server = spawn_scenario(
        ScenarioConfig::new(1, 24)
            .with_model_args(
                "model-a",
                8,
                // Headers must come back fast (well within the 1s budget)
                // and the SSE body must outlive it. canned streaming has
                // ~5 chunks; 400ms each = ~2s total body time.
                ModelMockArgs {
                    response_latency_ms: Some(0),
                    stream_chunk_delay_ms: Some(400),
                    ..ModelMockArgs::default()
                },
            )
            .with_request_timeout_secs(1),
    )
    .await;

    let started = Instant::now();
    let resp = server.post_chat_stream("model-a", "hi").await;
    assert_eq!(
        resp.status(),
        200,
        "streaming should succeed even though body takes longer than request_timeout_secs"
    );

    // Drain the stream to make sure the body is delivered cleanly. We don't
    // care about contents — only that no error short-circuits the body and
    // that draining genuinely outlives the timeout (proving body-time was
    // not bounded by the middleware).
    let bytes = resp
        .bytes()
        .await
        .expect("streaming body should drain without error");
    let elapsed = started.elapsed();
    assert!(!bytes.is_empty(), "streaming body should not be empty");
    assert!(
        elapsed > Duration::from_millis(1_200),
        "stream should have taken longer than the 1s request_timeout_secs (took {elapsed:?})"
    );

    server.shutdown().await;
}
