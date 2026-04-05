//! ROADMAP §6.3 scenario 7: `stream: true` requests return SSE chunks in
//! order and terminate with `[DONE]`.
//!
//! This scenario was deferred in PR #8 because the mock-backend-manager-
//! only harness couldn't serve real SSE. With the real
//! `mock-inference-backend` child process behind a real
//! `ProcessBackendManager`, the whole streaming pipeline is exercised:
//!
//! client → concerto `/v1/chat/completions` → reqwest byte stream →
//! axum `Body::from_stream` → back to the client as SSE chunks.
//!
//! We don't try to parse individual JSON deltas — only verify the
//! event-stream shape (each chunk starts with `data: `) and the terminal
//! `data: [DONE]` marker.

use concerto_scenarios::{spawn_scenario, ScenarioConfig};
use futures::StreamExt;

#[tokio::test]
async fn streaming_response_forwards_sse_chunks() {
    let server = spawn_scenario(ScenarioConfig::new(1, 24).with_model("model-a", 8)).await;

    let resp = server.post_chat_stream("model-a", "hi").await;
    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        content_type.contains("text/event-stream"),
        "expected text/event-stream, got {content_type}"
    );

    // Collect the full body and split into SSE events.
    let mut stream = resp.bytes_stream();
    let mut collected = Vec::new();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.expect("upstream chunk");
        collected.extend_from_slice(&bytes);
    }
    let body = String::from_utf8(collected).expect("UTF-8 SSE body");

    // At minimum we expect several `data: { ... }` JSON chunks and a
    // trailing `data: [DONE]` terminator.
    let data_lines: Vec<&str> = body.lines().filter(|l| l.starts_with("data: ")).collect();
    assert!(
        data_lines.len() >= 3,
        "expected at least 3 SSE data events, got {}: {body:?}",
        data_lines.len()
    );
    assert!(
        data_lines.iter().any(|l| l.trim() == "data: [DONE]"),
        "expected a terminating [DONE] event in stream:\n{body}"
    );

    server.shutdown().await;
}
