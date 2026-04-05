//! In-process integration tests for the mock inference backend.
//!
//! Each test spins the axum router up on a random OS-assigned port, talks to
//! it with `reqwest`, and shuts it down afterwards.

use std::net::{Ipv4Addr, SocketAddr};

use mock_inference_backend::{build_router, AppState, Args};
use serde_json::{json, Value};
use tokio::net::TcpListener;

/// Spawn a server on a random high port and return its base URL.
async fn spawn_test_server(args: Args) -> String {
    // Bind to port 0 so the OS picks a free port for us.
    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind random port");
    let addr = listener.local_addr().expect("local_addr");

    let state = AppState::from_args(&args);
    let app = build_router(state);

    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });

    // Yield once so the spawned task has a chance to start polling.
    tokio::task::yield_now().await;

    format!("http://{addr}")
}

fn test_args() -> Args {
    Args {
        port: 0,
        host: "127.0.0.1".to_string(),
        startup_delay_secs: 0,
        response_latency_ms: 0,
        fail_probability: 0.0,
        crash_after: None,
    }
}

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let base = spawn_test_server(test_args()).await;

    let resp = reqwest::get(format!("{base}/health"))
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn chat_completions_non_streaming_returns_valid_response() {
    let base = spawn_test_server(test_args()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "test-model",
            "messages": [{ "role": "user", "content": "hello" }]
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["object"], "chat.completion");
    assert!(body["choices"].is_array());
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("content is string");
    assert!(!content.is_empty());
}

#[tokio::test]
async fn chat_completions_streaming_returns_sse_chunks() {
    let base = spawn_test_server(test_args()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "stream-model",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": true
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get("content-type")
        .expect("content-type header")
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        content_type.starts_with("text/event-stream"),
        "unexpected content-type: {content_type}"
    );

    let body = resp.text().await.expect("body");
    let data_lines: Vec<&str> = body.lines().filter(|l| l.starts_with("data:")).collect();
    assert!(
        data_lines.len() >= 5,
        "expected several data lines, got {}",
        data_lines.len()
    );
    assert!(
        body.contains("[DONE]"),
        "stream should terminate with [DONE] marker"
    );
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_text() {
    let base = spawn_test_server(test_args()).await;

    let resp = reqwest::get(format!("{base}/metrics"))
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get("content-type")
        .expect("content-type")
        .to_str()
        .unwrap()
        .to_string();
    assert!(content_type.starts_with("text/plain"));

    let body = resp.text().await.expect("body");
    assert!(body.contains("mock_backend_requests_total"));
    assert!(body.contains("mock_backend_memory_bytes"));
    assert!(body.contains("mock_backend_up"));
}

#[tokio::test]
async fn request_counter_increments_across_calls() {
    let base = spawn_test_server(test_args()).await;

    let client = reqwest::Client::new();
    for _ in 0..3 {
        let resp = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&json!({
                "model": "counter-model",
                "messages": [{ "role": "user", "content": "hi" }]
            }))
            .send()
            .await
            .expect("request");
        assert_eq!(resp.status(), 200);
    }

    let metrics_body = reqwest::get(format!("{base}/metrics"))
        .await
        .expect("metrics request")
        .text()
        .await
        .expect("metrics body");

    // The counter should reflect 3 chat completion calls; the metrics endpoint
    // itself does not increment this counter.
    let line = metrics_body
        .lines()
        .find(|l| l.starts_with("mock_backend_requests_total "))
        .expect("requests_total line present");
    let value: usize = line
        .split_whitespace()
        .nth(1)
        .expect("metric value")
        .parse()
        .expect("parse metric");
    assert_eq!(value, 3);
}

#[tokio::test]
async fn fail_probability_one_always_returns_500() {
    let mut args = test_args();
    args.fail_probability = 1.0;
    let base = spawn_test_server(args).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "fail",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 500);
}
