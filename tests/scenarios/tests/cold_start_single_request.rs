//! ROADMAP §6.3 scenario 1: one request for a cold model.
//!
//! Drives the full stack end-to-end: HTTP → orchestrator → `ProcessBackendManager`
//! → spawned `mock-inference-backend` child → reverse proxy back to client.

use concerto_scenarios::{chat_json, spawn_scenario, ScenarioConfig};

#[tokio::test]
async fn cold_start_single_request() {
    let server = spawn_scenario(ScenarioConfig::new(2, 24).with_model("model-a", 8)).await;

    let resp = server.post_chat("model-a", "hello").await;
    assert_eq!(resp.status(), 200);
    let body = chat_json(resp).await;

    // Real OpenAI-shaped response from the mock inference backend.
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "model-a");
    assert!(body["choices"][0]["message"]["content"].is_string());

    // Exactly one backend launch, none stopped.
    assert_eq!(server.backend.launched_count(), 1);
    assert_eq!(server.backend.stopped_count(), 0);

    server.shutdown().await;
}
