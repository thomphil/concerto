//! ROADMAP §6.3 scenario 6 (§3 hard problem 3): a backend that dies mid
//! request is surfaced to the client as a clean 502, the health loop
//! evicts it from cluster state, and a subsequent request cold-starts a
//! fresh backend.
//!
//! We configure `mock-inference-backend` with `--crash-after 2` so:
//! 1. Request #1 — count=1, below limit → 200 OK (launch #1)
//! 2. Request #2 — count=2, hits limit → mock-backend calls
//!    `std::process::exit(1)` before responding → concerto sees the
//!    connection drop and returns 502.
//! 3. Health loop ticks (interval = 1s), notices the dead backend, and
//!    removes it from `state.backends` + `ClusterState`.
//! 4. Request #3 — cold start, launch #2, succeeds.

use std::time::Duration;

use concerto_scenarios::{spawn_scenario, ModelMockArgs, ScenarioConfig};

#[tokio::test]
async fn backend_crash_triggers_relaunch_on_next_request() {
    let cfg = ScenarioConfig::new(1, 24).with_model_args(
        "model-a",
        8,
        ModelMockArgs {
            crash_after: Some(2),
            ..Default::default()
        },
    );
    let server = spawn_scenario(cfg).await;

    // 1. Normal cold start.
    let first = server.post_chat("model-a", "hello").await;
    assert_eq!(first.status(), 200);
    assert_eq!(server.backend.launched_count(), 1);

    // 2. Crash request — backend exits before responding, concerto returns
    //    a clean 502 rather than a hung connection.
    let crash = server.post_chat("model-a", "die").await;
    assert_eq!(crash.status(), 502, "expected clean 502 on backend crash");

    // 3. Wait for a couple of health-loop ticks so the dead backend is
    //    evicted from state.backends + ClusterState.
    tokio::time::sleep(Duration::from_millis(2500)).await;

    {
        let cluster = server.state.cluster.lock().await;
        let any_loaded = cluster.gpus.iter().any(|g| !g.loaded_models.is_empty());
        assert!(
            !any_loaded,
            "health loop should have evicted the crashed backend"
        );
    }
    assert!(server.state.backends.lock().await.is_empty());

    // 4. Next request — fresh cold start, second launch.
    let recovered = server.post_chat("model-a", "again").await;
    assert_eq!(recovered.status(), 200);
    assert_eq!(server.backend.launched_count(), 2);

    server.shutdown().await;
}
