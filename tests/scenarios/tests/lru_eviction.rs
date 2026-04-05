//! ROADMAP §6.3 scenario 4: 2 GPUs × 24 GB, three 14 GB models. Loading
//! the third model forces LRU eviction of the least-recently-used one.

use concerto_core::ModelId;
use concerto_scenarios::{spawn_scenario, ScenarioConfig};

#[tokio::test]
async fn lru_eviction_when_memory_full() {
    let server = spawn_scenario(
        ScenarioConfig::new(2, 24)
            .with_model("alpha", 14)
            .with_model("beta", 14)
            .with_model("gamma", 14),
    )
    .await;

    // Load alpha + beta — one on each GPU (a 14 GB model + 1 GB headroom
    // leaves no room for a second 14 GB model on the same 24 GB GPU).
    assert_eq!(server.post_chat("alpha", "hi").await.status(), 200);
    // Give `last_request_at` a noticeable gap so beta is strictly newer
    // than alpha regardless of scheduler jitter.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(server.post_chat("beta", "hi").await.status(), 200);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    assert_eq!(server.backend.launched_count(), 2);
    assert_eq!(server.backend.stopped_count(), 0);

    // Load gamma — LRU must evict alpha (the older of the two) to make room.
    assert_eq!(server.post_chat("gamma", "hi").await.status(), 200);

    assert_eq!(server.backend.launched_count(), 3);
    assert_eq!(server.backend.stopped_count(), 1);

    // Verify the evicted one is alpha by inspecting the live backends map.
    let backends = server.state.backends.lock().await;
    assert!(backends.contains_key(&ModelId("beta".into())));
    assert!(backends.contains_key(&ModelId("gamma".into())));
    assert!(
        !backends.contains_key(&ModelId("alpha".into())),
        "alpha should have been evicted as LRU candidate"
    );
    drop(backends);

    server.shutdown().await;
}
