//! ROADMAP §6.3 scenario 4: 2 GPUs × 24 GB, three 14 GB models. Loading the
//! third model forces LRU eviction of the least-recently-used one — exactly
//! one stop, exactly three launches.

use concerto_api::orchestrator::route_and_dispatch;
use concerto_core::ModelId;
use concerto_scenarios::{build_harness, ScenarioConfig};

#[tokio::test]
async fn lru_eviction_when_memory_full() {
    let h = build_harness(ScenarioConfig {
        gpu_count: 2,
        memory_per_gpu_gb: 24,
        models: vec![
            ("alpha".into(), 14),
            ("beta".into(), 14),
            ("gamma".into(), 14),
        ],
    })
    .await;

    // Load alpha + beta — one per GPU (each GPU only has room for one 14 GB
    // model plus the 1 GB headroom).
    route_and_dispatch(&h.state, ModelId("alpha".into()))
        .await
        .expect("alpha loads");
    // Nudge last_request_at forward so beta is strictly newer than alpha.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    route_and_dispatch(&h.state, ModelId("beta".into()))
        .await
        .expect("beta loads");
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    assert_eq!(h.backend.launched_count(), 2);
    assert_eq!(h.backend.stopped_count(), 0);

    // Load gamma — LRU must evict alpha (the older of the two) to make room.
    route_and_dispatch(&h.state, ModelId("gamma".into()))
        .await
        .expect("gamma loads after eviction");

    assert_eq!(h.backend.launched_count(), 3, "three total launches");
    assert_eq!(h.backend.stopped_count(), 1, "exactly one eviction stop");

    // Verify the evicted one is alpha (LRU) by inspecting state.backends.
    let backends = h.state.backends.lock().await;
    assert!(backends.contains_key(&ModelId("beta".into())));
    assert!(backends.contains_key(&ModelId("gamma".into())));
    assert!(
        !backends.contains_key(&ModelId("alpha".into())),
        "alpha should have been evicted as LRU candidate"
    );
}
