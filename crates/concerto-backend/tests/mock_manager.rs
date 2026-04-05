//! Integration tests for [`MockBackendManager`].
//!
//! These tests exercise the mock manager end-to-end through the
//! [`BackendManager`] trait without spawning any real processes.

use std::time::Duration;

use bytesize::ByteSize;
use concerto_backend::{BackendError, BackendManager, MockBackendManager, PortAllocator};
use concerto_core::{EngineType, GpuId, ModelId, ModelSpec};

fn spec(id: &str) -> ModelSpec {
    ModelSpec {
        id: ModelId::from(id),
        name: id.to_string(),
        weight_path: format!("/models/{id}"),
        vram_required: ByteSize::gb(4),
        engine: EngineType::Mock,
        engine_args: vec![],
        pin: false,
    }
}

#[tokio::test]
async fn launch_returns_handle_with_valid_port() {
    let manager = MockBackendManager::new();
    let handle = manager
        .launch(&spec("model-a"), GpuId(0))
        .await
        .expect("launch should succeed");

    assert!((8100..9000).contains(&handle.port));
    assert_eq!(handle.model_id, ModelId::from("model-a"));
    assert_eq!(handle.gpu_id, GpuId(0));
    assert!(handle.pid >= 10_000, "pid should be from mock pid range");
}

#[tokio::test]
async fn health_check_true_while_running_false_after_stop() {
    let manager = MockBackendManager::new();
    let handle = manager.launch(&spec("model-a"), GpuId(0)).await.unwrap();

    assert!(manager.health_check(&handle).await);

    manager.stop(&handle).await.unwrap();
    assert!(!manager.health_check(&handle).await);
}

#[tokio::test]
async fn multiple_launches_get_unique_ports_and_pids() {
    let manager = MockBackendManager::new();
    let a = manager.launch(&spec("a"), GpuId(0)).await.unwrap();
    let b = manager.launch(&spec("b"), GpuId(0)).await.unwrap();
    let c = manager.launch(&spec("c"), GpuId(1)).await.unwrap();

    assert_ne!(a.port, b.port);
    assert_ne!(a.port, c.port);
    assert_ne!(b.port, c.port);

    assert_ne!(a.pid, b.pid);
    assert_ne!(a.pid, c.pid);
    assert_ne!(b.pid, c.pid);
}

#[tokio::test]
async fn counters_track_launches_and_stops() {
    let manager = MockBackendManager::new();
    assert_eq!(manager.launched_count(), 0);
    assert_eq!(manager.stopped_count(), 0);

    let h1 = manager.launch(&spec("a"), GpuId(0)).await.unwrap();
    let h2 = manager.launch(&spec("b"), GpuId(0)).await.unwrap();
    assert_eq!(manager.launched_count(), 2);
    assert_eq!(manager.running_count().await, 2);

    manager.stop(&h1).await.unwrap();
    assert_eq!(manager.stopped_count(), 1);
    assert_eq!(manager.running_count().await, 1);

    manager.stop(&h2).await.unwrap();
    assert_eq!(manager.stopped_count(), 2);
    assert_eq!(manager.running_count().await, 0);
}

#[tokio::test]
async fn launch_failure_is_surfaced() {
    let manager = MockBackendManager::new().with_launch_failure();
    let err = manager
        .launch(&spec("a"), GpuId(0))
        .await
        .expect_err("launch should fail");
    assert!(matches!(err, BackendError::LaunchFailed(_)));
    assert_eq!(manager.launched_count(), 0);
    assert_eq!(manager.running_count().await, 0);
}

#[tokio::test]
async fn launch_failure_toggle_recovers() {
    let manager = MockBackendManager::new().with_launch_failure();
    assert!(manager.launch(&spec("a"), GpuId(0)).await.is_err());

    manager.set_launch_failure(false);
    let handle = manager
        .launch(&spec("a"), GpuId(0))
        .await
        .expect("launch should now succeed");
    assert_eq!(handle.model_id, ModelId::from("a"));
}

#[tokio::test]
async fn forced_health_check_failure_returns_false_for_running_handle() {
    let manager = MockBackendManager::new().with_health_check_failure();
    let handle = manager.launch(&spec("a"), GpuId(0)).await.unwrap();
    assert!(!manager.health_check(&handle).await);
}

#[tokio::test]
async fn stop_releases_port_for_reuse() {
    let manager = MockBackendManager::with_port_allocator(PortAllocator::with_range(8100..8102));
    let a = manager.launch(&spec("a"), GpuId(0)).await.unwrap();
    let b = manager.launch(&spec("b"), GpuId(0)).await.unwrap();
    assert!(manager.launch(&spec("c"), GpuId(0)).await.is_err());

    manager.stop(&a).await.unwrap();
    let c = manager
        .launch(&spec("c"), GpuId(0))
        .await
        .expect("should reuse freed port");
    assert_eq!(c.port, a.port);

    manager.stop(&b).await.unwrap();
    manager.stop(&c).await.unwrap();
}

#[tokio::test]
async fn launch_latency_is_honoured() {
    let manager = MockBackendManager::new().with_launch_latency(Duration::from_millis(50));
    let start = std::time::Instant::now();
    let _ = manager.launch(&spec("a"), GpuId(0)).await.unwrap();
    assert!(start.elapsed() >= Duration::from_millis(50));
}

#[tokio::test]
async fn stop_unknown_handle_is_noop() {
    let manager = MockBackendManager::new();
    // A handle that was never returned by `launch`.
    let fake = concerto_backend::BackendHandle {
        pid: 1,
        port: 8500,
        model_id: ModelId::from("ghost"),
        gpu_id: GpuId(0),
    };
    manager.stop(&fake).await.expect("stop should be a no-op");
    assert_eq!(manager.stopped_count(), 0);
}

#[tokio::test]
async fn no_free_port_error_when_range_exhausted() {
    let manager = MockBackendManager::with_port_allocator(PortAllocator::with_range(8100..8101));
    let _a = manager.launch(&spec("a"), GpuId(0)).await.unwrap();
    let err = manager
        .launch(&spec("b"), GpuId(0))
        .await
        .expect_err("should run out of ports");
    assert!(matches!(err, BackendError::NoFreePort));
}
