//! Integration tests for [`MockGpuMonitor`] and [`classify_health`].

use bytesize::ByteSize;
use concerto_core::{GpuHealth, GpuId};
use concerto_gpu::{classify_health, GpuMonitor, GpuSnapshot, HealthThresholds, MockGpuMonitor};

fn sample_snapshot(id: usize) -> GpuSnapshot {
    GpuSnapshot {
        id: GpuId(id),
        memory_total: ByteSize::gb(24),
        memory_used: ByteSize::b(0),
        temperature_celsius: 40,
        utilisation_percent: 0,
        ecc_errors_uncorrected: 0,
    }
}

#[tokio::test]
async fn with_healthy_gpus_creates_expected_count_and_memory() {
    let monitor = MockGpuMonitor::with_healthy_gpus(2, 24);
    assert_eq!(monitor.gpu_count(), 2);

    let snaps = monitor.snapshot().await;
    assert_eq!(snaps.len(), 2);
    assert_eq!(snaps[0].id, GpuId(0));
    assert_eq!(snaps[1].id, GpuId(1));
    for snap in &snaps {
        assert_eq!(snap.memory_total, ByteSize::gb(24));
        assert_eq!(snap.memory_used, ByteSize::b(0));
        assert_eq!(snap.temperature_celsius, 40);
        assert_eq!(snap.ecc_errors_uncorrected, 0);
    }
}

#[tokio::test]
async fn new_preserves_explicit_snapshots() {
    let snaps = vec![sample_snapshot(0), sample_snapshot(1), sample_snapshot(2)];
    let monitor = MockGpuMonitor::new(snaps.clone());
    assert_eq!(monitor.gpu_count(), 3);
    assert_eq!(monitor.snapshot().await, snaps);
}

#[tokio::test]
async fn set_memory_used_is_reflected_in_next_snapshot() {
    let monitor = MockGpuMonitor::with_healthy_gpus(2, 24);
    monitor.set_memory_used(GpuId(1), ByteSize::gb(10)).await;

    let snaps = monitor.snapshot().await;
    assert_eq!(snaps[0].memory_used, ByteSize::b(0));
    assert_eq!(snaps[1].memory_used, ByteSize::gb(10));
}

#[tokio::test]
async fn set_temperature_is_reflected_in_next_snapshot() {
    let monitor = MockGpuMonitor::with_healthy_gpus(1, 24);
    monitor.set_temperature(GpuId(0), 88).await;

    let snaps = monitor.snapshot().await;
    assert_eq!(snaps[0].temperature_celsius, 88);
}

#[tokio::test]
async fn inject_ecc_error_increments_counter() {
    let monitor = MockGpuMonitor::with_healthy_gpus(1, 24);
    monitor.inject_ecc_error(GpuId(0)).await;
    monitor.inject_ecc_error(GpuId(0)).await;

    let snaps = monitor.snapshot().await;
    assert_eq!(snaps[0].ecc_errors_uncorrected, 2);
}

#[tokio::test]
async fn remove_gpu_simulates_disappearance() {
    let monitor = MockGpuMonitor::with_healthy_gpus(3, 24);
    assert_eq!(monitor.gpu_count(), 3);

    monitor.remove_gpu(GpuId(1)).await;

    assert_eq!(monitor.gpu_count(), 2);
    let snaps = monitor.snapshot().await;
    let ids: Vec<GpuId> = snaps.iter().map(|s| s.id).collect();
    assert_eq!(ids, vec![GpuId(0), GpuId(2)]);
}

#[tokio::test]
async fn mutators_on_missing_gpu_are_noops() {
    let monitor = MockGpuMonitor::with_healthy_gpus(1, 24);
    // GpuId(7) does not exist — this should not panic or mutate anything.
    monitor.set_temperature(GpuId(7), 99).await;
    monitor.set_memory_used(GpuId(7), ByteSize::gb(5)).await;
    monitor.inject_ecc_error(GpuId(7)).await;

    let snaps = monitor.snapshot().await;
    assert_eq!(snaps.len(), 1);
    assert_eq!(snaps[0].temperature_celsius, 40);
    assert_eq!(snaps[0].memory_used, ByteSize::b(0));
    assert_eq!(snaps[0].ecc_errors_uncorrected, 0);
}

#[tokio::test]
async fn mock_monitor_clones_share_state() {
    let monitor = MockGpuMonitor::with_healthy_gpus(1, 24);
    let handle = monitor.clone();
    handle.set_temperature(GpuId(0), 70).await;

    let snaps = monitor.snapshot().await;
    assert_eq!(snaps[0].temperature_celsius, 70);
}

#[test]
fn classify_health_healthy_under_all_thresholds() {
    let thresholds = HealthThresholds::default();
    let snap = sample_snapshot(0);
    assert_eq!(classify_health(&snap, &thresholds), GpuHealth::Healthy);
}

#[test]
fn classify_health_degraded_on_high_temperature() {
    let thresholds = HealthThresholds::default();
    let mut snap = sample_snapshot(0);
    snap.temperature_celsius = 80; // > 75 healthy cap, <= 85 degraded cap
    assert_eq!(classify_health(&snap, &thresholds), GpuHealth::Degraded);
}

#[test]
fn classify_health_unhealthy_on_very_high_temperature() {
    let thresholds = HealthThresholds::default();
    let mut snap = sample_snapshot(0);
    snap.temperature_celsius = 90; // > 85 degraded cap
    assert_eq!(classify_health(&snap, &thresholds), GpuHealth::Unhealthy);
}

#[test]
fn classify_health_unhealthy_on_ecc_errors() {
    let thresholds = HealthThresholds::default();
    let mut snap = sample_snapshot(0);
    snap.ecc_errors_uncorrected = 1;
    assert_eq!(classify_health(&snap, &thresholds), GpuHealth::Unhealthy);
}

#[test]
fn classify_health_respects_custom_thresholds() {
    let thresholds = HealthThresholds {
        max_healthy_temperature: 60,
        max_degraded_temperature: 70,
        max_tolerated_ecc: 10,
    };
    let mut snap = sample_snapshot(0);

    snap.temperature_celsius = 55;
    assert_eq!(classify_health(&snap, &thresholds), GpuHealth::Healthy);

    snap.temperature_celsius = 65;
    assert_eq!(classify_health(&snap, &thresholds), GpuHealth::Degraded);

    snap.temperature_celsius = 50;
    snap.ecc_errors_uncorrected = 10; // right at the limit — still tolerated
    assert_eq!(classify_health(&snap, &thresholds), GpuHealth::Healthy);

    snap.ecc_errors_uncorrected = 11;
    assert_eq!(classify_health(&snap, &thresholds), GpuHealth::Unhealthy);
}
