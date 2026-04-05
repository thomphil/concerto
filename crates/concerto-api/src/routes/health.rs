//! Liveness and readiness probes.
//!
//! - `GET /health` — always 200 while the process is accepting connections.
//!   Deliberately dumb so a stalled orchestrator still reports liveness
//!   (readiness is the probe that actually cares about state).
//! - `GET /ready` — 200 iff the GPU monitor is reachable and at least one
//!   GPU classifies as Healthy or Degraded using the configured thresholds.

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use concerto_core::GpuHealth;
use concerto_gpu::{classify_health, HealthThresholds};
use serde_json::json;

use crate::AppState;

pub async fn liveness() -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

pub async fn readiness(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    let snapshots = state.gpu.snapshot().await;
    if snapshots.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "unready", "reason": "no GPUs visible" })),
        );
    }

    let thresholds = HealthThresholds {
        max_healthy_temperature: state.config.routing.max_healthy_temperature,
        max_degraded_temperature: state.config.routing.max_degraded_temperature,
        max_tolerated_ecc: 0,
    };

    let any_usable = snapshots
        .iter()
        .any(|s| !matches!(classify_health(s, &thresholds), GpuHealth::Unhealthy));

    if any_usable {
        (
            StatusCode::OK,
            Json(json!({ "status": "ready", "gpus": snapshots.len() })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "unready", "reason": "all GPUs unhealthy" })),
        )
    }
}
