//! `GET /status` — human-oriented JSON dump of current cluster state.
//!
//! This is not part of the OpenAI-compatible surface; it's a local operator
//! aid. A human running `curl localhost:8000/status | jq` should be able to
//! see which models are loaded where, which GPUs are healthy, and how much
//! VRAM is in use.

use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::AppState;

pub async fn status(State(state): State<AppState>) -> Json<Value> {
    // Clone the cluster under a short critical section so we never serialise
    // while holding the mutex.
    let cluster = {
        let guard = state.cluster.lock().await;
        guard.clone()
    };

    let gpus: Vec<Value> = cluster
        .gpus
        .iter()
        .map(|g| {
            json!({
                "id": g.id.0,
                "memory_total": g.memory_total.to_string(),
                "memory_used": g.memory_used.to_string(),
                "memory_available": g.memory_available.to_string(),
                "temperature_celsius": g.temperature_celsius,
                "utilisation_percent": g.utilisation_percent,
                "health": format!("{:?}", g.health),
                "loaded_models": g
                    .loaded_models
                    .iter()
                    .map(|m| json!({
                        "model_id": m.model_id.0,
                        "vram_usage": m.vram_usage.to_string(),
                        "last_request_at": m.last_request_at.to_rfc3339(),
                        "request_count": m.request_count,
                        "backend_port": m.backend_port,
                    }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();

    Json(json!({
        "gpus": gpus,
        "registry_size": cluster.model_registry.len(),
    }))
}
