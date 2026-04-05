//! `GET /metrics` — Prometheus text-format scrape endpoint (ROADMAP §8).
//!
//! Refreshes gauge state (active backends, GPU memory) immediately before
//! rendering so each scrape reflects the current cluster snapshot rather
//! than the last event-driven write. Counters and histograms are emitted
//! inline at their call sites and don't need a refresh pass here.
//!
//! The endpoint is unauthenticated in v0.1 — operators who need to
//! restrict scrape access should put concerto behind a reverse proxy as
//! documented in `docs/deployment.md`.

use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::IntoResponse;

use crate::metrics as metrics_mod;
use crate::AppState;

/// Handler for `GET /metrics`. Returns Prometheus text-format exposition.
pub async fn scrape(State(state): State<AppState>) -> impl IntoResponse {
    metrics_mod::refresh_state_gauges(&state).await;
    let body = state.prometheus.render();
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        body,
    )
}
