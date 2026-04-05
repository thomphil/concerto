//! HTTP route definitions and the top-level [`router`] builder.

pub mod chat;
pub mod health;
pub mod models;
pub mod status;

use axum::routing::{get, post};
use axum::Router;

use crate::AppState;

/// Build the axum router with every route mounted and state attached.
///
/// Middleware layers (tracing, CORS, optional timeout, future auth/rate-limit
/// seam for open-core extensions per ROADMAP §6.4) are installed in
/// [`crate::server::serve`] — keeping them there means this function stays
/// plain enough to use directly from integration tests that want to serve
/// the router without the full serve loop.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::liveness))
        .route("/ready", get(health::readiness))
        .route("/status", get(status::status))
        .route("/v1/models", get(models::list))
        .route("/v1/chat/completions", post(chat::completions))
        .with_state(state)
}
