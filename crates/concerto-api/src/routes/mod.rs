//! HTTP route definitions and the top-level [`router`] builder.

pub mod chat;
pub mod health;
pub mod metrics;
pub mod models;
pub mod status;

use axum::routing::{get, post};
use axum::Router;

use crate::middleware::request_timeout;
use crate::AppState;

/// Build the axum router with every route mounted and state attached.
///
/// Cross-cutting middleware (tracing, CORS, future auth/rate-limit seam for
/// open-core extensions per ROADMAP §6.4) is installed in
/// [`crate::server::serve`]. Per-route middleware that depends on shared
/// state — currently the [`request_timeout`] layer applied to chat
/// completions — is wired in here so the merged sub-router stays the unit
/// of test against `routes::router(state)`.
pub fn router(state: AppState) -> Router {
    let chat_routes = Router::new()
        .route("/v1/chat/completions", post(chat::completions))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            request_timeout,
        ));

    Router::new()
        .route("/health", get(health::liveness))
        .route("/ready", get(health::readiness))
        .route("/status", get(status::status))
        .route("/metrics", get(metrics::scrape))
        .route("/v1/models", get(models::list))
        .merge(chat_routes)
        .with_state(state)
}
